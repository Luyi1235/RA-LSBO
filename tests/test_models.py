from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

from ra_lsbo import HIDDEN_MODES, normalize_evaluation


HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    import numpy as np
    import torch
    from ra_lsbo.features import LatentFeatures
    from ra_lsbo.risk_model import BinaryRiskHead, RiskHeads
    from ra_lsbo.responses import ResponseHeads
    from ra_lsbo.replay import prequential, summarize

# Portable test fixture; production algorithms and historical source records stay unchanged.
BASE = (Path(__file__).resolve().parents[1] /
        "runs/compact_mda_20261021T081006Z_20eb608d/qualification_recovered_20261025/frozen_base")
VARIABLES = [
    {"name": "layout", "kind": "nominal", "default": "single", "values": ["single", "distributed"]},
    {"name": "count", "kind": "ordinal", "default": 2, "values": [2, 4],
     "active_when": {"layout": ["distributed"]}},
    {"name": "x", "kind": "continuous", "default": .5, "bounds": [0., 1.]},
    {"name": "spacing", "kind": "continuous", "default": .5, "bounds": [0., 1.],
     "active_when": {"layout": ["distributed"]}},
]


def history(count=10):
    rows = []
    for i in range(1, count+1):
        failed = i in (3, 7, 9)
        x = i/12
        raw = {
            "objective": None if failed else 100 + 20*x,
            "solver_success": not failed,
            "constraints": {"capacity": x-.65, "span_min": -1.},
            "status": "mission_not_converged" if failed else
                      "constraint_violation" if x > .65 else "ok",
        }
        if i == 7:
            raw["constraint_observed"] = {"capacity": True}
        rows.append({
            "evaluation": i, "phase": "initial",
            "candidate": {"continuous": {"x": x, "spacing": .4},
                          "discrete": {"layout": "distributed" if i % 2 == 0 else "single", "count": 2}},
            "risk": normalize_evaluation(raw, ["capacity", "span_min"], ["span_min"]).to_dict(),
            "source_journal_sha256": "test", "evaluator_contract_sha256": "test",
        })
    return rows


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env for model tests")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def features(self, rows):
        return LatentFeatures(BASE, VARIABLES, seed=3).fit(rows, epochs=1)

    def test_prior_only_is_explicit_and_not_zero_or_one(self):
        for labels in ([], [0, 0, 0], [1, 1, 1], [0, 1]):
            head = BinaryRiskHead(np.zeros((len(labels), 2)), labels, min_per_class=3)
            self.assertTrue(head.diagnostics["status"].startswith("prior_only"))
            probability = head.predict(np.ones((2, 2)))
            self.assertTrue(np.all((probability > 0) & (probability < 1)))
            self.assertFalse(head.diagnostics["calibrated"])

    def test_two_class_forest_is_deterministic(self):
        x = np.arange(12.).reshape(6, 2)
        a = BinaryRiskHead(x, [0, 1]*3, seed=7)
        b = BinaryRiskHead(x, [0, 1]*3, seed=7)
        self.assertEqual(a.diagnostics["status"], "fitted_raw")
        np.testing.assert_array_equal(a.predict(x), b.predict(x))

    def test_invalid_binary_labels_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            BinaryRiskHead(np.zeros((1, 2)), [None])
        head = BinaryRiskHead(np.zeros((1, 2)), [0])
        with self.assertRaises(ValueError):
            head.predict(np.zeros((2, 3)))

    def test_inactive_values_do_not_change_latent_features(self):
        features = self.features(history(4))
        a = deepcopy(history(1)[0]["candidate"])
        b = deepcopy(a)
        b["continuous"]["spacing"] = .99
        b["discrete"]["count"] = 4
        torch.testing.assert_close(features.transform([a]), features.transform([b]))
        self.assertEqual(features.transform([a]).shape[1], 8)

    def test_candidate_out_of_bounds_not_silently_clipped(self):
        features = self.features(history(4))
        candidate = deepcopy(history(1)[0]["candidate"])
        candidate["continuous"]["x"] = 2.
        with self.assertRaisesRegex(ValueError, "bounds"):
            features.transform([candidate])

    def test_matching_prefix_required(self):
        rows = history(5)
        features = self.features(rows)
        with self.assertRaisesRegex(ValueError, "prefix"):
            RiskHeads(rows[:-1], features)
        with self.assertRaisesRegex(ValueError, "prefix"):
            ResponseHeads(rows[:-1], features)
        altered = deepcopy(rows)
        altered[0]["risk"]["objective"] += 1.
        with self.assertRaisesRegex(ValueError, "prefix"):
            RiskHeads(altered, features)
        with self.assertRaisesRegex(ValueError, "prefix"):
            ResponseHeads(altered, features)

    def test_mixed_contracts_rejected(self):
        rows = history(4)
        rows[-1]["evaluator_contract_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Mixed"):
            self.features(rows)

    def test_per_head_censoring_and_partial_constraint_observation(self):
        rows = history(10)
        features = self.features(rows)
        risks = RiskHeads(rows, features)
        responses = ResponseHeads(rows, features, steps=2)
        self.assertEqual(risks.diagnostics["hidden_success"]["samples"], 10)
        self.assertEqual(risks.diagnostics["mission_solver_nonconvergence"]["positive"], 3)
        self.assertEqual(risks.diagnostics["energy_audit_failure"]["samples"], 7)
        self.assertEqual(responses.diagnostics["objective"]["samples"], 7)
        self.assertEqual(responses.diagnostics["constraint:capacity"]["samples"], 8)
        self.assertNotIn("constraint:span_min", responses.diagnostics)
        predicted = responses.predict([rows[0]["candidate"]])
        self.assertGreater(predicted["objective"]["mean"][0], 0)
        self.assertTrue(np.isfinite(predicted["objective"]["std"]).all())
        self.assertEqual(set(risks.predict([rows[0]["candidate"]])),
                         set(HIDDEN_MODES) | {"hidden_success", "final_feasibility"})

    def test_refitting_representation_invalidates_heads(self):
        rows = history(5)
        features = self.features(rows)
        risks = RiskHeads(rows, features)
        responses = ResponseHeads(rows, features, steps=2)
        features.fit(rows, epochs=1)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            risks.predict([rows[0]["candidate"]])
        with self.assertRaisesRegex(RuntimeError, "changed"):
            responses.predict([rows[0]["candidate"]])

    def test_direct_base_refit_cannot_reuse_stale_feature_snapshot(self):
        rows = history(5)
        features = self.features(rows)
        risks = RiskHeads(rows, features)
        features.representation.fit([], epochs=1)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            risks.predict([rows[0]["candidate"]])

    def test_base_module_loading_does_not_import_physical_backend(self):
        import sys
        before = set(sys.modules)
        self.features(history(3))
        added = set(sys.modules) - before
        self.assertFalse(any(name.startswith("RCAIDE") or ".problems." in name for name in added))

    def test_no_valid_objectives_can_still_fit_risk_heads(self):
        rows = history(3)[2:]
        features = self.features(rows)
        risks = RiskHeads(rows, features)
        responses = ResponseHeads(rows, features, steps=2)
        self.assertEqual(responses.predict([rows[0]["candidate"]])["objective"], None)
        self.assertIn("prior_only", risks.diagnostics["hidden_success"]["status"])
        self.assertFalse(features.representation.performance_fitted)

    def test_future_outcome_does_not_change_earlier_forecasts(self):
        rows = history(6)
        forecasts = []
        first, _ = prequential(rows, {"design_variables": VARIABLES}, BASE,
                               initial=4, epochs=1, gp_steps=2, on_prediction=forecasts.append)
        altered = deepcopy(rows)
        altered[-1]["risk"]["objective"] = 10000.
        second, _ = prequential(altered, {"design_variables": VARIABLES}, BASE,
                                initial=4, epochs=1, gp_steps=2)
        for a, b in zip(first, second):
            self.assertEqual(a["representation_snapshot"], b["representation_snapshot"])
            self.assertEqual(a["risk_probabilities_raw"], b["risk_probabilities_raw"])
            self.assertEqual(a["response_predictions"], b["response_predictions"])
            self.assertTrue(all(i < a["evaluation"] for i in a["training_indices"]))
        self.assertTrue(all("observed" not in row for row in forecasts))
        self.assertEqual(summarize(first)["predictions"], 2)


if __name__ == "__main__":
    unittest.main()
