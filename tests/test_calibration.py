from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ra_lsbo import normalize_evaluation


HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    import numpy as np
    import torch
    from ra_lsbo.calibration import (
        CalibrationPolicy, CalibratedRiskHeads, SigmoidCalibrator, observed_target)
    from ra_lsbo.calibration_replay import blocked_calibration, calibration_scores, run_calibration
    from ra_lsbo.features import LatentFeatures
    from ra_lsbo.metrics import binary_metrics
    from ra_lsbo.risk_model import RiskHeads
    from test_models import BASE, VARIABLES, history


def synthetic_history(count=160):
    """Fixed seed, hierarchical candidates and stochastic terminal failures."""
    random = np.random.RandomState(31)
    rows = []
    for index in range(1, count+1):
        x = float(random.uniform())
        failed = random.uniform() < 1/(1+np.exp(7*(x-.35)))
        rows.append({
            "evaluation": index,
            "candidate": {"continuous": {"x": x, "spacing": .4},
                          "discrete": {"layout": "single" if index % 2 else "distributed", "count": 2}},
            "risk": normalize_evaluation({
                "objective": None if failed else 100+20*x,
                "solver_success": not failed,
                "constraints": {"capacity": x-.9, "span_min": -1.},
                "status": "mission_not_converged" if failed else
                          "constraint_violation" if x > .9 else "ok",
            }, ["capacity", "span_min"], ["span_min"]).to_dict(),
            "source_journal_sha256": "synthetic", "evaluator_contract_sha256": "synthetic",
        })
    return rows


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env for calibration tests")
class SigmoidTests(unittest.TestCase):
    def test_invalid_policy_rejected(self):
        for arguments in ({"min_samples": 0}, {"min_per_class": True},
                          {"min_distinct_scores": 1.5}, {"regularization_c": float("nan")},
                          {"regularization_c": 0}, {"probability_clip": .5}):
            with self.assertRaises(ValueError):
                CalibrationPolicy(**arguments)

    def test_invalid_scores_labels_rejected(self):
        for probabilities, labels in (([float("nan")], [0]), ([1.1], [1]), ([True], [1]),
                                      ([.5], [True]), ([.5], [2]), ([.5], [])):
            with self.assertRaises(ValueError):
                SigmoidCalibrator(probabilities, labels)

    def test_sample_class_and_variation_gates(self):
        cases = [
            ([], [], "insufficient_samples"),
            ([.1, .9]*4, [0, 1]*4, "insufficient_samples"),
            (np.linspace(.1, .9, 40).tolist(), [0]*40, "insufficient_classes"),
            ([.5]*40, [0, 1]*20, "insufficient_score_variation"),
        ]
        for probabilities, labels, reason in cases:
            calibrator = SigmoidCalibrator(probabilities, labels)
            self.assertIn(reason, calibrator.diagnostics["fallback_reasons"])
            self.assertIsNone(calibrator.predict([.2, .8]))
            self.assertFalse(calibrator.diagnostics["calibrator_fitted"])
            self.assertFalse(calibrator.diagnostics["is_confidence_bound"])

    def test_prior_base_cannot_be_called_calibrated(self):
        calibrator = SigmoidCalibrator(np.linspace(.1, .9, 40).tolist(),
                                      [0]*20+[1]*20, base_fitted=False)
        self.assertEqual(calibrator.diagnostics["status"], "fallback_base_unfitted")
        self.assertIsNone(calibrator.predict([.8]))

    def test_fitted_sigmoid_is_finite_monotone_and_reproducible(self):
        probabilities = np.linspace(.01, .99, 100).tolist()
        labels = [0]*50+[1]*50
        first, second = SigmoidCalibrator(probabilities, labels), SigmoidCalibrator(probabilities, labels)
        self.assertEqual(first.diagnostics["status"], "fitted_empirical")
        values = first.predict([0., .1, .5, .9, 1.])
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(sorted(values), values)
        self.assertEqual(values, second.predict([0., .1, .5, .9, 1.]))
        self.assertEqual(first.predict([]), [])
        self.assertFalse(first.diagnostics["reliability_verified"])

    def test_reverse_ordering_falls_back(self):
        calibrator = SigmoidCalibrator(np.linspace(.01, .99, 100).tolist(), [1]*50+[0]*50)
        self.assertEqual(calibrator.diagnostics["status"], "fallback_nonpositive_slope")
        self.assertIsNone(calibrator.predict([.2]))

    def test_convergence_failure_falls_back(self):
        from sklearn.exceptions import ConvergenceWarning
        with patch("ra_lsbo.calibration.LogisticRegression.fit", side_effect=ConvergenceWarning("test")):
            calibrator = SigmoidCalibrator(np.linspace(.01, .99, 100).tolist(), [0]*50+[1]*50)
        self.assertEqual(calibrator.diagnostics["status"], "fallback_fit_failed")

    def test_known_miscalibration_improves_on_independent_synthetic_test(self):
        def dataset(seed, count):
            random = np.random.RandomState(seed)
            score = random.uniform(-2, 2, count)
            true = 1/(1+np.exp(-(.6*score-.4)))
            raw = 1/(1+np.exp(-(2.5*score+1)))
            return raw.tolist(), (random.uniform(size=count) < true).astype(int).tolist()
        calibration_p, calibration_y = dataset(17, 1000)
        test_p, test_y = dataset(29, 2000)
        calibrator = SigmoidCalibrator(calibration_p, calibration_y)
        calibrated = calibrator.predict(test_p)
        before, after = binary_metrics(test_p, test_y), binary_metrics(calibrated, test_y)
        self.assertLess(after["brier"], before["brier"])
        self.assertLess(after["log_loss"], before["log_loss"])


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env for calibration tests")
class CalibrationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.rows = synthetic_history()
        cls.features = LatentFeatures(BASE, VARIABLES, seed=3).fit(cls.rows[:60], epochs=1)
        cls.risks = RiskHeads(cls.rows[:60], cls.features, seed=3, trees=16)

    def test_disjoint_blocks_and_known_mechanisms(self):
        calibrated = CalibratedRiskHeads(self.risks, self.rows[60:140])
        self.assertEqual(calibrated.diagnostics["hidden_success"]["status"], "fitted_empirical")
        self.assertEqual(calibrated.diagnostics["any_hidden_failure"]["positive"],
                         calibrated.diagnostics["hidden_success"]["negative"])
        geometry = calibrated.diagnostics["geometry_generation_failure"]
        self.assertGreater(geometry["excluded_unknown_or_precheck"], 0)
        self.assertEqual(geometry["positive"], 0)
        predicted = calibrated.predict([self.rows[-1]["candidate"]])
        self.assertIsNone(predicted["risk_probabilities_calibrated"]["geometry_generation_failure"])
        self.assertEqual(predicted["risk_probabilities_effective"]["geometry_generation_failure"],
                         predicted["risk_probabilities_raw"]["geometry_generation_failure"])
        self.assertAlmostEqual(
            predicted["risk_probabilities_calibrated"]["any_hidden_failure"][0],
            1-predicted["risk_probabilities_calibrated"]["hidden_success"][0], places=8)
        self.assertEqual(self.features.training_indices, list(range(1, 61)))

    def test_overlap_empty_and_source_mismatch_rejected(self):
        for rows in ([], self.rows[59:80]):
            with self.assertRaises(ValueError):
                CalibratedRiskHeads(self.risks, rows)
        for key in ("source_journal_sha256", "evaluator_contract_sha256"):
            changed = deepcopy(self.rows[60:80])
            for row in changed:
                row[key] = "different"
            with self.assertRaisesRegex(ValueError, "identity"):
                CalibratedRiskHeads(self.risks, changed)

    def test_precheck_and_unknown_outcomes_excluded(self):
        rows = deepcopy(self.rows[60:80])
        rows[0]["risk"] = normalize_evaluation(
            {"objective": None, "solver_success": False, "status": "timeout", "constraints": {}},
            ["capacity", "span_min"], ["span_min"]).to_dict()
        rows[1]["risk"] = normalize_evaluation(
            {"objective": None, "solver_success": False, "status": "precheck_rejected",
             "constraints": {"span_min": 1.}},
            ["capacity", "span_min"], ["span_min"]).to_dict()
        calibrated = CalibratedRiskHeads(self.risks, rows)
        self.assertEqual(calibrated.diagnostics["final_feasibility"]["samples"], 18)
        self.assertIsNone(observed_target(rows[1]["risk"], "final_feasibility"))

    def test_refit_invalidates_calibrator(self):
        features = LatentFeatures(BASE, VARIABLES, seed=3).fit(self.rows[:60], epochs=1)
        risks = RiskHeads(self.rows[:60], features, seed=3, trees=16)
        calibrated = CalibratedRiskHeads(risks, self.rows[60:100])
        features.fit(self.rows[:60], epochs=1)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            calibrated.predict([self.rows[-1]["candidate"]])

    def test_replaced_raw_model_invalidates_calibrator(self):
        calibrated = CalibratedRiskHeads(self.risks, self.rows[60:100])
        original = self.risks.model_id
        try:
            self.risks.model_id = "replaced"
            with self.assertRaisesRegex(RuntimeError, "changed"):
                calibrated.predict([self.rows[-1]["candidate"]])
        finally:
            self.risks.model_id = original

    def test_calibration_labels_do_not_train_representation_or_raw_heads(self):
        rows = deepcopy(self.rows[:140])
        first, metadata, _ = blocked_calibration(
            rows, {"design_variables": VARIABLES}, BASE, 60, 60, epochs=1, trees=8)
        for row in rows[60:120]:
            row["risk"] = deepcopy(self.rows[0]["risk"])
        second, changed, _ = blocked_calibration(
            rows, {"design_variables": VARIABLES}, BASE, 60, 60, epochs=1, trees=8)
        self.assertEqual(metadata["representation_snapshot"], changed["representation_snapshot"])
        self.assertEqual(metadata["raw_model_id"], changed["raw_model_id"])
        self.assertNotEqual(metadata["calibration_id"], changed["calibration_id"])
        for a, b in zip(first, second):
            self.assertEqual(a["risk_probabilities_raw"], b["risk_probabilities_raw"])

    def test_test_labels_never_change_calibration_or_predictions(self):
        rows = deepcopy(self.rows[:140])
        forecasts = []
        first, metadata, _ = blocked_calibration(
            rows, {"design_variables": VARIABLES}, BASE, 60, 60, epochs=1, trees=8,
            on_prediction=forecasts.append)
        for row in rows[120:]:
            row["risk"] = deepcopy(self.rows[0]["risk"])
        second, changed, _ = blocked_calibration(
            rows, {"design_variables": VARIABLES}, BASE, 60, 60, epochs=1, trees=8)
        self.assertEqual(metadata, changed)
        for a, b in zip(first, second):
            self.assertEqual({k: v for k, v in a.items() if k != "observed"},
                             {k: v for k, v in b.items() if k != "observed"})
        self.assertTrue(all("observed" not in row for row in forecasts))
        self.assertTrue(all(max(row["training_indices"]) < min(row["calibration_indices"])
                            and max(row["calibration_indices"]) < row["evaluation"] for row in forecasts))

    def test_insufficient_calibration_is_not_in_calibrated_scores(self):
        predictions, _, _ = blocked_calibration(
            history(), {"design_variables": VARIABLES}, BASE, 4, 3, epochs=1, trees=8)
        scores = calibration_scores(predictions)
        for name, head in scores.items():
            self.assertEqual(head["calibrated_only"]["samples"], 0)
            self.assertIsNone(head["calibrated_minus_raw_brier"])
            self.assertEqual(head["raw_all"], head["effective_all"])
        self.assertEqual(scores["final_feasibility"]["raw_all"]["samples"], 3)

    def test_invalid_split_rejected(self):
        for train, calibration in ((0, 1), (2, 0), (5, 5), (True, 2)):
            with self.assertRaisesRegex(ValueError, "blocks"):
                blocked_calibration(history(), {"design_variables": VARIABLES}, BASE, train, calibration)

    def test_export_and_failed_source_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            payload = b"frozen test journal"
            (source / "evaluations.jsonl").write_bytes(payload)
            import hashlib
            source_report = {
                "source_sha256": {"evaluations.jsonl": hashlib.sha256(payload).hexdigest()},
                "evaluator_contract_sha256": "test",
            }
            snapshot = (history(), {}, source_report, {"design_variables": VARIABLES})
            with patch("ra_lsbo.calibration_replay.build_snapshot", return_value=snapshot):
                destination, report = run_calibration(source, BASE, root / "outputs", 4, 3, epochs=1, trees=8)
                self.assertEqual(report["status"], "passed")
                self.assertEqual((source / "evaluations.jsonl").read_bytes(), payload)
                forecasts = [json.loads(line) for line in
                             (destination / "test_forecasts.jsonl").read_text().splitlines()]
                self.assertEqual(len(forecasts), 3)
                self.assertTrue(all("observed" not in row for row in forecasts))
                self.assertTrue((destination / "calibration_samples.jsonl").exists())
                source_report["source_sha256"]["evaluations.jsonl"] = "changed"
                with self.assertRaisesRegex(RuntimeError, "Source changed"):
                    run_calibration(source, BASE, root / "failed", 4, 3, epochs=1, trees=8)
            failed = next((root / "failed").glob("*/report.json"))
            self.assertEqual(json.loads(failed.read_text())["status"], "failed")

    def test_output_cannot_be_inside_source(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            run_calibration(BASE, BASE, BASE / "nested", 1, 1)


if __name__ == "__main__":
    unittest.main()
