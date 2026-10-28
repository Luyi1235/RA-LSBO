from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    import numpy as np
    import torch
    from ra_lsbo.budget import BudgetPolicy, RiskLedger
    from ra_lsbo.discovery_learning import FreshDiscoveryLearning
    from ra_lsbo.optimizer import OptimizerSettings, run_optimizer
    from ra_lsbo.selection import constraint_acquisition, select_candidate
    from ra_lsbo.synthetic import SyntheticProblem
    from test_models import BASE


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class ConstrainedDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_constraint_ranking_never_upgrades_raw_evidence_or_reduces_charge(self):
        pool = [{"candidate": {"continuous": {"x": value}, "discrete": {}}, "source": "test"}
                for value in (.1, .2)]
        raw = {"final_feasibility": [.99, .6], "hidden_success": [.9, .9]}
        risks = SimpleNamespace(predict=lambda candidates: raw,
                                heads={"final_feasibility": SimpleNamespace(model=None)},
                                model_id="raw_model")
        responses = SimpleNamespace(predict=lambda candidates: {
            "objective": {"mean": [1., 1.], "std": [1., 1.]},
            "constraint:g": {"p_satisfied_raw": [.1, .9]}})
        ledger = RiskLedger(self.root / "budget.jsonl", BudgetPolicy(4, 4, 0, 1, 0))
        self.addCleanup(ledger.close)
        rows = [{"risk": {"feasible_label": True, "objective": 2.}}]
        with patch("ra_lsbo.selection.log_expected_improvement", return_value=np.log([2., 1.])):
            original, _ = select_candidate(pool, rows, risks, responses, None, ledger, 1)
            changed, info = select_candidate(pool, rows, risks, responses, None, ledger, 1,
                                             discovery_rule="constraint_ei")
            self.assertEqual(original["candidate"], pool[0]["candidate"])
            self.assertEqual(changed["candidate"], pool[1]["candidate"])
            self.assertEqual(changed["mode"], "discovery")
            self.assertEqual(changed["evidence"]["basis"], "raw")
            self.assertIsNone(changed["evidence"]["calibration_id"])
            self.assertEqual(info["risk_charge"], 1.)
            ledger.reserve(changed["candidate"], changed["mode"], changed["evidence"], changed["source"])
            refused, _ = select_candidate(pool, rows, risks, responses, None, ledger, 1,
                                           discovery_rule="constraint_ei")
            self.assertIsNone(refused)

    def test_no_incumbent_uses_constraint_ranking_and_missing_output_rule(self):
        score = constraint_acquisition(
            {"hidden_success": [.8, .4]},
            {"constraint:known": {"p_satisfied_raw": [.1, .9]}, "constraint:missing": None},
            np.array([-np.inf, -np.inf]))
        np.testing.assert_allclose(score, [.04, .18])

    def test_integrated_rule_still_stops_after_eight_full_charge_discoveries(self):
        settings = OptimizerSettings(budget=24, bootstrap=10, calibration_window=4, pool_size=8,
                                     epochs=1, gp_steps=2, discovery_rule="constraint_ei")
        rows, summary = run_optimizer(
            SyntheticProblem(BASE, "no_initial_feasible"), BASE, settings,
            self.root / "run", model_strategy=FreshDiscoveryLearning())
        self.assertEqual(len(rows), 18)
        self.assertEqual(summary["ledger"]["risk_charge"], 18)
        self.assertEqual(summary["ledger"]["mode_calls"]["discovery"], 8)
        self.assertEqual(summary["status"], "stopped_by_policy")
        with self.assertRaisesRegex(ValueError, "RA"):
            replace(settings, algorithm="cbo")


if __name__ == "__main__":
    unittest.main()
