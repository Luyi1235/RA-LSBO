from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    import torch
    from ra_lsbo.discovery_learning import FreshDiscoveryLearning
    from ra_lsbo.optimizer import OptimizerSettings, run_optimizer
    from ra_lsbo.synthetic import SyntheticProblem
    from test_models import BASE


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class DiscoveryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.settings = OptimizerSettings(
            budget=24, bootstrap=10, calibration_window=4, pool_size=8,
            epochs=1, gp_steps=2, update_every=8)

    def test_fresh_learning_preserves_budget_and_uses_latest_observation(self):
        problem = SyntheticProblem(BASE, "no_initial_feasible")
        output = self.root / "fresh"
        rows, summary = run_optimizer(problem, BASE, self.settings, output,
                                      model_strategy=FreshDiscoveryLearning())
        self.assertEqual(summary["evaluations"], 18)
        self.assertEqual(summary["status"], "stopped_by_policy")
        self.assertEqual(summary["ledger"]["mode_calls"]["discovery"], 8)
        self.assertEqual(summary["ledger"]["risk_charge"], 18)
        fits = [json.loads(line) for line in (output / "model_fits.jsonl").read_text().splitlines()]
        self.assertEqual([fit["before_evaluation"] for fit in fits], list(range(11, 20)))
        for fit in fits:
            self.assertEqual(fit["training_indices"], list(range(1, fit["before_evaluation"])))
            self.assertEqual(fit["active_regime"], "raw_full_history_discovery")
            self.assertIsNone(fit["calibration_id"])
            self.assertEqual(fit["calibration_indices"], [])
        for row in rows[10:]:
            self.assertEqual(row["phase"], "discovery")
            self.assertEqual(row["proposal"]["evidence"]["basis"], "raw")
            self.assertIsNone(row["proposal"]["evidence"]["calibration_id"])

    def test_default_and_fresh_share_initialization_and_risk_policy(self):
        problem = SyntheticProblem(BASE, "mixed")
        for name, strategy in (("default", None), ("fresh", FreshDiscoveryLearning())):
            run_optimizer(problem, BASE, replace(self.settings, budget=12), self.root / name,
                          model_strategy=strategy)
        old = json.loads((self.root / "default" / "manifest.json").read_text())
        new = json.loads((self.root / "fresh" / "manifest.json").read_text())
        self.assertEqual(old["budget_policy"], new["budget_policy"])
        self.assertEqual(old["initial_candidates_sha256"], new["initial_candidates_sha256"])
        fits = [json.loads(line) for line in
                (self.root / "default" / "model_fits.jsonl").read_text().splitlines()]
        self.assertEqual(fits[0]["training_indices"], list(range(1, 7)))
        self.assertEqual(fits[0]["calibration_indices"], list(range(7, 11)))

    def test_discovery_strategy_cannot_silently_change_baseline(self):
        problem = SyntheticProblem(BASE)
        with self.assertRaisesRegex(ValueError, "RA"):
            run_optimizer(problem, BASE, replace(self.settings, algorithm="cbo"),
                          self.root / "invalid", model_strategy=FreshDiscoveryLearning())
        self.assertFalse((self.root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
