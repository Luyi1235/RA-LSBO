from copy import deepcopy
import importlib.util
from types import SimpleNamespace
import unittest

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    import torch
    from ra_lsbo.discovery_learning import FreshDiscoveryLearning
    from ra_lsbo.legacy import normalize_evaluation
    from test_calibration import synthetic_history
    from test_models import BASE, VARIABLES


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class FreshDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def settings(self, window=30):
        return SimpleNamespace(calibration_window=window, seed=0, epochs=1, gp_steps=2)

    def test_unavailable_calibration_uses_all_observed_rows_without_calibration_claim(self):
        rows = synthetic_history(20)
        strategy = FreshDiscoveryLearning()
        (risks, _, calibrated), metadata = strategy.fit(rows, VARIABLES, BASE, self.settings(10))
        self.assertIsNone(calibrated)
        self.assertEqual(risks.features.training_indices, list(range(1, 21)))
        self.assertEqual(metadata["active_regime"], "raw_full_history_discovery")
        self.assertFalse(strategy.should_refresh(rows))
        self.assertTrue(strategy.should_refresh(synthetic_history(21)))

    def test_new_feasible_observation_enters_next_discovery_fit(self):
        rows = synthetic_history(20)
        for row in rows:
            row["risk"] = normalize_evaluation({
                "status": "mission_not_converged", "solver_success": False,
                "objective": None, "constraints": {},
            }, ["capacity", "span_min"], ["span_min"]).to_dict()
        strategy = FreshDiscoveryLearning()
        (first, _, _), _ = strategy.fit(rows, VARIABLES, BASE, self.settings(10))
        next_row = deepcopy(rows[-1])
        next_row["evaluation"] = 21
        next_row["risk"] = normalize_evaluation({
            "status": "ok", "solver_success": True, "objective": 100.,
            "constraints": {"capacity": -.1, "span_min": -1.},
        }, ["capacity", "span_min"], ["span_min"]).to_dict()
        (second, _, calibrated), _ = strategy.fit(rows+[next_row], VARIABLES, BASE, self.settings(10))
        self.assertEqual(first.diagnostics["final_feasibility"]["positive"], 0)
        self.assertEqual(second.diagnostics["final_feasibility"]["positive"], 1)
        self.assertIn(21, second.features.training_indices)
        self.assertIsNone(calibrated)

    def test_future_rows_cannot_enter_model_fit(self):
        rows = synthetic_history(120)
        strategy = FreshDiscoveryLearning()
        (risks, _, calibrated), _ = strategy.fit(rows[:90], VARIABLES, BASE, self.settings())
        self.assertLessEqual(max(risks.features.training_indices), 90)
        if calibrated:
            self.assertLess(max(risks.features.training_indices), min(calibrated.calibration_indices))
            self.assertEqual(max(calibrated.calibration_indices), 90)


if __name__ == "__main__":
    unittest.main()
