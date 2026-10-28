import importlib.util
import unittest

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    import numpy as np
    import torch
    from ra_lsbo.calibration import CalibratedRiskHeads
    from ra_lsbo.constraint_risk import ConstraintRiskHeads
    from ra_lsbo.features import LatentFeatures
    from ra_lsbo.responses import ResponseHeads
    from ra_lsbo.risk_model import RiskHeads
    from test_calibration import synthetic_history
    from test_models import BASE, VARIABLES


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class ConstraintRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.rows = synthetic_history(90)
        self.features = LatentFeatures(BASE, VARIABLES, seed=0).fit(self.rows[:40], epochs=1)
        self.risks = RiskHeads(self.rows[:40], self.features)
        self.responses = ResponseHeads(self.rows[:40], self.features, steps=2)

    def test_score_uses_constraint_information_without_claiming_joint_probability(self):
        combined = ConstraintRiskHeads(self.risks, self.responses, ["capacity"])
        candidates = [row["candidate"] for row in self.rows[40:]]
        risk = self.risks.predict(candidates)
        response = self.responses.predict(candidates)
        score = combined.predict(candidates)
        np.testing.assert_allclose(
            score["final_feasibility"],
            np.minimum(risk["final_feasibility"], response["constraint:capacity"]["p_satisfied_raw"]))
        self.assertEqual(score["hidden_success"], risk["hidden_success"])
        self.assertFalse(combined.diagnostics["final_feasibility"]["raw_score_is_joint_probability"])
        old = CalibratedRiskHeads(self.risks, self.rows[40:])
        new = CalibratedRiskHeads(combined, self.rows[40:])
        self.assertNotEqual(old.model_id, new.model_id)
        self.assertNotEqual(old.calibration_id, new.calibration_id)
        self.assertEqual(new.calibration_indices, list(range(41, 91)))

    def test_missing_constraint_prevents_calibration_upgrade(self):
        combined = ConstraintRiskHeads(self.risks, self.responses, ["capacity", "missing"])
        calibrated = CalibratedRiskHeads(combined, self.rows[40:])
        self.assertEqual(calibrated.diagnostics["final_feasibility"]["status"], "fallback_base_unfitted")
        output = calibrated.predict([self.rows[-1]["candidate"]])
        self.assertIsNone(output["risk_probabilities_calibrated"]["final_feasibility"])

    def test_mutated_constraint_model_invalidates_calibration(self):
        combined = ConstraintRiskHeads(self.risks, self.responses, ["capacity"])
        calibrated = CalibratedRiskHeads(combined, self.rows[40:])
        model, center, scale, sign = self.responses.models["constraint:capacity"]
        self.responses.models["constraint:capacity"] = model, center+.01, scale, sign
        with self.assertRaisesRegex(RuntimeError, "constraint model changed"):
            calibrated.predict([self.rows[-1]["candidate"]])


if __name__ == "__main__":
    unittest.main()
