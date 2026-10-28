from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ra_lsbo.adaptive_risk import JointRiskEnsemble, JointRiskSettings, ReferenceTemperature
from ra_lsbo.margin_joint import MarginJointRisk, margin_joint_probabilities
from ra_lsbo.mechanism_system import FiveModeProblem, SystemSettings
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.structured_models import StructuredResponseHeads
from ra_lsbo.system_experiment import sample_rows


BASE = Path(__file__).resolve().parents[2]/"LSBO"/"X57-LSBO-refactored"


class MarginJointTests(unittest.TestCase):
    def test_chain_rule_preserves_hidden_mass_and_has_no_independence_product(self):
        terminal = np.array([[[.4, .4, .2]], [[.2, .6, .2]]])
        result, conditional = margin_joint_probabilities(terminal, [[.9, .8]])
        self.assertAlmostEqual(conditional[0], .7)
        np.testing.assert_allclose(result[:, :, 0], [[.56], [.56]])
        np.testing.assert_allclose(result[:, :, 2:], terminal[:, :, 2:])
        np.testing.assert_allclose(result.sum(-1), 1.)

    def test_complete_explicit_certainty_and_zero_surrogate_have_correct_limits(self):
        terminal = np.array([[[.1, .7, .2], [.2, .3, .5]]])
        result, _ = margin_joint_probabilities(terminal, [[1., 1.], [.1, .1]])
        self.assertAlmostEqual(result[0, 0, 0], .8)
        self.assertAlmostEqual(result[0, 1, 0], 0.)
        with self.assertRaises(ValueError):
            margin_joint_probabilities(terminal, [[1.1], [.2]])

    def test_all_GP_and_neural_parameters_round_trip_and_calibration_is_bound(self):
        torch.set_num_threads(1)
        problem = FiveModeProblem(BASE)
        rows = sample_rows(problem, 90, 334)
        settings = SystemSettings(encoder_epochs=3, gp_steps=2)
        features = StructuredFeatures(BASE, [asdict(v) for v in problem.space.variables],
                                      settings=settings.encoder()).fit(rows[:50])
        terminal = JointRiskEnsemble(rows[:50], features,
                                     settings=JointRiskSettings(members=2, epochs=2))
        responses = StructuredResponseHeads(rows[:50], features, steps=2)
        model = MarginJointRisk(terminal, responses)
        calibration = ReferenceTemperature(model, rows[50:])
        candidates = [r["candidate"] for r in rows[-4:]]
        predicted = model.predict(candidates)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"risk.pt"
            model.save_checkpoint(path)
            restored = MarginJointRisk.load_checkpoint(path, features, rows[:50])
            np.testing.assert_allclose(restored.predict(candidates)["probabilities"],
                                       predicted["probabilities"], rtol=1e-7, atol=1e-9)
            other = ReferenceTemperature(restored, rows[50:])
            self.assertEqual(other.calibration_id, calibration.calibration_id)
        gp = next(iter(responses.models.values()))[0]
        with torch.no_grad():
            next(gp.parameters()).add_(.01)
        with self.assertRaises(RuntimeError):
            model.predict(candidates)


if __name__ == "__main__":
    unittest.main()
