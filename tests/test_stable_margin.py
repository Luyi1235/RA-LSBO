from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ra_lsbo.adaptive_risk import JointRiskSettings
from ra_lsbo.mechanism_system import FiveModeProblem, SystemSettings
from ra_lsbo.stable_margin import StableMarginLearning, StableMarginRisk
from ra_lsbo.system_experiment import sample_rows


class StableMarginTests(unittest.TestCase):
    def test_online_factory_and_checkpoint_keep_calibration_and_normalization_identity(self):
        torch.set_num_threads(1)
        base = Path(__file__).resolve().parents[2]/"LSBO"/"X57-LSBO-refactored"
        problem = FiveModeProblem(base)
        rows = sample_rows(problem, 100, 839)
        settings = SystemSettings(encoder_epochs=3, gp_steps=2)
        learner = StableMarginLearning(
            "adaptive_full", settings, range(61, 101),
            risk_settings=JointRiskSettings(members=2, epochs=3))
        (risk, responses, calibration), metadata = learner.fit(
            rows, [asdict(v) for v in problem.space.variables], base)
        candidates = [r["candidate"] for r in rows[-5:]]
        predicted = np.asarray(risk.predict(candidates)["probabilities"])
        self.assertTrue(metadata["inference_normalization"])
        self.assertEqual(calibration.model_id, risk.model_id)
        self.assertEqual(metadata["calibration_id"], calibration.calibration_id)
        self.assertTrue(((predicted >= 0.) & (predicted <= 1.)).all())
        np.testing.assert_allclose(predicted.sum(1), 1., atol=1e-14, rtol=0.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"risk.pt"
            risk.save_checkpoint(path)
            restored = StableMarginRisk.load_checkpoint(path, learner.features, rows[:60])
            self.assertEqual(restored.model_id, risk.model_id)
            np.testing.assert_allclose(restored.predict(candidates)["probabilities"],
                                       predicted, rtol=1e-7, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
