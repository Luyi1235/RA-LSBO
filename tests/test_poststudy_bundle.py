from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from ra_lsbo.mechanism_system import FiveModeProblem, FullSystemLearning, SystemSettings
from ra_lsbo.system_experiment import sample_rows, save_new
from tools.release_system_training_data import (
    equivalent_observation, load_model_bundle, save_model_weights)


BASE = Path(__file__).resolve().parents[2]/"LSBO"/"X57-LSBO-refactored"


class PoststudyBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_weight_roundtrip_and_file_corruption_rejected_before_loading(self):
        problem = FiveModeProblem(BASE)
        rows = sample_rows(problem, 240, 791)
        learner = FullSystemLearning(
            "ra_full", SystemSettings(calibration_window=80, encoder_epochs=4, gp_steps=4), 0)
        models, metadata = learner.fit(
            rows, [asdict(v) for v in problem.space.variables], str(BASE), ["envelope", "load"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_new(output/"observations.json", rows)
            save_new(output/"model_metadata.json", metadata)
            learner.features.save_checkpoint(output/"encoder.pt")
            save_model_weights(output, models, metadata, str(BASE), rows, {})
            loaded = load_model_bundle(output)
            self.assertEqual(loaded[0].model_id, models[0].model_id)
            with (output/"gp_states.pt").open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaises((ValueError, RuntimeError)):
                load_model_bundle(output)

    def test_replica_consistency_does_not_turn_different_failure_into_same_supervision(self):
        problem = FiveModeProblem(BASE)
        rows = sample_rows(problem, 2, 1)
        risk = rows[0]["risk"]
        changed = deepcopy(risk)
        changed["elapsed_seconds"] = 1000.
        self.assertTrue(equivalent_observation(risk, changed))
        changed["hidden_success"] = 1-risk["hidden_success"]
        self.assertFalse(equivalent_observation(risk, changed))


if __name__ == "__main__":
    unittest.main()
