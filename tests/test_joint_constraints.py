from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from ra_lsbo.joint_constraints import JointConstraintPosterior, JointConstraintSettings
from ra_lsbo.joint_selection import select_joint_posterior
from ra_lsbo.posterior_risk import CompletionPosterior, PosteriorSettings
from ra_lsbo.structured_models import StructuredResponseHeads
from ra_lsbo.structured_encoder import EncoderSettings, StructuredFeatures
from ra_lsbo.mechanism_system import FiveModeProblem
from ra_lsbo.system_experiment import sample_rows

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent/"LSBO/X57-LSBO-refactored"


class JointConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem = FiveModeProblem(BASE)
        cls.rows = sample_rows(cls.problem, 40, 8876)
        cls.features = StructuredFeatures(
            BASE, [asdict(v) for v in cls.problem.space.variables], seed=11,
            settings=EncoderSettings(initial_epochs=2)).fit(cls.rows)
        cls.joint = JointConstraintPosterior(
            cls.rows, cls.features, cls.problem.analytical_constraints,
            seed=11, settings=JointConstraintSettings(steps=3))

    def test_joint_moments_samples_and_checkpoint_replay(self):
        candidates = [r["candidate"] for r in self.rows[:3]]
        mean, covariance = self.joint.pointwise_moments(candidates)
        self.assertEqual(tuple(mean.shape), (3, 2))
        self.assertEqual(tuple(covariance.shape), (3, 2, 2))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(covariance) >= -1e-10))
        draws = self.joint.sample(candidates, 97, samples=4)
        self.assertEqual(tuple(draws.shape), (4, 3, 2))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"joint.pt"
            self.joint.save_checkpoint(path)
            restored = JointConstraintPosterior.load_checkpoint(path, self.features, self.rows)
            self.assertEqual(restored.model_id, self.joint.model_id)
            torch.testing.assert_close(restored.sample(candidates, 97, 4), draws)
        result = self.joint.feasibility(candidates, 98, 64)
        self.assertFalse(result["probability_guarantee"])

    def test_diagonal_ablation_has_no_cross_output_covariance(self):
        model = JointConstraintPosterior(
            self.rows, self.features, self.problem.analytical_constraints,
            settings=JointConstraintSettings(steps=2, rank=0))
        _, covariance = model.pointwise_moments([self.rows[0]["candidate"]])
        self.assertAlmostEqual(float(covariance[0, 0, 1]), 0., places=10)

    def test_individual_output_missingness_does_not_drop_other_head_observation(self):
        rows = deepcopy(self.rows)
        index = next(i for i, r in enumerate(rows) if r["risk"]["state"] == "infeasible"
                     and r["risk"]["constraints"]["envelope"] > 0)
        rows[index]["risk"]["constraints"]["load"] = None
        rows[index]["risk"]["constraint_observed"]["load"] = False
        features = StructuredFeatures(
            BASE, [asdict(v) for v in self.problem.space.variables],
            settings=EncoderSettings(initial_epochs=2)).fit(rows)
        model = JointConstraintPosterior(rows, features, self.problem.analytical_constraints,
                                         settings=JointConstraintSettings(steps=2))
        evaluation = rows[index]["evaluation"]
        self.assertIn(evaluation, model.diagnostics["envelope"]["training_indices"])
        self.assertNotIn(evaluation, model.diagnostics["load"]["training_indices"])
        self.assertNotIn("packing", model.diagnostics)

    def test_future_history_and_missing_heads_rejected(self):
        with self.assertRaises((ValueError, RuntimeError)):
            JointConstraintPosterior(self.rows[:-1], self.features, self.problem.analytical_constraints)
        rows = deepcopy(self.rows)
        for row in rows:
            if row["risk"]["constraint_observed"]["load"]:
                row["risk"]["constraints"]["load"] = None
                row["risk"]["constraint_observed"]["load"] = False
                if not any(v is not None and v > 0 for v in row["risk"]["constraints"].values()):
                    row["risk"]["feasible_label"] = None
                    row["risk"]["state"] = "incomplete"
                row["risk"]["failure_modes"] = [
                    name for name in row["risk"]["failure_modes"] if name != "constraint:load"]
        features = StructuredFeatures(
            BASE, [asdict(v) for v in self.problem.space.variables],
            settings=EncoderSettings(initial_epochs=2)).fit(rows)
        model = JointConstraintPosterior(rows, features, self.problem.analytical_constraints,
                                         settings=JointConstraintSettings(steps=2))
        self.assertIn("load", model.unsupported)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            model.sample([rows[0]["candidate"]], 3)

    def test_joint_sampler_connects_to_objective_completion_and_selection(self):
        completion = CompletionPosterior(
            self.rows, self.features, settings=PosteriorSettings(steps=2, inducing=8))
        responses = StructuredResponseHeads(self.rows, self.features, steps=2)
        pool = [{"candidate": r["candidate"]} for r in self.rows[:4]]
        a, metadata = select_joint_posterior(
            pool, self.rows, responses, completion, self.joint, 61)
        b, replay = select_joint_posterior(
            pool, self.rows, responses, completion, self.joint, 61)
        self.assertEqual((a, metadata), (b, replay))
        self.assertEqual(metadata["constraint_names"], ["envelope", "load"])
        self.assertFalse(metadata["probability_guarantee"])
        self.assertIn(a["candidate"], [p["candidate"] for p in pool])


if __name__ == "__main__":
    unittest.main()
