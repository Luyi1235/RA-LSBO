import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ra_lsbo import x57_cold_risk as risk


class Point(int):
    def to_dict(self):
        return {"value": int(self)}


class RecoveryPoolTest(unittest.TestCase):
    def setUp(self):
        self.point = Point(1)
        self.neighbor = (Point(2), {"source": "feasible_neighborhood_recovery", "anchor_evaluation": 1})
        self.rep = SimpleNamespace(version=1, features=lambda points: torch.tensor(points, dtype=torch.double))
        self.native = SimpleNamespace(search_phase=lambda: "bo", representation=self.rep,
                                      space=SimpleNamespace(key=int), settings=SimpleNamespace(seed=1, update_every=20),
                                      last_representation_update=0)
        self.posterior = SimpleNamespace(log_probabilities=lambda points: np.array(points, dtype=float),
                                         probabilities=lambda points: np.ones(len(points)))

    def run_step(self, count, neighbor, score):
        row = SimpleNamespace(result=SimpleNamespace(has_objective=lambda: True), candidate=self.point)
        self.native.observations = [row] * count
        history = [{"feedback": {"usable": True}}] * count

        def native_propose(phase, refine_surrogate):
            refine_surrogate(object(), self.rep)
            return self.point, {"source": "native"}

        self.native.propose = native_propose
        ranking = SimpleNamespace(score=score)
        with patch.object(risk, "latent_pool", side_effect=lambda *args: [self.point]), \
             patch.object(risk, "risk_observation", return_value=SimpleNamespace(lower=None, upper=None)), \
             patch.object(risk, "CompletionPosterior", return_value=self.posterior), \
             patch.object(risk, "RiskWeightedRefinement", return_value=ranking):
            return risk.propose(self.native, history, "structured_sign", None, 512, recovery=neighbor)

    def test_completion_pool_competes_and_preserves_source(self):
        point, _, info = self.run_step(2, self.neighbor, None)
        self.assertEqual((point, info["anchor_evaluation"], info["candidate_count"]), (2, 1, 2))

    def test_performance_keeps_either_candidate_by_joint_score(self):
        for multiplier, expected in [(1., 2), (-1., 1), (0., 1)]:
            with self.subTest(multiplier=multiplier):
                point, _, _ = self.run_step(3, self.neighbor, lambda x: multiplier * x.numpy())
                self.assertEqual(point, expected)

    def test_disabled_arm_preserves_original_choice(self):
        for count in (2, 3):
            point, _, info = self.run_step(count, None, None)
            self.assertEqual(point, self.point)
            self.assertNotIn("recovery_candidate_id", info)


if __name__ == "__main__":
    unittest.main()
