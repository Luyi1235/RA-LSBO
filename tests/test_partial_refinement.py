from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.problems import HierarchicalBranin, candidate_pool
from ra_lsbo.partial_refinement.optimizer import RefinementSettings, RefiningLatentBO
from tools.run_partial_bo_development import BASE


class SmoothPosterior:
    def __init__(self, problem, base, records, seed, settings):
        self.best = min(record.observation.objective.value for record in records if record.usable_feasible)
        self.history_digest = digest([record.to_dict() for record in records])

    def predict(self, candidates):
        objective = np.asarray([(candidate["continuous"]["x"]-.67)**2+
                                (candidate["continuous"]["y"]-.21)**2 for candidate in candidates])
        count = len(candidates)
        return {"completion": {"mean": np.full(count, -2.), "variance": np.full(count, .01)},
                "constraints": {"region": {"mean": np.full(count, -1.), "variance": np.full(count, .01), "scale": 1.}},
                "objective": {"mean": objective, "variance": np.full(count, 1e-4)}}

    def diagnostics(self):
        return {"feature_identity": "mock_sealed", "history_digest": self.history_digest, "heads": {}}


class BranchRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        problem = HierarchicalBranin(BASE)
        anchor = problem.anchors()[0]
        history = [problem.record(anchor, 10., {"region": -.5, "resource": -.2},
                                  {"termination_reason": "mock_fixture"})]
        candidate = {"continuous": {**anchor["continuous"], "x": .5, "y": .4}, "discrete": anchor["discrete"]}
        return problem, history, [problem.canonical(candidate)]

    @patch("ra_lsbo.partial_refinement.optimizer.RefinementPosterior", SmoothPosterior)
    def test_local_continuous_search_improves_pool_without_oracle(self):
        problem, history, pool = self.fixture()
        settings = RefinementSettings(local_branches=1, local_starts=1, local_points=2)
        optimizer = RefiningLatentBO(problem, BASE, 14, settings)
        with patch.object(problem, "evaluate", side_effect=AssertionError("No oracle during search")):
            candidate, metadata = optimizer.propose(history, pool, 0)
        self.assertAlmostEqual(candidate["continuous"]["x"], .67, places=3)
        self.assertAlmostEqual(candidate["continuous"]["y"], .21, places=3)
        self.assertTrue(metadata["selected_from_local_augmentation"])
        self.assertTrue(metadata["local_reports"][0]["success"])
        self.assertEqual(metadata["local_reports"][0]["active_dimensions"], [0, 1])
        self.assertEqual(candidate["continuous"]["trim"], problem.space.default().continuous["trim"])
        with self.assertRaises(RuntimeError):
            optimizer.propose(history, pool, 1)
        with self.assertRaises(ValueError):
            optimizer.accept(history[0])
        settled = problem.record(candidate, 1., {"region": -.1, "resource": -.2}, {"termination_reason": "mock_fixture"})
        optimizer.accept(settled)
        self.assertAlmostEqual(next(iter(optimizer.radii.values())), .22)
        with self.assertRaisesRegex(ValueError, "prefix"):
            optimizer.propose(history, pool, 1)

    @patch("ra_lsbo.partial_refinement.optimizer.RefinementPosterior", SmoothPosterior)
    def test_global_ablation_preserves_pool_and_has_no_local_work(self):
        problem, history, pool = self.fixture()
        optimizer = RefiningLatentBO(problem, BASE, settings=RefinementSettings(local=False))
        candidate, metadata = optimizer.propose(history, pool, 0)
        self.assertEqual(candidate, pool[0])
        self.assertEqual(metadata["pool"], pool)
        self.assertEqual(metadata["local_reports"], [])

    def test_real_mixed_posterior_refinement_replays_with_mock_labels(self):
        problem = HierarchicalBranin(BASE)
        initial, _ = candidate_pool(problem, 72020, 6)
        history = [problem.record(candidate, 2.+number, {"region": -.2, "resource": -.1},
                                  {"termination_reason": "mock_fixture"}) for number, candidate in enumerate(initial)]
        pool, _ = candidate_pool(problem, 72021, 8, [record.observation.design_id for record in history])
        settings = RefinementSettings(encoder_epochs=3, continuous_grid=(.15,), discrete_grid=(.5,),
                                      local_branches=1, local_starts=1, local_points=2, local_maxiter=3)
        with patch.object(problem, "evaluate", side_effect=AssertionError("No oracle")):
            first = RefiningLatentBO(problem, BASE, 17, settings).propose(history, pool, 0)
            second = RefiningLatentBO(problem, BASE, 17, settings).propose(history, pool, 0)
        self.assertEqual(first, second)
        self.assertEqual(first[1]["history_digest"], digest([record.to_dict() for record in history]))

    @patch("ra_lsbo.partial_refinement.optimizer.RefinementPosterior", SmoothPosterior)
    def test_proposal_rejects_mutation_between_fit_and_feedback(self):
        problem, history, pool = self.fixture()
        optimizer = RefiningLatentBO(problem, BASE, settings=RefinementSettings(local=False))
        candidate, _ = optimizer.propose(history, pool, 0)
        history[0].diagnostics["mutated"] = True
        settled = problem.record(candidate, 1., {"region": -.1, "resource": -.2}, {"termination_reason": "mock_fixture"})
        with self.assertRaisesRegex(ValueError, "mutated"):
            optimizer.accept(settled)


if __name__ == "__main__":
    unittest.main()
