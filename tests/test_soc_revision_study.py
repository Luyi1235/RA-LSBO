"""Budget, feedback and no-oracle contracts for the new mechanism study."""

from unittest.mock import patch
import unittest

import torch

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.problems import HierarchicalBranin, candidate_pool
from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.soc_revision.optimizer import RevisionSettings, RiskRefiningBO
from tests.test_partial_refinement import SmoothPosterior
from tools.run_partial_bo_development import BASE
from tools.run_soc_revision_study import ARMS, MAXIMUM, settings_for


class RevisionStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_registered_budget_and_ablation_axes(self):
        self.assertEqual(MAXIMUM, 492)
        self.assertEqual(len(ARMS), 5)
        self.assertTrue(settings_for("revised_full").average_kernel)
        self.assertTrue(settings_for("revised_full").probability_exploitation)
        self.assertFalse(settings_for("average_only").probability_exploitation)
        self.assertFalse(settings_for("probability_only").average_kernel)
        self.assertFalse(settings_for("revised_no_intervals").use_intervals)

    @patch("ra_lsbo.soc_revision.optimizer.RiskPosterior", SmoothPosterior)
    def test_continuous_search_and_feedback_protection_retained(self):
        problem = HierarchicalBranin(BASE)
        anchor = problem.anchors()[0]
        history = [problem.record(anchor, 10., {"region": -.5, "resource": -.2},
                                  {"termination_reason": "mock_fixture"})]
        raw = {"continuous": {**anchor["continuous"], "x": .5, "y": .4}, "discrete": anchor["discrete"]}
        settings = RevisionSettings(local_branches=1, local_starts=1, local_points=2)
        optimizer = RiskRefiningBO(problem, BASE, 14, settings)
        with patch.object(problem, "evaluate", side_effect=AssertionError("No oracle")):
            candidate, metadata = optimizer.propose(history, [problem.canonical(raw)], 0)
        self.assertAlmostEqual(candidate["continuous"]["x"], .67, places=3)
        self.assertAlmostEqual(candidate["continuous"]["y"], .21, places=3)
        self.assertEqual(metadata["mode"], "probability_weighted_feasible_improvement")
        self.assertEqual(metadata["raw_history_digest"], digest([row.to_dict() for row in history]))
        with self.assertRaises(RuntimeError):
            optimizer.propose(history, [problem.canonical(raw)], 1)
        with self.assertRaises(ValueError):
            optimizer.accept(history[0])
        settled = problem.record(candidate, 1., {"region": -.1, "resource": -.2},
                                 {"termination_reason": "mock_fixture"})
        optimizer.accept(settled)
        self.assertAlmostEqual(next(iter(optimizer.radii.values())), .22)

    def test_interval_ablation_changes_training_only(self):
        problem = HierarchicalBranin(BASE)
        anchor, failed = problem.anchors()[0], candidate_pool(problem, 72031, 1)[0][0]
        history = [problem.record(anchor, 10., {"region": -.5, "resource": -.2},
                                  {"termination_reason": "mock_fixture"}),
                   problem.record(failed, OutputObservation("interval", "test:obj", lower=11.),
                       {"region": OutputObservation("interval", "test:g", lower=0., lower_inclusive=False),
                        "resource": -.2}, {"termination_reason": "mock_fixture"})]
        raw = {"continuous": {**anchor["continuous"], "x": .5, "y": .4}, "discrete": anchor["discrete"]}
        captured = []

        def construct(problem, base, records, seed, settings):
            captured.extend(records)
            return SmoothPosterior(problem, base, records, seed, settings)

        settings = RevisionSettings(local=False, use_intervals=False)
        optimizer = RiskRefiningBO(problem, BASE, 4, settings)
        with patch("ra_lsbo.soc_revision.optimizer.RiskPosterior", side_effect=construct):
            candidate, metadata = optimizer.propose(history, [problem.canonical(raw)], 0)
        self.assertEqual(captured[1].observation.objective.kind, "missing")
        self.assertEqual(captured[1].observation.constraint("region").kind, "missing")
        self.assertEqual(history[1].observation.objective.kind, "interval")
        self.assertNotEqual(metadata["history_digest"], metadata["raw_history_digest"])
        settled = problem.record(candidate, 1., {"region": -.1, "resource": -.2},
                                 {"termination_reason": "mock_fixture"})
        optimizer.accept(settled)
        self.assertEqual(optimizer.expected_prefix, digest([row.to_dict() for row in history+[settled]]))

    def test_real_mixture_local_search_replays_without_evaluate(self):
        problem = HierarchicalBranin(BASE)
        initial = candidate_pool(problem, 72032, 6)[0]
        history = [problem.record(candidate, 2.+index, {"region": -.2, "resource": -.1},
            {"termination_reason": "mock_fixture"}) for index, candidate in enumerate(initial)]
        pool = candidate_pool(problem, 72033, 8, [row.observation.design_id for row in history])[0]
        settings = RevisionSettings(encoder_epochs=3, continuous_grid=(.15, .4), discrete_grid=(.5,),
            local_branches=1, local_starts=1, local_points=2, local_maxiter=3)
        with patch.object(problem, "evaluate", side_effect=AssertionError("No oracle")):
            first = RiskRefiningBO(problem, BASE, 17, settings).propose(history, pool, 0)
            second = RiskRefiningBO(problem, BASE, 17, settings).propose(history, pool, 0)
        self.assertEqual(first, second)
        self.assertIn("components", first[1]["forecasts"]["constraints"]["region"])


if __name__ == "__main__":
    unittest.main()
