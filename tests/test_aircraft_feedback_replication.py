"""Check feedback masking, shared pools and predeclared contrasts."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_aircraft_coverage_factorial as coverage
import run_aircraft_feedback_replication as trial
from ra_lsbo.risk_core.observations import OutputObservation


class FeedbackReplicationTests(unittest.TestCase):
    def test_sign_head_requests_coarsened_observations(self):
        space = SimpleNamespace(continuous=(None,))
        problem = SimpleNamespace(space=space, settings={})
        record = SimpleNamespace(to_dict=lambda: {})
        bound = OutputObservation("interval", "smoke", lower=0.)
        with patch.object(coverage, "timing_feedback", return_value=bound) as labels, \
                patch.object(coverage, "EvidenceGP"):
            coverage.FixedTimingHead(problem, np.zeros((1, 2)), [record], 11, use_interval=False)
        self.assertFalse(labels.call_args.kwargs["use_interval"])

    def test_campaign_passes_each_arm_its_declared_feedback_modes(self):
        spec = trial.plan("smoke")
        initial = [{"x": i} for i in range(16)]
        covered = [{"x": 19}]
        record = SimpleNamespace(usable_feasible=False, to_dict=lambda: {})
        received = []

        def play(_, seed, history, pool, budget, policy, acquire, *, selector):
            self.assertIs(pool, covered)
            self.assertEqual(budget, 18)
            received.append((len(history), policy, selector.keywords))
            return {"capped_first": 19}

        with patch.object(trial, "rosters", return_value=(initial, {"covered": covered})), \
                patch.object(trial, "discovery_outcome", return_value="abort"), \
                patch.object(trial, "play", side_effect=play):
            trial.campaign(spec, None, lambda point, tag, role: record)
        self.assertEqual(received, [
            (size, policy, {"fit_interval": fit, "score_interval": score})
            for size in (8, 16) for policy, fit, score in trial.ARMS.values()
        ])

    def test_mechanism_contrasts_partition_the_active_feedback_difference(self):
        times = (23, 20, 28, 27, 24)
        row = {"arms": {k: {"first_usable": t, "capped_first": t, "trace": []}
                        for k, t in zip(trial.ARMS, times)}}
        comparisons = trial.summarize([row])["differences"]
        self.assertEqual(comparisons["feedback_by_policy_interaction"]["mean"], -2)
        total = comparisons["interval_active_minus_sign_active"]["mean"]
        fit = comparisons["interval_fit_sign_score_minus_sign_active"]["mean"]
        scoring = comparisons["interval_active_minus_interval_fit_sign_score"]["mean"]
        self.assertEqual(total, fit+scoring)


if __name__ == "__main__":
    unittest.main()
