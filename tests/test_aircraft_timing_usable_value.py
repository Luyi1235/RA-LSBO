"""Check the two-step utility and observation coverage semantics."""

import unittest

import numpy as np
from scipy.special import ndtr

from aircraft_timing_decision_value import two_call_value
from aircraft_timing_usable_value import timing_joint_cells, two_call_usable
from run_aircraft_timing_usable_value import summarize


class UsableValueTests(unittest.TestCase):
    def test_completed_violation_can_lead_to_a_usable_second_design(self):
        adaptive, static = two_call_usable(.8, .25, np.array([.8]), np.array([.2]),
                                           np.array([[.6], [.15]]))
        self.assertAlmostEqual(adaptive, .68)
        self.assertAlmostEqual(static, .68)

    def test_all_feasible_reduces_to_completion_lookahead(self):
        edges = np.array([0., .25, .50, .75])
        mean, sd, second_z, rho = .2, .4, np.array([-.2, .3]), np.array([.12, -.07])
        cells = timing_joint_cells(mean, sd, second_z, rho, edges)
        np.testing.assert_allclose(cells.sum(axis=0), ndtr(second_z))
        adaptive, static = two_call_usable(ndtr(-mean/sd), 1., np.ones(2), np.ones(2), cells)
        self.assertAlmostEqual(adaptive, two_call_value(mean, sd, second_z, rho, edges))
        self.assertGreaterEqual(adaptive, static)

    def test_aborts_can_change_the_best_continuation(self):
        adaptive, static = two_call_usable(.2, 1., np.ones(2), np.ones(2),
                                           np.array([[.1, .1], [.35, .10], [.10, .35]]))
        self.assertAlmostEqual(adaptive, .90)
        self.assertAlmostEqual(static, .65)

    def test_unobserved_endpoint_is_excluded_from_paired_comparison(self):
        known = {"first": {"id": "first", "outcome": "abort", "usable": False},
                 "second": {"id": "second", "outcome": "abort", "usable": False},
                 "any_usable_by_two": False}
        missing = {"first": {"id": "other", "outcome": None, "usable": None}, "second": None,
                   "any_usable_by_two": None}
        report = summarize([{"state": "paired", "arms": {
            "interval_greedy": known, "usable_greedy": known, "usable_lookahead": missing}}])
        paired = report["contrasts"]["usable_lookahead_vs_usable_greedy"]
        self.assertEqual(paired["changed_first"], 1)
        self.assertEqual(paired["both_endpoints_observed"], 0)
        self.assertEqual(paired["wins"], 0)
        self.assertEqual(paired["losses"], 0)


if __name__ == "__main__":
    unittest.main()
