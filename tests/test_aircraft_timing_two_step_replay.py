import unittest

import numpy as np
from scipy.special import ndtri

from run_aircraft_timing_two_step_replay import two_step_scores


class TwoStepCompletionScores(unittest.TestCase):
    def test_independent_candidates_use_best_distinct_followup(self):
        p = np.array([.6, .3, .1])
        np.testing.assert_allclose(two_step_scores(p, ndtri(p), np.eye(3)), [.72, .72, .64])

    def test_redundant_greedy_choice_can_lose_to_diverse_pair(self):
        p = np.array([.55, .50, .49])
        rho = np.array([[1., .7, .7], [.7, 1., .05], [.7, .05, 1.]])
        scores = two_step_scores(p, ndtri(p), rho)
        self.assertGreater(scores[1], scores[0])
        self.assertAlmostEqual(scores[1], scores[2])

    def test_single_candidate_has_no_continuation(self):
        np.testing.assert_array_equal(two_step_scores([.6], ndtri([.6]), np.eye(1)), [.6])
