import unittest

import numpy as np
from scipy.special import ndtr

from aircraft_timing_decision_value import two_call_value


class TestTwoCallValue(unittest.TestCase):
    def test_independent_feedback_matches_two_greedy_opportunities(self):
        mean, sd = .13, .46
        reference_z = np.array([-.2, .4])
        first = ndtr(-mean/sd)
        expected = first+(1-first)*ndtr(.4)
        value = two_call_value(mean, sd, reference_z, np.zeros(2), [0., .1, .2, .5])
        self.assertAlmostEqual(value, expected, places=13)

    def test_complementary_second_design_recovers_every_first_abort(self):
        z = np.array([0.])
        self.assertAlmostEqual(two_call_value(0., 1., z, np.array([-1.]), [0.]), 1.)
        self.assertAlmostEqual(two_call_value(0., 1., z, np.array([1.]), [0.]), .5)

    def test_refining_the_timing_feedback_does_not_reduce_model_value(self):
        z = np.array([.35, -.25])
        correlation = np.array([.75, -.75])
        coarse = two_call_value(.15, .6, z, correlation, [0.])
        refined = two_call_value(.15, .6, z, correlation, [0., .15, .4, .8])
        self.assertGreaterEqual(refined, coarse-1e-12)


if __name__ == "__main__":
    unittest.main()
