import unittest

import numpy as np

from aircraft_risk_acquisition import sign_variance_reduction
from aircraft_timing_interval_value import deficit_edges, interval_variance_reduction


class TestIntervalValue(unittest.TestCase):
    def test_coarsened_intervals_match_sign_acquisition(self):
        mean, sd = .15, .34
        reference = np.array([-1.3, 0., .8])
        rho = np.array([-.5, 0., .7])
        sign = sign_variance_reduction(np.array([-mean/sd]), reference, rho[None, :])[0]
        coarse = interval_variance_reduction(mean, sd, reference, rho, [0.])
        refined = interval_variance_reduction(mean, sd, reference, rho, [0., .1, .2, .3])
        np.testing.assert_allclose(coarse, sign, atol=1e-13)
        self.assertTrue(np.all(refined >= coarse-1e-13))
        self.assertGreater(refined[-1], coarse[-1])
        self.assertEqual(refined[1], 0.)

    def test_feedback_grid_covers_each_integration_step(self):
        candidate = {"continuous": {"cruise_airspeed": 50.}}
        settings = {"scenario": {"headwind_m_s": 0.}, "climb_height_m": 60.,
                    "nominal_cruise_distance_m": 2000., "reserve_duration_s": 10.,
                    "integration_step_s": 20.}
        edges = deficit_edges(candidate, settings)
        np.testing.assert_allclose(edges, np.array([0, 10, 30, 50, 70, 90, 105])/120)


if __name__ == "__main__":
    unittest.main()
