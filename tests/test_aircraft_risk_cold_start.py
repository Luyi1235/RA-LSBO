import unittest

import numpy as np
from scipy.special import ndtr

from aircraft_risk_acquisition import posterior_cross_covariance, sign_variance_reduction
from aircraft_risk_protocol import plan
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings, MixedObservationGP
from ra_lsbo.risk_core.observations import OutputObservation
from run_aircraft_risk_cold_start import checked_label, select


class AircraftRiskColdStartTests(unittest.TestCase):
    def test_budget(self):
        self.assertEqual(plan("smoke")["budget"], {"initial": 4, "online": 12, "heldout": 12})
        self.assertEqual(plan("pilot")["budget"], {"initial": 72, "online": 1296, "heldout": 768})

    def test_sign_information(self):
        z = np.array([0.])
        rho = np.array([[0., .5, -.5, 1., -1.]])
        actual = sign_variance_reduction(z, np.zeros(5), rho)
        expected = np.arcsin(rho)**2/np.pi**2
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        z = np.array([-1., .7, 2.])
        actual = sign_variance_reduction(z, z, np.eye(3))
        np.testing.assert_allclose(actual.diagonal(), ndtr(z)*ndtr(-z), atol=1e-13)

    def test_ep_cross_covariance_matches_prediction_variance(self):
        model = MixedObservationGP(
            [[.1], [.35]], [OutputObservation("exact", "observed", value=-.2),
                            OutputObservation("interval", "early_stop", lower=.1)],
            settings=MixedGPSettings(stable_tail_moments=True))
        query = np.array([[.12], [.3], [.9]])
        covariance = posterior_cross_covariance(model, query, query)
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        np.testing.assert_allclose(covariance.diagonal(), model.predict(query)["variance"], atol=1e-12)

    def test_observable_constraint_versus_numerical_failure(self):
        row = {"observation": {"status": "completed", "constraints": [
            ["soc", {"kind": "interval", "upper": None, "lower": .03, "lower_inclusive": True}],
            ["current", {"kind": "missing"}]]}, "diagnostics": {}}
        self.assertIs(checked_label(row), False)
        self.assertIsNone(checked_label(row, "current"))
        row["diagnostics"]["numerical_failure"] = True
        self.assertIsNone(checked_label(row))

    def test_selection_uses_candidate_aligned_scores(self):
        self.assertEqual(select([{"x": 1}, {"x": 2}], [.2, .5]), 1)
        with self.assertRaises(ValueError):
            select([{"x": 1}], [])


if __name__ == "__main__":
    unittest.main()
