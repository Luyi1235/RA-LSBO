import unittest

import numpy as np
from scipy.special import ndtr

from aircraft_risk_acquisition import posterior_cross_covariance, sign_variance_reduction
from aircraft_risk_protocol import plan
from aircraft_risk_local_search import check_augmentation, local_anchor
from aircraft_risk_energy import energy_scores, energy_pairs
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings, MixedObservationGP
from ra_lsbo.risk_core.observations import OutputObservation
from run_aircraft_risk_cold_start import checked_label, select


class AircraftRiskColdStartTests(unittest.TestCase):
    def test_budget(self):
        self.assertEqual(plan("smoke")["budget"], {"initial": 4, "online": 12, "heldout": 12})
        self.assertEqual(plan("pilot")["budget"], {"initial": 72, "online": 1296, "heldout": 768})

    def test_coverage_phase_budget_and_parent(self):
        smoke, pilot = plan("phase-smoke"), plan("phase-pilot")
        self.assertEqual(smoke["budget"], {"initial": 4, "online": 16, "heldout": 12})
        self.assertEqual(pilot["budget"], {"initial": 144, "online": 2304, "heldout": 1536})
        self.assertEqual(pilot["coverage_switch"], 24)
        self.assertEqual(pilot["total"], 48)
        self.assertLess(pilot["arms"].index(pilot["shared_coverage_arm"]),
                        pilot["arms"].index("coverage_brier"))
        self.assertFalse(set(pilot["seeds"]) & set(plan("pilot")["seeds"]))

    def test_energy_design_uses_matched_budgets_and_rules(self):
        p = plan("energy-pilot")
        self.assertEqual(p["budget"], {"initial": 144, "online": 1728, "heldout": 768})
        self.assertEqual(p["objective_update_every"], 8)
        self.assertFalse(set(p["seeds"]) & set(plan("phase-pilot")["seeds"]))
        self.assertEqual(energy_scores("ei", [2., 1.], [.1, .8], None), [2., 1.])
        self.assertEqual(energy_scores("cei", [2., 1.], [.1, .8], None), [.2, .8])
        self.assertEqual(energy_scores("cei", None, None, [.4, .1]), [.4, .1])

    def test_energy_pair_keeps_no_solution_and_uses_wh_only_when_both_solve(self):
        arms = plan("energy-pilot")["arms"]
        rows = [{"seed": 1, "arm": arm, "best_feasible_Wh": None if arm == "ei_latent" else 1.,
                 "first_usable_feasible_call": None, "online_feasible": 0, "online_infeasible": 1,
                 "online_unknown": 0} for arm in arms]
        first = energy_pairs(rows)[0]
        self.assertEqual(first["outcome"], "win")
        self.assertIsNone(first["delta_Wh"])

    def test_local_search_shares_prefix_budget_and_keeps_pool_sizes_equal(self):
        p = plan("local-pilot")
        self.assertEqual(p["budget"], {"initial": 144, "online": 576, "heldout": 384})
        self.assertEqual(p["local_switch"], 36)
        self.assertEqual(p["augmentation"], 24)
        a = {"continuous": {"x": .1}, "discrete": {"a": 1}}
        b = {"continuous": {"x": .2}, "discrete": {"a": 1}}
        from ra_lsbo.budget import digest
        row = {"event": "augmentation", "n": 36, "mode": "local", "anchor": digest(a),
               "candidates": [b]}
        check_augmentation(row, 36, 1, "local", a, {digest(a)})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            check_augmentation(row, 36, 1, "local", a, {digest(b)})
        row["candidates"] = [{**b, "discrete": {"a": 2}}]
        with self.assertRaisesRegex(ValueError, "configuration"):
            check_augmentation(row, 36, 1, "local", a, {digest(a)})
        self.assertIsNone(local_anchor([], checked_label))

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
