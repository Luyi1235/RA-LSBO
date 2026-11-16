import unittest

import numpy as np

from aircraft_risk_active import active_scores, configuration_brier, configuration_weights, reduction_components
from aircraft_risk_protocol import plan


class ActiveRiskTests(unittest.TestCase):
    def test_censored_branches_sum_to_expected_brier_reduction(self):
        p = {"mission": np.array([.5]), "feasible_given_mission": np.array([.5])}
        independent = {"mission": np.array([[.25]]), "feasible_given_mission": np.array([[.25]])}
        self.assertEqual(sum(row[0] for row in reduction_components(p, p, independent, [1.]).values()), 0.)
        correlated = {**independent, "mission": np.array([[.5]])}
        parts = reduction_components(p, p, correlated, [1.])
        self.assertEqual(parts, {"abort": [.03125], "violation": [.015625], "usable": [.015625]})
        receipt = {"mission": [.5], "feasible_given_mission": [.5], "coverage": [1.], "reductions": parts}
        self.assertEqual(active_scores("risk_information", receipt, 1), [.0625])
        with self.assertRaises(ValueError):
            active_scores("risk_greedy", receipt, 1)

    def test_configuration_weighting_and_unknowns(self):
        a, b = ({"discrete": {"layout": k}} for k in ("central", "distributed"))
        self.assertEqual(configuration_weights([a, a, b]), [.25, .25, .5])
        self.assertEqual(configuration_brier([(a, True, 0), (a, True, 0), (b, False, 0), (b, None, 1)]), (.5, 2))

    def test_registered_cold_start_budget(self):
        nominal, headwind = (plan("active-risk-"+v) for v in ("nominal", "headwind"))
        self.assertEqual(nominal["seeds"], headwind["seeds"])
        self.assertEqual(nominal["learning_checkpoints"], [8, 12, 16])
        self.assertEqual(nominal["budget"], {"initial": 64, "online": 960, "heldout": 1024})
        self.assertNotIn("objective_update_every", nominal)
