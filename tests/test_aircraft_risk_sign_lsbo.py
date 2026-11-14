import unittest

from aircraft_risk_protocol import SIGN_LSBO_ARMS, plan
from aircraft_risk_sign_lsbo import budget_trajectory, paired_endpoints, terminal_summary


class SignLSBOTests(unittest.TestCase):
    def test_best_uses_only_usable_results_and_keeps_missing_endpoints(self):
        records = [{"observation": {"objective": {"value": energy}}} for energy in (10., 17., 15., 2.)]
        self.assertEqual(budget_trajectory(records, ["abort", "usable", "usable", "violation"], [1, 2, 4]),
                         [{"n": 1, "best_feasible_Wh": None},
                          {"n": 2, "best_feasible_Wh": 17.},
                          {"n": 4, "best_feasible_Wh": 15.}])
        arms = []
        for arm in SIGN_LSBO_ARMS:
            a = None if arm == "lsbo_coverage_latent" else 15.
            arms.append({"seed": 10, "arm": arm, "initial_usable": 0,
                         "budget_trajectory": [{"n": 4, "best_feasible_Wh": None},
                                               {"n": 36, "best_feasible_Wh": a}]})
        rows = paired_endpoints(arms)
        self.assertEqual(len(rows), 10)
        terminal = terminal_summary(rows, 36)[0]
        self.assertEqual(terminal["wins"], 1)
        self.assertEqual(terminal["both_discovered"], 0)
        self.assertIsNone(terminal["mean_delta_Wh"])

    def test_frozen_scenarios_and_factorial(self):
        h, p = map(plan, ("sign-lsbo-headwind", "sign-lsbo-payload"))
        self.assertEqual(h["seeds"], p["seeds"])
        self.assertTrue(set(h["seeds"]).isdisjoint(plan("feedback-headwind")["seeds"]))
        self.assertEqual(h["budget"], {"initial": 64, "online": 2560, "heldout": 512})
        self.assertEqual(h["budget"], p["budget"])
        self.assertEqual((h["headwind_m_s"], p["payload_delta_kg"]), (4., 24.))
        self.assertEqual(sum(plan("sign-lsbo-smoke")["budget"].values()), 36)
        self.assertEqual(h["constraint_feedback"]["lsbo_sign_latent"], "sign")
        self.assertEqual(SIGN_LSBO_ARMS["lsbo_sign_explicit"], ("constraints", False))
