import unittest

from aircraft_risk_discovery import discovery_scores
from aircraft_risk_protocol import FEEDBACK_BASELINE_PAIRS, plan


class FeedbackBaselineTests(unittest.TestCase):
    def test_registered_profiles_match_extended_aircraft_conditions(self):
        wind = plan("feedback-baseline-headwind")
        payload = plan("feedback-baseline-payload")
        smoke = plan("feedback-baseline-smoke")
        self.assertEqual(wind["seeds"], payload["seeds"])
        self.assertTrue(set(wind["seeds"]).isdisjoint(plan("sign-lsbo-headwind")["seeds"]))
        self.assertTrue(wind["staged_discovery"])
        self.assertEqual(wind["arms"], ["lsbo_sign_latent", "discovery_staged", "discovery_direct"])
        self.assertEqual(wind["constraint_feedback"],
                         {"lsbo_sign_latent": "sign", "discovery_staged": "bounds", "discovery_direct": "bounds"})
        self.assertEqual((wind["headwind_m_s"], payload["payload_delta_kg"]), (4., 24.))
        self.assertEqual(wind["budget"], {"initial": 96, "online": 1440, "heldout": 768})
        self.assertEqual(wind["budget"], payload["budget"])
        self.assertEqual(sum(smoke["budget"].values()), 28)
        self.assertEqual(FEEDBACK_BASELINE_PAIRS[0], ("lsbo_sign_latent", "discovery_staged"))

    def test_three_cold_scores_use_different_feedback(self):
        values = {"mission": [.4, .6], "feasible_given_mission": [.5, .5],
                  "usable_factored": [.2, .3], "usable_direct": [.7, .1],
                  "joint": [.1, .9], "coverage": [.4, .3]}
        packet = {"probabilities": values, "rollout": [], "outcomes": {"abort": 1}}
        for mode, forecast in (("constraints", "joint"), ("staged", "usable_factored"), ("direct", "usable_direct")):
            self.assertEqual(discovery_scores(mode, packet, ["a", "b"], 4), values[forecast])
