import unittest

import numpy as np
from scipy.special import ndtri

from aircraft_risk_discovery import discovery_pairs, discovery_scores, label, rollout_components, shortlist
from aircraft_risk_protocol import plan
from run_aircraft_risk_cold_start import discovery_outcome


class StagedDiscoveryTests(unittest.TestCase):
    def test_completion_and_feasibility_have_distinct_evidence(self):
        row = {"observation": {"status": "completed", "objective": {"kind": "interval"},
                               "constraints": [["soc", {"kind": "exact", "value": .03}]]},
               "diagnostics": {"mission_completed": False}}
        self.assertEqual(discovery_outcome(row), "abort")
        self.assertIs(label("abort", "mission"), False)
        self.assertIsNone(label("abort", "feasible_given_mission"))
        self.assertIs(label("abort", "usable_direct"), False)
        row["diagnostics"]["mission_completed"] = True
        row["observation"]["objective"]["kind"] = "exact"
        self.assertEqual(discovery_outcome(row), "violation")
        self.assertIs(label("violation", "mission"), True)
        self.assertIs(label("violation", "feasible_given_mission"), False)
        row["diagnostics"]["numerical_failure"] = True
        self.assertEqual(discovery_outcome(row), "unknown")
        self.assertIsNone(label("unknown", "usable_direct"))

    def test_rollout_excludes_current_candidate_and_marginalizes_both_branches(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        c, f = np.array([.8, .4, .9]), np.array([.4, .9, .5])
        query = np.arange(3.).reshape(-1, 1)
        heads = {name: SimpleNamespace(model=None, divisor=1., scale=1.)
                 for name in ("mission", "feasible_given_mission")}
        moments = {name: {"variance": np.zeros(3)} for name in heads}
        probabilities = {"mission": c, "feasible_given_mission": f}
        with patch("aircraft_risk_acquisition.posterior_cross_covariance",
                   return_value=np.zeros((1, 3))):
            [[index, abort, violation, collapsed]] = rollout_components(heads, query, probabilities, moments,
                                                            {name: ndtri(probabilities[name]) for name in heads}, [2])
        self.assertEqual(index, 2)
        best_other = .36
        self.assertAlmostEqual(abort, (1-.9)*best_other)
        self.assertAlmostEqual(violation, .9*(1-.5)*best_other)
        self.assertAlmostEqual(collapsed, abort + violation)
        self.assertAlmostEqual(.45 + abort + violation, 1-(1-.45)*(1-best_other))

    def test_last_call_has_no_rollout_and_profile_is_paired_by_wind(self):
        p = {"mission": [.8, .5], "feasible_given_mission": [.5, .4],
             "usable_factored": [.4, .2], "usable_direct": [.3, .6],
             "joint": [.5, .7], "coverage": [.9, 1.1]}
        receipt = {"probabilities": p, "rollout": [], "outcomes": {"abort": 1}}
        self.assertEqual(discovery_scores("staged_lookahead", receipt, ["a", "b"], 1), [.4, .2])
        self.assertEqual(shortlist(p, ["a", "b"]), [0, 1])
        receipt["rollout"] = [[0, .05, .1, .11], [1, .05, .1, .12]]
        np.testing.assert_allclose(discovery_scores("staged_lookahead", receipt, ["a", "b"], 2), [.55, .35])
        np.testing.assert_allclose(discovery_scores("collapsed_lookahead", receipt, ["a", "b"], 2), [.51, .32])
        nominal, wind = plan("discovery-nominal"), plan("discovery-headwind")
        self.assertEqual(nominal["seeds"], wind["seeds"])
        self.assertEqual(nominal["budget"], {"initial": 64, "online": 1536, "heldout": 512})
        self.assertEqual(sum(plan("discovery-smoke")["budget"].values()), 38)

    def test_analysis_retains_nondiscovery(self):
        rows = [{"seed": 1, "arm": arm, "initial_outcomes": {"abort": 4}, "initial_usable": 0,
                 "first_usable_feasible_call": 10 if arm == "discovery_lookahead" else None}
                for arm in plan("discovery-headwind")["arms"]]
        first = discovery_pairs(rows, 20)[0]
        self.assertEqual(first["capped_calls_delta"], -11)
        self.assertIsNone(first["baseline_first_usable_call"])
        self.assertEqual(first["outcome"], "win")
