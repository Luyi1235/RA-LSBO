"""Check shared candidate rosters, per-policy budget and factorial contrasts."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_aircraft_coverage_factorial as trial


class CoverageFactorialTests(unittest.TestCase):
    def test_covered_roster_balances_branches_and_spreads_continuous_values(self):
        raw = [{"discrete": {"layout": layout}, "x": x}
               for layout, x in (("A", 0.), ("A", .1), ("A", .9), ("B", .3), ("B", .7))]
        with patch.object(trial, "active_distance", side_effect=lambda _, a, b: abs(a["x"]-b["x"])):
            roster = trial.covered_roster(None, raw, 4)
        self.assertEqual(roster, [raw[i] for i in (0, 3, 2, 4)])
        self.assertEqual(len(raw), 5)

    def test_two_pools_share_size_and_exclude_common_initialization(self):
        spec = trial.plan("smoke")
        problem = trial.UsableAircraft(spec)
        initial, pools = trial.rosters(problem, spec, spec["seeds"][0])
        self.assertEqual(len(initial), 16)
        seen = {trial.digest(p) for p in initial}
        for pool in pools.values():
            identifiers = {trial.digest(p) for p in pool}
            self.assertEqual(len(identifiers), spec["online_size"])
            self.assertTrue(seen.isdisjoint(identifiers))

    def test_policy_stops_at_first_usable_and_counts_logical_calls(self):
        initial = [SimpleNamespace(usable_feasible=False)]*8
        result = SimpleNamespace(usable_feasible=True, to_dict=lambda: {})
        with patch.object(trial, "choose", return_value=(0, {"rule": "greedy"})), \
                patch.object(trial, "discovery_outcome", return_value="usable"):
            receipt = trial.play(None, 1, initial, [{"x": 1}, {"x": 2}], 48,
                                 "greedy", lambda _, i: result)
        self.assertEqual(receipt["first_usable"], 9)
        self.assertEqual(receipt["online_evaluations"], 1)
        self.assertEqual(len(initial), 8)

    def test_interaction_uses_the_paired_active_effect_difference(self):
        times = {"standard_greedy": 22, "standard_active": 20,
                 "covered_greedy": 19, "covered_active": 14}
        row = {"arms": {name: {"first_usable": t, "capped_first": t, "trace": []}
                        for name, t in times.items()}}
        result = trial.contrasts([row])["delta_calls"]
        self.assertEqual(result["active_under_covered"]["mean"], -5)
        self.assertEqual(result["coverage_under_greedy"]["mean"], -3)
        self.assertEqual(result["interaction"]["mean"], -3)


if __name__ == "__main__":
    unittest.main()
