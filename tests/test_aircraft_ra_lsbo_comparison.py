import unittest

from run_aircraft_ra_lsbo_comparison import ARMS, compare_calibration, comparisons, plan


class ComparisonProtocolTests(unittest.TestCase):
    def test_frozen_budget_includes_original_and_two_risk_baselines(self):
        spec = plan("comparison-headwind")
        self.assertEqual(spec["seeds"], plan("comparison-payload")["seeds"])
        self.assertEqual(sum(spec["budget"].values()), 3744)
        self.assertTrue({"original_lsbo", "logcei", "scbo_adapted"}.issubset(ARMS))

    def test_early_native_stop_preserves_censoring_and_missing_energy(self):
        base = {"seed": 7, "initial_kind": "cold"}
        records = [
            {**base, "method": arm, "first_feasible_online_call": 3 if arm == "ra_separate" else None,
             "best_feasible": 100. if arm == "ra_separate" else None}
            for arm in ARMS
        ]
        output = comparisons(records, plan("comparison-headwind"))
        native = next(row for row in output if row["comparator"] == "original_lsbo" and row["cohort"] == "cold")
        self.assertEqual(native["capped_first_call_delta"]["mean"], -18.)
        self.assertEqual(native["discovery_by_8_delta"]["mean"], 1.)
        self.assertIsNone(native["energy_delta_Wh"]["mean"])

    def test_calibration_keeps_seed_pair_and_counts_failed_fit(self):
        rows = [{"seed": 8, "n": n, "method": arm, "completion": .25,
                 "usable": None if n == 8 and arm == "ra_joint_kernel" else .125}
                for n in (4, 8) for arm in ("ra_joint_kernel", "ra_separate")]
        result = compare_calibration(rows)
        self.assertEqual(result["completion"]["seed_deltas"], [[8, 0.]])
        self.assertEqual(result["usable"]["incomplete_seeds"], 1)


class HandoffProtocolTests(unittest.TestCase):
    def test_frozen_budget_and_backend_switch(self):
        from types import SimpleNamespace
        from run_aircraft_ra_lsbo_comparison import active_backend, FULL_NATIVE

        spec = plan("handoff-headwind")
        cold, warm = [SimpleNamespace(usable_feasible=usable) for usable in (False, True)]
        self.assertEqual(sum(spec["budget"].values()), 2976)
        self.assertEqual(spec["seeds"], plan("handoff-payload")["seeds"])
        self.assertTrue(set(spec["seeds"]).isdisjoint(plan("comparison-headwind")["seeds"]))
        self.assertEqual(active_backend("ra_to_lsbo", [cold]), "ra_separate")
        self.assertEqual(active_backend("ra_to_lsbo", [cold, warm]), FULL_NATIVE)
        self.assertEqual(active_backend("logcei_to_lsbo", [warm]), FULL_NATIVE)

    def test_prefix_audit_rejects_pre_handoff_changes(self):
        from types import SimpleNamespace
        from run_aircraft_ra_lsbo_comparison import check_handoff_paths, FULL_NATIVE

        a = SimpleNamespace(usable_feasible=False, design="a")
        b = SimpleNamespace(usable_feasible=False, design="b")
        winner = SimpleNamespace(usable_feasible=True, design="winner")
        paths = {"original_lsbo": [a], FULL_NATIVE: [a, b], "ra_separate": [b, winner, a],
                 "ra_to_lsbo": [b, winner, b], "logcei": [a, b], "logcei_to_lsbo": [a, b]}
        check_handoff_paths([a], paths)
        with self.assertRaisesRegex(ValueError, "declared cold"):
            check_handoff_paths([a], {**paths, "ra_to_lsbo": [a, winner, b]})
        with self.assertRaisesRegex(ValueError, "pre-stop"):
            check_handoff_paths([a], {**paths, FULL_NATIVE: [b, a]})
