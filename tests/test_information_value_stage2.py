import unittest

from ra_lsbo.experiments.information_value_stage2.protocol import (
    ARMS, CASES, SEEDS, all_candidate_support, feasible_first, plan,
)


class Stage2ProtocolTests(unittest.TestCase):
    def test_registered_roster(self):
        self.assertEqual(len(CASES), 4)
        self.assertEqual(len(SEEDS), 12)
        self.assertEqual(ARMS, ("soft_gp_eligible", "soft_gp_all_support"))

    def test_smoke_budget(self):
        p = plan("smoke", "a" * 40)
        self.assertEqual(p["initial_calls"], 48)
        self.assertEqual(p["online_calls"], 16)
        self.assertEqual(p["total_mathematical_calls"], 64)

    def test_full_budget(self):
        p = plan("full", "b" * 40)
        self.assertEqual(p["initial_calls"], 576)
        self.assertEqual(p["online_calls"], 576)
        self.assertEqual(p["total_mathematical_calls"], 1152)

    def test_all_support_keeps_pool_cardinality(self):
        original = [True, False, False, True]
        self.assertEqual(all_candidate_support(original, 4), [True] * 4)

    def test_all_support_does_not_mutate_original_mask(self):
        original = [False, True]
        all_candidate_support(original, 2)
        self.assertEqual(original, [False, True])

    def test_rejects_misaligned_support(self):
        with self.assertRaises(ValueError):
            all_candidate_support([True], 2)

    def test_feasible_first_missing_is_not_imputed(self):
        self.assertEqual(feasible_first({"final_best": 2.0}, {"final_best": None}), 1)
        self.assertEqual(feasible_first({"final_best": None}, {"final_best": None}), 0)

    def test_feasible_first_objective_order(self):
        self.assertEqual(feasible_first({"final_best": 1.0}, {"final_best": 2.0}), 1)
        self.assertEqual(feasible_first({"final_best": 2.0}, {"final_best": 1.0}), -1)

    def test_invalid_profile_rejected(self):
        with self.assertRaises(ValueError):
            plan("pilot", "c" * 40)


if __name__ == "__main__":
    unittest.main()
