import unittest

from ra_lsbo.research_statistics import holm, ordered_endpoint, paired_effect, run_metrics


def row(feasible, objective=None, hidden=1):
    return {"risk": {"feasible_label": feasible, "objective": objective, "hidden_success": hidden}}


class ResearchStatisticsTests(unittest.TestCase):
    def test_no_feasible_seed_is_retained_and_censored_at_budget(self):
        result = run_metrics([], [row(False), row(False, hidden=0)], 2)
        self.assertIsNone(result["endpoint_best"])
        self.assertEqual(result["restricted_first_feasible_cost"], 2)
        self.assertFalse(result["first_feasible_event"])
        self.assertTrue(result["administratively_censored"])
        self.assertEqual(ordered_endpoint(None, 10.), 1)

    def test_discovery_on_last_call_keeps_event_distinct_from_censoring(self):
        result = run_metrics([], [row(False), row(True, 9.)], 2)
        self.assertEqual(result["restricted_first_feasible_cost"], 2)
        self.assertTrue(result["first_feasible_event"])
        self.assertFalse(result["administratively_censored"])

    def test_early_stop_and_unknown_are_not_imputed_as_complete_failures(self):
        result = run_metrics([], [row(False)], 4)
        self.assertIsNone(result["restricted_first_feasible_cost"])
        self.assertTrue(result["early_stop_not_administrative_censoring"])
        result = run_metrics([], [row(None)], 1)
        self.assertIsNone(result["restricted_first_feasible_cost"])
        self.assertFalse(result["administratively_censored"])

    def test_initial_incumbent_and_monotone_trajectory(self):
        result = run_metrics([row(True, 10.)], [row(True, 12.), row(True, 9.)], 2)
        self.assertEqual(result["first_feasible_online_call"], 0)
        self.assertEqual([r["best_feasible"] for r in result["trajectory"]], [10., 10., 9.])

    def test_three_pairs_cannot_give_small_two_sided_exact_p(self):
        result = paired_effect([-1., -2., -3.], resamples=1000)
        self.assertEqual(result["exact_two_sided_sign_flip_p"], .25)
        self.assertEqual(result["n_pairs"], 3)
        self.assertTrue(result["small_sample_warning"])
        self.assertEqual(paired_effect([0., 0., 0.], resamples=1000)
                         ["exact_two_sided_sign_flip_p"], 1.)

    def test_ties_and_unavailable_endpoints(self):
        self.assertEqual(ordered_endpoint(None, None), 0)
        self.assertEqual(ordered_endpoint(4., None), -1)
        self.assertEqual(ordered_endpoint(4., 4.), 0)
        with self.assertRaises(ValueError):
            paired_effect([1., float("nan")])

    def test_holm_preserves_missing_planned_comparisons(self):
        adjusted = holm([.01, None, .04, .03])
        self.assertEqual(adjusted, [.04, None, .09, .09])


if __name__ == "__main__":
    unittest.main()
