from copy import deepcopy
import importlib.util
import unittest

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch", "matplotlib"))
if HAS_MODELS:
    from tools.analyze_boundary_validation import analyze, paired_difference


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class BoundaryAnalysisTests(unittest.TestCase):
    def fixture(self, count=60):
        proposals, forecasts = [], []
        for index in range(1, 121):
            raw = .1+.08*(index % 10)
            proposals.append({"raw_rf": raw, "raw_constraint": raw*.9})
            if index > 60:
                forecasts.append({
                    "index": index, "band": ("outer", "near", "inner")[(index-1) % 3],
                    "global_rf": .8, "local_rf": .9,
                    "global_constraint": .7, "local_constraint": None,
                })
        results = [{"index": index, "risk": {"feasible_label": index <= 60, "state": "feasible" if index <= 60 else "infeasible"}}
                   for index in range(1, 61+count)]
        return ({"proposals": proposals},
                {"results": results, "status": "complete" if count == 60 else "stopped_by_policy", "ledger": {}},
                {"forecasts": forecasts, "diagnostics": {}})

    def test_only_heldout_labels_are_used_for_loss(self):
        plan, result, local = self.fixture()
        report, _ = analyze(plan, result, local)
        self.assertEqual(report["completed_calibration"], 60)
        self.assertEqual(report["metrics"]["global_rf"]["samples"], 60)
        self.assertAlmostEqual(report["metrics"]["global_rf"]["brier"], .64)
        self.assertAlmostEqual(report["metrics"]["local_rf"]["brier"], .81)
        self.assertIsNone(report["metrics"]["local_constraint"]["brier"])
        self.assertEqual(report["paired_brier_differences"][1]["samples"], 0)

    def test_unknown_and_unexecuted_remain_separate(self):
        plan, result, local = self.fixture(3)
        result["results"][-1]["risk"]["feasible_label"] = None
        report, _ = analyze(plan, result, local)
        self.assertEqual(report["completed_test"], 3)
        self.assertEqual(report["known_test_labels"], 2)
        self.assertEqual(report["unknown_test_labels"], 1)
        self.assertEqual(report["unexecuted_test"], 57)
        self.assertEqual(report["metrics"]["global_rf"]["samples"], 2)

    def test_pairing_uses_same_rows_and_deterministic_stratum_resampling(self):
        rows = [{"band": band, "feasible": True, "a": .8, "b": .8}
                for band in ("outer", "near", "inner") for _ in range(3)]
        rows[0]["b"] = None
        first = paired_difference(rows, "a", "b", draws=100)
        second = paired_difference(rows, "a", "b", draws=100)
        self.assertEqual(first, second)
        self.assertEqual(first["samples"], 8)
        self.assertEqual(first["mean_brier_difference"], 0.)
        self.assertEqual(first["bootstrap_interval"], [0., 0.])

    def test_fixed_coverage_uses_underlying_score_not_saturated_probability(self):
        plan, result, local = self.fixture()
        before = deepcopy(local)
        report, _ = analyze(plan, result, local)
        for old, new in zip(report["fixed_acceptance"]["global_rf"], report["fixed_acceptance"]["local_rf"]):
            self.assertEqual(old["indices"], new["indices"])
        self.assertEqual(before, local)

    def test_precheck_rejection_is_not_a_completed_or_unknown_call(self):
        plan, result, local = self.fixture(3)
        result["results"][-1]["risk"] = None
        report, _ = analyze(plan, result, local)
        self.assertEqual(report["completed_test"], 2)
        self.assertEqual(report["unknown_test_labels"], 0)
        self.assertEqual(report["unexecuted_test"], 58)
        self.assertEqual(report["unsubmitted_precheck_test"], 1)

    def test_unknown_is_retained_in_fixed_count_and_threshold_acceptance(self):
        plan, result, local = self.fixture(20)
        plan["proposals"][79]["raw_rf"] = 1.
        result["results"][-1]["risk"]["feasible_label"] = None
        report, _ = analyze(plan, result, local)
        selected = report["fixed_acceptance"]["global_rf"][0]
        self.assertIn(80, selected["indices"])
        self.assertEqual(selected["unknown_accepted"], 1)
        self.assertEqual(selected["known_accepted"], 9)
        self.assertEqual(selected["infeasible"], 9)
        self.assertEqual(report["acceptance_at_0_8"]["global_rf"]["accepted"], 20)
        self.assertEqual(report["acceptance_at_0_8"]["global_rf"]["unknown_accepted"], 1)


if __name__ == "__main__":
    unittest.main()
