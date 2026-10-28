from dataclasses import asdict
import importlib.util
import unittest

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    from tools.encoder_framework_experiment import metrics, summarize, variants
    from tools.encoder_candidate_validation import risk_summary


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class EncoderExperimentTests(unittest.TestCase):
    def test_ablations_change_only_one_declared_setting(self):
        settings = variants()
        reference = asdict(settings["paper"])
        expected = {
            "no_performance": "performance_weight", "no_entropy": "entropy_weight",
            "constant_gamma": "performance_schedule", "no_mmd": "mmd_weight"}
        for name, changed in expected.items():
            self.assertEqual(
                [key for key, value in asdict(settings[name]).items() if value != reference[key]],
                [changed])

    def test_acceptance_retains_unknown_and_unavailable_predictions(self):
        score = metrics([.9, .9, None], [False, None, True])
        self.assertEqual(score["samples"], 1)
        self.assertEqual(score["accepted_at_0_8"], 2)
        self.assertEqual(score["unknown_accepted"], 1)
        self.assertEqual(score["infeasible_among_accepted"], 1)
        self.assertIsNone(metrics([None], [True])["brier"])
        with self.assertRaisesRegex(ValueError, "counts"):
            metrics([.5], [])

    def test_summary_preserves_missing_seed_counts(self):
        model = {
            "variant": "paper", "metrics": {
                cohort: {field: metrics(values, [False, True])
                         for field, values in (("global", [.2, .8]), ("local", [None, None]))}
                for cohort in ("historical_test", "boundary_test")}}
        row = summarize({"models": [model]})[0]
        self.assertEqual(row["historical_global"]["available"], 1)
        self.assertEqual(row["boundary_local"]["planned"], 1)
        self.assertEqual(row["boundary_local"]["available"], 0)
        self.assertIsNone(row["boundary_local"]["mean_brier"])

    def test_candidate_summary_keeps_all_missing_models_and_acceptance(self):
        def model(values):
            return {"metrics": {
                cohort: {calibration: metrics(values, [False, True])
                         for calibration in ("global", "local")}
                for cohort in ("historical_test", "boundary_test")}}
        result = risk_summary([model([.9, .8]), model([None, None])])
        for field in ("historical_global", "boundary_global", "boundary_local"):
            self.assertEqual(result[field]["available"], 1)
            self.assertEqual(result[field]["planned"], 2)
            self.assertEqual(result[field]["accepted_counts"], [2, 0])
            self.assertAlmostEqual(result[field]["mean_brier"], .425)
        self.assertIsNone(risk_summary([model([None, None])])["boundary_local"]["mean_brier"])


if __name__ == "__main__":
    unittest.main()
