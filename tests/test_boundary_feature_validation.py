from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_MODELS = all(importlib.util.find_spec(name) for name in
                 ("numpy", "torch", "sklearn", "botorch", "matplotlib"))
if HAS_MODELS:
    from tools.boundary_feature_validation import (
        MODES, SEEDS, analyze, calibrate, label_blind_result, local_fit, read_rows, save, save_rows, sha, watch)


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class BoundaryFeatureValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.campaign = Path(self.directory.name)
        self.folder = self.campaign / "feature_validation"
        self.folder.mkdir()
        self.plan = {"campaign": str(self.campaign), "models": {
            "latent/seed_0": {"raw_diagnostics": {"status": "fitted_raw"}}}, "source_sha256": {}}
        self.raw = [{"model": "latent/seed_0", "index": index, "candidate_key": str(index),
                     "band": ("outer", "near", "inner")[(index-1) % 3],
                     "raw": .1+.08*(index % 10), "global": .8} for index in range(1, 121)]
        self.results = [{"index": index, "risk": {"feasible_label": index % 10 >= 5}}
                        for index in range(1, 61)]
        save_rows(self.folder / "raw_forecasts.jsonl", self.raw)
        self.plan["raw_forecasts_sha256"] = sha(self.folder / "raw_forecasts.jsonl")
        save(self.folder / "plan.json", self.plan)
        save(self.campaign / "result.json", {"status": "running", "results": self.results})

    def test_locals_fit_only_first_60_and_forecasts_are_sealed_before_any_test_outcome(self):
        calibrate(self.folder)
        local = json.loads((self.folder / "local_calibration.json").read_text())
        forecasts = read_rows(self.folder / "test_forecasts.jsonl")
        self.assertEqual(local["status"], "saved_before_heldout_outcomes")
        self.assertEqual(local["result_counts_before_after"], [60, 60])
        self.assertEqual([row["index"] for row in forecasts], list(range(61, 121)))
        diagnostics, expected = local_fit(self.plan, self.raw, self.results)
        self.assertEqual(forecasts, expected)
        self.assertEqual(local["diagnostics"], diagnostics)
        with self.assertRaises(FileExistsError):
            calibrate(self.folder)

    def test_first_heldout_outcome_closes_registration_window(self):
        report = {"status": "running", "results": self.results+[
            {"index": 61, "risk": {"feasible_label": False}}]}
        (self.campaign / "result.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Held-out outcomes already available"):
            label_blind_result(self.campaign)
        with patch("tools.boundary_feature_validation.local_fit") as fitted:
            with self.assertRaises(ValueError):
                calibrate(self.folder)
            fitted.assert_not_called()
        self.assertFalse((self.folder / "test_forecasts.jsonl").exists())

    def test_watch_retains_missed_window_without_creating_predictions(self):
        report = {"status": "running", "results": self.results+[{"index": 61}]}
        (self.campaign / "result.json").write_text(json.dumps(report), encoding="utf-8")
        watch(self.folder)
        status = json.loads((self.folder / "local_calibration.json").read_text())
        self.assertEqual(status["status"], "missed_heldout_forecast_window")
        self.assertFalse((self.folder / "test_forecasts.jsonl").exists())

    def test_failed_class_gate_stays_unavailable_and_extra_calibration_rows_are_rejected(self):
        all_feasible = deepcopy(self.results)
        for item in all_feasible:
            item["risk"]["feasible_label"] = True
        diagnostics, forecasts = local_fit(self.plan, self.raw, all_feasible)
        self.assertEqual(diagnostics["latent/seed_0"]["status"], "fallback_insufficient_classes")
        self.assertTrue(all(row["local"] is None for row in forecasts))
        with self.assertRaisesRegex(ValueError, "first 60"):
            local_fit(self.plan, self.raw, self.results+[{"index": 61}])

    def test_arrival_during_calibration_is_explicitly_excluded(self):
        original = local_fit

        def fitted_then_arrived(plan, rows, results):
            fitted = original(plan, rows, results)
            (self.campaign / "result.json").write_text(json.dumps({
                "status": "running", "results": self.results+[{"index": 61}]}), encoding="utf-8")
            return fitted

        with patch("tools.boundary_feature_validation.local_fit", side_effect=fitted_then_arrived):
            watch(self.folder)
        status = json.loads((self.folder / "local_calibration.json").read_text())
        self.assertEqual(status["status"], "missed_heldout_forecast_window")
        self.assertFalse((self.folder / "test_forecasts.jsonl").exists())

    def test_full_secondary_pipeline_checks_frozen_predictions_without_mda(self):
        self.plan["models"] = {mode+"/seed_"+str(seed): {
            "raw_diagnostics": {"status": "fitted_raw"}} for mode in MODES for seed in SEEDS}
        rows = [dict(row, model=name) for name in self.plan["models"] for row in self.raw]
        (self.folder / "raw_forecasts.jsonl").write_text(
            "".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        self.plan["raw_forecasts_sha256"] = sha(self.folder / "raw_forecasts.jsonl")
        save(self.campaign / "plan.json", {"schema": "fake_parent"})
        self.plan["parent_plan_sha256"] = sha(self.campaign / "plan.json")
        save(self.folder / "registration_before.json", {"results": self.results[:30]})
        save(self.folder / "registration_after.json", {"results": self.results[:30]})
        self.plan["registration_witness_sha256"] = {
            name: sha(self.folder / name) for name in ("registration_before.json", "registration_after.json")}
        (self.folder / "plan.json").write_text(json.dumps(self.plan), encoding="utf-8")
        calibrate(self.folder)
        test_results = [{"index": index, "risk": {"feasible_label": index % 10 >= 5}}
                        for index in range(61, 121)]
        (self.campaign / "result.json").write_text(json.dumps({
            "status": "complete", "results": self.results+test_results}), encoding="utf-8")
        save(self.campaign / "audit.json", {
            "status": "passed", "input_sha256": {str(self.campaign / "result.json"): sha(self.campaign / "result.json")}})
        analyze(self.folder)
        result = json.loads((self.folder / "analysis.json").read_text())
        self.assertEqual(result["test_calls"], 60)
        self.assertEqual(result["known_test_labels"], 60)
        self.assertEqual(len(result["metrics"]), 20)
        self.assertEqual(len(result["paired_local_brier"]), 15)
        self.assertTrue(all(pair["mean_brier_difference"] == 0 for pair in result["paired_local_brier"]))
        self.assertEqual(result["new_mda_calls"], 0)


if __name__ == "__main__":
    unittest.main()
