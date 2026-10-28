from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    from ra_lsbo.boundary_validation import (
        BANDS, POLICY, execute_boundary_validation, fit_local_calibration, validate_plan)
    from ra_lsbo.budget import digest
    from ra_lsbo.legacy import normalize_evaluation


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class BoundaryValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.calls = []
        proposals = []
        for index in range(1, 121):
            candidate = {"continuous": {"x": index/121}, "discrete": {}}
            probability = .1+.08*(index % 10)
            proposals.append({
                "index": index, "phase": "calibration" if index <= 60 else "test",
                "band": BANDS[(index-1) % 3][0], "candidate": candidate,
                "raw_rf": probability, "raw_constraint": probability*.95,
                "global_rf": probability, "global_constraint": probability,
                "proposal": {
                    "candidate": candidate, "mode": "discovery", "source": "test",
                    "evidence": {"basis": "raw", "probability": probability, "model_id": "frozen",
                                 "calibration_id": None, "candidate_key": digest(candidate)}}})
        self.plan = {
            "schema": "ra_boundary_validation_plan_v1", "proposals": proposals,
            "budget_policy": asdict(POLICY), "worker_timeout_seconds": 1800,
            "base_project": "fake", "config_path": "fake", "contract_id": "fake", "source_sha256": {}}

    def request(self, approved=True):
        (self.root / "plan.json").write_text(json.dumps(self.plan), encoding="utf-8")
        request = {
            "schema": "ra_boundary_validation_request_v1", "operator_approved": approved,
            "plan_path": str(self.root / "plan.json"),
            "plan_sha256": hashlib.sha256((self.root / "plan.json").read_bytes()).hexdigest()}
        path = self.root / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        return path

    def problem(self, all_feasible=False, unknown=False):
        owner = self

        class FakeProblem:
            contract_id = "fake"
            required_constraints = ("g",)
            analytical_constraints = ()

            def __init__(self, *args, **kwargs):
                pass

            def canonical(self, candidate):
                return candidate

            def precheck(self, candidate):
                return {}

            def evaluate(self, candidate):
                index = len(owner.calls)+1
                if index == 61:
                    owner.assertTrue((owner.root / "test_forecasts.json").exists())
                owner.calls.append(candidate)
                feasible = all_feasible or index % 10 >= 5
                return normalize_evaluation({
                    "objective": None if unknown else 100.,
                    "solver_success": not unknown,
                    "constraints": {} if unknown else {"g": -1. if feasible else 1.},
                    "status": "exception:RuntimeError" if unknown else "ok" if feasible else "constraint_violation",
                }, self.required_constraints, self.analytical_constraints).to_dict()

        return FakeProblem

    def run_fake(self, **kwargs):
        with patch("ra_lsbo.boundary_validation.X57LiveProblem", self.problem(**kwargs)), redirect_stdout(io.StringIO()):
            return execute_boundary_validation(self.request())

    def test_large_campaign_requires_separate_explicit_approval(self):
        with patch("ra_lsbo.boundary_validation.X57LiveProblem") as factory:
            with self.assertRaisesRegex(ValueError, "separate approval"):
                execute_boundary_validation(self.request(False))
            factory.assert_not_called()
        self.assertFalse((self.root / "budget.jsonl").exists())

    def test_plan_tampering_is_rejected_before_evaluator_creation(self):
        request = self.request()
        self.plan["worker_timeout_seconds"] = 2000
        (self.root / "plan.json").write_text(json.dumps(self.plan), encoding="utf-8")
        with patch("ra_lsbo.boundary_validation.X57LiveProblem") as factory:
            with self.assertRaisesRegex(ValueError, "plan changed"):
                execute_boundary_validation(request)
            factory.assert_not_called()

    def test_fixed_quota_and_full_charge_rule_cannot_be_relaxed(self):
        for change in ("quota", "mode", "duplicate"):
            plan = deepcopy(self.plan)
            if change == "quota":
                plan["proposals"][0]["band"] = "inner"
            elif change == "mode":
                plan["proposals"][0]["proposal"]["mode"] = "development"
            else:
                plan["proposals"][1]["candidate"] = plan["proposals"][0]["candidate"]
                plan["proposals"][1]["proposal"]["candidate"] = plan["proposals"][0]["candidate"]
            with self.assertRaises(ValueError):
                validate_plan(plan)

    def test_forecasts_saved_before_test_calls_and_all_calls_charged(self):
        report = self.run_fake()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(self.calls), 120)
        self.assertEqual(report["ledger"]["charged_calls"], 120)
        self.assertEqual(report["ledger"]["risk_charge"], 120)
        self.assertEqual(report["ledger"]["unresolved_started_calls"], 0)
        local = json.loads((self.root / "test_forecasts.json").read_text())
        self.assertEqual(local["training_indices"], list(range(1, 61)))
        self.assertEqual(len(local["forecasts"]), 60)
        self.assertTrue(all("feasible" not in row for row in local["forecasts"]))
        self.assertEqual(report["test_metrics"]["all"]["local_rf"]["scored_samples"], 60)
        with patch("ra_lsbo.boundary_validation.X57LiveProblem", self.problem()):
            with self.assertRaises(FileExistsError):
                execute_boundary_validation(self.root / "request.json")
        self.assertEqual(len(self.calls), 120)

    def test_no_local_calibration_stops_after_sixty_without_test_calls(self):
        report = self.run_fake(all_feasible=True)
        self.assertEqual(report["status"], "stopped_by_policy")
        self.assertEqual(report["reason"], "local_calibration_unavailable")
        self.assertEqual(len(self.calls), 60)
        self.assertEqual(report["ledger"]["charged_calls"], 60)

    def test_unknown_stops_once_without_retry(self):
        report = self.run_fake(unknown=True)
        self.assertEqual(report["status"], "stopped_evaluator_error")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report["ledger"]["unknown_outcomes"], 1)

    def test_future_labels_are_not_inputs_to_local_calibration(self):
        results = [{"risk": {"feasible_label": index % 10 >= 5}} for index in range(1, 61)]
        before = fit_local_calibration(self.plan, results)
        self.plan["unused_test_labels"] = [False]*60
        after = fit_local_calibration(self.plan, results)
        self.assertEqual(before, after)
        with self.assertRaisesRegex(ValueError, "first 60"):
            fit_local_calibration(self.plan, results+[{"risk": {"feasible_label": True}}])


if __name__ == "__main__":
    unittest.main()
