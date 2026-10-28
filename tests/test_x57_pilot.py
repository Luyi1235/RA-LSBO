from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ra_lsbo.budget import BudgetPolicy, RiskLedger, digest
from ra_lsbo.legacy import normalize_evaluation
from ra_lsbo.x57_pilot import execute_pilot


class FakeProblem:
    required_constraints = ("g",)
    analytical_constraints = ()
    contract_id = "frozen"

    def __init__(self, result="ok"):
        self.calls = 0
        self.result = result

    def canonical(self, raw):
        return raw

    def precheck(self, raw):
        return {"g": -1.}

    def evaluate(self, raw):
        self.calls += 1
        unknown = self.result == "timeout"
        return normalize_evaluation({
            "status": self.result, "solver_success": not unknown,
            "objective": None if unknown else 100.,
            "constraints": {} if unknown else {"g": 1. if self.result == "constraint_violation" else -1.},
        }, self.required_constraints).to_dict()


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        proposals = []
        for index in range(8):
            candidate = {"continuous": {"x": index/8}, "discrete": {}}
            proposals.append({"proposal": {
                "candidate": candidate, "mode": "boundary" if index % 2 else "development",
                "source": "test", "evidence": {
                    "basis": "empirical_calibrated", "probability": .9,
                    "model_id": "model", "calibration_id": "cal", "candidate_key": digest(candidate)},
            }})
        self.plan = {
            "schema": "ra_x57_frozen_pilot_plan_v1", "base_project": str(self.root),
            "config_path": str(self.root / "config.json"), "contract_id": "frozen",
            "historical_source": str(self.root), "historical_source_sha256": {},
            "base_source_sha256": {}, "executor_source_sha256": {},
            "budget_policy": asdict(BudgetPolicy(8, 4.8, 0, 0, 4, .8, 3)),
            "worker_timeout_seconds": 1800, "proposals": proposals,
        }
        self.request = self.root / "request.json"
        self.save(True)

    def save(self, approved):
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(self.plan))
        self.request.write_text(json.dumps({
            "schema": "ra_x57_frozen_pilot_request_v1", "operator_approved": approved,
            "plan_path": str(plan_path), "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        }))

    def test_unapproved_never_constructs_physical_adapter(self):
        self.save(False)
        with patch("ra_lsbo.x57_pilot.X57LiveProblem") as constructor:
            with self.assertRaisesRegex(ValueError, "approval"):
                execute_pilot(self.request)
        constructor.assert_not_called()

    def test_changed_plan_refused(self):
        with (self.root / "plan.json").open("a") as stream:
            stream.write(" ")
        with patch("ra_lsbo.x57_pilot.X57LiveProblem") as constructor:
            with self.assertRaisesRegex(ValueError, "changed"):
                execute_pilot(self.request)
        constructor.assert_not_called()

    def test_expanded_scope_refused(self):
        self.plan["budget_policy"]["max_calls"] = 9
        self.save(True)
        with patch("ra_lsbo.x57_pilot.X57LiveProblem") as constructor:
            with self.assertRaisesRegex(ValueError, "scope"):
                execute_pilot(self.request)
        constructor.assert_not_called()

    def test_success_charged_once_and_request_cannot_repeat(self):
        problem = FakeProblem()
        with patch("ra_lsbo.x57_pilot.X57LiveProblem", return_value=problem):
            report = execute_pilot(self.request)
            with self.assertRaises(FileExistsError):
                execute_pilot(self.request)
        self.assertEqual(problem.calls, 8)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["ledger"], RiskLedger(self.root / "budget.jsonl").summary())
        self.assertEqual(report["ledger"]["charged_calls"], 8)

    def test_three_infeasible_calls_stop_without_retry(self):
        problem = FakeProblem("constraint_violation")
        with patch("ra_lsbo.x57_pilot.X57LiveProblem", return_value=problem):
            report = execute_pilot(self.request)
        self.assertEqual(problem.calls, 3)
        self.assertEqual(report["status"], "stopped_by_policy")
        self.assertEqual(report["ledger"]["known_infeasible"], 3)

    def test_timeout_stops_after_first_charged_call(self):
        problem = FakeProblem("timeout")
        with patch("ra_lsbo.x57_pilot.X57LiveProblem", return_value=problem):
            report = execute_pilot(self.request)
        self.assertEqual(problem.calls, 1)
        self.assertEqual(report["status"], "stopped_evaluator_error")
        self.assertEqual(report["ledger"]["charged_calls"], 1)
        self.assertEqual(report["ledger"]["unknown_outcomes"], 1)


if __name__ == "__main__":
    unittest.main()
