from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    from ra_lsbo.boundary_validation import BANDS, POLICY, execute_boundary_validation
    from ra_lsbo.budget import digest
    from ra_lsbo.legacy import normalize_evaluation
    from tools.audit_boundary_validation import audit


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class BoundaryAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        proposals = []
        for index in range(1, 121):
            candidate = {"continuous": {"x": index/121}, "discrete": {}}
            proposals.append({
                "index": index, "phase": "calibration" if index <= 60 else "test",
                "band": BANDS[(index-1) % 3][0], "candidate": candidate,
                "raw_rf": .5, "raw_constraint": .4, "global_rf": .5, "global_constraint": .5,
                "proposal": {"candidate": candidate, "mode": "discovery", "source": "test",
                             "evidence": {"basis": "raw", "probability": .4, "model_id": "test",
                                          "candidate_key": digest(candidate)}}})
        self.plan = {
            "schema": "ra_boundary_validation_plan_v1", "proposals": proposals,
            "budget_policy": asdict(POLICY), "worker_timeout_seconds": 1800,
            "base_project": "fake", "config_path": "fake", "contract_id": "fake", "source_sha256": {}}
        self.write("plan.json", self.plan)
        self.request = {
            "schema": "ra_boundary_validation_request_v1", "operator_approved": True,
            "plan_path": str(self.root / "plan.json"),
            "plan_sha256": hashlib.sha256((self.root / "plan.json").read_bytes()).hexdigest()}
        self.write("request.json", self.request)
        physical = SimpleNamespace(contract_id="fake", required_constraints=("g",),
                                   analytical_constraints=(), config={"test": "frozen"})
        factory = patch("tools.audit_boundary_validation.X57LiveProblem", return_value=physical)
        self.physical_factory = factory.start()
        self.addCleanup(factory.stop)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def execute(self, precheck_reject=False):
        owner = self

        class FakeProblem:
            contract_id = "fake"
            required_constraints = ("g",)
            analytical_constraints = ()

            def __init__(self, *args, **kwargs):
                self.calls = 0

            def canonical(self, candidate):
                return candidate

            def precheck(self, candidate):
                return {"g": 1.} if precheck_reject else {}

            def evaluate(self, candidate):
                self.calls += 1
                if self.calls == 1:
                    worker = owner.root / "worker"
                    worker.mkdir()
                    raw = {
                        "status": "ok", "objective": 100., "solver_success": True,
                        "constraints": {"g": -1.}, "metrics": {"worker_dir": str(worker)}}
                    (worker / "response.json").write_text(json.dumps(raw), encoding="utf-8")
                    (worker / "request.json").write_text(json.dumps({
                        "candidate": candidate, "config": {"test": "frozen"}}), encoding="utf-8")
                else:
                    raw = {"status": "exception:RuntimeError", "objective": None,
                           "solver_success": False, "constraints": {}}
                return normalize_evaluation(raw, self.required_constraints, self.analytical_constraints).to_dict()

        with patch("ra_lsbo.boundary_validation.X57LiveProblem", FakeProblem), redirect_stdout(io.StringIO()):
            return execute_boundary_validation(self.root / "request.json")

    def test_audit_matches_known_worker_and_retains_missing_error_response(self):
        self.execute()
        report = audit(self.root)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["raw_responses_matched"], 1)
        self.assertEqual(report["unavailable_error_responses"], [2])
        self.assertEqual(report["ledger"]["charged_calls"], 2)

    def test_extra_results_cannot_be_hidden_by_zip_truncation(self):
        result = self.execute()
        extra = dict(result["results"][-1], index=3, proposal=self.plan["proposals"][2]["proposal"])
        result["results"].append(extra)
        self.write("result.json", result)
        with (self.root / "submitted_proposals.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.plan["proposals"][2])+"\n")
        with self.assertRaisesRegex(ValueError, "candidate counts"):
            audit(self.root)

    def test_precheck_stop_has_no_physical_charge_or_worker(self):
        self.execute(precheck_reject=True)
        report = audit(self.root)
        self.assertEqual(report["ledger"]["charged_calls"], 0)
        self.assertEqual(report["unsubmitted_precheck_rejections"], [1])
        self.assertEqual(report["raw_responses_matched"], 0)

    def test_result_plan_identity_must_match_approved_request(self):
        result = self.execute()
        result["plan_sha256"] = "changed"
        self.write("result.json", result)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            audit(self.root)

    def test_result_stratum_must_match_frozen_plan(self):
        result = self.execute()
        result["results"][0]["band"] = "inner"
        self.write("result.json", result)
        with self.assertRaisesRegex(ValueError, "frozen forecast"):
            audit(self.root)

    def test_changed_frozen_source_is_rejected(self):
        source = self.root / "frozen.txt"
        source.write_text("original", encoding="utf-8")
        original = source.read_bytes()
        expected = hashlib.sha256(original).hexdigest()
        self.plan["source_sha256"] = {str(source): expected}
        self.write("plan.json", self.plan)
        self.request["plan_sha256"] = hashlib.sha256((self.root / "plan.json").read_bytes()).hexdigest()
        self.write("request.json", self.request)
        snapshot = self.root / "source_snapshot"
        snapshot.mkdir()
        (snapshot / (expected+".source")).write_bytes(original)
        self.execute()
        self.assertEqual(audit(self.root)["frozen_sources_verified"], 1)
        source.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen source changed"):
            audit(self.root)

    def test_physical_configuration_identity_is_recomputed_without_mda(self):
        self.execute()
        self.physical_factory.return_value.contract_id = "changed"
        with self.assertRaisesRegex(ValueError, "physical/configuration contract"):
            audit(self.root)
        self.assertIs(self.physical_factory.call_args.kwargs["allow_mda"], False)

    def test_actual_worker_candidate_must_match_the_approved_design(self):
        self.execute()
        request_path = self.root / "worker" / "request.json"
        worker = json.loads(request_path.read_text())
        worker["candidate"]["continuous"]["x"] = .99
        request_path.write_text(json.dumps(worker), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Raw worker input"):
            audit(self.root)

    def test_actual_worker_configuration_must_match_the_physical_contract(self):
        self.execute()
        request_path = self.root / "worker" / "request.json"
        worker = json.loads(request_path.read_text())
        worker["config"]["test"] = "changed"
        request_path.write_text(json.dumps(worker), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Raw worker input"):
            audit(self.root)


if __name__ == "__main__":
    unittest.main()
