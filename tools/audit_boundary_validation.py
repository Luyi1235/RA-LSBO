"""Audit an executed boundary campaign without invoking the physical solver."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import fit_local_calibration, test_metrics, validate_plan, verify_sources
from ra_lsbo.budget import RiskLedger, digest
from ra_lsbo.legacy import normalize_evaluation
from ra_lsbo.x57_live import X57LiveProblem


def audit(folder):
    request = load_json((folder / "request.json").read_bytes())
    plan_bytes = (folder / "plan.json").read_bytes()
    plan = load_json(plan_bytes)
    report = load_json((folder / "result.json").read_bytes())
    validate_plan(plan)
    if request.get("operator_approved") is not True:
        raise ValueError("Campaign lacks approval")
    if report["status"] not in ("complete", "stopped_by_policy", "stopped_evaluator_error"):
        raise ValueError("Campaign is not terminal")
    if hashlib.sha256(plan_bytes).hexdigest() != request["plan_sha256"]:
        raise ValueError("Approved plan changed")
    if (report["plan_sha256"] != request["plan_sha256"]
            or report["contract_id"] != plan["contract_id"]):
        raise ValueError("Result/approved plan identity mismatch")
    verify_sources(plan["source_sha256"])
    snapshot = folder / "source_snapshot"
    for expected in set(plan["source_sha256"].values()):
        if hashlib.sha256((snapshot / (expected+".source")).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source snapshot changed")
    problem = X57LiveProblem(plan["base_project"], plan["config_path"], folder / "workers", allow_mda=False)
    if problem.contract_id != plan["contract_id"]:
        raise ValueError("Current physical/configuration contract differs")
    ledger = RiskLedger(folder / "budget.jsonl")
    if ledger.summary() != report["ledger"] or report["ledger"]["unresolved_started_calls"]:
        raise ValueError("Ledger mismatch or unresolved calls")
    submitted = [load_json(line) for line in (folder / "submitted_proposals.jsonl").read_bytes().splitlines()]
    if len(submitted) != len(report["results"]) or len(report["results"]) > 120:
        raise ValueError("Submission/result mismatch")
    tickets = {ticket["candidate_key"]: ticket for ticket in ledger.tickets.values()}
    evaluated = {digest(item["proposal"]["candidate"]) for item in report["results"] if item["risk"] is not None}
    if set(tickets) != evaluated:
        raise ValueError("Result/ledger candidate counts differ")
    matched, absent, precheck = 0, [], []
    raw_hashes, request_hashes = {}, {}
    for position, (item, result) in enumerate(zip(submitted, report["results"]), 1):
        index = result["index"]
        if (index != position or item != plan["proposals"][index-1]
                or result["proposal"] != item["proposal"]
                or result["phase"] != item["phase"] or result["band"] != item["band"]):
            raise ValueError("Candidate or frozen forecast changed")
        risk = result["risk"]
        if risk is None:
            if (position != len(report["results"]) or report.get("reason") != "final_precheck_changed"
                    or report["status"] != "stopped_by_policy"):
                raise ValueError("Missing result without a terminal precheck rejection")
            precheck.append(index)
            continue
        ticket = tickets[digest(item["candidate"])]
        if any(ticket[name] != item["proposal"][name] for name in ("mode", "source", "evidence")):
            raise ValueError("Ledger proposal metadata mismatch")
        if risk != ticket.get("risk"):
            raise ValueError("Ledger result mismatch")
        directory = risk["evidence"].get("worker_dir")
        response = Path(directory) / "response.json" if directory else None
        if response is None or not response.exists():
            if risk["state"] not in ("unknown", "infrastructure_error"):
                raise ValueError("Completed worker response absent")
            absent.append(index)
            continue
        content = response.read_bytes()
        raw_hashes[str(response)] = hashlib.sha256(content).hexdigest()
        worker_request_path = response.parent / "request.json"
        worker_request_bytes = worker_request_path.read_bytes()
        worker_request = load_json(worker_request_bytes)
        if (worker_request["candidate"] != item["candidate"]
                or worker_request["config"] != problem.config):
            raise ValueError("Raw worker input candidate/configuration differs from frozen plan")
        request_hashes[str(worker_request_path)] = hashlib.sha256(worker_request_bytes).hexdigest()
        converted = normalize_evaluation(load_json(content), problem.required_constraints,
                                         problem.analytical_constraints).to_dict()
        for name in ("raw_status", "state", "objective", "objective_observed", "constraints",
                     "constraint_observed", "analytical_constraints", "hidden_success", "hidden_labels",
                     "failure_modes", "feasible_label", "mda_evaluated", "budget_charged", "elapsed_seconds"):
            if converted[name] != risk[name]:
                raise ValueError("Raw worker/adapter mismatch: " + name)
        matched += 1
    forecasts_verified = False
    if "test_forecasts_sha256" in report:
        content = (folder / "test_forecasts.json").read_bytes()
        if hashlib.sha256(content).hexdigest() != report["test_forecasts_sha256"]:
            raise ValueError("Test forecasts changed")
        local = load_json(content)
        reconstructed = fit_local_calibration(plan, report["results"][:60])
        if local != reconstructed or local["training_indices"] != list(range(1, 61)):
            raise ValueError("Test forecasts do not match calibration-only fit")
        if (all(item["risk"] is not None for item in report["results"])
                and test_metrics(local["forecasts"], report["results"]) != report["test_metrics"]):
            raise ValueError("Scored test metrics differ")
        forecasts_verified = True
    inputs = [folder / name for name in
              ("request.json", "plan.json", "result.json", "budget.jsonl", "submitted_proposals.jsonl")]
    if forecasts_verified:
        inputs.append(folder / "test_forecasts.json")
    return {"status": "passed", "campaign_status": report["status"], "ledger": report["ledger"],
            "plan_sha256": request["plan_sha256"],
            "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
            "raw_responses_matched": matched, "unavailable_error_responses": absent,
            "unsubmitted_precheck_rejections": precheck,
            "raw_worker_sha256": raw_hashes, "local_forecasts_verified": forecasts_verified,
            "raw_worker_request_sha256": request_hashes,
            "frozen_sources_verified": len(plan["source_sha256"]),
            "physical_contract_sha256": problem.contract_id,
            "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = audit(args.folder.resolve())
    with (args.folder / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
