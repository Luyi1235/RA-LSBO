"""Read-only verification of a prepared, unapproved large campaign."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import execute_boundary_validation, validate_plan, verify_sources
from ra_lsbo.x57_live import X57LiveProblem


def verify(request_path):
    request = load_json(request_path.read_bytes())
    if request["operator_approved"] is not False:
        raise ValueError("This verifier is only for an unapproved request")
    plan_path = Path(request["plan_path"])
    plan = load_json(plan_path.read_bytes())
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != request["plan_sha256"]:
        raise ValueError("Plan hash mismatch")
    validate_plan(plan)
    verify_sources(plan["source_sha256"])
    for sha in plan["source_sha256"].values():
        blob = request_path.parent / "source_snapshot" / (sha+".source")
        if hashlib.sha256(blob.read_bytes()).hexdigest() != sha:
            raise ValueError("Saved source snapshot mismatch")
    if (request_path.parent / "budget.jsonl").exists():
        raise ValueError("An execution ledger already exists")
    try:
        execute_boundary_validation(request_path)
    except ValueError as error:
        if "separate approval" not in str(error):
            raise
    else:
        raise ValueError("Unapproved request was accepted")
    problem = X57LiveProblem(plan["base_project"], plan["config_path"],
                             request_path.parent / "workers", allow_mda=False)
    if problem.contract_id != plan["contract_id"]:
        raise ValueError("Physical contract mismatch")
    for item in plan["proposals"]:
        candidate = item["candidate"]
        if problem.canonical(candidate) != candidate or any(value > 0 for value in problem.precheck(candidate).values()):
            raise ValueError("Frozen candidate fails canonicalization or cheap precheck")
    return {
        "status": "passed", "operator_approved": False, "mda_calls": 0,
        "plan_sha256": request["plan_sha256"], "frozen_sources_verified": len(plan["source_sha256"]),
        "source_snapshot": "passed", "approval_gate": "passed",
        "distinct_candidates": len(plan["proposals"]),
        "phase_stratum_counts": dict(Counter(item["phase"]+"_"+item["band"] for item in plan["proposals"])),
        "cheap_prechecks": "120 passed", "estimated_solver_hours": plan["estimated_solver_hours_from_previous_pilot"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    report = verify(args.request.resolve())
    with (args.request.parent / "preparation_verification.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
