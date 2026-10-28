"""Verify a terminal frozen pilot before changing its executor sources."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.budget import RiskLedger
from ra_lsbo.legacy import normalize_evaluation
from ra_lsbo.x57_live import X57LiveProblem


def audit_pilot(folder):
    folder = Path(folder).resolve()
    request = load_json((folder / "request.json").read_bytes())
    plan_bytes = (folder / "plan.json").read_bytes()
    plan = load_json(plan_bytes)
    result = load_json((folder / "result.json").read_bytes())
    if result["status"] not in ("complete", "stopped_by_policy", "stopped_evaluator_error"):
        raise ValueError("Pilot is not terminal")
    if hashlib.sha256(plan_bytes).hexdigest() != request["plan_sha256"]:
        raise ValueError("Approved plan changed")
    if result["plan_sha256"] != request["plan_sha256"] or not request["operator_approved"]:
        raise ValueError("Authorization identity mismatch")
    checked = {}
    for directory, hashes in (
        (Path(plan["historical_source"]), plan["historical_source_sha256"]),
        (Path(plan["base_project"]) / "lsbo", plan["base_source_sha256"]),
        (folder.parents[1] / "ra_lsbo", plan["executor_source_sha256"]),
    ):
        for name, expected in hashes.items():
            path = directory / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("Frozen source changed before audit: " + str(path))
            checked[str(path)] = actual
    problem = X57LiveProblem(plan["base_project"], plan["config_path"],
                             folder / "workers", allow_mda=False)
    if problem.contract_id != plan["contract_id"] or result["contract_id"] != plan["contract_id"]:
        raise ValueError("Physical contract mismatch")
    ledger = RiskLedger(folder / "budget.jsonl")
    if ledger.summary() != result["ledger"]:
        raise ValueError("Ledger replay mismatch")
    if any(result["ledger"][key] for key in ("reserved_calls", "unresolved_started_calls")):
        raise ValueError("Unresolved calls remain")
    submitted = [load_json(line) for line in (folder / "submitted_proposals.jsonl").read_bytes().splitlines()]
    if len(submitted) != len(result["results"]) or len(submitted) != result["ledger"]["charged_calls"]:
        raise ValueError("Submission/result/charge counts differ")
    tickets = list(ledger.tickets.values())
    records, raw_hashes = [], {}
    for index, (item, executed, ticket) in enumerate(zip(submitted, result["results"], tickets), 1):
        if item != plan["proposals"][index-1] or executed["proposal"] != item["proposal"]:
            raise ValueError("Frozen proposal changed")
        risk = executed["risk"]
        if ticket["risk"] != risk:
            raise ValueError("Ledger outcome differs")
        raw_path = Path(risk["evidence"]["worker_dir"]) / "response.json"
        raw_bytes = raw_path.read_bytes()
        raw_hashes[str(raw_path)] = hashlib.sha256(raw_bytes).hexdigest()
        converted = normalize_evaluation(load_json(raw_bytes), problem.required_constraints,
                                         problem.analytical_constraints).to_dict()
        for key in ("state", "objective", "objective_observed", "constraints", "constraint_observed",
                    "hidden_labels", "hidden_success", "feasible_label"):
            if converted[key] != risk[key]:
                raise ValueError("Raw worker/adapter mismatch: " + key)
        records.append({
            "index": index, "mode": item["proposal"]["mode"],
            "candidate": item["proposal"]["candidate"],
            "predicted_feasibility": item["proposal"]["evidence"]["probability"],
            "state": risk["state"], "feasible": risk["feasible_label"],
            "objective_Wh": risk["objective"], "elapsed_seconds": risk["elapsed_seconds"],
            "violations": {name: value for name, value in risk["constraints"].items()
                           if risk["constraint_observed"][name] and value > 0},
        })
    known = [row for row in records if row["feasible"] is not None]
    modes = {}
    for mode in sorted({row["mode"] for row in records}):
        group = [row for row in records if row["mode"] == mode]
        modes[mode] = {
            "calls": len(group), "feasible": sum(row["feasible"] is True for row in group),
            "predicted_nonfeasible_sum": sum(1-row["predicted_feasibility"] for row in group),
        }
    audit = {
        "status": "passed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_status": result["status"], "stop_reason": result.get("reason"),
        "planned_calls": len(plan["proposals"]), "completed_calls": len(records),
        "unexecuted_indices": list(range(len(records)+1, len(plan["proposals"])+1)),
        "ledger": result["ledger"], "source_sha256_verified_at_audit": checked,
        "raw_worker_sha256": raw_hashes, "physical_contract_sha256": problem.contract_id,
        "states": dict(Counter(row["state"] for row in records)),
        "descriptive_brier": sum((row["predicted_feasibility"]-int(row["feasible"]))**2
                                 for row in known)/len(known) if known else None,
        "by_mode": modes, "records": records,
        "limitations": [
            "A small, adaptively stopped, selected pilot is not a global calibration estimate.",
            "Predicted nonfeasible sum is descriptive, not a bound on actual violations.",
            "Unexecuted candidates have no outcome; do not impute them as successes.",
            "Frozen proposals do not establish closed-loop optimization improvement.",
            "Source hashes certify the files at this audit time; later development may change them.",
        ],
    }
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    report = audit_pilot(args.folder)
    destination = args.folder / "audit.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in
                      ("status", "pilot_status", "completed_calls", "states", "by_mode")}, indent=2))
    print("Audit:", destination)
