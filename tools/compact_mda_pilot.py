"""Prepare/execute six matched engineering MDA calls under a new 300-call cap."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import BudgetPolicy, RiskLedger, digest
from ra_lsbo.optimizer import append_json, dispatch
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem


BASE = ROOT.parent/"LSBO/X57-LSBO-refactored"
OLD = ROOT/"runs/system_validation_v5_20261020T192312Z_5f6531eb"
DISTANCES = (40, 55, 65)


def prepare():
    folder = ROOT/"runs"/("compact_mda_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                          +"_"+uuid.uuid4().hex[:8])
    folder.mkdir(exist_ok=False)
    eligible = []
    for path in sorted((OLD/"physical").rglob("observations.jsonl")):
        for line in path.read_bytes().splitlines():
            row = load_json(line)
            if row["risk"]["feasible_label"] is True:
                eligible.append((row["risk"]["objective"], str(path), row))
    if not eligible:
        raise ValueError("No documented physical anchor")
    _, source, best = min(eligible, key=lambda item: item[0])
    anchors = [deepcopy(best["candidate"]), deepcopy(best["candidate"])]
    anchors[1]["continuous"]["wing_area"] = 9.5
    # Preserve the original aspect ratio unless it violates the exact span check.
    anchors[1]["continuous"]["aspect_ratio"] = max(
        anchors[1]["continuous"]["aspect_ratio"], 10.6)
    tasks = {}
    for distance in DISTANCES:
        config = load_json((OLD/"frozen_config.json").read_bytes())
        config["name"] = "compact_wing_endurance_"+str(distance)+"nmi"
        config["mission"]["cruise_distance_nmi"] = float(distance)
        config["design_study"] = {
            "version": "compact_wing_bounds_study_20261012_v1",
            "wing_area_bounds_m2": [9., 17.],
            "scope": "conceptual_sensitivity_not_NASA_validated_bounds"}
        config_path = folder/("config_"+str(distance)+".json")
        save_new(config_path, config)
        problem = X57LiveProblem(BASE, config_path, folder/("workers_"+str(distance)))
        canonical = [problem.canonical(anchor) for anchor in anchors]
        checks = [problem.precheck(anchor) for anchor in canonical]
        if any(v > 0 or not np.isfinite(v) for item in checks for v in item.values()):
            raise ValueError("Engineering anchor fails an exact analytical check")
        tasks[str(distance)] = {
            "config_path": str(config_path), "contract_id": problem.contract_id,
            "anchors": canonical, "prechecks": checks,
            "variables": [asdict(v) for v in problem.space.variables]}
    paths = [Path(__file__).resolve(), ROOT/"docs/POSTERIOR_MDA_PROTOCOL_20261021.md"]
    paths += [ROOT/"ra_lsbo"/name for name in (
        "x57_live.py", "budget.py", "optimizer.py", "records.py", "legacy.py",
        "audit.py", "system_experiment.py", "boundary_validation.py")]
    save_new(folder/"plan.json", {
        "schema": "compact_endurance_pilot_v1", "base_project": str(BASE),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entire_new_study_max_real_calls": 300, "pilot_max_real_calls": 6,
        "prior_v1_v5_budget_remains_terminal": 500,
        "authorization": "standing user approval for complete plans <=500 evaluations",
        "anchor_source": source, "anchor_old_evaluation": best["evaluation"],
        "anchor_old_objective_Wh": best["risk"]["objective"],
        "purpose": "Matched engineering checks only; not sparse-region qualification",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "tasks": tasks, "distance_order": list(DISTANCES)})
    print(folder, flush=True)


def execute(folder):
    plan = load_json((folder/"plan.json").read_bytes())
    if plan["schema"] != "compact_endurance_pilot_v1" or plan["pilot_max_real_calls"] != 6:
        raise ValueError("Unexpected pilot scope")
    verify_sources(plan["source_sha256"])
    save_new(folder/"execution_started.json", {"utc": datetime.now(timezone.utc).isoformat()})
    status = "complete"
    rows = []
    # Separate contract namespaces prevent duplicated coordinates across missions
    # being mistaken for repeated evaluations of the same physical experiment.
    for distance in plan["distance_order"]:
        task = plan["tasks"][str(distance)]
        output = folder/("pilot_"+str(distance))
        output.mkdir(exist_ok=False)
        problem = X57LiveProblem(plan["base_project"], task["config_path"], output/"workers", True)
        if problem.contract_id != task["contract_id"]:
            raise ValueError("Frozen physical contract differs")
        with RiskLedger(output/"budget.jsonl", BudgetPolicy(2, 2., 2, 0, 0)) as ledger:
            for index, candidate in enumerate(task["anchors"]):
                if len(rows) >= 6:
                    raise RuntimeError("Global pilot cap reached")
                verify_sources(plan["source_sha256"])
                proposal = {"candidate": candidate, "mode": "bootstrap",
                            "source": "matched_engineering_anchor_"+str(index),
                            "evidence": {"basis": "unavailable", "probability": None}}
                append_json(folder/"submissions.jsonl", {
                    "call": len(rows)+1, "distance_nmi": distance, "anchor": index,
                    "contract_id": problem.contract_id, "candidate_key": digest(candidate),
                    "proposal": proposal})
                print("MDA start", len(rows)+1, distance, index, flush=True)
                risk = dispatch(problem, ledger, proposal)
                if risk is None:
                    raise RuntimeError("Frozen prechecked anchor was rejected")
                row = {"evaluation": len(rows)+1, "candidate": candidate, "risk": risk,
                       "evaluator_contract_sha256": problem.contract_id,
                       "distance_nmi": distance, "anchor": index}
                rows.append(row)
                append_json(folder/"observations.jsonl", row)
                print("MDA done", len(rows), risk["state"], risk["objective"],
                      risk["elapsed_seconds"], flush=True)
                if risk["feasible_label"] is None:
                    status = "stopped_unknown"
                    break
        if status != "complete":
            break
    charged = sum(RiskLedger(p).summary()["charged_calls"]
                  for p in folder.glob("pilot_*/budget.jsonl"))
    if charged != len(rows):
        raise RuntimeError("Physical accounting differs")
    verify_sources(plan["source_sha256"])
    report = {"status": status, "charged_real_calls": charged,
              "entire_study_remaining_cap": 300-charged,
              "feasible": sum(r["risk"]["feasible_label"] is True for r in rows),
              "sparse_region_qualified": False,
              "reason": "Engineering anchors are selected, not conditional-uniform samples",
              "results": [{"distance_nmi": r["distance_nmi"], "anchor": r["anchor"],
                           "state": r["risk"]["state"], "objective_Wh": r["risk"]["objective"],
                           "violations": {k: v for k, v in r["risk"]["constraints"].items()
                                          if v is not None and v > 0}}
                          for r in rows]}
    save_new(folder/"pilot_result.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.folder is None:
        parser.error("--folder required")
    else:
        execute(args.folder.resolve())
