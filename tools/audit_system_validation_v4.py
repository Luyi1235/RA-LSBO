"""Audit carried costs, targeted physical inputs, calibration and all comparisons."""

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import RiskLedger, digest
from ra_lsbo.features import history_identity
from ra_lsbo.mechanism_system import mechanism_support
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.structured_models import StructuredResponseHeads
from ra_lsbo.system_experiment import save_new
from ra_lsbo.partial_observation import normalize_with_partial_lift
from ra_lsbo.x57_live import X57LiveProblem
from tools.audit_system_validation import check_worker, verify_trial
from tools.run_system_validation import combine_history, read_lines
from tools.run_system_validation_v4 import TARGET, validate_plan
from tools import audit_system_validation as original_auditor


def audit_records(folder):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    report = load_json((folder/"report.json").read_bytes())
    if report["status"] != "terminal":
        raise ValueError("Study is still running")
    verify_sources(plan["source_sha256"])
    for sha in set(plan["source_sha256"].values()):
        if hashlib.sha256((folder/"source_snapshot"/(sha+".source")).read_bytes()).hexdigest() != sha:
            raise ValueError("Source snapshot changed")
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                             folder/"audit", allow_mda=False)
    if problem.contract_id != plan["physical"]["contract_id"]:
        raise ValueError("Physical contract differs")
    initial = load_json((folder/"initial_history.json").read_bytes())
    predecessor = {r["index"]: r for r in read_lines(Path(plan["predecessor"])/"probe_results.jsonl")}
    if plan["probes"] != load_json((Path(plan["predecessor"])/"plan.json").read_bytes())["probes"]:
        raise ValueError("Unexecuted physical candidates changed")
    seen = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in initial}
    anchors = {r["index"]: r for r in plan["carried_probes"]}
    for item in plan["probes"]:
        candidate = problem.candidate_class(**item["candidate"])
        if problem.canonical(item["candidate"]) != item["candidate"]:
            raise ValueError("Noncanonical planned probe")
        key = problem.space.key(candidate)
        if key in seen or any(v > 0 for v in problem.problem.precheck(candidate).values()):
            raise ValueError("Repeated or illegal planned physical probe")
        seen.add(key)
        if "anchor_index" in item:
            anchor = anchors[item["anchor_index"]]
            if anchor["risk"]["hidden_labels"][TARGET] != 1:
                raise ValueError("Unverified failure anchor")
            delta = np.abs(problem.space.unit(candidate)-problem.space.unit(
                problem.candidate_class(**anchor["candidate"])))
            if np.max(delta) > item["radius"]+1e-10:
                raise ValueError("Neighborhood exceeds frozen coordinate radius")
    probes = read_lines(folder/"probe_results.jsonl")
    cal_results = read_lines(folder/"calibration_results.jsonl")
    calibration_path = folder/"fresh_calibration_plan.json"
    cal_plan = load_json(calibration_path.read_bytes()) if calibration_path.exists() else None
    cal_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest() if cal_plan is not None else None
    ledgers, raw_hashes, total, carried = set(), {}, 0, 0
    rows = {"probe": [], "calibration": []}
    for kind, values in (("probe", probes), ("calibration", cal_results)):
        for result in values:
            if result["risk"] is None:
                continue
            directory = Path(result["directory"])
            ledger_path = directory/"budget.jsonl"
            if ledger_path in ledgers:
                raise ValueError("A physical ledger was counted twice")
            ledgers.add(ledger_path)
            ledger = RiskLedger(ledger_path)
            if (ledger.summary() != result["ledger"] or ledger.summary()["unresolved_started_calls"]
                    or ledger.summary()["reserved_calls"]):
                raise ValueError("Diagnostic ledger differs or is unresolved")
            proposal = load_json((directory/"proposal.json").read_bytes())
            tickets = [t for t in ledger.tickets.values() if t["state"] == "settled"]
            if (len(tickets) != 1 or tickets[0]["risk"] != result.get("original_risk", result["risk"])
                    or any(tickets[0][name] != proposal[name] for name in ("candidate", "source", "mode", "evidence"))
                    or proposal["candidate"] != result["candidate"]):
                raise ValueError("Diagnostic proposal/ledger/response differs")
            calls = ledger.summary()["charged_calls"]
            total += calls
            index = result["index"]
            if kind == "probe":
                frozen = plan["probes"][index-1]
                if proposal["candidate"] != frozen["candidate"] or proposal["source"] != frozen["source"]:
                    raise ValueError("Probe differs from frozen plan")
                if result in plan["carried_probes"]:
                    carried += calls
                    prior = predecessor[index]
                    if "original_risk" in result:
                        if (result["original_risk"] != prior["risk"]
                                or result["original_result_sha256"] != digest(prior)):
                            raise ValueError("Prior unknown outcome/record was changed")
                        raw = load_json((Path(result["risk"]["evidence"]["worker_dir"])/"response.json").read_bytes())
                        normalized = normalize_with_partial_lift(
                            raw, problem.required_constraints, problem.analytical_constraints).to_dict()
                        normalized["evidence"]["worker_dir"] = result["risk"]["evidence"]["worker_dir"]
                        if normalized != result["risk"]:
                            raise ValueError("Partial annotation differs from actual evidence")
                    elif result != prior:
                        raise ValueError("Unannotated carried record changed")
            else:
                frozen = cal_plan["proposals"][index-1]
                if (proposal["candidate"] != frozen["candidate"]
                        or proposal["source"] != frozen["source"]+":"+cal_sha):
                    raise ValueError("Fresh calibration differs from pre-submission plan hash")
            worker = result["risk"]["evidence"].get("worker_dir")
            if worker is not None and kind == "calibration" and calibration_path.stat().st_mtime > (
                    Path(worker)/"request.json").stat().st_mtime+.1:
                raise ValueError("Calibration plan saved after worker submission")
            if worker is not None and kind == "probe" and index > plan["carried_mda_calls"] and (
                    folder/"plan.json").stat().st_mtime > (Path(worker)/"request.json").stat().st_mtime+.1:
                raise ValueError("Neighborhood plan saved after submission")
            raw_hashes.update(check_worker(problem, result))
            if result["risk"]["state"] != "precheck_rejected":
                rows[kind].append({"evaluation": index, "candidate": result["candidate"], "risk": result["risk"]})
    if carried != plan["carried_mda_calls"]:
        raise ValueError("Prior-version cost differs")
    before_cal, _ = combine_history(problem, [initial, rows["probe"]])
    if before_cal != load_json((folder/"precalibration_training.json").read_bytes()):
        raise ValueError("Precalibration training differs from real observed labels")
    if cal_plan is not None:
        if history_identity(before_cal) != cal_plan["training_history_id"]:
            raise ValueError("Calibration planning leaked another history")
        features = StructuredFeatures.load_checkpoint(
            folder/"calibration_planning_encoder.pt", plan["physical"]["base_project"], before_cal)
        if features.snapshot_id != cal_plan["representation_snapshot"]:
            raise ValueError("Calibration planning representation differs")
        responses = StructuredResponseHeads(before_cal, features, seed=0, steps=45)
        predicted = responses.predict([p["candidate"] for p in cal_plan["proposals"]])
        margins = np.min([
            -np.asarray(v["mean"])/np.maximum(np.asarray(v["std"]), 1e-12)
            for name, v in predicted.items() if name.startswith("constraint:") and v is not None], axis=0)
        counts = {"outer": 0, "near": 0, "inner": 0}
        for i, item in enumerate(cal_plan["proposals"]):
            band = "outer" if margins[i] < 0 else "near" if margins[i] < 1 else "inner"
            if band != item["band"] or not np.isclose(margins[i], item["standardized_margin"], rtol=1e-10, atol=1e-10):
                raise ValueError("Calibration strata reconstruction differs")
            counts[band] += 1
        if counts != {"outer": 24, "near": 24, "inner": 24}:
            raise ValueError("Calibration stratum quota differs")
    shared, _ = combine_history(problem, [initial, rows["probe"], rows["calibration"]])
    if shared != load_json((folder/"shared_history.json").read_bytes()):
        raise ValueError("Shared history differs from actual inputs/responses")
    if {(r["method"], r["seed"]) for r in report["physical"]} != {
            (method, seed) for method in plan["methods"] for seed in plan["seeds"]}:
        raise ValueError("Missing optimizer repetitions")
    verified = []
    for run in report["physical"]:
        if run["status"] == "process_error":
            raise ValueError("Process accounting must be resolved")
        path = Path(run["directory"])
        ledger_path = path/"budget.jsonl"
        if ledger_path in ledgers:
            raise ValueError("Optimizer ledger counted twice")
        ledgers.add(ledger_path)
        result = verify_trial(path, plan, plan["physical"])
        raw_hashes.update(result.pop("raw_sha256"))
        total += run["ledger"]["charged_calls"]
        manifest = load_json((path/"manifest.json").read_bytes())
        if manifest["history_id"] != history_identity(shared):
            raise ValueError("Methods received different histories")
        verified.append(dict(result, method=run["method"], seed=run["seed"]))
    if (total != report["complete_new_mda_calls"] or total > 480
            or report["new_current_mda_calls"] != total-carried
            or sum(report["call_cost_by_version"].values()) != total
            or report["call_cost_by_version"] != dict(plan["prior_call_cost_by_version"], v4=total-carried)
            or report["label_contract_id"] != plan["label_contract_id"]):
        raise ValueError("Complete all-version physical cost differs")
    mechanism = load_json((folder/"mechanism_data_report.json").read_bytes())
    if (mechanism["training_support"] != mechanism_support(before_cal)
            or mechanism["probe_support"] != mechanism_support(rows["probe"])
            or mechanism["calibration_support"] != mechanism_support(rows["calibration"])):
        raise ValueError("Reported mechanism support differs")
    paths = [p for p in folder.rglob("*") if p.is_file()
             and not any(part in ("source_snapshot", "workers", "solver_work") for part in p.parts)
             and p.suffix in (".json", ".jsonl", ".pt")]
    inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    inputs.update(raw_hashes)
    return {
        "status": "passed", "complete_new_mda_calls": total, "carried_mda_calls": carried,
        "new_current_mda_calls": total-carried, "call_cost_by_version": report["call_cost_by_version"],
        "new_optimizer_runs": verified, "raw_worker_input_response_files": len(raw_hashes),
        "calibration_strata_reconstructed": cal_plan is not None,
        "targeted_neighborhoods_verified": 120-plan["carried_mda_calls"],
        "input_sha256": inputs, "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def audit(folder):
    with patch.object(original_auditor, "normalize_evaluation", normalize_with_partial_lift):
        return audit_records(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = audit(args.folder.resolve())
    save_new(args.folder.resolve()/"independent_audit.json", result)
    print("Audit", result["status"], "all-version MDA", result["complete_new_mda_calls"], flush=True)
