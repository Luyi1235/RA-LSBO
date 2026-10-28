"""Audit the complete amended study without rerunning any aircraft evaluation."""

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import sys

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
from ra_lsbo.x57_live import X57LiveProblem
from tools.audit_system_validation import check_worker, verify_trial
from tools.run_system_validation import read_lines
from tools.run_system_validation_v2 import validate_plan


def audit(folder):
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
        raise ValueError("Physical evaluator changed")
    raw_hashes, unique_ledgers, calls, carried = {}, set(), 0, 0
    initial = load_json((folder/"initial_history.json").read_bytes())
    before_calibration = load_json((folder/"precalibration_training.json").read_bytes())
    calibration_plan_path = folder/"fresh_calibration_plan.json"
    calibration_plan = load_json(calibration_plan_path.read_bytes()) if calibration_plan_path.exists() else None
    calibration_sha = (hashlib.sha256(calibration_plan_path.read_bytes()).hexdigest()
                       if calibration_plan is not None else None)
    probes = read_lines(folder/"probe_results.jsonl")
    cal_results = read_lines(folder/"calibration_results.jsonl")
    rows_for_support = {"probe": [], "calibration": []}
    for kind, values in (("probe", probes), ("calibration", cal_results)):
        for result in values:
            if result["risk"] is None:
                continue
            directory = Path(result["directory"])
            ledger_path = directory/"budget.jsonl"
            if ledger_path in unique_ledgers:
                raise ValueError("Physical call counted twice")
            unique_ledgers.add(ledger_path)
            ledger = RiskLedger(ledger_path)
            if ledger.summary() != result["ledger"] or ledger.summary()["unresolved_started_calls"]:
                raise ValueError("Diagnostic ledger mismatch")
            calls += ledger.summary()["charged_calls"]
            proposal = load_json((directory/"proposal.json").read_bytes())
            if result["candidate"] != proposal["candidate"]:
                raise ValueError("Diagnostic result/submitted candidate differs")
            index = result["index"]
            if kind == "probe":
                if proposal["candidate"] != plan["probes"][index-1]["candidate"]:
                    raise ValueError("Probe differs from frozen plan")
                if result in plan["carried_probes"]:
                    carried += ledger.summary()["charged_calls"]
            else:
                frozen = calibration_plan["proposals"][index-1]
                if (frozen["candidate"] != proposal["candidate"]
                        or proposal["source"] != frozen["source"]+":"+calibration_sha):
                    raise ValueError("Fresh calibration differs from the pre-submission plan hash")
                if directory.joinpath("workers").exists():
                    worker = result["risk"]["evidence"].get("worker_dir")
                    if worker is not None and calibration_plan_path.stat().st_mtime > (
                            Path(worker)/"request.json").stat().st_mtime+.1:
                        raise ValueError("Calibration plan timestamp follows worker submission")
            raw_hashes.update(check_worker(problem, result))
            rows_for_support[kind].append({"evaluation": index, "candidate": result["candidate"],
                                           "risk": result["risk"]})
    if carried != plan["carried_mda_calls"]:
        raise ValueError("Carried call count differs")
    if calibration_plan is not None:
        if history_identity(before_calibration) != calibration_plan["training_history_id"]:
            raise ValueError("Fresh calibration was designed using another training history")
        features = StructuredFeatures.load_checkpoint(
            folder/"calibration_planning_encoder.pt", plan["physical"]["base_project"], before_calibration)
        if features.snapshot_id != calibration_plan["representation_snapshot"]:
            raise ValueError("Calibration planning encoder differs")
        responses = StructuredResponseHeads(before_calibration, features, seed=0, steps=45)
        candidates = [p["candidate"] for p in calibration_plan["proposals"]]
        predicted = responses.predict(candidates)
        margins = np.min([
            -np.asarray(values["mean"])/np.maximum(np.asarray(values["std"]), 1e-12)
            for name, values in predicted.items() if name.startswith("constraint:") and values is not None], axis=0)
        counts = {"outer": 0, "near": 0, "inner": 0}
        for i, item in enumerate(calibration_plan["proposals"]):
            band = "outer" if margins[i] < 0 else "near" if margins[i] < 1 else "inner"
            if band != item["band"] or not np.isclose(margins[i], item["standardized_margin"], rtol=1e-10, atol=1e-10):
                raise ValueError("Fresh calibration stratum/model prediction differs")
            counts[band] += 1
        if counts != {"outer": 24, "near": 24, "inner": 24}:
            raise ValueError("Fresh calibration quotas differ")
    verified = []
    expected = {(method, seed) for method in plan["methods"] for seed in plan["seeds"]}
    if {(r["method"], r["seed"]) for r in report["physical"]} != expected:
        raise ValueError("Missing physical optimizer repetitions")
    shared = load_json((folder/"shared_history.json").read_bytes())
    if len(rows_for_support["calibration"]) == 72:
        expected_training = shared[:-72]
        if [r["risk"] for r in expected_training] != [r["risk"] for r in before_calibration]:
            raise ValueError("Mechanism probes were not all included before calibration")
    for run in report["physical"]:
        if run["status"] == "process_error":
            raise ValueError("Worker process error needs an explicit accounting audit")
        result = verify_trial(Path(run["directory"]), plan, plan["physical"])
        raw_hashes.update(result.pop("raw_sha256"))
        calls += run["ledger"]["charged_calls"]
        verified.append(dict(result, method=run["method"], seed=run["seed"]))
        manifest = load_json((Path(run["directory"])/"manifest.json").read_bytes())
        if manifest["history_id"] != history_identity(shared):
            raise ValueError("Methods did not receive the same initial observations")
    if calls != report["complete_new_mda_calls"] or calls > 480:
        raise ValueError("Complete physical call total differs or exceeds authorization")
    mechanism = load_json((folder/"mechanism_data_report.json").read_bytes())
    if (mechanism["training_support"] != mechanism_support(before_calibration)
            or mechanism["probe_support"] != mechanism_support(rows_for_support["probe"])
            or mechanism["calibration_support"] != mechanism_support(rows_for_support["calibration"])):
        raise ValueError("Reported physical mechanism support differs")
    paths = [p for p in folder.rglob("*") if p.is_file()
             and not any(part in ("source_snapshot", "workers", "solver_work") for part in p.parts)
             and p.suffix in (".json", ".jsonl", ".pt")]
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    inputs.update(raw_hashes)
    return {"status": "passed", "complete_new_mda_calls": calls, "carried_mda_calls": carried,
            "new_v2_mda_calls": calls-carried, "new_optimizer_runs": verified,
            "raw_worker_input_response_files": len(raw_hashes),
            "calibration_strata_reconstructed": calibration_plan is not None,
            "input_sha256": inputs, "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = audit(args.folder.resolve())
    save_new(args.folder.resolve()/"independent_audit.json", result)
    print("Audit", result["status"], "complete MDA", result["complete_new_mda_calls"])
