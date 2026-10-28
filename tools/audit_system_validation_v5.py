"""Independently audit v5 carried provenance, held-out inputs and all real runs."""

import argparse
from collections import Counter
from copy import deepcopy
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
from ra_lsbo.partial_observation import normalize_with_partial_lift
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.structured_models import StructuredResponseHeads
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem
from tools import audit_system_validation as original
from tools.run_system_validation import combine_history, read_lines
from tools.run_system_validation_v5 import CAP, QUOTAS, TARGET, validate_plan, rows
from tools.verify_system_v5_request import verify_candidates


def audit_records(folder):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    report = load_json((folder/"report.json").read_bytes())
    validate_plan(plan)
    if report["status"] != "terminal":
        raise ValueError("Study is not terminal")
    verify_sources(plan["source_sha256"])
    for sha in set(plan["source_sha256"].values()):
        if hashlib.sha256((folder/"source_snapshot"/(sha+".source")).read_bytes()).hexdigest() != sha:
            raise ValueError("Source snapshot differs")
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                             folder/"audit", allow_mda=False)
    if problem.contract_id != plan["physical"]["contract_id"]:
        raise ValueError("Physical contract differs")
    initial = load_json((folder/"initial_history.json").read_bytes())
    verify_candidates(plan, initial, problem)
    predecessor = Path(plan["predecessor"])
    old = {(stage, r["index"]): r for stage, filename in (
        ("probe", "probe_results.jsonl"), ("calibration", "calibration_results.jsonl"))
        for r in read_lines(predecessor/filename) if r["risk"] is not None}
    probes, calibration = read_lines(folder/"probe_results.jsonl"), read_lines(folder/"calibration_results.jsonl")
    cal_path = folder/"fresh_calibration_plan.json"
    cal_plan = load_json(cal_path.read_bytes()) if cal_path.exists() else None
    cal_sha = hashlib.sha256(cal_path.read_bytes()).hexdigest() if cal_plan else None
    ledgers, raw_hashes, cost, carried = set(), {}, 0, 0
    actual_keys = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in initial}
    for kind, values in (("probe", probes), ("calibration", calibration)):
        for position, result in enumerate(values, 1):
            if result["index"] != position:
                raise ValueError("Acquisition completion sequence differs")
            if result["risk"] is None:
                continue
            key = problem.space.key(problem.candidate_class(**result["candidate"]))
            if key in actual_keys:
                raise ValueError("Repeated real acquisition candidate")
            actual_keys.add(key)
            directory = Path(result["directory"])
            ledger_path = directory/"budget.jsonl"
            if ledger_path in ledgers:
                raise ValueError("Physical ledger counted twice")
            ledgers.add(ledger_path)
            ledger = RiskLedger(ledger_path)
            if (ledger.summary() != result["ledger"] or ledger.summary()["unresolved_started_calls"]
                    or ledger.summary()["reserved_calls"]):
                raise ValueError("Acquisition ledger differs or is unresolved")
            proposal = load_json((directory/"proposal.json").read_bytes())
            tickets = [t for t in ledger.tickets.values() if t["state"] == "settled"]
            if (len(tickets) != 1 or tickets[0]["risk"] != result.get("original_risk", result["risk"])
                    or any(tickets[0][k] != proposal[k] for k in ("candidate", "source", "mode", "evidence"))
                    or proposal["candidate"] != result["candidate"]):
                raise ValueError("Acquisition proposal, response and ledger differ")
            calls = ledger.summary()["charged_calls"]
            cost += calls
            index = result["index"]
            if kind == "probe":
                frozen = plan["probes"][index-1]
                if index <= plan["carried_mda_calls"]:
                    prior = old[(result["carried_stage"], result["carried_index"])]
                    expected = deepcopy(prior)
                    expected.update(index=index, carried_stage=result["carried_stage"],
                                    carried_index=prior["index"], predecessor_record_sha256=digest(prior))
                    if result != expected or result != plan["carried_probes"][index-1]:
                        raise ValueError("Carried outcome/provenance changed")
                    carried += calls
                elif (folder/"plan.json").stat().st_mtime > (
                        Path(result["risk"]["evidence"]["worker_dir"])/"request.json").stat().st_mtime+.1:
                    raise ValueError("Training design frozen after submission")
            else:
                if cal_plan is None:
                    raise ValueError("Missing frozen held-out design")
                frozen = dict(cal_plan["proposals"][index-1])
                frozen["source"] += ":"+cal_sha
                if cal_path.stat().st_mtime > (
                        Path(result["risk"]["evidence"]["worker_dir"])/"request.json").stat().st_mtime+.1:
                    raise ValueError("Held-out design frozen after submission")
            if proposal["candidate"] != frozen["candidate"] or proposal["source"] != frozen["source"]:
                raise ValueError("Actual acquisition differs from frozen design")
            raw_hashes.update(original.check_worker(problem, result))
    if carried != plan["carried_mda_calls"]:
        raise ValueError("Inherited all-version cost differs")
    training, duplicates = combine_history(problem, [initial, rows(probes)])
    if duplicates or training != load_json((folder/"precalibration_training.json").read_bytes()):
        raise ValueError("Training does not match actual settled observations")
    if cal_plan is not None:
        if (cal_plan["training_history_id"] != history_identity(training)
                or cal_plan["new_calibration_labels_available"] is not False):
            raise ValueError("Calibration design training differs")
        features = StructuredFeatures.load_checkpoint(
            folder/"calibration_planning_encoder.pt", plan["physical"]["base_project"], training)
        if features.snapshot_id != cal_plan["representation_snapshot"]:
            raise ValueError("Planning representation differs")
        responses = StructuredResponseHeads(training, features, seed=0, steps=45)
        proposals = cal_plan["proposals"]
        predicted = responses.predict([p["candidate"] for p in proposals])
        margins = np.min([-np.asarray(v["mean"])/np.maximum(np.asarray(v["std"]), 1e-12)
                          for name, v in predicted.items() if name.startswith("constraint:") and v is not None], axis=0)
        seen = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in training}
        for i, item in enumerate(proposals):
            candidate = problem.candidate_class(**item["candidate"])
            key = problem.space.key(candidate)
            if (key in seen or problem.canonical(item["candidate"]) != item["candidate"]
                    or any(not np.isfinite(v) or v > 0 for v in problem.problem.precheck(candidate).values())):
                raise ValueError("Repeated/noncanonical/illegal held-out design")
            seen.add(key)
            if item["band"] == "hidden_neighborhood":
                anchor = training[item["anchor_evaluation"]-1]
                if (anchor["risk"]["hidden_labels"][TARGET] != 1
                        or item["candidate"]["discrete"] != anchor["candidate"]["discrete"]
                        or np.max(np.abs(problem.space.unit(candidate)-problem.space.unit(
                            problem.candidate_class(**anchor["candidate"])))) > item["radius"]+1e-10):
                    raise ValueError("Held-out hidden neighborhood lacks prior failure evidence")
            else:
                band = "outer" if margins[i] < 0 else "near" if margins[i] < 1 else "inner"
                if band != item["band"] or not np.isclose(
                        margins[i], item["standardized_margin"], rtol=1e-10, atol=1e-10):
                    raise ValueError("Held-out explicit GP stratum differs")
        if dict(Counter(p["band"] for p in proposals)) != QUOTAS or cal_plan["quotas"] != QUOTAS:
            raise ValueError("Held-out quotas differ")
    shared, duplicates = combine_history(problem, [initial, rows(probes), rows(calibration)])
    if duplicates or shared != load_json((folder/"shared_history.json").read_bytes()):
        raise ValueError("Shared history differs")
    if {(r["method"], r["seed"]) for r in report["physical"]} != {
            (method, seed) for method in plan["methods"] for seed in plan["seeds"]}:
        raise ValueError("Missing optimizer repetitions")
    verified = []
    for run in report["physical"]:
        if run["status"] == "process_error":
            raise ValueError("Process accounting needs resolution")
        path = Path(run["directory"])
        if path/"budget.jsonl" in ledgers:
            raise ValueError("Optimizer ledger counted twice")
        ledgers.add(path/"budget.jsonl")
        result = original.verify_trial(path, plan, plan["physical"])
        raw_hashes.update(result.pop("raw_sha256"))
        cost += run["ledger"]["charged_calls"]
        if (load_json((path/"manifest.json").read_bytes())["history_id"] != history_identity(shared)
                or load_json((path/"summary.json").read_bytes()) != run):
            raise ValueError("Optimizer shared history or summary differs")
        verified.append(dict(result, method=run["method"], seed=run["seed"]))
    if (cost != report["complete_new_mda_calls"] or cost > CAP
            or report["maximum_complete_mda_calls"] != CAP
            or report["new_current_mda_calls"] != cost-carried
            or report["call_cost_by_version"] != dict(plan["prior_call_cost_by_version"], v5=cost-carried)
            or sum(report["call_cost_by_version"].values()) != cost
            or report["label_contract_id"] != plan["label_contract_id"]):
        raise ValueError("Cumulative all-version physical cost differs")
    mechanism = load_json((folder/"mechanism_data_report.json").read_bytes())
    if (mechanism["training_support"] != mechanism_support(training)
            or mechanism["probe_support"] != mechanism_support(rows(probes))
            or mechanism["calibration_support"] != mechanism_support(rows(calibration))):
        raise ValueError("Mechanism class counts differ")
    paths = [p for p in folder.rglob("*") if p.is_file()
             and not any(part in ("source_snapshot", "workers", "solver_work") for part in p.parts)
             and p.suffix in (".json", ".jsonl", ".pt")]
    inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    inputs.update(raw_hashes)
    return dict(status="passed", complete_new_mda_calls=cost, carried_mda_calls=carried,
        new_current_mda_calls=cost-carried, call_cost_by_version=report["call_cost_by_version"],
        new_optimizer_runs=verified, raw_worker_input_response_files=len(raw_hashes),
        calibration_strata_reconstructed=cal_plan is not None, input_sha256=inputs,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def audit(folder):
    with patch.object(original, "normalize_evaluation", normalize_with_partial_lift):
        return audit_records(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    folder = parser.parse_args().folder.resolve()
    result = audit(folder)
    save_new(folder/"independent_audit.json", result)
    print("Audit", result["status"], "all-version MDA", result["complete_new_mda_calls"], flush=True)
