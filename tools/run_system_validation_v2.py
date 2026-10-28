"""Continue the full study with mechanism training followed by fresh calibration."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import uuid

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import RiskLedger, digest
from ra_lsbo.mechanism_system import FullSystemLearning, SystemSettings, mechanism_support
from ra_lsbo.optimizer import append_json
from ra_lsbo.selection import candidate_pool
from ra_lsbo.system_experiment import save_new, signal_stop
from ra_lsbo.x57_live import X57LiveProblem
from tools.run_system_validation import combine_history, diagnostic_candidates, probe_one, read_lines, run_trials
from tools.snapshot_experiment_sources import snapshot


def validate_plan(plan):
    from ra_lsbo.mechanism_system import METHODS
    if (plan.get("schema") != "ra_system_validation_v2" or plan.get("operator_approved") is not True
            or plan.get("complete_plan_mda_cap") != 480
            or plan.get("probe_call_cap_including_carried") != 120
            or plan.get("fresh_calibration_calls") != 72
            or plan.get("methods") != list(METHODS) or plan.get("seeds") != [31, 32, 33]
            or plan.get("settings") != asdict(replace(SystemSettings(), calibration_window=72))
            or plan.get("parallel_workers") != 4
            or type(plan.get("carried_mda_calls")) is not int or not 0 <= plan["carried_mda_calls"] <= 4
            or plan.get("remaining_mda_cap") != 480-plan["carried_mda_calls"]
            or len(plan.get("carried_probes", [])) != plan["carried_mda_calls"]
            or len(plan.get("probes", [])) != 120):
        raise ValueError("Scope differs from the reviewed 480-call complete plan")
    if [p["index"] for p in plan["probes"]] != list(range(1, 121)):
        raise ValueError("Probe indices differ")
    if len({digest(p["candidate"]) for p in plan["probes"]}) != 120:
        raise ValueError("Duplicated diagnostic candidate")
    if any(p["candidate"] != plan["probes"][p["index"]-1]["candidate"] for p in plan["carried_probes"]):
        raise ValueError("Carried diagnostic candidate differs")


def prepare(predecessor, output_root):
    predecessor = predecessor.resolve()
    old = load_json((predecessor/"plan.json").read_bytes())
    result = load_json((predecessor/"report.json").read_bytes())
    audit = load_json((predecessor/"independent_audit.json").read_bytes())
    if result["status"] != "terminal" or audit["status"] != "passed":
        raise ValueError("Predecessor must be terminal and independently audited")
    verify_sources(old["source_sha256"])
    verify_sources(audit["input_sha256"])
    if any(run["evaluations"] for run in result["physical"]):
        raise ValueError("Predecessor physical optimizers must not have executed")
    carried = [r for r in read_lines(predecessor/"probe_results.jsonl") if r["risk"] is not None]
    if any(r["risk"]["state"] in ("unknown", "infrastructure_error") for r in carried):
        raise ValueError("A physical evaluator error requires diagnosis before a new physical plan")
    if not 0 <= len(carried) <= 4 or audit["new_mda_calls"] != len(carried):
        raise ValueError("Unexpected carried physical scope")
    project = Path(__file__).resolve().parents[1]
    folder = output_root.resolve()/("system_validation_v2_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                     +"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    config_path = folder/"frozen_config.json"
    config_path.write_bytes(Path(old["physical"]["config_path"]).read_bytes())
    initial = load_json((predecessor/"initial_history.json").read_bytes())
    save_new(folder/"initial_history.json", initial)
    problem = X57LiveProblem(old["physical"]["base_project"], config_path, folder/"planning", allow_mda=False)
    if problem.contract_id != old["physical"]["contract_id"]:
        raise ValueError("Physical contract changed")
    # The inherited accepted proposals are the only old-version failure inputs.
    old_inputs = [{"candidate": item["candidate"]} for item in old["probes"]
                  if item["source"] == "old_version_failure_input_new_label_required"]
    probes = diagnostic_candidates(problem, initial, old_inputs, count=120)
    if probes[:48] != old["probes"]:
        raise ValueError("The original diagnostic sequence changed")
    for row in carried:
        if row["candidate"] != probes[row["index"]-1]["candidate"]:
            raise ValueError("Carried call candidate differs")
    sources = dict(old["source_sha256"])
    sources.update(audit["input_sha256"])
    paths = [Path(__file__).resolve(), project/"tools/audit_system_validation_v2.py",
             project/"docs/SYSTEM_VALIDATION_V2_PROTOCOL_20261021.md",
             predecessor/"plan.json", predecessor/"report.json", predecessor/"independent_audit.json",
             predecessor/"physical_STOP.json", config_path, folder/"initial_history.json"]
    sources.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    verify_sources(sources)
    plan = {
        "schema": "ra_system_validation_v2", "operator_approved": True,
        "complete_plan_mda_cap": 480, "carried_mda_calls": len(carried),
        "remaining_mda_cap": 480-len(carried), "probe_call_cap_including_carried": 120,
        "fresh_calibration_calls": 72, "methods": old["methods"], "seeds": old["seeds"],
        "parallel_workers": 4, "settings": asdict(replace(SystemSettings(), calibration_window=72)),
        "predecessor": str(predecessor), "source_sha256": sources, "probes": probes,
        "carried_probes": carried,
        "physical": dict(old["physical"], config_path=str(config_path)),
        "source_note": "Inherited controlled results; physical training probes precede all fresh calibration."}
    validate_plan(plan)
    save_new(folder/"plan.json", plan)
    snapshot(folder/"plan.json")
    print("Prepared", folder, "complete cap", 480, "carried", len(carried), flush=True)
    return folder


def collect_batch(pool, folder, spec, items, stop, output_path):
    futures = [pool.submit(probe_one, spec, item,
                           str(folder/("call_"+str(item["index"]))), str(stop)) for item in items]
    values = []
    for future in futures:
        try:
            value = future.result()
        except Exception as error:
            signal_stop(stop, {"reason": f"Diagnostic process error: {type(error).__name__}: {error}"})
            raise
        values.append(value)
        append_json(output_path, value)
        print(folder.name, value["index"], value["status"],
              None if value["risk"] is None else value["risk"]["state"], flush=True)
    return values


def freeze_calibration(folder, plan, problem, training):
    settings = SystemSettings(**plan["settings"])
    learner = FullSystemLearning("cbo", settings, 0)
    models, metadata = learner.fit(
        training, [asdict(v) for v in problem.space.variables], plan["physical"]["base_project"],
        sorted(set(problem.required_constraints)-set(problem.analytical_constraints)))
    learner.features.save_checkpoint(folder/"calibration_planning_encoder.pt")
    save_new(folder/"calibration_planning_model.json", metadata)
    responses = models[1]
    seen = {digest(problem.canonical(r["candidate"])) for r in training}
    rng, groups = np.random.default_rng(20261022), {"outer": [], "near": [], "inner": []}
    for _ in range(40):
        problem.verify_frozen()
        pool, _ = candidate_pool(
            problem.space, rng, 512, seen, training,
            lambda raw: problem.problem.precheck(problem.candidate_class(**raw)))
        candidates = [item["candidate"] for item in pool]
        if not candidates:
            continue
        predictions = responses.predict(candidates)
        margins = np.min([
            -np.asarray(values["mean"])/np.maximum(np.asarray(values["std"]), 1e-12)
            for name, values in predictions.items() if name.startswith("constraint:") and values is not None], axis=0)
        for i, item in enumerate(pool):
            seen.add(digest(item["candidate"]))
            band = "outer" if margins[i] < 0 else "near" if margins[i] < 1 else "inner"
            if len(groups[band]) < 24:
                groups[band].append(dict(item, band=band, standardized_margin=float(margins[i])))
        if all(len(items) == 24 for items in groups.values()):
            break
    if any(len(items) != 24 for items in groups.values()):
        save_new(folder/"calibration_preparation_failed.json", {
            "counts": {name: len(items) for name, items in groups.items()},
            "extra_mda_calls": 0})
        raise ValueError("Unable to fill the frozen fresh calibration strata")
    proposals = []
    for i in range(24):
        for band in ("outer", "near", "inner"):
            item = groups[band][i]
            proposals.append(dict(item, index=len(proposals)+1,
                                  source="fresh_calibration_"+band))
    save_new(folder/"fresh_calibration_plan.json", {
        "training_indices": learner.features.training_indices,
        "training_history_id": learner.features.history_id,
        "representation_snapshot": learner.features.snapshot_id,
        "proposals": proposals, "source_sha256": plan["source_sha256"],
        "new_calibration_labels_available": False})
    plan_sha = hashlib.sha256((folder/"fresh_calibration_plan.json").read_bytes()).hexdigest()
    return [dict(item, source=item["source"]+":"+plan_sha) for item in proposals]


def execute(folder):
    torch.set_num_threads(1)
    folder = folder.resolve()
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    verify_sources(plan["source_sha256"])
    save_new(folder/"execution_started.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest()})
    stop = folder/"physical_STOP.json"
    probes = list(plan["carried_probes"])
    for value in probes:
        append_json(folder/"probe_results.jsonl", value)
    with ProcessPoolExecutor(max_workers=plan["parallel_workers"]) as pool:
        for offset in range(len(probes), 120, 4):
            if stop.exists():
                break
            probes.extend(collect_batch(pool, folder/"probes", plan["physical"],
                                        plan["probes"][offset:offset+4], stop, folder/"probe_results.jsonl"))
            complete = [r for r in probes if r["risk"] is not None]
            if len(complete) >= 24 and not any(r["risk"]["state"] == "hidden_failure" for r in complete):
                save_new(folder/"probe_zero_positive_stop.json", {
                    "completed": len(complete), "reason": "Predeclared zero-positive stop",
                    "not_a_proof_of_impossibility": True})
                break
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                             folder/"planning", allow_mda=False)
    initial = load_json((folder/"initial_history.json").read_bytes())
    probe_rows = [{"evaluation": r["index"], "candidate": r["candidate"], "risk": r["risk"]}
                  for r in probes if r["risk"] is not None and r["risk"]["state"] != "precheck_rejected"]
    training, duplicate_probes = combine_history(problem, [initial, probe_rows])
    save_new(folder/"precalibration_training.json", training)
    calibration = []
    if not stop.exists():
        proposals = freeze_calibration(folder, plan, problem, training)
        with ProcessPoolExecutor(max_workers=plan["parallel_workers"]) as pool:
            for offset in range(0, 72, 4):
                if stop.exists():
                    break
                calibration.extend(collect_batch(
                    pool, folder/"calibration", plan["physical"], proposals[offset:offset+4],
                    stop, folder/"calibration_results.jsonl"))
    calibration_rows = [
        {"evaluation": r["index"], "candidate": r["candidate"], "risk": r["risk"]}
        for r in calibration if r["risk"] is not None and r["risk"]["state"] != "precheck_rejected"]
    shared, duplicates = combine_history(problem, [initial, probe_rows, calibration_rows])
    save_new(folder/"shared_history.json", shared)
    save_new(folder/"mechanism_data_report.json", {
        "scope": "Current physical contract only; controlled outcomes excluded",
        "training_support": mechanism_support(training),
        "calibration_support": mechanism_support(calibration_rows),
        "probe_support": mechanism_support(probe_rows), "probe_calls_including_carried": len(probe_rows),
        "fresh_calibration_calls": len(calibration_rows), "duplicates_removed": duplicate_probes+duplicates,
        "training_support_flags": {name: values["positive"] >= 20 and values["negative"] >= 20
                                   for name, values in mechanism_support(training).items()}})
    if len(calibration_rows) != 72:
        signal_stop(stop, {"reason": "Fresh calibration block incomplete; no optimization submissions"})
    physical = run_trials(folder, plan, "physical",
                          {seed: folder/"shared_history.json" for seed in plan["seeds"]}, plan["settings"])
    ledger_paths = [*sorted((folder/"probes").rglob("budget.jsonl")),
                    *sorted((folder/"calibration").rglob("budget.jsonl")),
                    *sorted((folder/"physical").rglob("budget.jsonl"))]
    newly_charged = sum(RiskLedger(path).summary()["charged_calls"] for path in ledger_paths)
    total = newly_charged+plan["carried_mda_calls"]
    if total > 480:
        raise RuntimeError("Complete physical budget exceeded")
    verify_sources(plan["source_sha256"])
    old = load_json((Path(plan["predecessor"])/"report.json").read_bytes())
    save_new(folder/"report.json", {
        "schema": plan["schema"], "status": "terminal", "physical": physical,
        "controlled_source": plan["predecessor"], "controlled": old["controlled"],
        "carried_mda_calls": plan["carried_mda_calls"], "new_v2_mda_calls": newly_charged,
        "complete_new_mda_calls": total, "maximum_complete_mda_calls": 480,
        "probe_calls_including_carried": len(probe_rows), "fresh_calibration_calls": len(calibration_rows),
        "completed_utc": datetime.now(timezone.utc).isoformat()})
    print("Terminal report", folder/"report.json", "complete MDA", total, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"runs")
    args = parser.parse_args()
    if (args.prepare is None) == (args.execute is None):
        parser.error("Choose exactly one of --prepare or --execute")
    if args.prepare is not None:
        prepare(args.prepare, args.output)
    else:
        execute(args.execute)
