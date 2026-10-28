"""Freeze and execute mechanism-data and paired full-system validation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import build_snapshot, load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import BudgetPolicy, RiskLedger, digest
from ra_lsbo.calibration import SigmoidCalibrator, observed_target
from ra_lsbo.features import history_identity
from ra_lsbo.legacy import normalize_evaluation
from ra_lsbo.mechanism_system import FiveModeProblem, HIDDEN_MODES, METHODS, SystemSettings, mechanism_support
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.optimizer import append_json, dispatch
from ra_lsbo.preflight import inspect_x57
from ra_lsbo.system_experiment import (
    FullSystemLearning, run_system_trial, sample_rows, save_new, signal_stop, update_json)
from ra_lsbo.x57_live import X57LiveProblem
from tools.snapshot_experiment_sources import snapshot


def read_lines(path):
    return [load_json(line) for line in Path(path).read_bytes().splitlines()] if Path(path).exists() else []


def combine_history(problem, groups):
    seen, result, excluded = set(), [], []
    identity = digest([[history_identity(rows), len(rows)] for rows in groups])
    for group, rows in enumerate(groups):
        for row in rows:
            canonical = problem.canonical(row["candidate"])
            key = problem.space.key(problem.candidate_class(**canonical))
            if key in seen:
                excluded.append({"group": group, "evaluation": row["evaluation"], "candidate_key": key})
                continue
            seen.add(key)
            result.append(dict(
                row, evaluation=len(result)+1, candidate=canonical, source_journal_sha256=identity,
                evaluator_contract_sha256=problem.contract_id,
                source_origin={"group": group, "evaluation": row["evaluation"],
                               "journal": row.get("source_journal_sha256")}))
    return result, excluded


def diagnostic_candidates(problem, history, old_failed_rows, count=48):
    rng = np.random.default_rng(20261021)
    seen = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in history}
    result = []

    def add(raw, source):
        try:
            canonical = problem.canonical(raw)
            key = problem.space.key(problem.candidate_class(**canonical))
            checks = problem.problem.precheck(problem.candidate_class(**canonical))
        except (ValueError, KeyError, TypeError):
            return
        if key in seen or any(not math.isfinite(v) or v > 0 for v in checks.values()):
            return
        seen.add(key)
        result.append({"index": len(result)+1, "candidate": canonical, "source": source})

    for row in old_failed_rows:
        if len(result) == 8:
            break
        add(row["candidate"], "old_version_failure_input_new_label_required")
    configurations = problem.space.configurations()
    for attempt in range(30000):
        if len(result) == count:
            break
        d = configurations[int(rng.integers(len(configurations)))]
        if len(result) < 24:
            unit = rng.choice([.02, .08, .92, .98], len(problem.space.continuous))
            candidate = problem.space.from_unit(unit, d)
            source = "legal_domain_corner_stress"
        else:
            candidate = problem.space.sample(rng, d)
            source = "legal_domain_global_coverage"
        add(candidate.to_dict(), source)
    if len(result) != count:
        raise ValueError("Cannot freeze the complete diagnostic candidate set")
    return result


def prepare(boundary_folder, output_root):
    boundary_folder = boundary_folder.resolve()
    parent = load_json((boundary_folder/"plan.json").read_bytes())
    audit = load_json((boundary_folder/"audit.json").read_bytes())
    completed = load_json((boundary_folder/"result.json").read_bytes())
    if audit["status"] != "passed" or completed["status"] != "complete":
        raise ValueError("Require the completed, audited parent physical study")
    verify_sources(parent["source_sha256"])
    verify_sources(audit["input_sha256"])
    preflight = inspect_x57(parent["historical_source"], parent["base_project"], parent["config_path"])
    if not preflight["historical_data_reusable_without_revalidation"]:
        raise ValueError("Historical physical/configuration contract differs")
    folder = output_root.resolve()/("system_validation_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                     +"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    config_path = folder/"frozen_config.json"
    config_path.write_bytes(Path(parent["config_path"]).read_bytes())
    problem = X57LiveProblem(parent["base_project"], config_path, folder/"probe_workers", allow_mda=False)
    if problem.contract_id != parent["contract_id"]:
        raise ValueError("Live contract differs from the audited parent")
    history, _, _, contract = build_snapshot(parent["historical_source"])
    boundary = [{"evaluation": r["index"], "candidate": r["proposal"]["candidate"], "risk": r["risk"]}
                for r in completed["results"]]
    shared, duplicates = combine_history(problem, [history, boundary])
    save_new(folder/"initial_history.json", shared)
    save_new(folder/"preflight.json", preflight)
    old_paths = [
        Path(parent["base_project"])/"runs/native_lift_trial/run_20261004_172404_d44184d4",
        Path(parent["base_project"])/"runs/native_lift_trial/run_20261004_212210_d1e7bacf"]
    old_failed, old_hashes = [], {}
    for source in old_paths:
        rows, _, report, _ = build_snapshot(source)
        old_failed.extend(r for r in rows if r["risk"]["state"] == "hidden_failure")
        old_hashes.update({str(source/name): sha for name, sha in report["source_sha256"].items()})
    probes = diagnostic_candidates(problem, shared, old_failed)
    project = Path(__file__).resolve().parents[1]
    sources = dict(parent["source_sha256"])
    sources.update(old_hashes)
    paths = [Path(__file__).resolve(), project/"tools/audit_system_validation.py",
             project/"docs/SYSTEM_VALIDATION_PROTOCOL_20261021.md",
             folder/"initial_history.json", folder/"preflight.json", config_path,
             boundary_folder/"plan.json", boundary_folder/"audit.json", boundary_folder/"result.json",
             *sorted((project/"ra_lsbo").glob("*.py")),
             *sorted((Path(parent["base_project"])/"lsbo").glob("*.py"))]
    sources.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    verify_sources(sources)
    plan = {
        "schema": "ra_system_validation_v1", "operator_approved": True,
        "authorization": "User requested both tasks and previously authorized complete plans <=500 MDA.",
        "maximum_new_mda_calls": 336, "probe_call_cap": 48, "parallel_workers": 4,
        "source_sha256": sources, "physical": {
            "kind": "physical", "base_project": parent["base_project"],
            "config_path": str(config_path), "contract_id": problem.contract_id},
        "methods": list(METHODS), "seeds": [31, 32, 33],
        "settings": asdict(SystemSettings()), "probes": probes,
        "probe_stop": "After 24 completed probes, stop if no terminal hidden failure was found.",
        "initial_history_count": len(shared), "initial_history_duplicates": duplicates,
        "initial_support": mechanism_support(shared), "controlled_static_split": [600, 300, 600],
        "controlled_settings": asdict(SystemSettings(
            calls=24, calibration_window=80, discovery_cap=8, boundary_cap=8, failure_cap=12)),
        "controlled_initial_count": 240, "historical_aircraft_evaluation_cost": 420,
        "scope": "Same aircraft task, three randomized repetitions, no guaranteed superiority."}
    save_new(folder/"plan.json", plan)
    snapshot(folder/"plan.json")
    print("Prepared", folder, "MDA cap", plan["maximum_new_mda_calls"], flush=True)
    return folder


def probe_one(spec, item, folder, stop_path):
    import torch
    torch.set_num_threads(1)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    if Path(stop_path).exists():
        return {"index": item["index"], "status": "not_started_global_stop", "risk": None}
    with (folder/"console.log").open("x", encoding="utf-8") as log, redirect_stdout(log):
        problem = X57LiveProblem(spec["base_project"], spec["config_path"], folder/"workers", allow_mda=True)
        if problem.contract_id != spec["contract_id"]:
            raise ValueError("Probe physical identity differs")
        proposal = {"candidate": item["candidate"], "source": item["source"], "mode": "baseline",
                    "evidence": {"basis": "unavailable", "probability": None}}
        save_new(folder/"proposal.json", proposal)
        with RiskLedger(folder/"budget.jsonl", BudgetPolicy(1, 1., 0, 0, 0, allow_baseline=True)) as ledger:
            if Path(stop_path).exists():
                return {"index": item["index"], "status": "not_started_global_stop", "risk": None}
            risk = dispatch(problem, ledger, proposal)
            result = {"index": item["index"], "candidate": item["candidate"], "risk": risk,
                      "ledger": ledger.summary(), "directory": str(folder), "status": "evaluated"}
        save_new(folder/"result.json", result)
        if risk is None or risk["state"] in ("unknown", "infrastructure_error"):
            signal_stop(Path(stop_path), {"reason": "Diagnostic physical outcome unavailable",
                                         "probe": item["index"], "directory": str(folder)})
        return result


def trial_logged(spec, method, seed, settings, history_path, output, stop_path):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".log").open("x", encoding="utf-8") as log, redirect_stdout(log):
        return run_system_trial(spec, method, seed, settings, history_path, output, stop_path)


def run_trials(folder, plan, kind, history_paths, settings):
    spec = plan["physical"] if kind == "physical" else {
        "kind": "controlled", "base_project": plan["physical"]["base_project"],
        "contract_id": FiveModeProblem(plan["physical"]["base_project"]).contract_id}
    stop = folder/(kind+"_STOP.json")
    results = []
    with ProcessPoolExecutor(max_workers=plan["parallel_workers"]) as pool:
        futures = {}
        for seed in plan["seeds"]:
            for method in plan["methods"]:
                output = folder/kind/("seed_"+str(seed))/method
                future = pool.submit(trial_logged, spec, method, seed, settings,
                                     str(history_paths[seed]), str(output), str(stop))
                futures[future] = (method, seed)
        for future in as_completed(futures):
            method, seed = futures[future]
            try:
                result = future.result()
            except Exception as error:
                signal_stop(stop, {"reason": f"Trial process error: {type(error).__name__}: {error}",
                                   "method": method, "seed": seed})
                result = {"method": method, "seed": seed, "status": "process_error", "kind": kind,
                          "error": str(error), "directory": str(folder/kind/("seed_"+str(seed))/method)}
            results.append(result)
            append_json(folder/(kind+"_completed.jsonl"), result)
            print(kind, method, seed, result["status"], result.get("evaluations"),
                  result.get("best_feasible_objective"), flush=True)
    return sorted(results, key=lambda r: (r["seed"], plan["methods"].index(r["method"])))


def controlled_static(folder, plan):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    problem = FiveModeProblem(plan["physical"]["base_project"])
    rows = sample_rows(problem, 1500, 20261021)
    rows, excluded = combine_history(problem, [rows])
    if excluded:
        raise ValueError("Duplicate controlled diagnostics")
    save_new(folder/"controlled_static_history.json", rows)
    training, calibration, test = rows[:600], rows[600:900], rows[900:]
    variables = [asdict(v) for v in problem.space.variables]
    settings = SystemSettings(calibration_window=300)
    report = {"scope": "Controlled five-mode process, never aircraft data",
              "support": {name: mechanism_support(block) for name, block in
                          (("training", training), ("calibration", calibration), ("test", test))},
              "models": []}
    for method in ("ra_full", "ra_no_mechanisms", "ra_onehot_risk"):
        learner = FullSystemLearning(method, settings, 0)
        models, metadata = learner.fit(rows[:900], variables, plan["physical"]["base_project"],
                                      ["envelope", "load"])
        risks, _, _ = models
        learner.features.save_checkpoint(folder/(method+"_controlled_static.pt"))
        cal_raw = risks.predict([r["candidate"] for r in calibration])
        test_raw = risks.predict([r["candidate"] for r in test])
        calibrators, calibrated, scores = {}, {}, {}
        for name in ("final_feasibility", "hidden_success", *HIDDEN_MODES):
            labels = [observed_target(r["risk"], name) for r in calibration]
            known = [i for i, y in enumerate(labels) if y is not None]
            value = SigmoidCalibrator([cal_raw[name][i] for i in known],
                                      [labels[i] for i in known],
                                      base_fitted=risks.diagnostics[name]["status"] == "fitted_raw")
            calibrators[name] = value.diagnostics
            calibrated[name] = value.predict(test_raw[name])
        for i, row in enumerate(test):
            append_json(folder/"controlled_forecasts.jsonl", {
                "method": method, "evaluation": row["evaluation"], "candidate": row["candidate"],
                "raw": {name: test_raw[name][i] for name in calibrated},
                "calibrated": {name: None if values is None else values[i]
                               for name, values in calibrated.items()}})
        for name in calibrated:
            labels = [observed_target(r["risk"], name) for r in test]
            known = [i for i, label in enumerate(labels) if label is not None]
            values = calibrated[name]
            scores[name] = {
                "raw": binary_metrics([test_raw[name][i] for i in known], [labels[i] for i in known]),
                "calibrated": binary_metrics(
                    [] if values is None else [values[i] for i in known],
                    [] if values is None else [labels[i] for i in known]),
                "unknown_labels": len(labels)-len(known)}
        report["models"].append({"method": method, "metadata": metadata,
                                 "calibration": calibrators, "scores": scores})
        print("controlled static", method, "fitted mechanisms", list(risks.active_mechanisms), flush=True)
    forecasts = read_lines(folder/"controlled_forecasts.jsonl")
    lookup = {r["evaluation"]: r for r in test}
    for forecast in forecasts:
        append_json(folder/"controlled_scored.jsonl", dict(
            forecast, observed=lookup[forecast["evaluation"]]["risk"]))
    report["training_support_flags"] = {
        name: counts["positive"] >= 20 and counts["negative"] >= 20
        for name, counts in report["support"]["training"].items()}
    save_new(folder/"controlled_static_report.json", report)


def execute(folder):
    folder = folder.resolve()
    plan = load_json((folder/"plan.json").read_bytes())
    if (plan["schema"] != "ra_system_validation_v1" or plan["maximum_new_mda_calls"] != 336
            or plan["methods"] != list(METHODS) or plan["seeds"] != [31, 32, 33]
            or plan["operator_approved"] is not True):
        raise ValueError("Execution differs from the frozen complete scope")
    verify_sources(plan["source_sha256"])
    save_new(folder/"execution_started.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest()})
    controlled_static(folder, plan)
    control = FiveModeProblem(plan["physical"]["base_project"])
    controlled_paths = {}
    for seed in plan["seeds"]:
        rows = sample_rows(control, plan["controlled_initial_count"], seed+300000)
        rows, duplicates = combine_history(control, [rows])
        if duplicates:
            raise ValueError("Duplicate controlled warm start")
        path = folder/("controlled_history_seed_"+str(seed)+".json")
        save_new(path, rows)
        controlled_paths[seed] = path
    controlled = run_trials(folder, plan, "controlled", controlled_paths, plan["controlled_settings"])
    if any(r["status"] in ("process_error", "stopped_model_error", "stopped_accounting_error")
           for r in controlled):
        save_new(folder/"controlled_preflight_failed.json", {
            "status": "not_started_physical", "new_mda_calls": 0, "controlled": controlled})
        raise RuntimeError("Controlled integration failed; inspect before any real submissions")
    initial = load_json((folder/"initial_history.json").read_bytes())
    probe_results = []
    stop_path = folder/"physical_STOP.json"
    with ProcessPoolExecutor(max_workers=plan["parallel_workers"]) as pool:
        for offset in range(0, 48, 4):
            if stop_path.exists():
                break
            futures = [pool.submit(probe_one, plan["physical"], item,
                                   str(folder/"probes"/("probe_"+str(item["index"]))), str(stop_path))
                       for item in plan["probes"][offset:offset+4]]
            for future in futures:
                try:
                    value = future.result()
                except Exception as error:
                    signal_stop(stop_path, {"reason": f"Probe process error: {type(error).__name__}: {error}"})
                    raise
                probe_results.append(value)
                append_json(folder/"probe_results.jsonl", value)
                print("probe", value["index"], value["status"],
                      None if value["risk"] is None else value["risk"]["state"], flush=True)
            completed = [r for r in probe_results if r["risk"] is not None]
            if len(completed) >= 24 and not any(r["risk"]["state"] == "hidden_failure" for r in completed):
                save_new(folder/"probe_zero_positive_stop.json", {
                    "completed": len(completed), "reason": "Predeclared no-hidden-positive stop",
                    "not_a_proof_of_impossibility": True})
                break
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                             folder/"history_precheck", allow_mda=False)
    probes = [{"evaluation": r["index"], "candidate": r["candidate"], "risk": r["risk"]}
              for r in probe_results if r["risk"] is not None and r["risk"]["state"] != "precheck_rejected"]
    shared, duplicates = combine_history(problem, [initial, probes])
    save_new(folder/"shared_history.json", shared)
    save_new(folder/"mechanism_data_report.json", {
        "initial_support": plan["initial_support"], "probe_support": mechanism_support(probes),
        "combined_support": mechanism_support(shared), "new_mda_calls": len(probes),
        "duplicates_removed": duplicates,
        "training_support_flags": {name: v["positive"] >= 20 and v["negative"] >= 20
                                   for name, v in mechanism_support(shared).items()}})
    # Real runs and controlled runs are distinct; neither supplies labels to the other.
    physical = run_trials(folder, plan, "physical",
                          {seed: folder/"shared_history.json" for seed in plan["seeds"]}, plan["settings"])
    verify_sources(plan["source_sha256"])
    actual_ledgers = [*sorted((folder/"probes").rglob("budget.jsonl")),
                      *sorted((folder/"physical").rglob("budget.jsonl"))]
    actual_calls = sum(RiskLedger(path).summary()["charged_calls"] for path in actual_ledgers)
    report = {"schema": plan["schema"], "status": "terminal",
              "physical": physical, "controlled": controlled,
              "probe_mda_calls": len(probes),
              "new_mda_calls": actual_calls,
              "maximum_new_mda_calls": plan["maximum_new_mda_calls"],
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    if report["new_mda_calls"] > 336:
        raise RuntimeError("Complete physical budget exceeded")
    save_new(folder/"report.json", report)
    print("Terminal report", folder/"report.json", "MDA", report["new_mda_calls"], flush=True)


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
