"""Frozen supplemental controlled comparison; never launches a physical MDA."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch
import uuid

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.controlled_sparse import SparseInteriorFiveModeProblem, enable_sparse_case
from ra_lsbo.features import history_identity
from ra_lsbo.mechanism_system import METHODS, SystemSettings
from ra_lsbo.system_experiment import sample_rows, save_new
from tools import analyze_system_validation as analysis
from tools import audit_system_validation as auditor
from tools import run_system_validation as original
from tools.reconstruct_system_forecasts import reconstruct
from tools.snapshot_experiment_sources import snapshot


SETTINGS = asdict(SystemSettings(calls=16, calibration_window=300))


def validate_plan(plan):
    if (plan.get("schema") != "ra_sparse_controlled_validation_v1"
            or plan.get("additional_real_mda_calls") != 0
            or plan.get("methods") != list(METHODS) or plan.get("seeds") != [31, 32, 33]
            or plan.get("controlled_settings") != SETTINGS or plan.get("parallel_workers") != 1
            or plan.get("controlled_initial_count") != 900
            or plan.get("maximum_new_controlled_optimizer_calls") != 384
            or plan.get("sparsity_sample_count") != 10000):
        raise ValueError("Frozen controlled supplement scope differs")


def prepare(parent, output):
    physical = load_json((parent/"plan.json").read_bytes())
    verify_sources(physical["source_sha256"])
    folder = output/("sparse_controlled_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                     +"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    paths = [*sorted((project/"ra_lsbo").glob("*.py")), Path(__file__).resolve(),
             project/"tools/analyze_system_validation.py", project/"tools/audit_system_validation.py",
             project/"tools/reconstruct_system_forecasts.py", project/"tools/encoder_framework_experiment.py",
             project/"tools/run_system_validation.py", project/"tools/snapshot_experiment_sources.py",
             project/"docs/SPARSE_CONTROLLED_PROTOCOL_20261021.md", parent/"plan.json"]
    sources = {p: sha for p, sha in physical["source_sha256"].items()
               if "/runs/" not in p.replace("\\", "/").lower()}
    sources.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    problem = SparseInteriorFiveModeProblem(physical["physical"]["base_project"])
    plan = {
        "schema": "ra_sparse_controlled_validation_v1", "additional_real_mda_calls": 0,
        "physical": physical["physical"], "controlled_contract_id": problem.contract_id,
        "methods": list(METHODS), "seeds": [31, 32, 33], "controlled_settings": SETTINGS,
        "parallel_workers": 1, "controlled_initial_count": 900,
        "maximum_new_controlled_optimizer_calls": 384, "sparsity_sample_count": 10000,
        "source_sha256": sources,
        "scope": "Supplementary newly designed controlled case; preserves all earlier unfavorable outcomes"}
    validate_plan(plan)
    verify_sources(sources)
    save_new(folder/"plan.json", plan)
    snapshot(folder/"plan.json")
    print("Prepared zero-real-MDA supplement", folder, flush=True)
    return folder


def sparse_pool(*args, **kwargs):
    return ProcessPoolExecutor(*args, initializer=enable_sparse_case, **kwargs)


def execute(folder):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    verify_sources(plan["source_sha256"])
    problem = SparseInteriorFiveModeProblem(plan["physical"]["base_project"])
    if problem.contract_id != plan["controlled_contract_id"]:
        raise ValueError("Controlled process changed")
    save_new(folder/"execution_started.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest()})
    screened = sample_rows(problem, 10000, 20261025)
    save_new(folder/"sparsity_sample.json", screened)
    save_new(folder/"sparsity_summary.json", {
        "samples": len(screened), "sampling": "Independent uniform draw conditional on packing precheck",
        "states": dict(Counter(r["risk"]["state"] for r in screened)),
        "feasible_fraction": sum(r["risk"]["feasible_label"] is True for r in screened)/len(screened),
        "new_real_mda_calls": 0})
    paths = {}
    for seed in plan["seeds"]:
        rows, duplicates = original.combine_history(problem, [sample_rows(problem, 900, seed+300000)])
        if duplicates:
            raise ValueError("Duplicate initial controlled observations")
        paths[seed] = folder/("controlled_history_seed_"+str(seed)+".json")
        save_new(paths[seed], rows)
    with patch.object(original, "FiveModeProblem", SparseInteriorFiveModeProblem), \
            patch.object(original, "ProcessPoolExecutor", sparse_pool):
        original.controlled_static(folder, plan)
        runs = original.run_trials(folder, plan, "controlled", paths, SETTINGS)
    verify_sources(plan["source_sha256"])
    save_new(folder/"report.json", {
        "schema": plan["schema"], "status": "terminal", "controlled": runs, "physical": [],
        "new_mda_calls": 0, "additional_real_mda_calls": 0,
        "new_controlled_optimizer_calls": sum(r.get("evaluations", 0) for r in runs),
        "completed_utc": datetime.now(timezone.utc).isoformat()})


def audit(folder):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    report = load_json((folder/"report.json").read_bytes())
    verify_sources(plan["source_sha256"])
    if report["status"] != "terminal" or report["new_mda_calls"] != 0 or report["physical"]:
        raise ValueError("Physical work or incomplete supplement")
    for sha in set(plan["source_sha256"].values()):
        if hashlib.sha256((folder/"source_snapshot"/(sha+".source")).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen source snapshot differs")
    problem = SparseInteriorFiveModeProblem(plan["physical"]["base_project"])
    if problem.contract_id != plan["controlled_contract_id"]:
        raise ValueError("Controlled contract differs")
    truth_count = 0
    expected_sizes = {"sparsity_sample.json": 10000, "controlled_static_history.json": 1500,
                      **{"controlled_history_seed_"+str(seed)+".json": 900 for seed in plan["seeds"]}}
    for filename, expected_size in expected_sizes.items():
        rows = load_json((folder/filename).read_bytes())
        if len(rows) != expected_size:
            raise ValueError("Controlled sample quota differs")
        for row in rows:
            if problem.evaluate(row["candidate"]) != row["risk"]:
                raise ValueError("Controlled source truth differs")
            truth_count += 1
        if filename == "sparsity_sample.json":
            sparsity = load_json((folder/"sparsity_summary.json").read_bytes())
            if (sparsity["states"] != dict(Counter(r["risk"]["state"] for r in rows))
                    or sparsity["feasible_fraction"] != sum(
                        r["risk"]["feasible_label"] is True for r in rows)/len(rows)):
                raise ValueError("Screened-domain sparsity estimate differs")
    expected = {(m, seed) for m in plan["methods"] for seed in plan["seeds"]}
    if {(r["method"], r["seed"]) for r in report["controlled"]} != expected:
        raise ValueError("Missing method/seed outcomes")
    verified = []
    with patch.object(auditor, "FiveModeProblem", SparseInteriorFiveModeProblem):
        for run in report["controlled"]:
            if run["status"] in ("process_error", "stopped_model_error", "stopped_accounting_error"):
                raise ValueError("An implementation error requires resolution")
            directory = Path(run["directory"])
            manifest = load_json((directory/"manifest.json").read_bytes())
            history_path = folder/("controlled_history_seed_"+str(run["seed"])+".json")
            history = load_json(history_path.read_bytes())
            if manifest["history_id"] != history_identity(history):
                raise ValueError("Method initial history differs")
            verified.append(auditor.verify_trial(directory, plan, plan["physical"]))
    calls = sum(r["ledger"]["charged_calls"] for r in report["controlled"])
    if calls != report["new_controlled_optimizer_calls"] or calls > 384:
        raise ValueError("Controlled optimizer cost differs")
    inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob("*")
              if p.is_file() and "source_snapshot" not in p.parts and p.suffix in (".json", ".jsonl", ".pt")}
    result = dict(status="passed", new_mda_calls=0, new_controlled_optimizer_calls=calls,
                  controlled_source_truths_verified=truth_count, runs=verified, input_sha256=inputs,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_new(folder/"independent_audit.json", result)
    return result


def finish(folder, output):
    audit(folder)
    with patch.object(analysis, "FiveModeProblem", SparseInteriorFiveModeProblem):
        analysis.analyze(folder, output, controlled_only=True)
    reconstruct(folder, output/"forecast_reconstruction.json", controlled_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.prepare is None) == (args.execute is None):
        parser.error("Choose one of --prepare or --execute")
    if args.prepare:
        prepare(args.prepare.resolve(), (args.output or Path("runs")).resolve())
    else:
        execute(args.execute.resolve())
        finish(args.execute.resolve(), (args.output or args.execute.with_name(args.execute.name+"_analysis")).resolve())
