"""Freeze and run a zero-real-MDA development comparison of revised RA-LSBO."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.adaptive_risk import JointRiskEnsemble, ReferenceTemperature
from ra_lsbo.adaptive_system import ADAPTIVE_METHODS, AdaptiveLearning, LocalRegion, local_pool, select_adaptive
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import BudgetPolicy, BudgetRejected, RiskLedger, digest
from ra_lsbo.controlled_sparse import SparseInteriorFiveModeProblem
from ra_lsbo.features import history_identity
from ra_lsbo.mechanism_system import FiveModeProblem, FullSystemLearning, SystemSettings
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.optimizer import append_json, dispatch
from ra_lsbo.selection import candidate_pool, incumbent
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.system_experiment import (OriginalLSBOWarmStart, policy_for, sample_rows, save_new,
                                       select_system_candidate, update_json)


ROOT = Path(__file__).resolve().parents[1]
METHODS = ADAPTIVE_METHODS+("old_ra", "cbo", "lsbo", "random")
CASES = ("five_mode", "sparse_interior")
SEEDS = (61, 62, 63)
SETTINGS = asdict(SystemSettings(calls=24, failure_cap=12, calibration_window=200, pool_size=256,
                                  encoder_epochs=200, incremental_epochs=20, gp_steps=25,
                                  update_every=4, boundary_cap=3))


def problem_for(case, base):
    if case not in CASES:
        raise ValueError("Only controlled cases are allowed: no physical evaluator entry point")
    return (FiveModeProblem if case == "five_mode" else SparseInteriorFiveModeProblem)(base)


def lines(path):
    return [load_json(line) for line in Path(path).read_bytes().splitlines()] if Path(path).exists() else []


def prepare(parent, output):
    previous = load_json((parent/"plan.json").read_bytes())
    verify_sources(previous["source_sha256"])
    folder = output/("adaptive_research_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                     +"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    paths = [*sorted((ROOT/"ra_lsbo").glob("*.py")), Path(__file__).resolve(),
             ROOT/"tools/diagnose_adaptive_physical.py",
             ROOT/"tools/release_system_training_data.py", ROOT/"tools/run_system_validation.py",
             ROOT/"tools/validate_mechanism_transfer.py",
             ROOT/"docs/METHOD_REVISION_RESEARCH_20261021.md",
             ROOT/"docs/ADAPTIVE_RESEARCH_PROTOCOL_20261021.md"]
    sources = {p: sha for p, sha in previous["source_sha256"].items()
               if "/runs/" not in p.replace("\\", "/").lower()}
    sources.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    plan = {
        "schema": "ra_adaptive_research_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "additional_real_mda_calls": 0, "base_project": previous["physical"]["base_project"],
        "physical_parent": str(parent), "settings": SETTINGS, "methods": list(METHODS),
        "cases": list(CASES), "seeds": list(SEEDS), "initial_count": 600, "static_test_count": 300,
        "max_controlled_online_calls": len(CASES)*len(SEEDS)*len(METHODS)*SETTINGS["calls"],
        "parallel_workers": 4, "source_sha256": sources,
        "contracts": {case: problem_for(case, previous["physical"]["base_project"]).contract_id for case in CASES},
        "scope": "Development on known equations with fresh seeds; not independent confirmation",
        "retrospective": {"seed": 0, "reference_bundle": str(ROOT/"runs/system_shared_reference_v5_20261021"),
                          "held_out_anchor_families": [5, 19],
                          "optimizer_rows": "all 288, common fixed historical test; not prospective outcomes"}}
    save_new(folder/"plan.json", plan)
    (folder/"source_snapshot").mkdir()
    for i, path in enumerate(paths):
        (folder/"source_snapshot"/(str(i)+"_"+path.name)).write_bytes(path.read_bytes())
    print("Prepared", folder, flush=True)
    return folder


def validate_plan(plan):
    if (plan["schema"] != "ra_adaptive_research_v1" or plan["additional_real_mda_calls"] != 0
            or plan["settings"] != SETTINGS or plan["methods"] != list(METHODS)
            or plan["cases"] != list(CASES) or plan["seeds"] != list(SEEDS)
            or plan["initial_count"] != 600 or plan["static_test_count"] != 300
            or plan["max_controlled_online_calls"] != 1584):
        raise ValueError("Frozen study scope changed")
    verify_sources(plan["source_sha256"])


def run_trial(folder, case, method, seed):
    torch.set_num_threads(1)
    folder = Path(folder)
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    problem = problem_for(case, plan["base_project"])
    if problem.contract_id != plan["contracts"][case]:
        raise ValueError("Controlled truth changed")
    output = folder/case/(method+"_"+str(seed))
    output.mkdir(parents=True, exist_ok=False)
    with (output/"execution.log").open("x", encoding="utf-8") as stream, redirect_stdout(stream):
        return _trial(plan, folder, output, problem, case, method, seed)


def _trial(plan, folder, output, problem, case, method, seed):
    settings = SystemSettings(**plan["settings"])
    rows = load_json((folder/case/("initial_"+str(seed)+".json")).read_bytes())
    initial_count = len(rows)
    static = load_json((folder/case/("test_"+str(seed)+".json")).read_bytes())
    reference_ids = list(range(initial_count-settings.calibration_window+1, initial_count+1))
    variables = [asdict(v) for v in problem.space.variables]
    rng, region = np.random.default_rng(seed+200000), LocalRegion()
    seen = {digest(r["candidate"]) for r in rows}
    learner, engine, models = None, None, None
    old_method = "ra_full" if method == "old_ra" else method
    if method in ADAPTIVE_METHODS:
        learner = AdaptiveLearning(method, settings, reference_ids, seed)
    elif method in ("old_ra", "cbo"):
        learner = FullSystemLearning(old_method, settings, seed)
    elif method == "lsbo":
        engine = OriginalLSBOWarmStart(problem, plan["base_project"], rows, seed, settings, output)
        save_new(output/"original_settings.json", engine.settings)
    policy = (policy_for("ra_full", settings) if method == "old_ra" else
              BudgetPolicy(settings.calls, float(settings.calls), 0, 0, 0,
                           max_observed_nonfeasible=settings.failure_cap, allow_baseline=True))
    save_new(output/"manifest.json", {
        "case": case, "method": method, "seed": seed, "history_id": history_identity(rows),
        "settings": asdict(settings), "policy": asdict(policy),
        "reference_ids": reference_ids if method in ADAPTIVE_METHODS else None})
    start, status, stop = time.perf_counter(), "running", None
    new_rows, expected_risk, information_used, fit_number = [], 0., 0, -1
    ledger = RiskLedger(output/"budget.jsonl", policy)
    try:
        for iteration in range(settings.calls):
            if learner is not None and (models is None or iteration % settings.update_every == 0):
                fit_number += 1
                if method in ADAPTIVE_METHODS:
                    models, metadata = learner.fit(rows, variables, plan["base_project"])
                else:
                    models, metadata = learner.fit(rows, variables, plan["base_project"], ["envelope", "load"])
                metadata.update(fit_number=fit_number, before_evaluation=len(rows)+1)
                learner.features.save_checkpoint(output/("encoder_"+str(fit_number)+".pt"))
                if method in ADAPTIVE_METHODS:
                    models[0].save_checkpoint(output/("risk_"+str(fit_number)+".pt"))
                    if fit_number == 0:
                        predicted = models[0].predict([r["candidate"] for r in static])
                        transformed = models[2].transform(predicted["probabilities"], models[0].model_id)
                        save_new(output/"static_forecasts.json", {
                            "raw": predicted, "transformed": transformed.tolist(),
                            "calibration": models[2].to_dict(), "history_id": history_identity(rows)})
                append_json(output/"model_fits.jsonl", metadata)
            if method == "lsbo":
                proposal, selection = engine.propose()
                pool, rejected = [], {}
            else:
                if method in ADAPTIVE_METHODS:
                    pool, rejected = local_pool(problem, rng, settings.pool_size, seen, rows, region,
                                                enabled=method != "adaptive_no_trust")
                    proposal, selection = select_adaptive(
                        pool, rows, models, iteration, method, settings,
                        ledger.summary()["observed_nonfeasible"], expected_risk, information_used)
                    if proposal is not None:
                        proposal = dict(proposal, mode="baseline", evidence={
                            "basis": "raw", "probability": selection["used_final"],
                            "candidate_key": digest(proposal["candidate"]), "model_id": models[0].model_id})
                else:
                    pool, rejected = candidate_pool(problem.space, rng, settings.pool_size, seen,
                                                     rows, problem.precheck)
                    if method == "random":
                        proposal = None if not pool else dict(pool[0], mode="baseline",
                                     evidence={"basis": "unavailable", "probability": None})
                        selection = {"reason": "random"}
                    else:
                        proposal, selection = select_system_candidate(
                            pool, rows, models, ledger, iteration, old_method, settings)
            if proposal is None:
                status, stop = "stopped_by_policy", selection
                break
            append_json(output/"proposals.jsonl", {
                "new_call": iteration+1, "evaluation": len(rows)+1, "fit_number": fit_number,
                "prediction_utc": datetime.now(timezone.utc).isoformat(),
                "proposal": proposal, "selection": selection, "region": asdict(region),
                "history_id": history_identity(rows), "precheck_rejections": rejected})
            previous_best = incumbent(rows)
            risk = dispatch(problem, ledger, proposal)
            if risk is None:
                raise RuntimeError("Candidate changed or failed its final precheck")
            row = {"evaluation": len(rows)+1, "candidate": proposal["candidate"], "risk": risk,
                   "evaluator_contract_sha256": problem.contract_id, "phase": selection.get("mode", "baseline")}
            append_json(output/"observations.jsonl", row)
            rows.append(row)
            new_rows.append(row)
            seen.add(digest(row["candidate"]))
            region.observe(previous_best, risk)
            if engine is not None:
                engine.accept(row, selection)
            expected_risk += selection.get("estimated_risk_debit", 0.)
            information_used += int(selection.get("mode") == "information")
            update_json(output/"progress.json", {
                "new_calls": len(new_rows), "best": incumbent(rows), "fit_number": fit_number,
                "ledger": ledger.summary(), "expected_risk": expected_risk})
            if ledger.summary()["observed_nonfeasible"] >= settings.failure_cap:
                status, stop = "stopped_by_policy", {"reason": "observed_failure_stop"}
                break
        else:
            status = "complete"
    except BudgetRejected as error:
        status, stop = "stopped_by_policy", {"reason": str(error)}
    except Exception as error:
        status, stop = "error", {"reason": f"{type(error).__name__}: {error}"}
        (output/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        ledger.close()
    accounting = RiskLedger(output/"budget.jsonl").summary()
    if accounting["charged_calls"] != len(new_rows) or accounting["unresolved_started_calls"]:
        status, stop = "error", {"reason": "Accounting mismatch"}
    initial_best, best = incumbent(rows[:initial_count]), incumbent(rows)
    summary = {
        "case": case, "method": method, "seed": seed, "status": status, "stop": stop,
        "evaluations": len(new_rows), "feasible": sum(r["risk"]["feasible_label"] is True for r in new_rows),
        "initial_best": initial_best, "best": best,
        "improvement_percent": None if initial_best is None else 100*(initial_best-best)/abs(initial_best),
        "states": dict(Counter(r["risk"]["state"] for r in new_rows)), "ledger": accounting,
        "expected_risk": expected_risk, "information_calls": information_used,
        "wall_seconds": time.perf_counter()-start}
    save_new(output/"summary.json", summary)
    return summary


def execute(folder):
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    save_new(folder/"execution_started.json", {"created_utc": datetime.now(timezone.utc).isoformat()})
    for case in CASES:
        (folder/case).mkdir()
        problem = problem_for(case, plan["base_project"])
        for seed in SEEDS:
            save_new(folder/case/("initial_"+str(seed)+".json"), sample_rows(problem, plan["initial_count"], seed))
            save_new(folder/case/("test_"+str(seed)+".json"), sample_rows(problem, plan["static_test_count"], seed+5000))
    results = []
    with ProcessPoolExecutor(max_workers=plan["parallel_workers"]) as executor:
        jobs = {executor.submit(run_trial, str(folder), case, method, seed): (case, method, seed)
                for case in CASES for seed in SEEDS for method in METHODS}
        for job in as_completed(jobs):
            try:
                result = job.result()
            except Exception as error:
                result = {"case": jobs[job][0], "method": jobs[job][1], "seed": jobs[job][2],
                          "status": "error", "exception": f"{type(error).__name__}: {error}"}
            results.append(result)
            append_json(folder/"completed.jsonl", result)
            print(len(results), "/", len(jobs), result, flush=True)
    analyze(folder)


def analyze(folder):
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    results = lines(folder/"completed.jsonl")
    aggregate, metrics, replay_count = [], [], 0
    for case in CASES:
        problem = problem_for(case, plan["base_project"])
        for method in METHODS:
            trials = [r for r in results if r["case"] == case and r["method"] == method]
            if len(trials) != len(SEEDS):
                raise ValueError("Missing trials")
            aggregate.append({
                "case": case, "method": method, "calls": sum(t.get("evaluations", 0) for t in trials),
                "feasible": sum(t.get("feasible", 0) for t in trials),
                "mean_improvement_percent": (float(np.mean([t["improvement_percent"] for t in trials]))
                                             if all("improvement_percent" in t for t in trials) else None),
                "statuses": dict(Counter(t["status"] for t in trials))})
            for seed in SEEDS:
                output = folder/case/(method+"_"+str(seed))
                initial = load_json((folder/case/("initial_"+str(seed)+".json")).read_bytes())
                observations, proposals = lines(output/"observations.jsonl"), lines(output/"proposals.jsonl")
                if len(observations) != len(proposals):
                    raise ValueError("Proposals and observations differ")
                for row, proposal in zip(observations, proposals):
                    if (row["candidate"] != proposal["proposal"]["candidate"]
                            or row["risk"] != problem.evaluate(row["candidate"])):
                        raise ValueError("Controlled outcome mismatch")
                fits = lines(output/"model_fits.jsonl")
                if method not in ADAPTIVE_METHODS:
                    continue
                for fit in fits:
                    number = fit["fit_number"]
                    reference = set(fit["reference_indices"])
                    past = initial+[r for r in observations if r["evaluation"] < fit["before_evaluation"]]
                    eligible = [r for r in past if r["evaluation"] not in reference]
                    if fit["training_indices"] != [r["evaluation"] for r in eligible]:
                        raise ValueError("New labels were delayed or reference leaked")
                    features = StructuredFeatures.load_checkpoint(output/("encoder_"+str(number)+".pt"),
                                                                  plan["base_project"], eligible)
                    risk = JointRiskEnsemble.load_checkpoint(output/("risk_"+str(number)+".pt"), features)
                    calibration = ReferenceTemperature(risk, [r for r in initial if r["evaluation"] in reference])
                    if calibration.calibration_id != fit["calibration_id"]:
                        raise ValueError("Calibration replay differs")
                    selected = [p for p in proposals if p["fit_number"] == number]
                    predicted = risk.predict([p["proposal"]["candidate"] for p in selected])
                    raw = np.asarray(predicted["probabilities"])
                    transformed = calibration.transform(raw, risk.model_id)
                    for i, proposal in enumerate(selected):
                        recorded = proposal["selection"]
                        np.testing.assert_allclose(raw[i], recorded["raw_events"], atol=2e-7, rtol=1e-6)
                        np.testing.assert_allclose(transformed[i], recorded["transformed_events"], atol=2e-7, rtol=1e-6)
                        if recorded["candidate_key"] != digest(proposal["proposal"]["candidate"]):
                            raise ValueError("Forecast binding differs")
                        replay_count += 1
                static = load_json((folder/case/("test_"+str(seed)+".json")).read_bytes())
                saved = load_json((output/"static_forecasts.json").read_bytes())
                known = [i for i, r in enumerate(static) if r["risk"]["feasible_label"] is not None]
                labels = [int(static[i]["risk"]["feasible_label"]) for i in known]
                p = np.asarray(saved["raw"]["probabilities"])
                q = np.asarray(saved["transformed"])
                metrics.append({
                    "case": case, "method": method, "seed": seed,
                    "raw": binary_metrics(p[known, 0].tolist(), labels),
                    "transformed": binary_metrics(q[known, 0].tolist(), labels),
                    "used": binary_metrics((p if method == "adaptive_no_calibration" else q)[known, 0].tolist(), labels),
                    "calibration_status": saved["calibration"]["status"]})
    result = {"status": "completed" if all(r["status"] != "error" for r in results) else "errors_present",
              "additional_real_mda_calls": 0, "trials": results, "aggregate": aggregate,
              "static_metrics": metrics, "selected_risk_forecasts_replayed": replay_count,
              "controlled_online_calls": sum(r.get("evaluations", 0) for r in results),
              "scope": plan["scope"], "source_verified": True}
    save_new(folder/"analysis.json", result)
    print("Analysis", result["status"], "calls", result["controlled_online_calls"],
          "replayed", replay_count, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "analyze"))
    parser.add_argument("--parent", type=Path,
                        default=ROOT/"runs/system_validation_v5_20261020T192312Z_5f6531eb")
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    if args.action == "prepare":
        prepare(args.parent.resolve(), ROOT/"runs")
    else:
        if args.folder is None:
            parser.error("--folder is required")
        torch.set_num_threads(1)
        (execute if args.action == "execute" else analyze)(args.folder.resolve())
