"""Small-history paired development: 120 shared labels + <=360 online labels."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import traceback
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import BudgetPolicy, RiskLedger, digest
from ra_lsbo.controlled_sparse import SparseInteriorFiveModeProblem
from ra_lsbo.features import history_identity
from ra_lsbo.mechanism_system import FiveModeProblem, SystemSettings
from ra_lsbo.optimizer import append_json, dispatch
from ra_lsbo.posterior_risk import CompletionPosterior, PosteriorSettings
from ra_lsbo.posterior_selection import posterior_pool, select_posterior
from ra_lsbo.selection import incumbent
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.structured_models import StructuredResponseHeads
from ra_lsbo.system_experiment import OriginalLSBOWarmStart, sample_rows, save_new


BASE = ROOT.parent/"LSBO/X57-LSBO-refactored"
METHODS = ("posterior_full", "posterior_no_completion", "posterior_onehot_risk", "lsbo", "random")
CASES, SEEDS = ("five_mode", "sparse_interior"), (81, 82, 83)
SETTINGS = SystemSettings(calls=12, failure_cap=12, calibration_window=1,
                          encoder_epochs=120, incremental_epochs=20, gp_steps=25,
                          pool_size=128, update_every=4)
RISK_SETTINGS = PosteriorSettings(steps=80, inducing=40)


def problem_for(case):
    return (FiveModeProblem if case == "five_mode" else SparseInteriorFiveModeProblem)(BASE)


def lines(path):
    return [load_json(line) for line in path.read_bytes().splitlines()] if path.exists() else []


def prepare():
    folder = ROOT/"runs"/("posterior_development_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                          +"_"+uuid.uuid4().hex[:8])
    folder.mkdir(exist_ok=False)
    paths = [*sorted((ROOT/"ra_lsbo").glob("*.py")), Path(__file__).resolve()]
    save_new(folder/"plan.json", {
        "schema": "posterior_development_v1", "additional_real_mda_calls": 0,
        "cases": list(CASES), "seeds": list(SEEDS), "methods": list(METHODS),
        "initial_count": 20, "max_online_calls": 360, "max_new_truth_labels": 480,
        "settings": asdict(SETTINGS), "risk_settings": asdict(RISK_SETTINGS),
        "contracts": {case: problem_for(case).contract_id for case in CASES},
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "scope": "Small-history development on known equations; no selection of feasible initial histories",
        "representation_ablation_scope": "Risk view only; response GP retains structured inputs"})
    for case in CASES:
        (folder/case).mkdir()
        for seed in SEEDS:
            save_new(folder/case/("initial_"+str(seed)+".json"),
                     sample_rows(problem_for(case), 20, seed))
    print(folder, flush=True)


def validate(folder):
    plan = load_json((folder/"plan.json").read_bytes())
    if (plan["schema"] != "posterior_development_v1" or plan["settings"] != asdict(SETTINGS)
            or plan["risk_settings"] != asdict(RISK_SETTINGS)
            or plan["methods"] != list(METHODS) or plan["seeds"] != list(SEEDS)
            or plan["max_new_truth_labels"] != 480):
        raise ValueError("Development scope differs")
    verify_sources(plan["source_sha256"])
    return plan


def fit_bundle(rows, problem, method, seed, features=None):
    if features is None:
        features = StructuredFeatures(BASE, [asdict(v) for v in problem.space.variables],
                                      seed, SETTINGS.encoder()).fit(rows)
    else:
        features.update(rows)
    view = "hierarchical_onehot" if method == "posterior_onehot_risk" else "paper_gp"
    risk = CompletionPosterior(rows, features, seed,
                               PosteriorSettings(**dict(asdict(RISK_SETTINGS), view=view)))
    responses = StructuredResponseHeads(rows, features, seed, SETTINGS.gp_steps)
    return features, risk, responses


def save_responses(path, responses):
    with path.open("xb") as stream:
        torch.save({name: {"state": gp.state_dict(), "normalization": [center, scale, sign]}
                    for name, (gp, center, scale, sign) in responses.models.items()}, stream)


def restore_responses(path, rows, features, seed):
    responses = StructuredResponseHeads(rows, features, seed, steps=1)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if set(saved) != set(responses.models):
        raise ValueError("Response checkpoint model names differ")
    for name, item in saved.items():
        gp, center, scale, sign = responses.models[name]
        if item["normalization"] != [center, scale, sign]:
            raise ValueError("Response checkpoint normalization differs")
        gp.train()
        gp.load_state_dict(item["state"], strict=True)
        gp.eval()
    return responses


def run_trial(folder, case, seed, method):
    validate(folder)
    problem = problem_for(case)
    output = folder/case/(method+"_"+str(seed))
    output.mkdir(exist_ok=False)
    initial = load_json((folder/case/("initial_"+str(seed)+".json")).read_bytes())
    rows, seen = list(initial), {digest(r["candidate"]) for r in initial}
    rng = np.random.default_rng(seed+200000)
    engine = (OriginalLSBOWarmStart(problem, BASE, rows, seed, SETTINGS, output)
              if method == "lsbo" else None)
    features = risk = responses = None
    save_new(output/"manifest.json", {"case": case, "method": method, "seed": seed,
                                     "initial_history": history_identity(initial),
                                     "initial_feasible": sum(r["risk"]["feasible_label"] is True for r in initial)})
    status, error, fit_number = "complete", None, -1
    with RiskLedger(output/"budget.jsonl", BudgetPolicy(12, 12., 0, 0, 0, allow_baseline=True)) as ledger:
        try:
            for iteration in range(SETTINGS.calls):
                if method.startswith("posterior") and iteration % SETTINGS.update_every == 0:
                    fit_number += 1
                    features, risk, responses = fit_bundle(rows, problem, method, seed, features)
                    features.save_checkpoint(output/("encoder_"+str(fit_number)+".pt"))
                    risk.save_checkpoint(output/("risk_"+str(fit_number)+".pt"))
                    save_responses(output/("responses_"+str(fit_number)+".pt"), responses)
                    append_json(output/"fits.jsonl", {
                        "fit_number": fit_number, "before_evaluation": len(rows)+1,
                        "training_indices": [r["evaluation"] for r in rows],
                        "history_id": history_identity(rows), "model_id": risk.model_id,
                        "diagnostics": risk.diagnostics})
                pool = []
                if engine is not None:
                    proposal, selection = engine.propose()
                else:
                    pool_rows = [] if method == "random" else rows
                    pool, _ = posterior_pool(problem, rng, SETTINGS.pool_size, seen, pool_rows)
                    if not pool:
                        raise RuntimeError("Empty candidate pool")
                    if method == "random":
                        proposal = dict(pool[0], mode="baseline", evidence={"basis": "unavailable"})
                        selection = {"reason": "random"}
                    else:
                        proposal, selection = select_posterior(
                            pool, rows, responses, risk, problem.analytical_constraints,
                            seed*10000+iteration, method != "posterior_no_completion")
                append_json(output/"proposals.jsonl", {
                    "evaluation": len(rows)+1, "fit_number": fit_number, "pool": pool,
                    "proposal": proposal, "selection": selection, "history_id": history_identity(rows)})
                result = dispatch(problem, ledger, proposal)
                if result is None:
                    raise RuntimeError("Final precheck rejected")
                row = {"evaluation": len(rows)+1, "candidate": proposal["candidate"], "risk": result,
                       "evaluator_contract_sha256": problem.contract_id}
                append_json(output/"observations.jsonl", row)
                rows.append(row)
                seen.add(digest(row["candidate"]))
                if engine is not None:
                    engine.accept(row, selection)
        except Exception as exception:
            status, error = "error", str(exception)
            (output/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
        accounting = ledger.summary()
    new = rows[len(initial):]
    initial_best, best = incumbent(initial), incumbent(rows)
    first = next((i+1 for i, r in enumerate(new) if r["risk"]["feasible_label"] is True), None)
    summary = {
        "case": case, "seed": seed, "method": method, "status": status, "error": error,
        "new_calls": len(new), "new_feasible": sum(r["risk"]["feasible_label"] is True for r in new),
        "initial_feasible": sum(r["risk"]["feasible_label"] is True for r in initial),
        "initial_best": initial_best, "best": best, "first_new_feasible_call": first,
        "discovered_from_no_feasible": initial_best is None and best is not None,
        "improvement_percent": None if initial_best is None else 100*(initial_best-best)/abs(initial_best),
        "ledger": accounting}
    save_new(output/"summary.json", summary)
    return summary


def audit(folder):
    plan = validate(folder)
    results, replayed, outcomes = [], 0, 0
    for case in CASES:
        problem = problem_for(case)
        if problem.contract_id != plan["contracts"][case]:
            raise ValueError("Controlled truth changed")
        for seed in SEEDS:
            initial = load_json((folder/case/("initial_"+str(seed)+".json")).read_bytes())
            for method in METHODS:
                output = folder/case/(method+"_"+str(seed))
                manifest = load_json((output/"manifest.json").read_bytes())
                if manifest["initial_history"] != history_identity(initial):
                    raise ValueError("Initial histories differ")
                results.append(load_json((output/"summary.json").read_bytes()))
                proposals, observations = lines(output/"proposals.jsonl"), lines(output/"observations.jsonl")
                if len(proposals) != len(observations):
                    raise ValueError("Unsettled proposal")
                ledger = RiskLedger(output/"budget.jsonl").summary()
                if ledger["charged_calls"] != len(observations) or ledger["unresolved_started_calls"]:
                    raise ValueError("Ledger differs")
                for row, p in zip(observations, proposals):
                    if row["candidate"] != p["proposal"]["candidate"] or row["risk"] != problem.evaluate(row["candidate"]):
                        raise ValueError("Truth replay differs")
                    outcomes += 1
                if not method.startswith("posterior"):
                    continue
                for fit in lines(output/"fits.jsonl"):
                    n = fit["fit_number"]
                    past = initial+[r for r in observations if r["evaluation"] < fit["before_evaluation"]]
                    if (history_identity(past) != fit["history_id"]
                            or [r["evaluation"] for r in past] != fit["training_indices"]):
                        raise ValueError("Training data leakage")
                    features = StructuredFeatures.load_checkpoint(output/("encoder_"+str(n)+".pt"), BASE, past)
                    risk = CompletionPosterior.load_checkpoint(output/("risk_"+str(n)+".pt"), features, past)
                    responses = restore_responses(output/("responses_"+str(n)+".pt"), past, features, seed)
                    for p in (p for p in proposals if p["fit_number"] == n):
                        iteration = p["evaluation"]-len(initial)-1
                        current = initial+[r for r in observations if r["evaluation"] < p["evaluation"]]
                        proposal, metadata = select_posterior(
                            p["pool"], current, responses, risk, problem.analytical_constraints,
                            seed*10000+iteration, method != "posterior_no_completion")
                        if proposal["candidate"] != p["proposal"]["candidate"]:
                            raise ValueError("Full acquisition replay selected a different design")
                        for key in ("objective_sample", "sampled_violation", "raw_completion",
                                    "independent_head_feasibility_raw"):
                            if key in metadata:
                                np.testing.assert_allclose(metadata[key], p["selection"][key], rtol=2e-6, atol=1e-8)
                        replayed += 1
    aggregate = []
    for case in CASES:
        for method in METHODS:
            group = [r for r in results if r["case"] == case and r["method"] == method]
            aggregate.append({
                "case": case, "method": method, "calls": sum(r["new_calls"] for r in group),
                "feasible": sum(r["new_feasible"] for r in group),
                "initial_zero_feasible_runs": sum(r["initial_best"] is None for r in group),
                "discoveries_from_zero": sum(r["discovered_from_no_feasible"] for r in group),
                "mean_best": None if any(r["best"] is None for r in group)
                             else float(np.mean([r["best"] for r in group])),
                "statuses": dict(Counter(r["status"] for r in group))})
    save_new(folder/"audit.json", {
        "status": "passed", "truth_replays": outcomes, "full_acquisition_replays": replayed,
        "shared_initial_truth_labels": 120, "additional_real_mda_calls": 0,
        "total_new_controlled_truth_labels": outcomes+120, "aggregate": aggregate,
        "limitations": "Known equations, three seeds, 12 online calls; development only"})
    print("AUDIT PASSED", outcomes, replayed, flush=True)


def execute(folder):
    validate(folder)
    save_new(folder/"execution_started.json", {"utc": datetime.now(timezone.utc).isoformat()})
    for case in CASES:
        for seed in SEEDS:
            for method in METHODS:
                result = run_trial(folder, case, seed, method)
                append_json(folder/"completed.jsonl", result)
                print(case, seed, method, result["status"], result["new_feasible"],
                      result["best"], flush=True)
    audit(folder)


if __name__ == "__main__":
    import os
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "audit"))
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.folder is None:
        parser.error("--folder required")
    elif args.action == "execute":
        execute(args.folder.resolve())
    else:
        audit(args.folder.resolve())
