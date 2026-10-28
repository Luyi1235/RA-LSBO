"""Twenty paired seeds, six initial + three times six online = 480 cheap labels."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import BudgetPolicy, RiskLedger, digest
from ra_lsbo.controlled_sparse import SparseInteriorFiveModeProblem
from ra_lsbo.discovery_selection import select_feasibility_first
from ra_lsbo.features import history_identity
from ra_lsbo.mechanism_system import SystemSettings
from ra_lsbo.optimizer import append_json, dispatch
from ra_lsbo.posterior_risk import CompletionPosterior, PosteriorSettings
from ra_lsbo.posterior_selection import posterior_pool, select_posterior
from ra_lsbo.research_statistics import holm, ordered_endpoint, paired_effect, run_metrics
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.structured_models import StructuredResponseHeads
from ra_lsbo.system_experiment import OriginalLSBOWarmStart, save_new
from tools.run_posterior_development import lines, restore_responses, save_responses

METHODS = ("posterior_full", "feasibility_first", "lsbo")
SEEDS = tuple(range(2001, 2021))
SETTINGS = SystemSettings(calls=6, failure_cap=6, calibration_window=1,
                          encoder_epochs=120, incremental_epochs=20, gp_steps=25,
                          pool_size=128, update_every=4)
RISK = PosteriorSettings(steps=80, inducing=40)
BASE = (ROOT/"runs/compact_mda_20261021T081006Z_20eb608d/"
        "qualification_recovered_20261025/frozen_base")


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=False)
    problem = SparseInteriorFiveModeProblem(BASE)
    candidates = {}
    for seed in SEEDS:
        rng, accepted, seen = np.random.default_rng(seed), [], set()
        for _ in range(10000):
            candidate = problem.canonical(problem.space.sample(rng).to_dict())
            key = digest(candidate)
            if key in seen or any(v > 0 for v in problem.precheck(candidate).values()):
                continue
            accepted.append(candidate)
            seen.add(key)
            if len(accepted) == 6:
                break
        if len(accepted) != 6:
            raise ValueError("Initial uniform candidate sampling failed")
        candidates[str(seed)] = accepted
    paths = [*sorted((ROOT/"ra_lsbo").glob("*.py")), Path(__file__).resolve(),
             ROOT/"tools/run_posterior_development.py"]
    paths += [p for p in BASE.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    save_new(folder/"plan.json", {
        "schema": "discovery_policy_diagnostic_v1", "date": "2026-10-25",
        "scope": "new seed development on an already-inspected controlled equation; not engineering validation",
        "maximum_new_truth_labels": 480, "maximum_new_real_mda_calls": 0,
        "initial_count": 6, "online_calls_per_run": 6,
        "seeds": list(SEEDS), "methods": list(METHODS),
        "primary_endpoint": "mean min(first-feasible online call,6), paired seeds with zero initial feasible",
        "secondary_endpoints": ["online_feasible_count", "hidden_failure_count", "endpoint_order"],
        "direction_primary": "smaller is better; no-event runs retained with administrative cost6",
        "primary_comparison": "feasibility_first versus posterior_full",
        "secondary_family": "six tests: two references times three endpoints; Holm correction",
        "no_hyperparameter_tuning_or_seed_extension": True,
        "settings": asdict(SETTINGS), "risk_settings": asdict(RISK),
        "base_project": str(BASE), "contract_id": problem.contract_id,
        "initial_candidates": candidates,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    print(folder/"plan.json", flush=True)


def validate(folder):
    plan = load_json((folder/"plan.json").read_bytes())
    if (plan["schema"] != "discovery_policy_diagnostic_v1"
            or plan["maximum_new_truth_labels"] != 480 or plan["maximum_new_real_mda_calls"] != 0
            or plan["seeds"] != list(SEEDS) or plan["methods"] != list(METHODS)
            or plan["settings"] != asdict(SETTINGS) or plan["risk_settings"] != asdict(RISK)):
        raise ValueError("Diagnostic scope differs")
    verify_sources(plan["source_sha256"])
    return plan


def selector(method):
    return select_feasibility_first if method == "feasibility_first" else select_posterior


def fit(rows, problem, seed, features):
    if features is None:
        features = StructuredFeatures(BASE, [asdict(v) for v in problem.space.variables],
                                      seed, SETTINGS.encoder()).fit(rows)
    else:
        features.update(rows)
    return (features, CompletionPosterior(rows, features, seed, RISK),
            StructuredResponseHeads(rows, features, seed, SETTINGS.gp_steps))


def execute(folder):
    plan = validate(folder)
    save_new(folder/"execution_started.json", {"utc": datetime.now(timezone.utc).isoformat()})
    problem = SparseInteriorFiveModeProblem(BASE)
    if problem.contract_id != plan["contract_id"]:
        raise ValueError("Controlled truth contract changed")
    calls = 0
    for seed in SEEDS:
        initial_folder = folder/("initial_"+str(seed))
        initial_folder.mkdir()
        initial = []
        with RiskLedger(initial_folder/"budget.jsonl", BudgetPolicy(6, 6., 6, 0, 0)) as ledger:
            for candidate in plan["initial_candidates"][str(seed)]:
                risk = dispatch(problem, ledger, {"candidate": candidate, "mode": "bootstrap",
                                "evidence": {"basis": "unavailable"}, "source": "shared_uniform_initial"})
                if risk is None:
                    raise ValueError("Initial candidate rejected")
                calls += 1
                initial.append({"evaluation": len(initial)+1, "candidate": candidate, "risk": risk,
                                "evaluator_contract_sha256": problem.contract_id})
        save_new(initial_folder/"rows.json", initial)
        for method in METHODS:
            validate(folder)
            output = folder/(method+"_"+str(seed))
            output.mkdir()
            rows, seen = list(initial), {digest(r["candidate"]) for r in initial}
            rng, features, risk_model, responses = np.random.default_rng(seed+200000), None, None, None
            engine = OriginalLSBOWarmStart(problem, BASE, rows, seed, SETTINGS, output) if method == "lsbo" else None
            if engine is not None:
                save_new(output/"original_settings.json", engine.settings)
            save_new(output/"manifest.json", {"seed": seed, "method": method,
                                             "initial_history": history_identity(initial)})
            with RiskLedger(output/"budget.jsonl",
                            BudgetPolicy(6, 6., 0, 0, 0, allow_baseline=True)) as ledger:
                for iteration in range(6):
                    if calls >= 480:
                        raise RuntimeError("Whole diagnostic truth budget reached")
                    fit_number = iteration//SETTINGS.update_every
                    if engine is not None:
                        proposal, metadata = engine.propose()
                        pool = []
                    else:
                        if iteration % SETTINGS.update_every == 0:
                            features, risk_model, responses = fit(rows, problem, seed, features)
                            features.save_checkpoint(output/("encoder_"+str(fit_number)+".pt"))
                            risk_model.save_checkpoint(output/("risk_"+str(fit_number)+".pt"))
                            save_responses(output/("responses_"+str(fit_number)+".pt"), responses)
                            append_json(output/"fits.jsonl", {
                                "fit_number": fit_number, "before_evaluation": len(rows)+1,
                                "history_id": history_identity(rows)})
                        pool, _ = posterior_pool(problem, rng, SETTINGS.pool_size, seen, rows)
                        proposal, metadata = selector(method)(pool, rows, responses, risk_model,
                                                             problem.analytical_constraints,
                                                             seed*10000+iteration)
                    append_json(output/"proposals.jsonl", {
                        "evaluation": len(rows)+1, "fit_number": fit_number, "pool": pool,
                        "proposal": proposal, "selection": metadata, "history_id": history_identity(rows)})
                    risk = dispatch(problem, ledger, proposal)
                    if risk is None or risk["feasible_label"] is None:
                        raise RuntimeError("Unknown or rejected controlled outcome")
                    calls += 1
                    row = {"evaluation": len(rows)+1, "candidate": proposal["candidate"], "risk": risk,
                           "evaluator_contract_sha256": problem.contract_id}
                    append_json(output/"observations.jsonl", row)
                    rows.append(row)
                    seen.add(digest(row["candidate"]))
                    if engine is not None:
                        engine.accept(row, metadata)
            metrics = run_metrics(initial, rows[6:], 6)
            metrics.pop("trajectory")
            summary = {"seed": seed, "method": method, **metrics}
            save_new(output/"summary.json", summary)
            append_json(folder/"completed.jsonl", summary)
            print(seed, method, metrics["initial_feasible"], metrics["online_feasible_count"],
                  metrics["first_feasible_online_call"], flush=True)
    save_new(folder/"execution_result.json", {"status": "complete", "new_truth_calls": calls,
                                             "new_real_mda_calls": 0})


def audit(folder):
    plan = validate(folder)
    truth_checks, acquisition_checks, native_checks, total = 0, 0, 0, 0
    summaries = lines(folder/"completed.jsonl")
    if len(summaries) != 60:
        raise ValueError("All planned seeds and methods must be retained")
    for seed in SEEDS:
        initial = load_json((folder/("initial_"+str(seed))/"rows.json").read_bytes())
        if [r["candidate"] for r in initial] != plan["initial_candidates"][str(seed)]:
            raise ValueError("Shared initial candidates differ")
        for method in METHODS:
            problem = SparseInteriorFiveModeProblem(BASE)
            output = folder/(method+"_"+str(seed))
            observations, proposals = lines(output/"observations.jsonl"), lines(output/"proposals.jsonl")
            if len(observations) != 6 or len(proposals) != 6:
                raise ValueError("Missing planned outcomes")
            manifest = load_json((output/"manifest.json").read_bytes())
            if manifest["initial_history"] != history_identity(initial):
                raise ValueError("Initial histories differ")
            for row, proposal in zip(observations, proposals):
                if (row["candidate"] != proposal["proposal"]["candidate"]
                        or history_identity(initial+[r for r in observations if r["evaluation"] < row["evaluation"]])
                        != proposal["history_id"]
                        or row["risk"] != problem.evaluate(row["candidate"])):
                    raise ValueError("Truth, selected point or chronological history mismatch")
                truth_checks += 1
            if method == "lsbo":
                replay = output/"native_replay"
                replay.mkdir(exist_ok=False)
                engine = OriginalLSBOWarmStart(problem, BASE, initial, seed, SETTINGS, replay)
                for row, recorded in zip(observations, proposals):
                    proposal, metadata = engine.propose()
                    if proposal != recorded["proposal"] or metadata != recorded["selection"]:
                        raise ValueError("Native LSBO replay mismatch")
                    engine.accept(row, metadata)
                    native_checks += 1
                continue
            for fitted in lines(output/"fits.jsonl"):
                number = fitted["fit_number"]
                past = initial+[r for r in observations if r["evaluation"] < fitted["before_evaluation"]]
                if history_identity(past) != fitted["history_id"]:
                    raise ValueError("Future observations in fit")
                features = StructuredFeatures.load_checkpoint(output/("encoder_"+str(number)+".pt"), BASE, past)
                risk = CompletionPosterior.load_checkpoint(output/("risk_"+str(number)+".pt"), features, past)
                responses = restore_responses(output/("responses_"+str(number)+".pt"), past, features, seed)
                for recorded in (p for p in proposals if p["fit_number"] == number):
                    current = initial+[r for r in observations if r["evaluation"] < recorded["evaluation"]]
                    proposal, metadata = selector(method)(
                        recorded["pool"], current, responses, risk, problem.analytical_constraints,
                        seed*10000+recorded["evaluation"]-7)
                    if proposal != recorded["proposal"] or metadata != recorded["selection"]:
                        raise ValueError("Acquisition checkpoint replay mismatch")
                    acquisition_checks += 1
        for path in [folder/("initial_"+str(seed))/"budget.jsonl",
                     *(folder/(method+"_"+str(seed))/"budget.jsonl" for method in METHODS)]:
            ledger = RiskLedger(path).summary()
            if ledger["charged_calls"] != 6 or ledger["unresolved_started_calls"]:
                raise ValueError("Incomplete budget ledger")
            total += ledger["charged_calls"]
    lookup = {(r["seed"], r["method"]): r for r in summaries}
    zero = [seed for seed in SEEDS if lookup[(seed, METHODS[0])]["initial_feasible"] == 0]
    primary_differences = [lookup[(s, "feasibility_first")]["restricted_first_feasible_cost"]
                           -lookup[(s, "posterior_full")]["restricted_first_feasible_cost"] for s in zero]
    primary = paired_effect(primary_differences) if zero else None
    secondary = []
    for reference in ("posterior_full", "lsbo"):
        for metric in ("online_feasible_count", "hidden_failure_count", "endpoint_order"):
            pairs = [(lookup[(s, "feasibility_first")], lookup[(s, reference)]) for s in SEEDS]
            differences = ([ordered_endpoint(a["endpoint_best"], b["endpoint_best"]) for a, b in pairs]
                           if metric == "endpoint_order" else [a[metric]-b[metric] for a, b in pairs])
            secondary.append({"reference": reference, "metric": metric, **paired_effect(differences)})
    for result, adjusted in zip(secondary, holm([r["exact_two_sided_sign_flip_p"] for r in secondary])):
        result["holm_adjusted_p"] = adjusted
    save_new(folder/"audit.json", {
        "status": "passed", "new_truth_labels": total, "new_real_mda_calls": 0,
        "online_truth_checks": truth_checks, "full_acquisition_checks": acquisition_checks,
        "native_lsbo_proposal_checks": native_checks, "zero_initial_seeds": zero,
        "primary_paired_difference_feasibility_first_minus_posterior": primary,
        "secondary_comparisons": secondary,
        "groups": [{"method": m, "n": 20,
                    "online_feasible": sum(r["online_feasible_count"] for r in summaries if r["method"] == m),
                    "hidden_failures": sum(r["hidden_failure_count"] for r in summaries if r["method"] == m),
                    "discovery_events_from_zero": sum(lookup[(s, m)]["first_feasible_event"] for s in zero),
                    "restricted_mean_discovery_calls_from_zero": float(np.mean(
                        [lookup[(s, m)]["restricted_first_feasible_cost"] for s in zero])) if zero else None}
                   for m in METHODS],
        "scope": "predeclared diagnostic on known equations; not an independent engineering task",
        "plan_sha256": hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest()})
    print("AUDIT PASSED", total, truth_checks, acquisition_checks, native_checks, flush=True)


if __name__ == "__main__":
    import os
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "audit"))
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "execute": execute, "audit": audit}[args.action](args.folder.resolve())
