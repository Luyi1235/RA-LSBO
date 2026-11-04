"""PR54: prospective ON-POLICY risk-budget experiment on analytical mixed tasks.

Unlike PR45/PR52/PR53 retrospective logs, EVERY method genuinely selects
its next candidate and receives the mathematical evaluator's result.
This is NOT a flight/aircraft simulation or RCAIDE MDA evaluation.

All methods see the SAME 12 shared random starts and deterministic candidate
pool draws. The same hard cap of SIX observed online nonfeasible evaluations
applies to all three arms; only acquisition re-ranking differs. The predicted
budget policy is NOT an ex-ante or statistical failure guarantee.
"""
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.problems import candidate_pool, SimplifiedElectricAircraft
from ra_lsbo.experiments.core.problems import MathematicalCase, AircraftCase
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.core.runner import plain
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import StableCorpusDiscoveryBO

from .risk_budget54_engine import BudgetedDualViewMathBO, settings_for_math54
from .risk_budget54_policy import observed_failures, SCHEMA as POLICY_SCHEMA

SCHEMA = "ra_lsbo_pr54_analytic_onpolicy_budget_trial_v1"
CASES = ("branin_ordinary","branin_sparse")
ARMS = ("legacy","semantic_only","budgeted")
SEEDS = tuple(range(995401,995409))
INITIAL = 12
HORIZON = 12
FAILURE_CAP = 6
POOL = 128
SHARED_DRAW = 194000000
ONLINE_DRAW = 195000000


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f:
        json.dump(plain(obj),f,sort_keys=True,indent=2,allow_nan=False)


def _append(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf8") as f:
        f.write(json.dumps(plain(obj),sort_keys=True,allow_nan=False)+"\n")
        f.flush()
        os.fsync(f.fileno())


def clean_commit():
    root=Path(__file__).resolve().parents[2]
    commands=[
        ["git","rev-parse","--show-toplevel"],
        ["git","rev-parse","HEAD"],
        ["git","status","--porcelain","--untracked-files=no"],
    ]
    replies=[]
    for cmd in commands:
        result=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
        if result.returncode!=0:
            raise RuntimeError("PR54 requires a tracked clean Git worktree")
        replies.append(result.stdout.strip())
    if Path(replies[0]).resolve()!=root or replies[2]:
        raise ValueError("Use one clean checkout at a frozen PR54 commit")
    return replies[1]


def frozen_registration(profile, source_sha):
    if profile not in ("smoke","full") or not source_sha:
        raise ValueError("Only frozen smoke and full analytical study profiles")
    seeds = list(SEEDS[:1] if profile=="smoke" else SEEDS)
    steps = 2 if profile=="smoke" else HORIZON
    return {
        "schema":SCHEMA,"source_commit":source_sha,
        "profile":profile,"cases":list(CASES),"arms":list(ARMS),
        "seeds":seeds,"initial_shared":INITIAL,"online_horizon":HORIZON,
        "online_steps_to_run":steps,"observed_online_failure_cap":FAILURE_CAP,
        "pool_per_step":POOL,
        "objective_view":"frozen_2000_structure_VAE_latent",
        "risk_view":"onehot_constraints",
        "pr45_status_completion_removed_from_new_policy_ranking":True,
        "hard_observed_online_failure_stop_shared_by_all_arms":True,
        "future_information_available_to_optimizer":False,
        "same_pool_draw_stream_across_arms":True,
        "analytical_problems_only":True,
        "model_kernel_grid":"4x2 frozen stable GP/EP",
        "new_reduced_aircraft_calls":0,"new_rcaide_calls":0,
        "max_new_analytical_calls":len(CASES)*len(seeds)*(INITIAL+len(ARMS)*steps),
        "prespecified_results":"descriptive paired energy, feasibility discovery, failures, charged online calls",
        "science_status":"exploratory on-policy methodology verification, NOT confirmed aircraft superiority",
    }


def _problem(case):
    family,difficulty=case.split("_")
    if family!="branin" or difficulty not in ("ordinary","sparse"):
        raise ValueError("Forbidden nonanalytical PR54 task")
    return MathematicalCase(BASE,family,difficulty)


def _initial(case,seed,problem):
    case_index=CASES.index(case)
    designs,_=candidate_pool(problem,SHARED_DRAW+1000000*case_index+seed,
                             INITIAL)
    if len(designs)!=INITIAL:
        raise AssertionError("Incomplete first 12 designs")
    rows=[problem.evaluate(x) for x in designs]
    for row in rows:
        if (row.observation.status!="completed" or
                row.physical_feasible is None):
            raise RuntimeError("Unknown/non-diagnostic analytical initial outcome")
    return rows


def _pool(case,seed,iteration,problem,history):
    # Raw stream is independent of method and outcomes; a method's already
    # observed design IDs are removed, never replaced by truth-screening.
    case_index=CASES.index(case)
    draw=ONLINE_DRAW+case_index*10000000+seed*100+iteration
    raw,_=candidate_pool(problem,draw,POOL+HORIZON)
    seen={digest(r.candidate) for r in history}
    candidates=[x for x in raw if digest(x) not in seen][:POOL]
    if len(candidates)!=POOL:
        raise RuntimeError("Unseen analytically-screened pool exhausted")
    return candidates


def _engine(problem,seed,arm,path):
    if arm=="legacy":
        return StableCorpusDiscoveryBO(problem,BASE,seed,settings_for_math54(),
                                       path,allocation="onehot_risk")
    return BudgetedDualViewMathBO(problem,BASE,seed,path,policy=arm,
                                  initial_count=INITIAL,failure_cap=FAILURE_CAP)


def _status(history):
    feasible=[row.observation.objective.value for row in history
              if row.usable_feasible]
    return min(feasible) if feasible else None


def _trial(case,seed,arm,initial,folder,steps):
    problem=_problem(case)
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    engine=_engine(problem,seed,arm,folder)
    history=list(initial)
    first_feasible_online=None
    stops=None
    trajectory=[_status(history)]
    meta_modes=Counter()
    selected_risk=[]
    failures=[]
    tic=time.perf_counter()
    for iteration in range(steps):
        spent=observed_failures(history,INITIAL,iteration)
        if spent>=FAILURE_CAP:
            stops="observed_online_failure_cap"
            break
        candidates=_pool(case,seed,iteration,problem,history)
        prefix=digest([r.to_dict() for r in history])
        candidate,meta=engine.propose(history,candidates,iteration)
        if (digest([r.to_dict() for r in history])!=prefix or
                candidate!=problem.canonical(candidate) or
                candidate not in candidates):
            raise RuntimeError("Optimizer altered history or selected outside pool")
        _append(folder/"proposals.jsonl", {
            "iteration":iteration,"candidate":candidate,"candidate_id":digest(candidate),
            "history_prefix_digest":prefix,"pool_candidate_ids_sha256":digest([digest(x) for x in candidates]),
            "metadata":meta})
        # A *new*, actual analytical call. No lookahead or stored table truth.
        record=problem.evaluate(candidate)
        if (record.candidate!=candidate or record.observation.status!="completed" or
                record.physical_feasible is None):
            raise RuntimeError("Untrusted analytical simulator return")
        _append(folder/"observations.jsonl", {
            "iteration":iteration,"record":record.to_dict(),
            "evidence_sha256":digest(record.to_dict())})
        engine.accept(record)
        history.append(record)
        if first_feasible_online is None and record.usable_feasible:
            first_feasible_online=iteration
        failures.append(int(not record.usable_feasible))
        mode=meta.get("mode","unknown")
        meta_modes[mode]+=1
        decision=meta.get("risk_budget")
        selected_risk.append({
            "iteration":iteration,"policy":arm,"mode":mode,
            "predicted_nonfeasible_proxy":(
                decision["selected_predicted_failure"] if decision else None),
            "forecast_available":decision is not None,
            "realized_nonfeasible":int(not record.usable_feasible),
            "budget_gate_passed":(
                decision["candidate_risk_within_allowance"] if decision else None),
            "budget_gate_fallback":(
                decision["selection_rule"] if decision else None),
            "observation_not_available_at_selection_time":True})
        trajectory.append(_status(history))
    if stops is None:
        stops="smoke_two_steps_only" if steps<HORIZON else "horizon_or_last_budget_evaluation"
    spent=observed_failures(history,INITIAL,len(history)-INITIAL)
    if spent>FAILURE_CAP:
        raise AssertionError("Observed online failure cap violated")
    result={
        "case":case,"seed":seed,"arm":arm,"source":"ON_POLICY_NEW_ANALYTICAL_TRUTH",
        "initial_records":INITIAL,
        "online_calls":len(history)-INITIAL,"online_nonfeasible":spent,
        "hard_online_failure_cap":FAILURE_CAP,
        "stopped_reason":stops,"initial_best":_status(initial),
        "final_best":_status(history),
        "first_online_feasible_iteration":first_feasible_online,
        "best_feasible_after_each_actual_online_call":trajectory,
        "mode_counts":dict(meta_modes),
        "pre_evaluation_risk_receipts":selected_risk,
        "final_recorded_history_digest":digest([r.to_dict() for r in history]),
        "training_or_evaluation_failures_fabricated":False,
        "elapsed_seconds":time.perf_counter()-tic,
    }
    _write(folder/"result.json",result)
    return result


def _comparisons(results,steps):
    by_key={(r["case"],r["seed"],r["arm"]):r for r in results}
    summary={}
    for case in CASES:
        for candidate,reference in (
            ("budgeted","semantic_only"),("semantic_only","legacy"),
            ("budgeted","legacy")):
            pairs=[]
            for seed in sorted({x["seed"] for x in results if x["case"]==case}):
                a=by_key[(case,seed,candidate)]
                b=by_key[(case,seed,reference)]
                ea,eb=a["final_best"],b["final_best"]
                pairs.append({
                    "seed":seed,"candidate_arm":candidate,"control_arm":reference,
                    "energy_difference_candidate_minus_control":ea-eb if ea is not None and eb is not None else None,
                    "candidate_only_feasible":ea is not None and eb is None,
                    "control_only_feasible":ea is None and eb is not None,
                    "paired_feasible":ea is not None and eb is not None,
                    "online_nonfeasible_difference":a["online_nonfeasible"]-b["online_nonfeasible"],
                    "online_call_difference":a["online_calls"]-b["online_calls"],
                })
            both=[row["energy_difference_candidate_minus_control"] for row in pairs
                  if row["energy_difference_candidate_minus_control"] is not None]
            summary[f"{case}/{candidate}_vs_{reference}"]={
                "n_pairs":len(pairs),"both_final_feasible":len(both),
                "mean_energy_difference_conditional_on_both_feasible":sum(both)/len(both) if both else None,
                "candidate_only_feasible":sum(x["candidate_only_feasible"] for x in pairs),
                "control_only_feasible":sum(x["control_only_feasible"] for x in pairs),
                "mean_online_failure_count_difference":sum(x["online_nonfeasible_difference"] for x in pairs)/len(pairs),
                "mean_actual_online_call_count_difference":sum(x["online_call_difference"] for x in pairs)/len(pairs),
                "per_seed":pairs,
                "no_randomized_confirmatory_claim":True,
                "different_actual_online_call_counts_are_reported_not_hidden":True,
            }
    return summary


def run(output, *, profile="smoke"):
    """Write-once on-policy study. No user aircraft/RCAIDE or historical DB access."""
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("New PR54 study output path required")
    source_sha=clean_commit()
    registration=frozen_registration(profile,source_sha)
    output.mkdir(parents=True,exist_ok=False)
    _write(output/"plan.json",registration)
    _write(output/"source_receipt.json",{
        "code_sha":source_sha,"PR53_base_sha":"a7151a7bb44e2570eaec64f1ce3125024deadffc",
        "policy_source_sha256":_sha(Path(__file__).with_name("risk_budget54_policy.py")),
        "engine_source_sha256":_sha(Path(__file__).with_name("risk_budget54_engine.py")),
        "new_aircraft_or_rcaide_calls_prohibited":True,
    })
    results=[]
    calls=0
    def prohibited(*_args,**_kwargs):
        raise RuntimeError("PR54 mathematical experiment PROHIBITS aircraft evaluations")
    with ExitStack() as stack:
        stack.enter_context(patch.object(AircraftCase,"evaluate",prohibited))
        stack.enter_context(patch.object(SimplifiedElectricAircraft,"evaluate",prohibited))
        for case in CASES:
            problem=_problem(case)
            for seed in registration["seeds"]:
                original=_initial(case,seed,problem)
                calls+=len(original)
                _write(output/"shared_initial"/case/(str(seed)+".json"),{
                    "case":case,"seed":seed,"shared_between_all_arms":True,
                    "history_digest":digest([r.to_dict() for r in original]),
                    "records":[r.to_dict() for r in original]})
                for arm in ARMS:
                    result=_trial(case,seed,arm,original,
                                  output/"trials"/case/str(seed)/arm,
                                  registration["online_steps_to_run"])
                    calls+=result["online_calls"]
                    results.append(result)
                    _append(output/"progress.jsonl",{
                        "case":case,"seed":seed,"arm":arm,
                        "calls_charged_to_analytical_evaluator":result["online_calls"],
                        "stopped_reason":result["stopped_reason"],
                        "final_best":result["final_best"],
                        "observed_online_nonfeasible":result["online_nonfeasible"]})
                    print("PR54:",case,seed,arm,"calls",result["online_calls"],
                          "nonfeasible",result["online_nonfeasible"],
                          "best",result["final_best"],flush=True)
    if calls > registration["max_new_analytical_calls"]:
        raise RuntimeError("Analytical evaluation count exceeded frozen campaign bound")
    report={
        "schema":SCHEMA,
        "status":"COMPLETE_ON_POLICY_EXPLORATORY_MATH" if profile=="full" else "SMOKE_ONLY",
        "source_sha":source_sha,"registration":registration,
        "source_pr53_result_commit":"f022871cfe79b997ca808b68d4bdffae5c382890",
        "actual_new_analytical_evaluations":calls,
        "new_aircraft_evaluations":0,"new_rcaide_calls":0,
        "reused_original_PR45_physical_truth":False,
        "on_policy_actual_candidates_and_sequential_feedback":True,
        "synthetic_mathematical_cases_not_original_upstream_benchmarks":True,
        "hard_online_observed_failures_at_most_six":all(
            x["online_nonfeasible"]<=FAILURE_CAP for x in results),
        "per_arm_results":results,
        "paired_descriptive_comparisons":_comparisons(
            results,registration["online_steps_to_run"]),
        "limitations":[
            "This is newly sampled analytically evaluated hierarchical Branin, NOT reduced-aircraft/RCAIDE.",
            "The three arms share an observed six-failure stop; the budgeted arm additionally changes candidate ranking.",
            "Prediction-based per-call allowance is a heuristic, NOT an expected/observed failure guarantee.",
            "All controls share initial data and pool streams; subsequent histories and GP/VAE training diverge on-policy.",
            "Different call counts under hard stop are retained; no budget-free superiority claim.",
            "This study has 8 independent seeds/task in full, descriptive and NOT confirmatory.",
            "PR53 phase/pooled calibration is NOT inserted because PR45 online never had other-seed results.",
            "No independent sparse-domain qualification and no high-fidelity aircraft claim.",
        ],
    }
    _write(output/"risk_budget_math_report.json",report)
    return report
