"""PR55 independent-seed and transfer-family *new on-policy* mathematical trial.

Runs the exact PR54 three engines without altering their decision rule.
Only fresh analytical MathematicalCase evaluations are permitted. Never reads
PR45's SQLite or previously paid-for physical truth. Every arm trains from
its own sequentially settled observations, not a counterfactual label table.
"""
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.problems import candidate_pool, SimplifiedElectricAircraft
from ra_lsbo.experiments.core.problems import MathematicalCase, AircraftCase
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.core.runner import plain

from .risk_budget54_runner import (
    _engine, _status, _write, _append, _sha, clean_commit,
)
from .risk_budget54_policy import observed_failures, SCHEMA as ACQUISITION_SCHEMA
from .risk_budget55_protocol import (
    SCHEMA, CASES, ARMS, SEEDS, INITIAL, ONLINE, OBSERVED_FAILURE_CAP,
    POOL_SIZE, SEED_INITIAL_STREAM, SEED_ONLINE_STREAM,
    PARENT_PR54_SHA, registration, summarize,
)


def _problem(case):
    bits=case.split("_")
    if len(bits)!=2 or case not in CASES:
        raise ValueError("Only four declared mathematical screening tasks are authorized")
    return MathematicalCase(BASE,bits[0],bits[1])


def _initial(problem,case,seed):
    index=CASES.index(case)
    proposals,rejected = candidate_pool(
        problem,SEED_INITIAL_STREAM + 1000000*index + seed,INITIAL)
    if len(proposals)!=INITIAL or len({digest(c) for c in proposals})!=INITIAL:
        raise ValueError("Initial candidate stream has duplicates or missing points")
    result=[]
    for raw in proposals:
        rec=problem.evaluate(raw)
        if (rec.candidate!=raw or rec.observation.status!="completed" or
                rec.physical_feasible is None or rec.diagnostics.get("numerical_failure")):
            raise RuntimeError("Untrusted mathematical initial observation")
        result.append(rec)
    return result, rejected


def _pool(problem,case,seed,iteration,history):
    """Matched *unlabelled* draw stream; remove method-specific observed points."""
    index=CASES.index(case)
    raw,_ = candidate_pool(problem,
             SEED_ONLINE_STREAM+index*10000000+seed*100+iteration,
             POOL_SIZE+ONLINE)
    seen={digest(record.candidate) for record in history}
    novel=[]
    visited=set()
    for candidate in raw:
        key=digest(candidate)
        if key not in seen and key not in visited:
            novel.append(candidate)
            visited.add(key)
        if len(novel)==POOL_SIZE:
            break
    if len(novel)!=POOL_SIZE:
        raise RuntimeError("Insufficient distinct unseen analytical candidates")
    return novel


def _trial(case,seed,arm,initial,output,steps):
    """All online analytic observations are made AFTER that arm's proposal."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    problem=_problem(case)
    engine=_engine(problem,seed,arm,output)
    history=list(initial)
    start=time.perf_counter()
    stop=None
    modes=Counter()
    online=[]
    risk_receipts=[]
    selected=[]
    incumbent_curve=[_status(history)]
    first_online=None
    for step in range(steps):
        used=observed_failures(history,INITIAL,step)
        if used>=OBSERVED_FAILURE_CAP:
            stop="observed_six_nonfeasible_stop"
            break
        pool=_pool(problem,case,seed,step,history)
        before=digest([r.to_dict() for r in history])
        chosen,meta=engine.propose(history,pool,step)
        if (before!=digest([r.to_dict() for r in history]) or
                chosen!=problem.canonical(chosen) or chosen not in pool or
                any(digest(chosen)==digest(r.candidate) for r in history)):
            raise RuntimeError("Current engine output changed history or selected invalid candidate")
        chosen_id=digest(chosen)
        selected.append(chosen_id)
        _append(output/"proposals.jsonl",{
            "iteration":step, "candidate":chosen,
            "candidate_sha256":chosen_id,
            "history_prefix_sha256":before,
            "unlabelled_pool_sha256":digest([digest(x) for x in pool]),
            "metadata":meta,
            "future_observation_not_available":True,
        })
        rec=problem.evaluate(chosen)
        if (rec.candidate!=chosen or rec.observation.status!="completed" or
                rec.physical_feasible is None or rec.diagnostics.get("numerical_failure")):
            raise RuntimeError("Current actual mathematical evaluation is untrusted")
        payload=rec.to_dict()
        _append(output/"observations.jsonl",{
            "iteration":step,"record":payload,
            "record_sha256":digest(payload),
            "proposal_candidate_sha256":chosen_id,
        })
        engine.accept(rec)
        online.append(rec)
        history.append(rec)
        if first_online is None and rec.usable_feasible:
            first_online=step
        modes[meta.get("mode","unknown")]+=1
        decision=meta.get("risk_budget")
        receipt={
            "iteration":step,
            "selected_design_sha256":chosen_id,
            "mode":meta.get("mode","unknown"),
            "prediction_available_for_selected":decision is not None,
            "selected_predicted_nonfeasible":decision["selected_predicted_failure"]
                if decision else None,
            "candidate_risk_within_allowance":decision["candidate_risk_within_allowance"]
                if decision else None,
            "budget_gate_fallback":decision["selection_rule"] if decision else None,
            "observed_online_nonfeasible_before":used,
            "realized_nonfeasible":int(not rec.usable_feasible),
            "risk_selection_bypassed":meta.get("risk_selection_bypassed"),
            "event_was_unobserved_when_candidate_selected":True,
            "probability_guarantee":False,
        }
        risk_receipts.append(receipt)
        incumbent_curve.append(_status(history))
    calls=len(online)
    used=observed_failures(history,INITIAL,calls)
    if used>OBSERVED_FAILURE_CAP:
        raise RuntimeError("Observed hard online nonfeasible stop violated")
    if stop is None:
        if steps<ONLINE:
            stop="smoke_two_steps_only"
        elif used==OBSERVED_FAILURE_CAP:
            stop="full_horizon_with_failure_cap_reached_on_last_call"
        else:
            stop="full_online_horizon"
    result={
        "case":case,"seed":seed,"arm":arm,
        "source":"ACTUAL_NEW_MATHEMATICAL_EVALUATIONS_ON_POLICY",
        "initial_count":INITIAL,"online_calls":calls,
        "online_nonfeasible":used,
        "observed_online_cap":OBSERVED_FAILURE_CAP,
        "stop_reason":stop,
        "initial_best":_status(initial),"final_best":_status(history),
        "initial_was_feasible":bool(any(r.usable_feasible for r in initial)),
        "first_online_feasible_iteration":first_online,
        "best_feasible_after_each_actual_online_call":incumbent_curve,
        "mode_counts":dict(modes),
        "pre_evaluation_risk_receipts":risk_receipts,
        "selected_path_digest":digest(selected),
        "final_history_digest":digest([r.to_dict() for r in history]),
        "elapsed_seconds":time.perf_counter()-start,
        "no_future_truth_used_in_posterior":True,
        "probability_guarantee":False,
    }
    _write(output/"result.json",result)
    return result


def run(output, *, profile="smoke"):
    """Write-once new analytical evidence, with no prior experimental truth reads."""
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("A fresh analytical results folder is required")
    sha=clean_commit()
    planned=registration(profile,sha)
    output.mkdir(parents=True,exist_ok=False)
    _write(output/"plan.json",planned)
    code=Path(__file__).resolve()
    _write(output/"source_receipt.json",{
        "experiment_sha":sha,
        "base_pr54_code_sha":PARENT_PR54_SHA,
        "protocol_sha256":_sha(code.with_name("risk_budget55_protocol.py")),
        "runner_sha256":_sha(code),
        "PR54_fixed_policy_sha256":_sha(code.with_name("risk_budget54_policy.py")),
        "PR54_fixed_engine_sha256":_sha(code.with_name("risk_budget54_engine.py")),
        "risk_selection_rule_not_retrained_or_modified":True,
        "new_aircraft_or_rcaide_calls":0,
    })
    results=[]
    total=0
    def forbidden(*_args,**_kwargs):
        raise RuntimeError("PR55 mathematical-only experiment PROHIBITS aircraft evaluation")
    with ExitStack() as stack:
        stack.enter_context(patch.object(AircraftCase,"evaluate",forbidden))
        stack.enter_context(patch.object(SimplifiedElectricAircraft,"evaluate",forbidden))
        for case in CASES:
            problem=_problem(case)
            for seed in planned["seeds"]:
                initial,rejections=_initial(problem,case,seed)
                total+=INITIAL
                _write(output/"shared_initial"/case/(str(seed)+".json"),{
                    "case":case,"seed":seed,"shared_initial":True,
                    "initial_sha256":digest([r.to_dict() for r in initial]),
                    "analytic_screen_rejections":rejections,
                    "records":[r.to_dict() for r in initial],
                })
                for arm in ARMS:
                    trial=_trial(case,seed,arm,initial,
                                 output/"trials"/case/str(seed)/arm,
                                 planned["online_steps_to_execute"])
                    results.append(trial)
                    total+=trial["online_calls"]
                    _append(output/"progress.jsonl",{
                        "case":case,"seed":seed,"arm":arm,
                        "online_calls":trial["online_calls"],
                        "observed_online_nonfeasible":trial["online_nonfeasible"],
                        "final_best":trial["final_best"],
                        "stop_reason":trial["stop_reason"],
                        "selected_path_digest":trial["selected_path_digest"],
                    })
                    print("PR55",case,seed,arm,"calls",trial["online_calls"],
                          "fail",trial["online_nonfeasible"],
                          "best",trial["final_best"],flush=True)
    if total>planned["new_analytical_evaluation_cap"]:
        raise RuntimeError("Frozen new analytical call count exceeded")
    by_arm, comparisons=summarize(results,profile)
    report={
        "schema":SCHEMA,
        "status":"COMPLETE_INDEPENDENT_ON_POLICY_ANALYTICAL_VALIDATION"
            if profile=="full" else "SMOKE_ONLY_NOT_A_VALIDATION",
        "source_sha":sha,"plan":planned,
        "parent_pr54_code_sha":PARENT_PR54_SHA,
        "real_new_analytical_calls":total,
        "new_aircraft_calls":0,
        "new_rcaide_calls":0,
        "old_PR45_54_truth_replayed":False,
        "independent_PR54_seed_block":True,
        "new_rosenbrock_test_family_included":True,
        "actual_on_policy_feedback":True,
        "observed_six_failure_stop_identical_all_arms":True,
        "all_paths_respect_observed_stop":all(
            row["online_nonfeasible"]<=OBSERVED_FAILURE_CAP for row in results),
        "method_summary_by_case":by_arm,
        "paired_seed_comparisons":comparisons,
        "trial_details":results,
        "caveats":[
            "Four user-defined hierarchical mathematical variants, NOT original benchmark or aircraft/RCAIDE.",
            "PR54 showed heterogeneous seed effects; this independent seed block is prospective follow-up, not retrospective p-value fishing.",
            "Different actual online call counts after six observed nonfeasible outcomes must be reported.",
            "Observed stop at six failures is a trivial accounting property, not a probabilistic safety guarantee.",
            "Risk eligibility depends on uncalibrated marginal constraint probability products.",
            "All arms share initial data and candidate stream but update on their OWN selected observations.",
            "Conditional objective averages exclude infeasible cases; interpret paired discovery separately.",
            "No independent sparse-aircraft domain qualification, no claims of RCAIDE fidelity or guaranteed online advantage.",
        ],
    }
    _write(output/"risk_budget55_report.json",report)
    return report
