"""PR58: genuine NEW equal-effort on-policy mathematical BO at exactly six calls.

Each algorithm uses the REAL PR54 GP/VAE/acquisition from the pinned source,
an independent 24-seed panel, exactly 12 shared initial plus 6 individually
chosen and evaluated analytical designs per arm. No PR45/54/55 data replay.
No aircraft or RCAIDE calls. Real GP and VAE fits DO occur.

The original risk model plans a 12-online-call horizon and can stop after six
observed online nonfeasible. We stop the whole experiment at call six for ALL
arms (no method can reach failure cap before then), giving equal actual effort.
"""
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import time
from unittest.mock import patch

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.problems import candidate_pool, SimplifiedElectricAircraft
from ra_lsbo.experiments.core.problems import MathematicalCase, AircraftCase
from ra_lsbo.experiments.core.protocol import BASE
from .risk_budget54_runner import _engine, _status, _write, _append, _sha, clean_commit
from .risk_budget54_policy import observed_failures
from .risk_budget58_protocol import (
    SCHEMA, CASES, ARMS, SEEDS, INITIAL, REAL_ONLINE_CALLS,
    ENGINE_PLANNING_HORIZON, OBSERVED_FAILURE_CAP, POOL, INIT_DRAW_BASE,
    ONLINE_DRAW_BASE, PR54_POLICY_SHA256, PR54_ENGINE_SHA256,
    plan, summarize,
)


def _case(case):
    if case not in CASES:
        raise ValueError("Unregistered PR58 analytical model")
    family, difficulty=case.split("_")
    return MathematicalCase(BASE,family,difficulty)


def _shared_initial(problem,case,seed):
    draw=INIT_DRAW_BASE+1000000*CASES.index(case)+seed
    designs, rejected=candidate_pool(problem,draw,INITIAL)
    if len(designs)!=INITIAL or len({digest(x) for x in designs})!=INITIAL:
        raise RuntimeError("PR58 initial pool unavailable or duplicate")
    records=[]
    for raw in designs:
        rec=problem.evaluate(raw)
        if (rec.candidate!=raw or rec.observation.status!="completed" or
                rec.physical_feasible is None or
                rec.diagnostics.get("numerical_failure")):
            raise RuntimeError("Untrusted analytical shared-initial evaluation")
        records.append(rec)
    return records, rejected


def _candidates(problem,case,seed,iteration,records):
    draw=ONLINE_DRAW_BASE+10000000*CASES.index(case)+100*seed+iteration
    options, _=candidate_pool(problem,draw,POOL+ENGINE_PLANNING_HORIZON)
    prior={digest(r.candidate) for r in records}
    chosen, ids=[],set()
    for raw in options:
        token=digest(raw)
        if token not in prior and token not in ids:
            chosen.append(raw)
            ids.add(token)
            if len(chosen)==POOL:break
    if len(chosen)!=POOL:
        raise RuntimeError("PR58 not enough distinct unevaluated legal candidates")
    return chosen


def _trial(problem,case,seed,arm,initial,output,steps):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    engine=_engine(problem,seed,arm,output)
    history=list(initial)
    modes=Counter()
    receipts=[]
    chosen_ids=[]
    best=[_status(history)]
    first=None
    tic=time.perf_counter()
    init_hash=digest([r.to_dict() for r in initial])
    for iteration in range(steps):
        # The hard failure stop cannot fire before the sixth evaluation;
        # do not bypass or relax the underlying PR54 observed-cap rule.
        used=observed_failures(history,INITIAL,iteration)
        if used>=OBSERVED_FAILURE_CAP:
            raise RuntimeError("PR58 fixed-six trial could not reach mandated next call")
        pool=_candidates(problem,case,seed,iteration,history)
        before=digest([r.to_dict() for r in history])
        candidate,meta=engine.propose(history,pool,iteration)
        if (digest([r.to_dict() for r in history])!=before or
                candidate not in pool or candidate != problem.canonical(candidate) or
                digest(candidate) in {digest(r.candidate) for r in history}):
            raise RuntimeError("Untrusted or non-candidate PR58 online proposal")
        candidate_id=digest(candidate)
        chosen_ids.append(candidate_id)
        _append(output/"proposals.jsonl",{
            "iteration":iteration,"candidate":candidate,
            "candidate_sha256":candidate_id,
            "before_proposal_history_sha256":before,
            "full_unlabelled_candidate_pool_sha256":digest([digest(p) for p in pool]),
            "future_result_unseen":True,"metadata":meta,
        })
        rec=problem.evaluate(candidate)    # ACTUAL new analytical objective/constraint call
        if (rec.candidate!=candidate or rec.observation.status!="completed" or
                rec.physical_feasible is None or
                rec.diagnostics.get("numerical_failure")):
            raise RuntimeError("PR58 real mathematical evaluator returned invalid evidence")
        packet=rec.to_dict()
        _append(output/"observations.jsonl",{
            "iteration":iteration,
            "proposal_candidate_sha256":candidate_id,
            "actual_record":packet,"actual_record_sha256":digest(packet),
        })
        engine.accept(rec)
        history.append(rec)
        if first is None and rec.usable_feasible:
            first=iteration
        modes[meta.get("mode","unknown")]+=1
        receipt=meta.get("risk_budget")
        receipts.append({
            "iteration":iteration,"candidate_sha256":candidate_id,
            "mode":meta.get("mode","unknown"),
            "forecast_available_for_selected":receipt is not None,
            "predicted_nonfeasible":receipt["selected_predicted_failure"]
                if receipt else None,
            "budget_rule":receipt["selection_rule"] if receipt else None,
            "predicted_gate_passed":receipt["candidate_risk_within_allowance"]
                if receipt else None,
            "observed_online_nonfeasible_before":used,
            "realized_nonfeasible":int(not rec.usable_feasible),
            "risk_model_decision_bypassed":meta.get("risk_selection_bypassed"),
            "prediction_is_not_a_guarantee":True,
            "outcome_not_available_when_selected":True,
        })
        best.append(_status(history))
    if len(receipts)!=steps:
        raise RuntimeError("PR58 did not actually evaluate the planned fixed online calls")
    failed=observed_failures(history,INITIAL,steps)
    if failed>OBSERVED_FAILURE_CAP:
        raise RuntimeError("PR58 fixed-six observed failure count exceeded call count")
    result={
        "case":case,"seed":seed,"arm":arm,
        "actual_on_policy_new_analytical_evaluations":True,
        "online_calls":steps,"observed_nonfeasible":failed,
        "initial_best":_status(initial),"final_best":_status(history),
        "initial_prefix_sha256":init_hash,
        "best_feasible_at_each_actual_call":best,
        "first_online_feasible_iteration":first,
        "risk_receipts":receipts,
        "selected_path_sha256":digest(chosen_ids),
        "final_history_sha256":digest([r.to_dict() for r in history]),
        "six_call_horizon":REAL_ONLINE_CALLS,
        "forecast_internal_horizon":ENGINE_PLANNING_HORIZON,
        "mode_counts":dict(modes),
        "stop_reason":"planned_equal_six_real_calls"
                      if steps==REAL_ONLINE_CALLS else "smoke_two_real_calls",
        "evidence_before_result_was_future_blind":True,
        "probability_guarantee":False,
        "fit_and_analytical_evaluation_seconds":time.perf_counter()-tic,
    }
    _write(output/"result.json",result)
    return result


def run(output, *, profile="smoke"):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("PR58 requires a NEW write-once output folder")
    sha=clean_commit()
    here=Path(__file__).resolve()
    # Required exact source provenance from completed PR54/55. The algorithm
    # modules cannot silently change when making the new prospective claim.
    if (_sha(here.with_name("risk_budget54_policy.py"))!=PR54_POLICY_SHA256 or
            _sha(here.with_name("risk_budget54_engine.py"))!=PR54_ENGINE_SHA256):
        raise ValueError("PR58 refuses modified original PR54 policy/engine")
    spec=plan(profile,sha)
    steps=spec["actual_online_calls_per_arm"]
    output.mkdir(parents=True,exist_ok=False)
    _write(output/"plan.json",spec)
    _write(output/"source_receipt.json",{
        "experiment_head_sha":sha,
        "pr58_protocol_sha256":_sha(here.with_name("risk_budget58_protocol.py")),
        "pr58_runner_sha256":_sha(here),
        "frozen_pr54_policy_sha256":PR54_POLICY_SHA256,
        "frozen_pr54_engine_sha256":PR54_ENGINE_SHA256,
        "source_evaluator":"real custom MathematicalCase.evaluate",
        "no_pr45_or_pr54_pr55_data_read":True,
        "new_aircraft_calls":0,"new_rcaide_calls":0,
    })
    def forbidden(*_a,**_k):
        raise RuntimeError("PR58 is mathematical ONLY: aircraft evaluation forbidden")
    records=[]
    charged=0
    with ExitStack() as stack:
        stack.enter_context(patch.object(AircraftCase,"evaluate",forbidden))
        stack.enter_context(patch.object(SimplifiedElectricAircraft,"evaluate",forbidden))
        for case in CASES:
            problem=_case(case)
            for seed in spec["seeds"]:
                initial,rejections=_shared_initial(problem,case,seed)
                charged+=INITIAL
                _write(output/"shared_initial"/case/(str(seed)+".json"),{
                    "case":case,"seed":seed,
                    "all_three_arms_share_same_initial":True,
                    "records":[r.to_dict() for r in initial],
                    "initial_prefix_sha256":digest([r.to_dict() for r in initial]),
                    "analytical_precheck_rejections":rejections,
                })
                for arm in ARMS:
                    result=_trial(problem,case,seed,arm,initial,
                                  output/"trials"/case/str(seed)/arm,steps)
                    records.append(result)
                    charged+=result["online_calls"]
                    _append(output/"progress.jsonl",{
                        "case":case,"seed":seed,"arm":arm,
                        "actual_online_calls":result["online_calls"],
                        "actual_online_nonfeasible":result["observed_nonfeasible"],
                        "best_actual_feasible":result["final_best"],
                        "selected_path_sha256":result["selected_path_sha256"],
                    })
                    print("PR58",case,seed,arm,
                          "calls",result["online_calls"],
                          "nonfeasible",result["observed_nonfeasible"],
                          "best",result["final_best"],flush=True)
    if charged!=spec["new_analytic_evaluation_ceiling"]:
        raise RuntimeError("PR58 fixed exact real analytical call budget not reconciled")
    by_case,paired=summarize(records,profile=profile)
    report={
        "schema":SCHEMA,
        "status":"COMPLETE_FRESH_FIXED_SIX_ON_POLICY" if profile=="full" else
                 "SMOKE_TWO_CALLS_NOT_FULL_STUDY",
        "code_sha":sha,
        "plan":spec,
        "n_real_analytical_calls":charged,
        "n_original_on_policy_trials":len(records),
        "new_aircraft_calls":0,"new_rcaide_calls":0,
        "new_gp_vae_fits": "real actual fits, NOT zero",
        "same_real_online_call_count":True,
        "no_early_stop_or_treatment_dependent_censoring":True,
        "identical_pr54_acquisition_and_planning_horizon":True,
        "result_summary_by_case_arm":by_case,
        "paired_primary_and_secondary":paired,
        "trial_details":records,
        "interpretation_limits":[
            "After seeing PR54-PR57, we froze this fresh seed block and six-call objective; exploratory methodology, not pre-registered formal confirmation.",
            "Only custom hierarchical Branin/Rosenbrock mathematical tasks; not aircraft or a standardized benchmark replica.",
            "Each arm truly made the same number of evaluations, but chose DIFFERENT designs and refit on its OWN observed history.",
            "The legacy completion factor was one-class in PR45 aircraft, but non-aircraft task structure differs; treat ablation as a policy change, not a universal probability theorem.",
            "The budgeted policy still uses a possibly optimistic marginal GP probability product, NOT calibrated physical reliability.",
            "The stop cap of six cannot truncate BEFORE the sixth call; that fact is not a safety guarantee.",
            "Results are stratified by case, including all paired infeasible and only-one-feasible seeds; no unsupported p-value or cross-case pooling.",
            "No mission-scale 140 km independent sparsity qualification or RCAIDE/flying vehicle validation.",
        ],
    }
    _write(output/"fixed_six_onpolicy_report.json",report)
    return report
