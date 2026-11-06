"""PR63: genuine equal-effort candidate-level credibility/value on-policy BO.

Three unchanged PR54 controls and three opt-in SOFT selection rules share
exactly the same real VAE/GP and mathematical evaluator. New algorithms
blend candidate-local observed failure evidence and optional benefit-gated
information value, NEVER a hard probability gate. Six actual online calls
per arm are performed with arm-specific posterior retraining and feedback.
There are no RCAIDE/aircraft calls or reads of earlier result tables.
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
from .risk_budget54_runner import _engine as _legacy_engine, _status, _write, _append, _sha, clean_commit
from .risk_budget63_local_value_engine import LocalCredibilityValueBO
from .risk_budget54_policy import observed_failures
from .risk_budget63_protocol import (
    SCHEMA, CASES, ARMS, SEEDS, INITIAL, REAL_ONLINE_CALLS,
    ENGINE_PLANNING_HORIZON, OBSERVED_FAILURE_CAP, POOL, INIT_DRAW_BASE,
    ONLINE_DRAW_BASE, PR54_POLICY_SHA256, PR54_ENGINE_SHA256,
    plan, summarize,
)


def _engine(problem, seed, arm, output):
    if arm in ("legacy","semantic_only","budgeted"):
        return _legacy_engine(problem,seed,arm,output)
    return LocalCredibilityValueBO(
        problem,BASE,seed,output,variant=arm,initial_count=INITIAL,
        failure_cap=OBSERVED_FAILURE_CAP)


def _case(case):
    if case not in CASES:
        raise ValueError("Unregistered PR63 analytical model")
    family, difficulty=case.split("_")
    return MathematicalCase(BASE,family,difficulty)


def _shared_initial(problem,case,seed):
    draw=INIT_DRAW_BASE+1000000*CASES.index(case)+seed
    designs, rejected=candidate_pool(problem,draw,INITIAL)
    if len(designs)!=INITIAL or len({digest(x) for x in designs})!=INITIAL:
        raise RuntimeError("PR63 initial pool unavailable or duplicate")
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
        raise RuntimeError("PR63 not enough distinct unevaluated legal candidates")
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
            raise RuntimeError("PR63 fixed-six trial could not reach mandated next call")
        pool=_candidates(problem,case,seed,iteration,history)
        before=digest([r.to_dict() for r in history])
        candidate,meta=engine.propose(history,pool,iteration)
        if (digest([r.to_dict() for r in history])!=before or
                candidate not in pool or candidate != problem.canonical(candidate) or
                digest(candidate) in {digest(r.candidate) for r in history}):
            raise RuntimeError("Untrusted or non-candidate PR63 online proposal")
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
            raise RuntimeError("PR63 real mathematical evaluator returned invalid evidence")
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
        receipt=(meta.get("credibility_decision")
                 if arm in ("soft_gp","local_blend","local_value")
                 else meta.get("risk_budget"))
        if arm in ("soft_gp","local_blend","local_value") and receipt is not None:
            if (receipt.get("selected_candidate_sha256")!=candidate_id or
                    receipt.get("no_hard_risk_gate") is not True or
                    receipt.get("actual_action")!="evaluate_under_frozen_six_call_protocol"):
                raise ValueError("PR63 soft decision not bound to actual evaluated design")
        receipts.append({
            "iteration":iteration,"candidate_sha256":candidate_id,
            "mode":meta.get("mode","unknown"),
            "forecast_available_for_selected":receipt is not None,
            "predicted_nonfeasible":receipt["selected_predicted_failure"]
                if receipt else None,
            "budget_rule":receipt["selection_rule"] if receipt else None,
            "predicted_gate_passed":receipt.get("candidate_risk_within_allowance")
                if receipt else None,
            "original_gp_predicted_failure":receipt.get(
                "selected_original_GP_failure_proxy") if receipt else None,
            "selected_local_evidence":receipt.get(
                "selected_local_evidence") if receipt else None,
            "selected_net_utility":receipt.get("selected_net_utility")
                if receipt else None,
            "selected_gain":receipt.get("selected_objective_gain_normalized")
                if receipt else None,
            "selected_information_bonus":receipt.get("selected_information_bonus")
                if receipt else None,
            "would_defer_in_unconstrained_utility_model":receipt.get(
                "would_defer_in_unconstrained_utility_model") if receipt else None,
            "no_hard_risk_gate":receipt.get("no_hard_risk_gate")
                if receipt else None,
            "actual_action":receipt.get("actual_action") if receipt else None,
            "observed_online_nonfeasible_before":used,
            "realized_nonfeasible":int(not rec.usable_feasible),
            "risk_model_decision_bypassed":meta.get("risk_selection_bypassed"),
            "prediction_is_not_a_guarantee":True,
            "outcome_not_available_when_selected":True,
        })
        best.append(_status(history))
    if len(receipts)!=steps:
        raise RuntimeError("PR63 did not actually evaluate the planned fixed online calls")
    failed=observed_failures(history,INITIAL,steps)
    if failed>OBSERVED_FAILURE_CAP:
        raise RuntimeError("PR63 fixed-six observed failure count exceeded call count")
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
    if output.exists():raise FileExistsError("PR63 requires a NEW write-once output folder")
    sha=clean_commit()
    here=Path(__file__).resolve()
    # Required exact source provenance from completed PR54/55. The algorithm
    # modules cannot silently change when making the new prospective claim.
    if (_sha(here.with_name("risk_budget54_policy.py"))!=PR54_POLICY_SHA256 or
            _sha(here.with_name("risk_budget54_engine.py"))!=PR54_ENGINE_SHA256):
        raise ValueError("PR63 refuses modified original PR54 policy/engine")
    spec=plan(profile,sha)
    steps=spec["actual_online_calls_per_arm"]
    output.mkdir(parents=True,exist_ok=False)
    _write(output/"plan.json",spec)
    _write(output/"source_receipt.json",{
        "experiment_head_sha":sha,
        "pr63_protocol_sha256":_sha(here.with_name("risk_budget63_protocol.py")),
        "pr63_policy_sha256":_sha(here.with_name("risk_budget63_local_value_policy.py")),
        "pr63_engine_sha256":_sha(here.with_name("risk_budget63_local_value_engine.py")),
        "pr63_runner_sha256":_sha(here),
        "frozen_pr54_policy_sha256":PR54_POLICY_SHA256,
        "frozen_pr54_engine_sha256":PR54_ENGINE_SHA256,
        "source_evaluator":"real custom MathematicalCase.evaluate",
        "no_pr45_to_pr62_historical_truth_read":True,
        "new_aircraft_calls":0,"new_rcaide_calls":0,
    })
    def forbidden(*_a,**_k):
        raise RuntimeError("PR63 is mathematical ONLY: aircraft evaluation forbidden")
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
                    "all_six_arms_share_same_initial":True,
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
                    print("PR63",case,seed,arm,
                          "calls",result["online_calls"],
                          "nonfeasible",result["observed_nonfeasible"],
                          "best",result["final_best"],flush=True)
    if charged!=spec["new_analytic_evaluation_ceiling"]:
        raise RuntimeError("PR63 fixed exact real analytical call budget not reconciled")
    by_case,paired=summarize(records,profile=profile)
    report={
        "schema":SCHEMA,
        "status":"COMPLETE_FRESH_FIXED_SIX_ON_POLICY" if profile=="full" else
                 "SMOKE_TWO_CALLS_NOT_FULL_STUDY",
        "code_sha":sha,
        "plan":spec,
        "n_real_analytical_calls":charged,
        "n_new_on_policy_trials":len(records),
        "new_aircraft_calls":0,"new_rcaide_calls":0,
        "new_gp_vae_fits": "real actual fits, NOT zero",
        "same_real_online_call_count":True,
        "no_early_stop_or_treatment_dependent_censoring":True,
        "original_pr54_three_control_arms_byte_identical":True,
        "new_credibility_value_arms_are_opt_in":True,
        "new_arms_use_no_hard_risk_gate":True,
        "internal_planning_horizon_12_unmodified":True,
        "result_summary_by_case_arm":by_case,
        "paired_primary_and_secondary":paired,
        "trial_details":records,
        "interpretation_limits":[
            "Methods were designed after observing PR50–61 and preparing PR62; prospective new seed experiment is exploratory, not proof of independent confirmatory superiority.",
            "These are custom hierarchical Branin/Rosenbrock analytical tasks, not real aircraft or RCAIDE, and not official benchmark replicas.",
            "All arms have identical six actual online calls but adaptively choose different designs and retrain their own GP/VAE from settled observations.",
            "The candidate-local kernel-weighted error frequency is a spatial heuristic, NOT an IID/Beta posterior, calibrated probability, confidence set, or safety bound.",
            "Soft expected-gain minus failure-cost utility has no hard risk gate; an outside-option negative-net flag is ONLY a counterfactual preference, not an actual abstention in this six-call trial.",
            "The optimistic marginal GP feasibility product may remain misspecified, so no safe-BO or physical failure guarantee applies.",
            "Posterior fitting/representation numerical failure or coverage decisions have NO model probability receipt; they still cost actual evaluations and failures.",
            "All paired infeasible and one-sided feasible seeds are retained; objective mean is conditional on both feasible; no posthoc multiple-testing superiority claim.",
            "Flight propulsion/control collaboration requires separately qualified 100/140km physical constraints, scenarios and independent on-policy airplane experiments.",
        ],
    }
    _write(output/"fixed_six_onpolicy_report.json",report)
    return report
