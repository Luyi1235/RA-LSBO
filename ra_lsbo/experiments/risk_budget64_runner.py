"""PR64: genuine equal-effort candidate-level credibility/value on-policy BO.

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
from .risk_budget64_abstention_engine import SoftGPAbstainBO
from .risk_budget54_policy import observed_failures
from .risk_budget64_protocol import (
    SCHEMA, CASES, ARMS, SEEDS, INITIAL, MAX_ONLINE_CALLS,
    POOL, INIT_DRAW_BASE, ONLINE_DRAW_BASE, PR54_POLICY_SHA256,
    PR54_ENGINE_SHA256, plan, summarize,
)
from .risk_budget60_protocol import ENGINE_PLANNING_HORIZON
)


def _engine(problem, seed, arm, output):
    if arm in ("semantic_only","budgeted"):
        return _legacy_engine(problem,seed,arm,output)
    if arm=="soft_gp_continue":
        return LocalCredibilityValueBO(problem,BASE,seed,output,variant="soft_gp",
            initial_count=INITIAL,failure_cap=MAX_ONLINE_CALLS)
    if arm=="soft_gp_abstain":
        return SoftGPAbstainBO(problem,BASE,seed,output,
            initial_count=INITIAL,failure_cap=MAX_ONLINE_CALLS)
    raise ValueError("Unregistered PR64 arm")


def _case(case):
    if case not in CASES:
        raise ValueError("Unregistered PR64 analytical model")
    family, difficulty=case.split("_")
    return MathematicalCase(BASE,family,difficulty)


def _shared_initial(problem,case,seed):
    draw=INIT_DRAW_BASE+1000000*CASES.index(case)+seed
    designs, rejected=candidate_pool(problem,draw,INITIAL)
    if len(designs)!=INITIAL or len({digest(x) for x in designs})!=INITIAL:
        raise RuntimeError("PR64 initial pool unavailable or duplicate")
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
        raise RuntimeError("PR64 not enough distinct unevaluated legal candidates")
    return chosen


def _trial(problem,case,seed,arm,initial,output,max_steps):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    engine=_engine(problem,seed,arm,output)
    history=list(initial)
    modes=Counter()
    decisions=[]
    chosen_ids=[]
    best=[_status(history)]
    first=None
    tic=time.perf_counter()
    init_hash=digest([r.to_dict() for r in initial])
    stop_reason="max_budget_exhausted"
    for iteration in range(max_steps):
        # The hard failure stop cannot fire before the sixth evaluation;
        # do not bypass or relax the underlying PR54 observed-cap rule.
        used=observed_failures(history,INITIAL,iteration)
        pool=_candidates(problem,case,seed,iteration,history)
        before=digest([r.to_dict() for r in history])
        candidate,meta=engine.propose(history,pool,iteration)
        if digest([r.to_dict() for r in history])!=before:
            raise RuntimeError("PR64 proposal modified history")
        if candidate is None:
            if arm!="soft_gp_abstain" or meta.get("pr64_action")!="abstain_stop_before_evaluator":
                raise RuntimeError("Unregistered abstention")
            receipt=meta.get("credibility_decision")
            rejected=meta.get("abstained_candidate")
            cid=digest(rejected)
            if (not isinstance(receipt,dict) or receipt.get("selected_candidate_sha256")!=cid
                or receipt.get("selected_net_utility",1.)>0
                or receipt.get("abstention_performed") is not True):
                raise RuntimeError("Invalid pre-outcome abstention receipt")
            chosen_ids.append(cid)
            _append(output/"proposals.jsonl",{"iteration":iteration,"candidate":None,
                "counterfactual_candidate_sha256":cid,"evaluation_performed":False,
                "action":"abstain_stop_before_evaluator","metadata":meta})
            decisions.append({"iteration":iteration,"candidate_sha256":cid,
                "evaluation_performed":False,"action":"abstain_stop_before_evaluator",
                "selected_net_utility":receipt["selected_net_utility"],
                "predicted_nonfeasible":receipt["selected_predicted_failure"],
                "realized_nonfeasible":None,"no_evaluator_call":True})
            stop_reason="soft_utility_outside_option"
            break
        if candidate not in pool or candidate!=problem.canonical(candidate) or digest(candidate) in {digest(r.candidate) for r in history}:
            raise RuntimeError("Invalid PR64 candidate")
        cid=digest(candidate);chosen_ids.append(cid)
        _append(output/"proposals.jsonl",{"iteration":iteration,"candidate":candidate,
            "candidate_sha256":cid,"evaluation_performed":True,"metadata":meta})
        rec=problem.evaluate(candidate)    # ACTUAL new analytical objective/constraint call
        if (rec.candidate!=candidate or rec.observation.status!="completed" or
                rec.physical_feasible is None or
                rec.diagnostics.get("numerical_failure")):
            raise RuntimeError("PR64 real mathematical evaluator returned invalid evidence")
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
        receipt=(meta.get("credibility_decision") if arm.startswith("soft_gp_") else meta.get("risk_budget"))
        if arm.startswith("soft_gp_") and receipt is not None:
            if (receipt.get("selected_candidate_sha256")!=cid or
                    receipt.get("actual_action")!=("evaluate" if arm=="soft_gp_abstain"
                    else "evaluate_under_frozen_six_call_protocol")):
                raise RuntimeError("PR64 selected risk receipt mismatch")
        decisions.append({"iteration":iteration,"candidate_sha256":cid,
            "evaluation_performed":True,"action":"evaluate",
            "forecast_available_for_selected":receipt is not None,
            "predicted_nonfeasible":receipt.get("selected_predicted_failure") if receipt else None,
            "selected_net_utility":receipt.get("selected_net_utility") if receipt else None,
            "realized_nonfeasible":int(not rec.usable_feasible),
            "outcome_not_available_when_selected":True})
        best.append(_status(history))
    actual=sum(d["evaluation_performed"] for d in decisions)
    failed=observed_failures(history,INITIAL,actual)
    if failed!=sum(d.get("realized_nonfeasible")==1 for d in decisions):
        raise RuntimeError("PR64 failures mismatch actual evaluations")
    result={"case":case,"seed":seed,"arm":arm,"max_online_calls":max_steps,
        "online_calls":actual,"observed_nonfeasible":failed,
        "initial_best":_status(initial),"final_best":_status(history),
        "initial_prefix_sha256":init_hash,"best_feasible_at_each_actual_call":best,
        "first_online_feasible_iteration":first,"decisions":decisions,
        "selected_path_sha256":digest(chosen_ids),
        "final_history_sha256":digest([r.to_dict() for r in history]),
        "stop_reason":stop_reason,"probability_guarantee":False,
        "evidence_before_result_was_future_blind":True,
        "fit_and_analytical_evaluation_seconds":time.perf_counter()-tic}
    _write(output/"result.json",result)
    return result


def run(output, *, profile="smoke"):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("PR64 requires a NEW write-once output folder")
    sha=clean_commit()
    here=Path(__file__).resolve()
    # Required exact source provenance from completed PR54/55. The algorithm
    # modules cannot silently change when making the new prospective claim.
    if (_sha(here.with_name("risk_budget54_policy.py"))!=PR54_POLICY_SHA256 or
            _sha(here.with_name("risk_budget54_engine.py"))!=PR54_ENGINE_SHA256):
        raise ValueError("PR64 refuses modified original PR54 policy/engine")
    spec=plan(profile,sha)
    steps=spec["max_online_calls_per_arm"]
    output.mkdir(parents=True,exist_ok=False)
    _write(output/"plan.json",spec)
    _write(output/"source_receipt.json",{
        "experiment_head_sha":sha,
        "pr64_protocol_sha256":_sha(here.with_name("risk_budget64_protocol.py")),
        "pr64_abstention_engine_sha256":_sha(here.with_name("risk_budget64_abstention_engine.py")),
        "pr63_policy_sha256":_sha(here.with_name("risk_budget63_local_value_policy.py")),
        "pr63_engine_sha256":_sha(here.with_name("risk_budget63_local_value_engine.py")),
        "pr64_runner_sha256":_sha(here),
        "frozen_pr54_policy_sha256":PR54_POLICY_SHA256,
        "frozen_pr54_engine_sha256":PR54_ENGINE_SHA256,
        "source_evaluator":"real custom MathematicalCase.evaluate",
        "no_pr45_to_pr63_historical_truth_read":True,
        "new_aircraft_calls":0,"new_rcaide_calls":0,
    })
    def forbidden(*_a,**_k):
        raise RuntimeError("PR64 is mathematical ONLY: aircraft evaluation forbidden")
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
                    "all_four_arms_share_same_initial":True,
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
                    print("PR64",case,seed,arm,
                          "calls",result["online_calls"],
                          "nonfeasible",result["observed_nonfeasible"],
                          "best",result["final_best"],flush=True)
    if charged>spec["max_new_analytic_evaluation_budget"]:
        raise RuntimeError("PR64 actual analytical calls exceeded maximum budget")
    by_case,paired=summarize(records,profile=profile)
    report={
        "schema":SCHEMA,
        "status":"COMPLETE_FRESH_MAX_BUDGET_ABSTENTION" if profile=="full" else
                 "SMOKE_MAX_BUDGET_ABSTENTION_NOT_FULL",
        "code_sha":sha,
        "plan":spec,
        "n_real_analytical_calls":charged,
        "max_new_analytic_evaluation_budget":spec["max_new_analytic_evaluation_budget"],
        "unused_analytic_evaluation_budget":spec["max_new_analytic_evaluation_budget"]-charged,
        "max_new_analytic_evaluation_budget":spec["max_new_analytic_evaluation_budget"],
        "n_new_on_policy_trials":len(records),
        "new_aircraft_calls":0,"new_rcaide_calls":0,
        "new_gp_vae_fits": "real actual fits, NOT zero",
        "same_maximum_online_call_budget":True,
        "actual_online_calls_may_be_lower_for_abstention_arm":True,
        "no_early_stop_or_treatment_dependent_censoring":True,
        "original_pr54_three_control_arms_byte_identical":True,
        "new_credibility_value_arms_are_opt_in":True,
        "new_arms_use_no_hard_risk_gate":True,
        "real_abstention_performed_only_before_evaluator":True,
        "internal_planning_horizon_12_unmodified":True,
        "result_summary_by_case_arm":by_case,
        "paired_primary_and_secondary":paired,
        "trial_details":records,
        "interpretation_limits":[
            "PR64 is designed after PR63 results; exploratory prospective evidence, not independent confirmatory superiority.",
            "This is same MAXIMUM budget, not same actual evaluation count; report cost and quality together.",
            "Abstention is not a success or failure label and has no objective value.",
            "These are custom hierarchical Branin/Rosenbrock analytical tasks, not real aircraft or RCAIDE, and not official benchmark replicas.",
            "All arms have identical six actual online calls but adaptively choose different designs and retrain their own GP/VAE from settled observations.",
            "PR64 tests only PR63 soft-GP utility and real abstention; local blend and information bonus were not carried forward after negative ablations.",
            "GP feasibility and net utility remain uncalibrated heuristics, not safety bounds.",
            "Abstentions have no evaluator outcome; do not impute objective or feasible labels.",
            "The optimistic marginal GP feasibility product may remain misspecified, so no safe-BO or physical failure guarantee applies.",
            "Posterior fitting/representation numerical failure or coverage decisions have NO model probability receipt; they still cost actual evaluations and failures.",
            "All paired infeasible and one-sided feasible seeds are retained; objective mean is conditional on both feasible; no posthoc multiple-testing superiority claim.",
            "Flight propulsion/control collaboration requires separately qualified 100/140km physical constraints, scenarios and independent on-policy airplane experiments.",
        ],
    }
    _write(output/"fixed_six_onpolicy_report.json",report)
    return report
