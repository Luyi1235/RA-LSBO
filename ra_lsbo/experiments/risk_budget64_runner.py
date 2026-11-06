"""PR64 real selective evaluation: fixed-vs-abstain soft GP at max six calls.

The PR63 soft_gp score is reused unchanged. PR64 compares:
- soft_gp_fixed128: PR63 pool breadth and always evaluate;
- soft_gp_fixed512: expanded cheap screening, always evaluate;
- soft_gp_abstain512: same 512 candidates and same GP score, but if the
  maximum eligible candidate net utility is <= 0, abstain BEFORE evaluator.

An abstained candidate receives no objective/constraint label and no actual
evaluation call. Model-free coverage remains a separate, explicitly unscored
action and is evaluated. This is a maximum-budget/Pareto study, NOT equal
realized-call comparison. No aircraft/RCAIDE calls.
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
from .risk_budget64_protocol import (
    SCHEMA, CASES, ARMS, SEEDS, INITIAL, MAX_ONLINE_CALLS,
    ENGINE_PLANNING_HORIZON, OBSERVED_FAILURE_CAP,
    POOL_NARROW, POOL_WIDE, INIT_DRAW_BASE, ONLINE_DRAW_BASE,
    PR54_POLICY_SHA256, PR54_ENGINE_SHA256, PR63_POLICY_SHA256,
    PR63_ENGINE_SHA256, plan, summarize,
)


def _engine(problem,seed,arm,output):
    if arm in ("legacy","semantic_only","budgeted"):
        return _legacy_engine(problem,seed,arm,output)
    return LocalCredibilityValueBO(
        problem,BASE,seed,output,variant="soft_gp",
        initial_count=INITIAL,failure_cap=OBSERVED_FAILURE_CAP)


def _case(case):
    if case not in CASES:
        raise ValueError("Unregistered PR64 analytical model")
    family,difficulty=case.split("_")
    return MathematicalCase(BASE,family,difficulty)


def _shared_initial(problem,case,seed):
    draw=INIT_DRAW_BASE+1000000*CASES.index(case)+seed
    designs,rejected=candidate_pool(problem,draw,INITIAL)
    if len(designs)!=INITIAL or len({digest(x) for x in designs})!=INITIAL:
        raise RuntimeError("PR64 shared initial design pool unavailable or duplicated")
    records=[]
    for raw in designs:
        rec=problem.evaluate(raw)
        if (rec.candidate!=raw or rec.observation.status!="completed" or
                rec.physical_feasible is None or rec.diagnostics.get("numerical_failure")):
            raise RuntimeError("PR64 untrusted analytical shared-initial evaluation")
        records.append(rec)
    return records,rejected


def _candidate_pools(problem,case,seed,iteration,records):
    """One deterministic 512 pool; narrow arms use its exact first 128 points."""
    draw=ONLINE_DRAW_BASE+10000000*CASES.index(case)+100*seed+iteration
    prior={digest(r.candidate) for r in records}
    options,rejected=candidate_pool(
        problem,draw,POOL_WIDE+ENGINE_PLANNING_HORIZON,prior)
    if len(options)<POOL_WIDE:
        raise RuntimeError("PR64 exact-screened wide candidate pool too small")
    wide=options[:POOL_WIDE]
    narrow=wide[:POOL_NARROW]
    if (len({digest(x) for x in wide})!=POOL_WIDE or
            any(digest(x) in prior for x in wide)):
        raise RuntimeError("PR64 candidate pool contains duplicate/previously evaluated designs")
    return narrow,wide,rejected


def _trial(problem,case,seed,arm,initial,output,max_calls):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    engine=_engine(problem,seed,arm,output)
    history=list(initial)
    decision_receipts=[]
    evaluated_ids=[]
    decision_ids=[]
    best=[_status(history)]
    first=None
    mode_counts=Counter()
    tic=time.perf_counter()
    initial_hash=digest([r.to_dict() for r in initial])
    stop_reason=None
    total_screened=0
    for iteration in range(max_calls):
        used=observed_failures(history,INITIAL,iteration)
        if used>=OBSERVED_FAILURE_CAP:
            raise RuntimeError("PR64 reached observed failure cap before maximum call budget")
        narrow,wide,rejected=_candidate_pools(problem,case,seed,iteration,history)
        pool=wide if arm in ("soft_gp_fixed512","soft_gp_abstain512") else narrow
        total_screened+=len(pool)
        before=digest([r.to_dict() for r in history])
        candidate,meta=engine.propose(history,pool,iteration)
        if (digest([r.to_dict() for r in history])!=before or
                candidate not in pool or candidate!=problem.canonical(candidate) or
                digest(candidate) in {digest(r.candidate) for r in history}):
            raise RuntimeError("PR64 proposal is not a fresh canonical member of its actual GP pool")
        candidate_id=digest(candidate)
        decision_ids.append(candidate_id)
        receipt=(meta.get("credibility_decision")
                 if arm.startswith("soft_gp") else meta.get("risk_budget"))
        forecast=receipt is not None
        is_soft=arm.startswith("soft_gp")
        if is_soft and forecast:
            if (receipt.get("selected_candidate_sha256")!=candidate_id or
                    receipt.get("no_hard_risk_gate") is not True or
                    receipt.get("candidate_filter_applied") is not False or
                    not isinstance(receipt.get("would_defer_in_unconstrained_utility_model"),bool)):
                raise ValueError("PR64 soft-GP risk/utility receipt not bound to the selected candidate")
        abstain=(arm=="soft_gp_abstain512" and forecast and
                 receipt["would_defer_in_unconstrained_utility_model"] is True)
        if receipt is not None:
            receipt["actual_action"]=(
                "abstain_before_evaluation" if abstain else
                "evaluate_under_maximum_budget")
            receipt["actual_evaluation_performed"]=not abstain
        action=("abstain_before_evaluation" if abstain else
                "evaluate_under_maximum_budget" if forecast else
                "evaluate_coverage_without_risk_forecast")
        _append(output/"proposals.jsonl",{
            "proposal_step":iteration,"candidate":candidate,
            "candidate_sha256":candidate_id,
            "before_proposal_history_sha256":before,
            "candidate_pool_size":len(pool),
            "candidate_pool_sha256":digest([digest(p) for p in pool]),
            "narrow_pool_size":POOL_NARROW,"wide_pool_size":POOL_WIDE,
            "pool_exact_screen_rejections":rejected,
            "action_before_truth_evaluation":action,
            "evaluated":not abstain,
            "future_result_unseen":True,
            "metadata":meta,
        })
        decision={
            "proposal_step":iteration,"candidate_sha256":candidate_id,
            "candidate_pool_size":len(pool),
            "mode":meta.get("mode","unknown"),
            "evaluated":not abstain,
            "actual_call_index":iteration,
            "actual_action":action,
            "forecast_available_for_selected":forecast,
            "predicted_nonfeasible":receipt["selected_predicted_failure"] if receipt else None,
            "original_gp_predicted_failure":receipt.get(
                "selected_original_GP_failure_proxy") if receipt else None,
            "selected_net_utility":receipt.get("selected_net_utility") if receipt else None,
            "selected_gain":receipt.get("selected_objective_gain_normalized") if receipt else None,
            "selected_information_bonus":receipt.get("selected_information_bonus") if receipt else None,
            "would_defer_in_unconstrained_utility_model":receipt.get(
                "would_defer_in_unconstrained_utility_model") if receipt else None,
            "no_hard_risk_gate":receipt.get("no_hard_risk_gate") if receipt else None,
            "candidate_filter_applied":receipt.get("candidate_filter_applied") if receipt else None,
            "observed_online_nonfeasible_before":used,
            "realized_nonfeasible":None,
            "prediction_is_not_a_guarantee":True,
            "outcome_not_available_when_selected":True,
        }
        if abstain:
            decision["actual_call_index"]=len(history)-INITIAL
            decision_receipts.append(decision)
            stop_reason="outside_option_nonpositive"
            break
        rec=problem.evaluate(candidate)   # only this line incurs a real analytical truth call
        if (rec.candidate!=candidate or rec.observation.status!="completed" or
                rec.physical_feasible is None or rec.diagnostics.get("numerical_failure")):
            raise RuntimeError("PR64 real mathematical evaluator returned invalid evidence")
        packet=rec.to_dict()
        _append(output/"observations.jsonl",{
            "proposal_step":iteration,"actual_call_index":iteration,
            "proposal_candidate_sha256":candidate_id,
            "actual_record":packet,"actual_record_sha256":digest(packet),
        })
        engine.accept(rec)
        history.append(rec)
        evaluated_ids.append(candidate_id)
        if first is None and rec.usable_feasible:
            first=iteration
        decision["realized_nonfeasible"]=int(not rec.usable_feasible)
        decision_receipts.append(decision)
        mode_counts[meta.get("mode","unknown")]+=1
        best.append(_status(history))
    online_calls=len(history)-INITIAL
    if stop_reason is None:
        stop_reason="maximum_online_call_budget"
    if len(best)!=online_calls+1 or len(evaluated_ids)!=online_calls:
        raise RuntimeError("PR64 actual evaluator calls and incumbent curve do not reconcile")
    failed=observed_failures(history,INITIAL,online_calls)
    if failed>OBSERVED_FAILURE_CAP:
        raise RuntimeError("PR64 actual online failures exceed the registered cap")
    result={
        "case":case,"seed":seed,"arm":arm,
        "actual_on_policy_new_analytical_evaluations":True,
        "online_calls":online_calls,"maximum_online_calls":max_calls,
        "observed_nonfeasible":failed,
        "initial_best":_status(initial),"final_best":_status(history),
        "initial_prefix_sha256":initial_hash,
        "best_feasible_at_each_actual_call":best,
        "first_online_feasible_iteration":first,
        "decision_receipts":decision_receipts,
        "evaluated_path_sha256":digest(evaluated_ids),
        "decision_path_sha256":digest(decision_ids),
        "six_call_maximum_horizon":MAX_ONLINE_CALLS,
        "forecast_internal_horizon":ENGINE_PLANNING_HORIZON,
        "stop_reason":stop_reason,
        "mode_counts_actual_evaluations":dict(mode_counts),
        "candidate_points_screened":total_screened,
        "fit_and_analytical_evaluation_seconds":time.perf_counter()-tic,
        "evidence_before_result_was_future_blind":True,
        "probability_guarantee":False,
    }
    _write(output/"result.json",result)
    return result


def run(output,*,profile="smoke"):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("PR64 requires a new write-once output directory")
    sha=clean_commit()
    here=Path(__file__).resolve()
    if (_sha(here.with_name("risk_budget54_policy.py"))!=PR54_POLICY_SHA256 or
            _sha(here.with_name("risk_budget54_engine.py"))!=PR54_ENGINE_SHA256):
        raise ValueError("PR64 refuses modified frozen PR54 policy/engine")
    if (_sha(here.with_name("risk_budget63_local_value_policy.py"))!=PR63_POLICY_SHA256 or
            _sha(here.with_name("risk_budget63_local_value_engine.py"))!=PR63_ENGINE_SHA256):
        raise ValueError("PR64 refuses modified frozen PR63 GP-only soft utility source")
    spec=plan(profile,sha)
    max_calls=spec["maximum_online_calls_per_arm"]
    output.mkdir(parents=True,exist_ok=False)
    _write(output/"plan.json",spec)
    _write(output/"source_receipt.json",{
        "experiment_head_sha":sha,
        "pr64_protocol_sha256":_sha(here.with_name("risk_budget64_protocol.py")),
        "pr64_runner_sha256":_sha(here),
        "frozen_pr54_policy_sha256":PR54_POLICY_SHA256,
        "frozen_pr54_engine_sha256":PR54_ENGINE_SHA256,
        "frozen_pr63_soft_policy_sha256":PR63_POLICY_SHA256,
        "frozen_pr63_soft_engine_sha256":PR63_ENGINE_SHA256,
        "source_evaluator":"real custom MathematicalCase.evaluate",
        "no_pr45_to_pr63_historical_truth_read":True,
        "new_aircraft_calls":0,"new_rcaide_calls":0,
        "actual_abstention_before_problem_evaluate":True,
    })
    def forbidden(*_a,**_k):
        raise RuntimeError("PR64 mathematical-only study: aircraft evaluation forbidden")
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
                    trial=_trial(problem,case,seed,arm,initial,
                                 output/"trials"/case/str(seed)/arm,max_calls)
                    records.append(trial)
                    charged+=trial["online_calls"]
                    _append(output/"progress.jsonl",{
                        "case":case,"seed":seed,"arm":arm,
                        "maximum_online_calls":max_calls,
                        "actual_online_calls":trial["online_calls"],
                        "actual_online_nonfeasible":trial["observed_nonfeasible"],
                        "stop_reason":trial["stop_reason"],
                        "best_actual_feasible":trial["final_best"],
                        "evaluated_path_sha256":trial["evaluated_path_sha256"],
                        "decision_path_sha256":trial["decision_path_sha256"],
                    })
                    print("PR64",case,seed,arm,
                          "calls",trial["online_calls"],"/",max_calls,
                          "nonfeasible",trial["observed_nonfeasible"],
                          "stop",trial["stop_reason"],
                          "best",trial["final_best"],flush=True)
    if charged>spec["new_analytic_evaluation_ceiling"]:
        raise RuntimeError("PR64 actual analytical calls exceeded the frozen maximum ceiling")
    by_case,paired,shadow=summarize(records,profile=profile)
    report={
        "schema":SCHEMA,
        "status":"COMPLETE_FRESH_MAX_BUDGET_SELECTIVE_EVALUATION" if profile=="full" else
                 "SMOKE_MAX_BUDGET_SELECTIVE_EVALUATION_NOT_FULL_STUDY",
        "code_sha":sha,"plan":spec,
        "n_real_analytical_calls":charged,
        "maximum_analytic_evaluation_ceiling":spec["new_analytic_evaluation_ceiling"],
        "n_new_on_policy_trials":len(records),
        "n_actual_online_evaluations":sum(t["online_calls"] for t in records),
        "n_maximum_online_evaluations":len(records)*max_calls,
        "new_aircraft_calls":0,"new_rcaide_calls":0,
        "new_gp_vae_fits":"real actual fits, NOT zero",
        "same_maximum_real_call_budget":True,
        "actual_calls_variable_only_for_soft_gp_abstain512":True,
        "original_pr54_three_control_arms_byte_identical":True,
        "pr63_gp_only_soft_utility_source_byte_identical":True,
        "result_summary_by_case_arm":by_case,
        "paired_primary_and_secondary":paired,
        "abstention_shadow_audit":shadow,
        "interpretation_limits":[
            "This is an exploratory prospective new-seed study designed after PR63; not an independent preregistration.",
            "Four custom hierarchical Branin/Rosenbrock mathematical tasks only; not official benchmark replicas or aircraft/RCAIDE.",
            "The primary contrast isolates actual pre-evaluation abstention from always-evaluate using the exact same 512-candidate GP soft utility and same maximum six-call budget.",
            "The 128-vs-512 comparison changes cheap candidate screening breadth and must be interpreted separately from abstention.",
            "The soft utility and zero outside-option threshold remain heuristic; no expected monetary value or safety guarantee is established.",
            "Coverage actions with no risk forecast are evaluated as explicit unscored exploration; no fake probability is attached.",
            "A fixed512 shadow label for an abstained candidate is post-run counterfactual audit evidence only and is never available to the abstention algorithm.",
            "Actual evaluator calls differ by design; report both quality and real call cost, not equal realized-cost superiority.",
            "GP risk remains an uncalibrated marginal-probability product; no Safe BO or aircraft safety claim.",
            "All paired seeds, one-sided feasible outcomes, failures and abstentions are retained; objective means are per task because task scales differ.",
        ],
    }
    _write(output/"selective_evaluation_report.json",report)
    return report
