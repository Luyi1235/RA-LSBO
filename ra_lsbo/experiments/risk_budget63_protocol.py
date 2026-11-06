"""PR63: frozen NEW six-arm candidate-local soft utility on-policy validation.

Three PR54 controls remain unchanged. Three pure acquisition-only new arms
share original GP/VAE and candidate pools; they differ ONLY in (i) soft
risk/gain utility, (ii) past-neighbor risk blending, (iii) a bounded and
promise-gated information bonus. There is NO hard risk gate in any new arm.

The study is knowingly devised after observing PR50–61 failures, thus
exploratory, not independent preregistered superiority or safety evidence.
All 6 arms perform SIX actual new online analytical evaluations per seed.
"""
from collections import Counter
import math
import statistics

SCHEMA = "ra_lsbo_pr63_candidate_credibility_soft_value_onpolicy_v1"
CASES = ("branin_ordinary", "branin_sparse",
         "rosenbrock_ordinary", "rosenbrock_sparse")
ARMS = ("legacy", "semantic_only", "budgeted",
        "soft_gp", "local_blend", "local_value")
SEEDS = tuple(range(997201, 997213))
INITIAL = 12
REAL_ONLINE_CALLS = 6
ENGINE_PLANNING_HORIZON = 12
OBSERVED_FAILURE_CAP = 6
POOL = 128
INIT_DRAW_BASE = 598000000
ONLINE_DRAW_BASE = 599000000
PR54_POLICY_SHA256 = "d2c32b77d096d46ca4d20b74d165b0e601877c3aa28741bc16a1f2bf255b1437"
PR54_ENGINE_SHA256 = "3f6105f073f01918b95de413a46cedda8b46c5aa56ced78b0eec8f1302c6a677"
PARENT_PR62_HEAD = "df9d2b5e8ea30d778e6754faf1912f09271b58ce"
PAIRS = (("local_value", "budgeted"),
         ("soft_gp", "semantic_only"),
         ("local_blend", "soft_gp"),
         ("local_value", "local_blend"),
         ("local_value", "soft_gp"),
         ("local_value", "legacy"),
         ("budgeted", "semantic_only"))


def plan(profile, sha):
    if profile not in ("smoke", "full") or not isinstance(sha, str) or len(sha) != 40:
        raise ValueError("Registered PR63 smoke/full and code SHA are mandatory")
    seeds = list(SEEDS[:1] if profile == "smoke" else SEEDS)
    limit = 2 if profile == "smoke" else REAL_ONLINE_CALLS
    return {
        "schema": SCHEMA,
        "source_sha": sha,
        "profile": profile,
        "cases": list(CASES), "arms": list(ARMS), "seeds": seeds,
        "initial": INITIAL, "actual_online_calls_per_arm": limit,
        "fixed_full_online_calls": REAL_ONLINE_CALLS,
        "engine_internal_horizon_unmodified": ENGINE_PLANNING_HORIZON,
        "observed_failure_cap": OBSERVED_FAILURE_CAP,
        "pool_per_online_iteration": POOL,
        "shared_initial_stream": INIT_DRAW_BASE,
        "online_candidate_stream": ONLINE_DRAW_BASE,
        "fresh_nonoverlap_with_pr54_pr55_pr58_pr60": all(
            seed not in range(995401,995409) and
            seed not in range(996101,996117) and
            seed not in range(997001,997025) and
            seed not in range(997101,997113) for seed in seeds),
        "new_analytic_evaluation_ceiling": len(CASES) * len(seeds) *
            (INITIAL + len(ARMS) * limit),
        "new_reduced_aircraft_calls": 0, "new_rcaide_calls": 0,
        "immutable_policy_sha256": PR54_POLICY_SHA256,
        "immutable_engine_sha256": PR54_ENGINE_SHA256,
        "matched_same_six_real_evaluations": profile == "full",
        "initial_shared_once_per_case_seed": True,
        "arms_train_their_own_real_observed_history": True,
        "method_definition": {
            "legacy": "exact PR54 original completion+constraints",
            "semantic_only": "exact PR54 constraint-only ranking",
            "budgeted": "exact PR54 hard risk gate and fallback",
            "soft_gp": "soft expected gain minus GP failure cost, no hard risk gate",
            "local_blend": "soft_gp plus continuous active-metric local observed risk blend",
            "local_value": "local_blend plus small benefit-gated candidate information bonus",
        },
        "local_support_length_scale": .25,
        "local_risk_prior": .5,
        "local_prior_mass": 1.,
        "local_trust_pseudo_mass": 2.,
        "soft_failure_price_base": .15,
        "soft_failure_price_slope_per_six_observed_failures": .25,
        "bounded_information_bonus_coefficient": .08,
        "soft_choice_always_evaluates_in_six_call_experiment": True,
        "model_may_flag_unattractive_evaluation_without_actual_abstaining": True,
        "new_arms_never_use_a_hard_risk_probability_gate": True,
        "local_support_uses_only_candidate_specific_pre_outcome_active_continuous_neighbors": True,
        "primary_contrast": "local_value_vs_budgeted_per_task",
        "predeclared_primary_metric": "per task 12-paired-seed feasible-first wins/losses/ties of local_value vs budgeted",
        "predeclared_secondary_metrics": [
            "soft_gp vs semantic_only; local_blend vs soft_gp; local_value vs local_blend",
            "paired both-feasible objective deltas and all-seed unique feasibility discovery",
            "real nonfeasible counts and loss of actual known feasible incumbents",
            "true GP risk and candidate-local blended risk scored on each arm's own selections",
            "actual source of model-free coverage and abstention counterfactual flags",
            "soft-value regret cases where all candidate scores are below outside option zero",
        ],
        "design_after_pr61_is_exploratory": True,
        "no_probability_safety_guarantee": True,
    }


def _valid_score(x):
    return x is None or (type(x) in (int,float) and math.isfinite(x))


def check_trial(row, *, profile):
    cap = 2 if profile == "smoke" else REAL_ONLINE_CALLS
    if (row.get("case") not in CASES or row.get("seed") not in SEEDS or
        row.get("arm") not in ARMS or type(row.get("online_calls")) is not int or
        row["online_calls"] != cap or
        type(row.get("observed_nonfeasible")) is not int or
        row["observed_nonfeasible"] not in range(cap+1) or
        row.get("six_call_horizon") != REAL_ONLINE_CALLS or
        row.get("forecast_internal_horizon") != ENGINE_PLANNING_HORIZON):
        raise ValueError("PR63 actual fixed-call trial roster/observation budget changed")
    curve=row.get("best_feasible_at_each_actual_call")
    if not isinstance(curve,list) or len(curve)!=cap+1:
        raise ValueError("PR63 incumbent prefix must match actual call count")
    if (curve[0] != row.get("initial_best") or curve[-1] != row.get("final_best") or
        any(not _valid_score(x) for x in curve)):
        raise ValueError("PR63 initial/final objective or finite evidence mismatch")
    for a,b in zip(curve,curve[1:]):
        if a is not None and (b is None or b > a+1e-9):
            raise ValueError("PR63 known best feasible objective increased")
    receipts=row.get("risk_receipts")
    if not isinstance(receipts,list) or len(receipts)!=cap or [
        r.get("iteration") for r in receipts] != list(range(cap)):
        raise ValueError("PR63 sequential online evidence missing or reordered")
    if (any(type(r.get("realized_nonfeasible")) is not int or
             r["realized_nonfeasible"] not in (0,1) for r in receipts) or
        sum(r["realized_nonfeasible"] for r in receipts)!=row["observed_nonfeasible"]):
        raise ValueError("PR63 actual nonfeasible count differs from serial receipts")
    for i,(before,after) in enumerate(zip(curve,curve[1:])):
        receipt=receipts[i]
        forecast=receipt.get("forecast_available_for_selected")
        predicted=receipt.get("predicted_nonfeasible")
        if type(forecast) is not bool:
            raise ValueError("PR63 explicit forecast coverage must be Boolean")
        if (forecast and
                (type(predicted) not in (int,float) or not math.isfinite(predicted)
                 or not 0 <= predicted <= 1)):
            raise ValueError("PR63 selected GP risk proxy is missing, nonfinite or invalid")
        if not forecast and predicted is not None:
            raise ValueError("PR63 model-free coverage cannot fabricate probability")
        if receipt["realized_nonfeasible"]==1 and before != after:
            raise ValueError("PR63 infeasible actual evaluation cannot improve feasible objective")
        if row["arm"] in ("soft_gp","local_blend","local_value") and forecast:
            ep=receipt.get("selected_local_evidence")
            raw=receipt.get("original_gp_predicted_failure")
            utility=receipt.get("selected_net_utility")
            if (not isinstance(ep,dict) or
                    type(raw) not in (float,int) or not math.isfinite(raw) or
                    not 0<=raw<=1 or
                    not isinstance(utility,(int,float)) or not math.isfinite(utility) or
                    type(ep.get("same_discrete_branch_observations")) is not int or
                    ep["same_discrete_branch_observations"]>INITIAL+i or
                    not 0<=ep.get("local_trust_weight",-1)<=1 or
                    not 0<=ep.get("candidate_local_ambiguity",-1)<=1 or
                    not 0<=ep.get("candidate_blended_failure_proxy",-1)<=1):
                raise ValueError("PR63 actual GP selected candidate lacks valid local trust/utility receipt")
            if (row["arm"]=="soft_gp" and
                    abs(predicted-raw)>1e-9):
                raise ValueError("GP-only arm unexpectedly changed raw GP risk")
            if (row["arm"]!="soft_gp" and
                    abs(predicted-ep["candidate_blended_failure_proxy"])>1e-9):
                raise ValueError("Local-fusion arm risk doesn't match candidate-specific receipt")
            if receipt.get("no_hard_risk_gate") is not True or
                    receipt.get("actual_action")!="evaluate_under_frozen_six_call_protocol":
                raise ValueError("Soft value arm illegally enabled a hard gate or abstention")
    first=row.get("first_online_feasible_iteration")
    if first is not None and (type(first) is not int or not 0 <= first < cap):
        raise ValueError("PR63 first online feasible discovery out of range")
    observed_first=next((i for i,r in enumerate(receipts)
                         if r["realized_nonfeasible"]==0),None)
    if first!=observed_first:
        raise ValueError("PR63 first feasible discovery differs from actual outcome receipts")
    if row["initial_best"] is None:
        derived=next((i for i,v in enumerate(curve[1:]) if v is not None),None)
        if first!=derived:
            raise ValueError("PR63 first feasible discovery not bound to actual evaluation")
    if type(row.get("initial_prefix_sha256")) is not str or len(row["initial_prefix_sha256"])!=64:
        raise ValueError("PR63 shared initial identity missing")
    if type(row.get("selected_path_sha256")) is not str or len(row["selected_path_sha256"])!=64:
        raise ValueError("PR63 actual selected path digest missing")
    if not row.get("evidence_before_result_was_future_blind",False):
        raise ValueError("No registered future-blind proposal/evaluation contract")
    return row


def _compare(a,b):
    x,y=a["final_best"],b["final_best"]
    if x is None and y is None:return 0
    if x is not None and y is None:return 1
    if x is None and y is not None:return -1
    return (1 if x < y - 1e-10 else
           -1 if x > y + 1e-10 else 0)


def _mean(values):
    return statistics.mean(values) if values else None


def _median(values):
    return statistics.median(values) if values else None


def summarize(rows, *, profile):
    if profile not in ("full","smoke"):
        raise ValueError("Unknown PR63 registration")
    seeds=SEEDS[:1] if profile=="smoke" else SEEDS
    saved={}
    for t in rows:
        check_trial(t,profile=profile)
        key=(t["case"],t["seed"],t["arm"])
        if key in saved:
            raise ValueError("Duplicate PR63 actual on-policy trial key")
        saved[key]=t
    expect={(c,s,a) for c in CASES for s in seeds for a in ARMS}
    if set(saved)!=expect:
        raise ValueError("Wrong PR63 actual task/seed/arm roster")
    case_summary={}
    pair_summary={}
    for case in CASES:
        for seed in seeds:
            vals=[saved[case,seed,arm] for arm in ARMS]
            ids=[v.get("initial_prefix_sha256") for v in vals]
            if not ids[0] or len(set(ids))!=1 or len({
                    v["initial_best"] for v in vals})!=1:
                raise ValueError("PR63 matched initial observations are not identical")
        for arm in ARMS:
            group=[saved[case,s,arm] for s in seeds]
            scores=[t["final_best"] for t in group if t["final_best"] is not None]
            model_receipts=[r for t in group for r in t["risk_receipts"]
                            if r.get("forecast_available_for_selected")]
            fallbacks=[r for r in model_receipts if str(
                       r.get("budget_rule","")).startswith("no_candidate_below_allowance_")]
            defer=[r for r in model_receipts if
                   r.get("would_defer_in_unconstrained_utility_model") is True]
            invalid=[r for t in group for r in t["risk_receipts"]
                     if r.get("forecast_available_for_selected") and
                     (r.get("predicted_nonfeasible") is None or
                      not 0<=r["predicted_nonfeasible"]<=1)]
            if invalid:raise ValueError("Risk probabilities must be finite valid if present")
            case_summary[case+"/"+arm]={
                "n_independent_seed_groups":len(group),
                "initial_feasible":sum(t["initial_best"] is not None for t in group),
                "final_feasible":len(scores),
                "initially_infeasible":sum(t["initial_best"] is None for t in group),
                "cold_start_feasible_discovered":sum(
                    t["initial_best"] is None and t["final_best"] is not None for t in group),
                "mean_exact_feasible_objective_conditional":_mean(scores),
                "median_exact_feasible_objective_conditional":_median(scores),
                "mean_actual_online_nonfeasible":_mean([
                    t["observed_nonfeasible"] for t in group]),
                "online_calls_per_seed":group[0]["online_calls"],
                "risk_forecast_receipts":len(model_receipts),
                "risk_gate_fallbacks":len(fallbacks),
                "fallback_observed_nonfeasible":sum(
                    r["realized_nonfeasible"] for r in fallbacks),
                "soft_outside_option_preferred_but_evaluated":len(defer),
                "actual_failure_after_soft_outside_option_preferred":sum(
                    r["realized_nonfeasible"] for r in defer),
                "original_gp_predicted_failure_mean_on_soft_model_calls":_mean([
                    r["original_gp_predicted_failure"] for r in model_receipts
                    if r.get("original_gp_predicted_failure") is not None]),
                "local_evidence_mass_mean_on_soft_model_calls":_mean([
                    r["selected_local_evidence"]["weighted_local_support_mass"]
                    for r in model_receipts
                    if r.get("selected_local_evidence") is not None]),
                "forecasted_nonfeasible_mean":_mean([
                    r["predicted_nonfeasible"] for r in model_receipts]),
                "realized_nonfeasible_on_scored_calls":_mean([
                    r["realized_nonfeasible"] for r in model_receipts]),
                "no_safety_guarantee":True,
                "phase_selected_mode_counts":dict(Counter(
                    r.get("mode") for t in group for r in t["risk_receipts"])),
            }
        for candidate,control in PAIRS:
            comparisons=[]
            for seed in seeds:
                x=saved[case,seed,candidate]
                y=saved[case,seed,control]
                outcome=_compare(x,y)
                d=(x["final_best"]-y["final_best"]
                   if x["final_best"] is not None and y["final_best"] is not None
                   else None)
                comparisons.append({
                    "seed":seed,"winner":candidate if outcome>0 else
                        control if outcome<0 else "tied",
                    "candidate_final_feasible":x["final_best"] is not None,
                    "control_final_feasible":y["final_best"] is not None,
                    "objective_delta_if_both_feasible":d,
                    "candidate_online_nonfeasible":x["observed_nonfeasible"],
                    "control_online_nonfeasible":y["observed_nonfeasible"],
                    "candidate_selected_path_sha256":x["selected_path_sha256"],
                    "control_selected_path_sha256":y["selected_path_sha256"],
                })
            deltas=[x["objective_delta_if_both_feasible"] for x in comparisons
                    if x["objective_delta_if_both_feasible"] is not None]
            pair_summary[f"{case}/{candidate}_vs_{control}"]={
                "pair_count":len(comparisons),
                "primary_all_seed_feasible_first_wins":sum(
                    r["winner"]==candidate for r in comparisons),
                "primary_all_seed_feasible_first_losses":sum(
                    r["winner"]==control for r in comparisons),
                "primary_all_seed_feasible_first_ties":sum(
                    r["winner"]=="tied" for r in comparisons),
                "both_feasible":len(deltas),
                "only_candidate_feasible":sum(
                    r["candidate_final_feasible"] and not r["control_final_feasible"]
                    for r in comparisons),
                "only_control_feasible":sum(
                    r["control_final_feasible"] and not r["candidate_final_feasible"]
                    for r in comparisons),
                "both_infeasible":sum(
                    not r["candidate_final_feasible"] and not r["control_final_feasible"]
                    for r in comparisons),
                "mean_objective_delta_if_both_feasible":_mean(deltas),
                "median_objective_delta_if_both_feasible":_median(deltas),
                "mean_observed_nonfeasible_delta":_mean([
                    r["candidate_online_nonfeasible"]-
                    r["control_online_nonfeasible"] for r in comparisons]),
                "changed_actual_selected_paths":sum(
                    r["candidate_selected_path_sha256"]!=
                    r["control_selected_path_sha256"] for r in comparisons),
                "per_seed":comparisons,
                "n_real_online_calls_per_arm":2 if profile=="smoke" else REAL_ONLINE_CALLS,
                "same_real_online_call_count":True,
            }
    return case_summary,pair_summary
