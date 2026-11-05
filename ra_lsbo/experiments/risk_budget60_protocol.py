"""PR60: prospectively frozen equal-six-call on-policy mechanism verification.

PR54-59 results motivated this independent new-seed fixed-effort design.
The original PR54 acquisition and its internal 12-step planning horizon are
unchanged for the three controls; three new, fully opt-in branch-evidence
acquisitions also use the identical GP/VAE surrogate. All six strategies
execute SIX real online evaluations, without early-stop censoring.

New data and a different seed block are used. All rankings and evaluations
happen on-policy, with no counterfactual labels or post-stop extrapolation.
This is an exploratory mathematical mechanism validation, NOT aircraft safety.
"""
from collections import Counter
import math
import statistics

SCHEMA = "ra_lsbo_pr60_branch_evidence_dispersion_onpolicy_v1"
CASES = ("branin_ordinary", "branin_sparse",
         "rosenbrock_ordinary", "rosenbrock_sparse")
ARMS = ("legacy", "semantic_only", "budgeted",
        "branch_mean", "branch_dispersion", "branch_explore")
SEEDS = tuple(range(997101, 997113))
INITIAL = 12
REAL_ONLINE_CALLS = 6
ENGINE_PLANNING_HORIZON = 12
OBSERVED_FAILURE_CAP = 6
POOL = 128
INIT_DRAW_BASE = 498000000
ONLINE_DRAW_BASE = 499000000
PR54_POLICY_SHA256 = "d2c32b77d096d46ca4d20b74d165b0e601877c3aa28741bc16a1f2bf255b1437"
PR54_ENGINE_SHA256 = "3f6105f073f01918b95de413a46cedda8b46c5aa56ced78b0eec8f1302c6a677"
PARENT_PR59_HEAD = "b3c29ccdcbef932ad961bc50f4bb688c92a5d48e"
PAIRS = (("budgeted", "semantic_only"),
         ("branch_mean", "budgeted"),
         ("branch_dispersion", "branch_mean"),
         ("branch_explore", "branch_dispersion"),
         ("branch_explore", "budgeted"),
         ("branch_dispersion", "budgeted"),
         ("branch_explore", "legacy"))


def plan(profile, sha):
    if profile not in ("smoke", "full") or not isinstance(sha, str) or len(sha) != 40:
        raise ValueError("Registered PR60 smoke/full and code SHA are mandatory")
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
        "fresh_nonoverlap_with_pr54_pr55_pr58": all(
            seed not in range(995401,995409) and
            seed not in range(996101,996117) and
            seed not in range(997001,997025) for seed in seeds),
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
            "semantic_only": "exact PR54 constraint-only re-ranking",
            "budgeted": "exact PR54 estimated-failure budget gating",
            "branch_mean": "GP risk upper-envelope with Beta(1,1) mean by discrete branch",
            "branch_dispersion": "same but Beta mean+one posterior standard deviation",
            "branch_explore": "branch_dispersion with explicit high-uncertainty fallback if no candidate passes",
        },
        "branch_beta_prior": [1.,1.],
        "branch_dispersion_multiplier": 1.,
        "branch_evidence_only_from_this_arm_settled_prefix": True,
        "predeclared_primary_metric": "paired feasible-first wins/losses/ties at six true online calls for each registered risk-UQ ablation",
        "predeclared_secondary_metrics": [
            "paired objective delta only where both have known feasible exact values",
            "pairwise real nonfeasible count difference at six charged calls",
            "new feasible discovery when shared initial set is all nonfeasible",
            "branch posterior mean/dispersion receipts from same-arm earlier outcomes",
            "fallback failure rate and scored-forecast-vs-realized failure gap",
        ],
        "design_after_pr59_is_exploratory": True,
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
        raise ValueError("PR60 actual fixed-call trial roster/observation budget changed")
    curve=row.get("best_feasible_at_each_actual_call")
    if not isinstance(curve,list) or len(curve)!=cap+1:
        raise ValueError("PR60 incumbent prefix must match actual call count")
    if (curve[0] != row.get("initial_best") or curve[-1] != row.get("final_best") or
        any(not _valid_score(x) for x in curve)):
        raise ValueError("PR60 initial/final objective or finite evidence mismatch")
    for a,b in zip(curve,curve[1:]):
        if a is not None and (b is None or b > a+1e-9):
            raise ValueError("PR60 known best feasible objective increased")
    receipts=row.get("risk_receipts")
    if not isinstance(receipts,list) or len(receipts)!=cap or [
        r.get("iteration") for r in receipts] != list(range(cap)):
        raise ValueError("PR60 sequential online evidence missing or reordered")
    if (any(type(r.get("realized_nonfeasible")) is not int or
             r["realized_nonfeasible"] not in (0,1) for r in receipts) or
        sum(r["realized_nonfeasible"] for r in receipts)!=row["observed_nonfeasible"]):
        raise ValueError("PR60 actual nonfeasible count differs from serial receipts")
    for i,(before,after) in enumerate(zip(curve,curve[1:])):
        receipt=receipts[i]
        forecast=receipt.get("forecast_available_for_selected")
        predicted=receipt.get("predicted_nonfeasible")
        if type(forecast) is not bool:
            raise ValueError("PR60 explicit forecast coverage must be Boolean")
        if (forecast and
                (type(predicted) not in (int,float) or not math.isfinite(predicted)
                 or not 0 <= predicted <= 1)):
            raise ValueError("PR60 selected GP risk proxy is missing, nonfinite or invalid")
        if not forecast and predicted is not None:
            raise ValueError("PR60 model-free coverage cannot fabricate probability")
        if receipt["realized_nonfeasible"]==1 and before != after:
            raise ValueError("PR60 infeasible actual evaluation cannot improve feasible objective")
        if row["arm"].startswith("branch_") and forecast:
            ep=receipt.get("selected_branch_posterior")
            raw=receipt.get("original_gp_predicted_failure")
            if (not isinstance(ep,dict) or
                type(ep.get("n_settled_in_branch")) is not int or
                ep["n_settled_in_branch"]<0 or
                type(raw) not in (float,int) or not math.isfinite(raw) or
                not 0<=raw<=1 or
                not (0<=ep.get("posterior_failure_mean",-1)<=1) or
                not (0<=ep.get("posterior_failure_sd",-1)<=1)):
                raise ValueError("PR60 branch posterior selection receipt absent or invalid")
    first=row.get("first_online_feasible_iteration")
    if first is not None and (type(first) is not int or not 0 <= first < cap):
        raise ValueError("PR60 first online feasible discovery out of range")
    observed_first=next((i for i,r in enumerate(receipts)
                         if r["realized_nonfeasible"]==0),None)
    if first!=observed_first:
        raise ValueError("PR60 first feasible discovery differs from actual outcome receipts")
    if row["initial_best"] is None:
        derived=next((i for i,v in enumerate(curve[1:]) if v is not None),None)
        if first!=derived:
            raise ValueError("PR60 first feasible discovery not bound to actual evaluation")
    if type(row.get("initial_prefix_sha256")) is not str or len(row["initial_prefix_sha256"])!=64:
        raise ValueError("PR60 shared initial identity missing")
    if type(row.get("selected_path_sha256")) is not str or len(row["selected_path_sha256"])!=64:
        raise ValueError("PR60 actual selected path digest missing")
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
        raise ValueError("Unknown PR60 registration")
    seeds=SEEDS[:1] if profile=="smoke" else SEEDS
    saved={}
    for t in rows:
        check_trial(t,profile=profile)
        key=(t["case"],t["seed"],t["arm"])
        if key in saved:
            raise ValueError("Duplicate PR60 actual on-policy trial key")
        saved[key]=t
    expect={(c,s,a) for c in CASES for s in seeds for a in ARMS}
    if set(saved)!=expect:
        raise ValueError("Wrong PR60 actual task/seed/arm roster")
    case_summary={}
    pair_summary={}
    for case in CASES:
        for seed in seeds:
            vals=[saved[case,seed,arm] for arm in ARMS]
            ids=[v.get("initial_prefix_sha256") for v in vals]
            if not ids[0] or len(set(ids))!=1 or len({
                    v["initial_best"] for v in vals})!=1:
                raise ValueError("PR60 matched initial observations are not identical")
        for arm in ARMS:
            group=[saved[case,s,arm] for s in seeds]
            scores=[t["final_best"] for t in group if t["final_best"] is not None]
            model_receipts=[r for t in group for r in t["risk_receipts"]
                            if r.get("forecast_available_for_selected")]
            fallbacks=[r for r in model_receipts if str(
                       r.get("budget_rule","")).startswith("no_candidate_below_allowance_")]
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
