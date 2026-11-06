"""PR64 max-budget (not equal-real-call) pre-evaluation abstention protocol."""
import math
import statistics

SCHEMA="ra_lsbo_pr64_soft_gp_real_abstention_max_budget_v1"
CASES=("branin_ordinary","branin_sparse","rosenbrock_ordinary","rosenbrock_sparse")
ARMS=("semantic_only","budgeted","soft_gp_continue","soft_gp_abstain")
SEEDS=tuple(range(998101,998113))
INITIAL=12
MAX_ONLINE_CALLS=6
SMOKE_CALLS=2
POOL=128
INIT_DRAW_BASE=698000000
ONLINE_DRAW_BASE=699000000
from .risk_budget60_protocol import PR54_POLICY_SHA256,PR54_ENGINE_SHA256

PAIRS=(("soft_gp_abstain","soft_gp_continue"),
       ("soft_gp_abstain","budgeted"),
       ("soft_gp_continue","budgeted"),
       ("budgeted","semantic_only"))


def plan(profile,sha):
    if profile not in ("smoke","full") or not isinstance(sha,str) or len(sha)!=40:
        raise ValueError("Registered PR64 smoke/full and fixed code SHA required")
    cap=SMOKE_CALLS if profile=="smoke" else MAX_ONLINE_CALLS
    seeds=SEEDS[:1] if profile=="smoke" else SEEDS
    max_calls=len(CASES)*len(seeds)*(INITIAL+len(ARMS)*cap)
    return {
        "schema":SCHEMA,"source_sha":sha,"profile":profile,
        "cases":list(CASES),"arms":list(ARMS),"seeds":list(seeds),
        "initial":INITIAL,"max_online_calls_per_arm":cap,
        "full_max_online_calls_per_arm":MAX_ONLINE_CALLS,
        "max_new_analytic_evaluation_budget":max_calls,
        "initial_shared_once_per_case_seed":True,
        "same_maximum_online_evaluation_budget":True,
        "actual_calls_may_be_lower_only_for_abstention_arm":True,
        "abstention_evaluator_call_count":0,
        "abstention_uses_outside_option_zero":True,
        "abstention_rule":"soft_gp maximum net utility <= 0; only when genuine GP risk receipt exists",
        "abstention_stops_run_no_restart":True,
        "no_unexecuted_evaluation_is_counted_as_success":True,
        "seed_block_disjoint_pr54_pr55_pr58_pr60_pr63":True,
        "immutable_pr54_policy_sha256":PR54_POLICY_SHA256,
        "immutable_pr54_engine_sha256":PR54_ENGINE_SHA256,
        "new_aircraft_calls":0,"new_rcaide_calls":0,
        "primary_contrast":"soft_gp_abstain_vs_soft_gp_continue",
        "secondary_contrasts":[list(x) for x in PAIRS[1:]],
        "primary_outcomes":[
            "per-case paired feasible-first wins/losses/ties at run end",
            "actual online evaluator calls and budget saved",
            "actual nonfeasible calls, never count abstentions as failures or successes",
            "final best observed feasible objective and cold-start discovery",
            "stopped runs that would have discovered feasibility later in continue arm",
        ],
        "no_probability_safety_guarantee":True,
        "exploratory_design_informed_by_PR63":True,
    }


def _cmp(a,b):
    x,y=a["final_best"],b["final_best"]
    if x is None and y is None:return 0
    if x is not None and y is None:return 1
    if x is None and y is not None:return -1
    return 1 if x<y-1e-10 else -1 if x>y+1e-10 else 0


def check_trial(row,*,profile):
    cap=SMOKE_CALLS if profile=="smoke" else MAX_ONLINE_CALLS
    arm=row.get("arm")
    if (row.get("case") not in CASES or row.get("seed") not in SEEDS or
        arm not in ARMS or type(row.get("max_online_calls")) is not int or
        row["max_online_calls"]!=cap or type(row.get("online_calls")) is not int or
        not 0<=row["online_calls"]<=cap):
        raise ValueError("PR64 trial violates registered maximum-call budget")
    curve=row.get("best_feasible_at_each_actual_call")
    if (not isinstance(curve,list) or len(curve)!=row["online_calls"]+1 or
        curve[0]!=row.get("initial_best") or curve[-1]!=row.get("final_best") or
        any(v is not None and (type(v) not in (int,float) or not math.isfinite(v))
            for v in curve)):
        raise ValueError("PR64 incumbent curve must include actual evaluations only")
    for a,b in zip(curve,curve[1:]):
        if a is not None and (b is None or b>a+1e-9):
            raise ValueError("PR64 feasible incumbent worsened")
    decisions=row.get("decisions")
    if not isinstance(decisions,list) or len(decisions) not in (
            row["online_calls"],row["online_calls"]+1):
        raise ValueError("PR64 decision roster must include actual calls and at most one stop")
    if sum(bool(d.get("evaluation_performed")) for d in decisions)!=row["online_calls"]:
        raise ValueError("PR64 actual evaluator count differs from decision receipts")
    abst=[d for d in decisions if d.get("action")=="abstain_stop_before_evaluator"]
    if arm=="soft_gp_abstain":
        if len(abst)>1 or (abst and decisions[-1] is not abst[0]):
            raise ValueError("PR64 abstention must occur once and terminate the run")
        if abst:
            d=abst[0]
            if d.get("evaluation_performed") is not False or d.get("realized_nonfeasible") is not None:
                raise ValueError("PR64 abstention must not create an evaluator result or label")
            if d.get("selected_net_utility") is None or d["selected_net_utility"]>0:
                raise ValueError("PR64 abstention without nonpositive selected utility")
    elif abst:
        raise ValueError("Only registered abstain arm may abstain")
    if row.get("observed_nonfeasible")!=sum(
        d.get("realized_nonfeasible")==1 for d in decisions if d.get("evaluation_performed")):
        raise ValueError("PR64 nonfeasible count includes unevaluated candidate or mismatches labels")
    if row.get("stop_reason") not in ("max_budget_exhausted","soft_utility_outside_option"):
        raise ValueError("PR64 unknown stop reason")
    if (row["stop_reason"]=="soft_utility_outside_option") != bool(abst):
        raise ValueError("PR64 stop reason and actual abstention disagree")
    if type(row.get("initial_prefix_sha256")) is not str or len(row["initial_prefix_sha256"])!=64:
        raise ValueError("PR64 shared initial digest missing")
    return row


def summarize(rows,*,profile):
    spec_seeds=SEEDS[:1] if profile=="smoke" else SEEDS
    saved={}
    for r in rows:
        check_trial(r,profile=profile)
        key=(r["case"],r["seed"],r["arm"])
        if key in saved:raise ValueError("Duplicate PR64 trial")
        saved[key]=r
    expect={(c,s,a) for c in CASES for s in spec_seeds for a in ARMS}
    if set(saved)!=expect:raise ValueError("PR64 trial roster incomplete")
    by={}
    pairs={}
    for c in CASES:
        for s in spec_seeds:
            ids={saved[c,s,a]["initial_prefix_sha256"] for a in ARMS}
            if len(ids)!=1:raise ValueError("PR64 shared initial prefix mismatch")
        for a in ARMS:
            group=[saved[c,s,a] for s in spec_seeds]
            vals=[x["final_best"] for x in group if x["final_best"] is not None]
            actual=sum(x["online_calls"] for x in group)
            budget=len(group)*(SMOKE_CALLS if profile=="smoke" else MAX_ONLINE_CALLS)
            by[f"{c}/{a}"]={
                "trials":len(group),
                "final_feasible":len(vals),
                "cold_start_feasible_discovered":sum(
                    x["initial_best"] is None and x["final_best"] is not None for x in group),
                "actual_online_calls":actual,"max_online_call_budget":budget,
                "budget_saved":budget-actual,
                "budget_saved_fraction":(budget-actual)/budget if budget else 0.,
                "observed_nonfeasible":sum(x["observed_nonfeasible"] for x in group),
                "stopped_by_abstention":sum(x["stop_reason"]=="soft_utility_outside_option" for x in group),
                "mean_actual_calls_per_trial":statistics.mean(x["online_calls"] for x in group),
                "mean_final_objective_if_feasible":statistics.mean(vals) if vals else None,
                "median_final_objective_if_feasible":statistics.median(vals) if vals else None,
                "all_seed_final_best":[x["final_best"] for x in group],
            }
        for cand,ctrl in PAIRS:
            records=[]
            for s in spec_seeds:
                x,y=saved[c,s,cand],saved[c,s,ctrl]
                k=_cmp(x,y)
                records.append({
                    "seed":s,"winner":cand if k>0 else ctrl if k<0 else "tied",
                    "candidate_final_best":x["final_best"],
                    "control_final_best":y["final_best"],
                    "candidate_actual_calls":x["online_calls"],
                    "control_actual_calls":y["online_calls"],
                    "candidate_nonfeasible":x["observed_nonfeasible"],
                    "control_nonfeasible":y["observed_nonfeasible"],
                    "candidate_abstained":x["stop_reason"]=="soft_utility_outside_option",
                    "control_abstained":y["stop_reason"]=="soft_utility_outside_option",
                    "both_feasible_objective_delta":(
                        x["final_best"]-y["final_best"]
                        if x["final_best"] is not None and y["final_best"] is not None else None),
                })
            pairs[f"{c}/{cand}_vs_{ctrl}"]={
                "pair_count":len(records),
                "wins":sum(x["winner"]==cand for x in records),
                "losses":sum(x["winner"]==ctrl for x in records),
                "ties":sum(x["winner"]=="tied" for x in records),
                "candidate_actual_calls":sum(x["candidate_actual_calls"] for x in records),
                "control_actual_calls":sum(x["control_actual_calls"] for x in records),
                "candidate_nonfeasible":sum(x["candidate_nonfeasible"] for x in records),
                "control_nonfeasible":sum(x["control_nonfeasible"] for x in records),
                "candidate_abstentions":sum(x["candidate_abstained"] for x in records),
                "both_feasible_objective_delta_mean":(
                    statistics.mean([x["both_feasible_objective_delta"] for x in records
                                     if x["both_feasible_objective_delta"] is not None])
                    if any(x["both_feasible_objective_delta"] is not None for x in records)
                    else None),
                "per_seed":records,
            }
    return by,pairs
