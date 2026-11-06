"""PR64 prospective max-budget protocol with actual candidate abstention.

Six arms share the same 12 truly evaluated initial designs and the same
maximum six online truth calls. Only soft_gp_abstain512 may decline an
unattractive GP-scored candidate BEFORE the evaluator is called. Fixed arms
continue to evaluate, even for negative net utility, as controls.

No unobserved label is ever attached to an abstained candidate. A separately
matched soft_gp_fixed512 arm supplies a post-run SHADOW outcome only for
counterfactual audit; that outcome is never fed to the abstention algorithm.
"""
from collections import Counter
import math
import statistics

SCHEMA="ra_lsbo_pr64_selective_evaluation_abstention_v1"
CASES=("branin_ordinary","branin_sparse",
       "rosenbrock_ordinary","rosenbrock_sparse")
ARMS=("legacy","semantic_only","budgeted",
      "soft_gp_fixed128","soft_gp_fixed512","soft_gp_abstain512")
SEEDS=tuple(range(997301,997325))
INITIAL=12
MAX_ONLINE_CALLS=6
ENGINE_PLANNING_HORIZON=12
OBSERVED_FAILURE_CAP=6
POOL_NARROW=128
POOL_WIDE=512
INIT_DRAW_BASE=698000000
ONLINE_DRAW_BASE=699000000
PR54_POLICY_SHA256="d2c32b77d096d46ca4d20b74d165b0e601877c3aa28741bc16a1f2bf255b1437"
PR54_ENGINE_SHA256="3f6105f073f01918b95de413a46cedda8b46c5aa56ced78b0eec8f1302c6a677"
PR63_POLICY_SHA256="af69caa7f4fce51497de4c2af8832643a051540ffc7f009e925863c3a4810a91"
PR63_ENGINE_SHA256="4b64f2afc16786d3adec1932c6311dc3d2863b5683fa875d9b68cf5a5116ebd9"

PAIRS=(
    ("soft_gp_abstain512","soft_gp_fixed512"),
    ("soft_gp_fixed512","soft_gp_fixed128"),
    ("soft_gp_abstain512","budgeted"),
    ("soft_gp_fixed512","budgeted"),
    ("budgeted","semantic_only"),
    ("soft_gp_abstain512","legacy"),
)


def plan(profile,sha):
    if profile not in ("smoke","full") or not isinstance(sha,str) or len(sha)!=40:
        raise ValueError("PR64 profile and immutable code SHA are required")
    seeds=SEEDS[:1] if profile=="smoke" else SEEDS
    maximum=2 if profile=="smoke" else MAX_ONLINE_CALLS
    max_calls=len(CASES)*len(seeds)*(INITIAL+len(ARMS)*maximum)
    return {
        "schema":SCHEMA,"source_sha":sha,"profile":profile,
        "cases":list(CASES),"arms":list(ARMS),"seeds":list(seeds),
        "initial":INITIAL,"maximum_online_calls_per_arm":maximum,
        "internal_planning_horizon_unmodified":ENGINE_PLANNING_HORIZON,
        "observed_failure_cap":OBSERVED_FAILURE_CAP,
        "narrow_candidate_pool":POOL_NARROW,"wide_candidate_pool":POOL_WIDE,
        "shared_initial_stream":INIT_DRAW_BASE,
        "online_candidate_stream":ONLINE_DRAW_BASE,
        "fresh_nonoverlap_with_pr54_pr55_pr58_pr60_pr63":all(
            seed not in range(995401,995409) and
            seed not in range(996101,996117) and
            seed not in range(997001,997025) and
            seed not in range(997101,997113) and
            seed not in range(997201,997213) for seed in seeds),
        "new_analytic_evaluation_ceiling":max_calls,
        "initial_shared_once_per_case_seed":True,
        "all_arms_same_maximum_real_call_budget":True,
        "actual_calls_may_differ_only_for_soft_gp_abstain512":True,
        "method_definition":{
            "legacy":"exact frozen PR54 original completion+constraints, 128 candidates",
            "semantic_only":"exact frozen PR54 constraint-only selector, 128 candidates",
            "budgeted":"exact frozen PR54 hard risk-budget selector, 128 candidates",
            "soft_gp_fixed128":"PR63 GP-only soft gain/failure-cost utility, 128 candidates; always evaluate",
            "soft_gp_fixed512":"same PR63 GP-only utility, 512 candidates; always evaluate",
            "soft_gp_abstain512":"same 512-candidate GP-only utility; if max eligible net utility <= 0, abstain before real evaluation and stop the trial",
        },
        "soft_gp_formula":"U(x)=G(x)*(1-r_GP(x))-lambda_t*r_GP(x)",
        "outside_option_utility":0.,
        "abstention_is_real_no_evaluator_call":True,
        "coverage_without_risk_forecast":"explicitly evaluate and label as unscored coverage; never fabricate risk probability",
        "shadow_outcome_source":"matched soft_gp_fixed512 same-case/seed/proposal-step candidate only; post-run audit, never visible to abstention arm",
        "primary_contrast":"soft_gp_abstain512_vs_soft_gp_fixed512 per task, 24 matched seeds",
        "secondary_contrasts":[
            "soft_gp_fixed512_vs_soft_gp_fixed128 (candidate-screening breadth)",
            "soft_gp_abstain512_vs_budgeted (selective evaluation vs hard budget)",
            "soft_gp_fixed512_vs_budgeted",
            "budgeted_vs_semantic_only",
        ],
        "predeclared_metrics":[
            "feasible-first wins/losses/ties at same maximum budget",
            "actual expensive evaluator calls and calls saved",
            "actual nonfeasible evaluations and failure fraction per actual call",
            "final feasible incumbent and cold-start feasibility discovery",
            "shadow outcome for the exact candidate abstained by soft_gp_abstain512",
            "model-scored versus model-free coverage calls",
            "selected GP risk, utility, and no-probability coverage receipts",
            "objective delta only when both paired final incumbents are feasible",
        ],
        "no_probability_safety_guarantee":True,
        "new_aircraft_calls":0,"new_rcaide_calls":0,
    }


def _finite(x):
    return type(x) in (int,float) and math.isfinite(x)


def _compare(a,b):
    x,y=a["final_best"],b["final_best"]
    if x is None and y is None:return 0
    if x is not None and y is None:return 1
    if x is None and y is not None:return -1
    return 1 if x<y-1e-10 else -1 if x>y+1e-10 else 0


def _mean(values):
    return statistics.mean(values) if values else None


def _median(values):
    return statistics.median(values) if values else None


def check_trial(row,*,profile):
    cap=2 if profile=="smoke" else MAX_ONLINE_CALLS
    arm=row.get("arm")
    if (row.get("case") not in CASES or row.get("seed") not in SEEDS or
            arm not in ARMS or type(row.get("online_calls")) is not int or
            not 0<=row["online_calls"]<=cap or
            (arm!="soft_gp_abstain512" and row["online_calls"]!=cap) or
            type(row.get("observed_nonfeasible")) is not int or
            not 0<=row["observed_nonfeasible"]<=row["online_calls"] or
            row.get("maximum_online_calls")!=cap or
            row.get("forecast_internal_horizon")!=ENGINE_PLANNING_HORIZON):
        raise ValueError("PR64 actual call count / registered roster / maximum budget invalid")
    decisions=row.get("decision_receipts")
    if not isinstance(decisions,list) or len(decisions) not in (
            row["online_calls"],row["online_calls"]+1):
        raise ValueError("PR64 decisions must include each evaluation and optional final abstention")
    if arm!="soft_gp_abstain512" and len(decisions)!=row["online_calls"]:
        raise ValueError("Only the abstention arm may have a non-evaluated final decision")
    abstained=[i for i,v in enumerate(decisions) if v.get("evaluated") is False]
    if row.get("stop_reason")=="outside_option_nonpositive":
        if (arm!="soft_gp_abstain512" or len(abstained)!=1 or
                abstained[0]!=len(decisions)-1 or
                row["online_calls"]>=cap):
            raise ValueError("PR64 invalid abstention action/stop reason")
    elif row.get("stop_reason")=="maximum_online_call_budget":
        if abstained or row["online_calls"]!=cap:
            raise ValueError("PR64 maximum-budget completion mismatch")
    else:
        raise ValueError("PR64 unknown stop reason")
    actual=[v for v in decisions if v.get("evaluated") is True]
    if len(actual)!=row["online_calls"]:
        raise ValueError("PR64 real evaluator receipts do not equal actual call count")
    expected_curve=row.get("best_feasible_at_each_actual_call")
    if (not isinstance(expected_curve,list) or len(expected_curve)!=row["online_calls"]+1 or
            expected_curve[0]!=row.get("initial_best") or
            expected_curve[-1]!=row.get("final_best")):
        raise ValueError("PR64 incumbent curve does not match actual evaluation count")
    if any(not (v is None or _finite(v)) for v in expected_curve):
        raise ValueError("PR64 incumbent curve contains invalid objective")
    for x,y in zip(expected_curve,expected_curve[1:]):
        if x is not None and (y is None or y>x+1e-9):
            raise ValueError("PR64 best feasible incumbent increased")
    failures=0
    call_index=0
    observed_first=None
    for i,r in enumerate(decisions):
        if (r.get("proposal_step")!=i or
                type(r.get("evaluated")) is not bool or
                type(r.get("forecast_available_for_selected")) is not bool or
                r.get("outcome_not_available_when_selected") is not True):
            raise ValueError("PR64 proposal/forecast/temporal receipt malformed")
        if r["forecast_available_for_selected"]:
            if (not _finite(r.get("predicted_nonfeasible")) or
                    not 0<=r["predicted_nonfeasible"]<=1):
                raise ValueError("PR64 scored candidate lacks finite risk proxy")
        elif r.get("predicted_nonfeasible") is not None:
            raise ValueError("PR64 coverage cannot fabricate a risk probability")
        if arm.startswith("soft_gp") and r["forecast_available_for_selected"]:
            if (r.get("no_hard_risk_gate") is not True or
                    not _finite(r.get("selected_net_utility")) or
                    type(r.get("would_defer_in_unconstrained_utility_model")) is not bool):
                raise ValueError("PR64 soft utility receipt is incomplete or reintroduced a hard gate")
        if r["evaluated"]:
            if (r.get("actual_call_index")!=call_index or
                    type(r.get("realized_nonfeasible")) is not int or
                    r["realized_nonfeasible"] not in (0,1) or
                    r.get("actual_action") not in (
                        "evaluate_under_maximum_budget",
                        "evaluate_coverage_without_risk_forecast")):
                raise ValueError("PR64 real observation lacks correct call/action receipt")
            failures+=r["realized_nonfeasible"]
            if observed_first is None and r["realized_nonfeasible"]==0:
                observed_first=call_index
            call_index+=1
        else:
            if (i!=len(decisions)-1 or
                    r.get("actual_call_index")!=call_index or
                    r.get("actual_action")!="abstain_before_evaluation" or
                    r.get("realized_nonfeasible") is not None or
                    r.get("would_defer_in_unconstrained_utility_model") is not True):
                raise ValueError("PR64 abstention must be pre-evaluation with UNKNOWN outcome")
    if failures!=row["observed_nonfeasible"] or call_index!=row["online_calls"]:
        raise ValueError("PR64 observed failures/calls do not reconcile")
    if row.get("first_online_feasible_iteration")!=observed_first:
        raise ValueError("PR64 first feasible call does not match actual labels")
    if (not isinstance(row.get("initial_prefix_sha256"),str) or
            len(row["initial_prefix_sha256"])!=64 or
            not isinstance(row.get("evaluated_path_sha256"),str) or
            len(row["evaluated_path_sha256"])!=64 or
            not isinstance(row.get("decision_path_sha256"),str) or
            len(row["decision_path_sha256"])!=64):
        raise ValueError("PR64 immutable initial/evaluated/decision identities missing")
    if not row.get("evidence_before_result_was_future_blind",False):
        raise ValueError("PR64 future-blind proposal contract absent")
    return row


def summarize(rows,*,profile):
    if profile not in ("smoke","full"):
        raise ValueError("Unknown PR64 profile")
    seeds=SEEDS[:1] if profile=="smoke" else SEEDS
    saved={}
    for t in rows:
        check_trial(t,profile=profile)
        key=(t["case"],t["seed"],t["arm"])
        if key in saved:raise ValueError("Duplicate PR64 trial key")
        saved[key]=t
    expect={(c,s,a) for c in CASES for s in seeds for a in ARMS}
    if set(saved)!=expect:raise ValueError("PR64 incomplete/mismatched task-seed-arm roster")
    by_case_arm={}
    paired={}
    shadow={}
    for case in CASES:
        for seed in seeds:
            group=[saved[case,seed,a] for a in ARMS]
            prefixes={r["initial_prefix_sha256"] for r in group}
            initial={r["initial_best"] for r in group}
            if len(prefixes)!=1 or len(initial)!=1:
                raise ValueError("PR64 algorithms do not share exact same initial evidence")
        for arm in ARMS:
            group=[saved[case,s,arm] for s in seeds]
            decisions=[r for t in group for r in t["decision_receipts"]]
            evaluated=[r for r in decisions if r["evaluated"]]
            scored=[r for r in evaluated if r["forecast_available_for_selected"]]
            coverage=[r for r in evaluated if not r["forecast_available_for_selected"]]
            abstained=[r for r in decisions if not r["evaluated"]]
            all_scored_decisions=[r for r in decisions if r["forecast_available_for_selected"]]
            negative=[r for r in all_scored_decisions if r["would_defer_in_unconstrained_utility_model"]]
            finals=[t["final_best"] for t in group if t["final_best"] is not None]
            by_case_arm[f"{case}/{arm}"]={
                "n_seed_groups":len(group),
                "initial_feasible":sum(t["initial_best"] is not None for t in group),
                "final_feasible":len(finals),
                "cold_start_feasible_discovered":sum(
                    t["initial_best"] is None and t["final_best"] is not None for t in group),
                "mean_exact_feasible_objective_conditional":_mean(finals),
                "median_exact_feasible_objective_conditional":_median(finals),
                "actual_online_calls_total":sum(t["online_calls"] for t in group),
                "maximum_online_calls_total":len(group)*(2 if profile=="smoke" else MAX_ONLINE_CALLS),
                "mean_actual_online_calls":_mean([t["online_calls"] for t in group]),
                "actual_nonfeasible_total":sum(t["observed_nonfeasible"] for t in group),
                "failure_fraction_per_actual_call":(
                    sum(t["observed_nonfeasible"] for t in group)/
                    max(1,sum(t["online_calls"] for t in group))),
                "abstentions":len(abstained),
                "scored_evaluations":len(scored),
                "coverage_evaluations_without_risk_probability":len(coverage),
                "negative_utility_scored_decisions":len(negative),
                "negative_utility_decisions":len(negative),
                "negative_utility_evaluated":sum(r["evaluated"] for r in negative),
                "negative_utility_abstained":sum(not r["evaluated"] for r in negative),
                "negative_utility_actual_failures":sum(
                    r["realized_nonfeasible"] for r in negative if r["evaluated"]),
                "no_safety_guarantee":True,
            }
        for a,b in PAIRS:
            comparisons=[]
            for seed in seeds:
                x,y=saved[case,seed,a],saved[case,seed,b]
                c=_compare(x,y)
                delta=(x["final_best"]-y["final_best"]
                       if x["final_best"] is not None and y["final_best"] is not None
                       else None)
                comparisons.append({
                    "seed":seed,"winner":a if c>0 else b if c<0 else "tied",
                    "candidate_final_feasible":x["final_best"] is not None,
                    "control_final_feasible":y["final_best"] is not None,
                    "objective_delta_if_both_feasible":delta,
                    "candidate_actual_calls":x["online_calls"],
                    "control_actual_calls":y["online_calls"],
                    "candidate_actual_nonfeasible":x["observed_nonfeasible"],
                    "control_actual_nonfeasible":y["observed_nonfeasible"],
                    "candidate_evaluated_path_sha256":x["evaluated_path_sha256"],
                    "control_evaluated_path_sha256":y["evaluated_path_sha256"],
                })
            ds=[r["objective_delta_if_both_feasible"] for r in comparisons
                if r["objective_delta_if_both_feasible"] is not None]
            key=f"{case}/{a}_vs_{b}"
            paired[key]={
                "pair_count":len(comparisons),
                "feasible_first_wins":sum(r["winner"]==a for r in comparisons),
                "feasible_first_losses":sum(r["winner"]==b for r in comparisons),
                "feasible_first_ties":sum(r["winner"]=="tied" for r in comparisons),
                "both_feasible":len(ds),
                "only_candidate_feasible":sum(
                    r["candidate_final_feasible"] and not r["control_final_feasible"]
                    for r in comparisons),
                "only_control_feasible":sum(
                    r["control_final_feasible"] and not r["candidate_final_feasible"]
                    for r in comparisons),
                "both_infeasible":sum(
                    not r["candidate_final_feasible"] and not r["control_final_feasible"]
                    for r in comparisons),
                "mean_objective_delta_if_both_feasible":_mean(ds),
                "median_objective_delta_if_both_feasible":_median(ds),
                "mean_actual_call_delta":_mean([
                    r["candidate_actual_calls"]-r["control_actual_calls"]
                    for r in comparisons]),
                "total_actual_call_delta":sum(
                    r["candidate_actual_calls"]-r["control_actual_calls"]
                    for r in comparisons),
                "total_actual_nonfeasible_delta":sum(
                    r["candidate_actual_nonfeasible"]-r["control_actual_nonfeasible"]
                    for r in comparisons),
                "same_maximum_call_budget":True,
                "per_seed":comparisons,
            }
    # The matched fixed512 arm is a post-run shadow only for the exact
    # candidate that the abstention arm declined. Never expose this label to
    # the abstention algorithm or mix it into its own observed failure count.
    shadow_rows=[]
    for case in CASES:
        for seed in seeds:
            abst=saved[case,seed,"soft_gp_abstain512"]
            fixed=saved[case,seed,"soft_gp_fixed512"]
            for decision in abst["decision_receipts"]:
                if decision["evaluated"]:
                    continue
                match=next((r for r in fixed["decision_receipts"]
                            if r["proposal_step"]==decision["proposal_step"]),None)
                same=bool(match and match["evaluated"] and
                          match["candidate_sha256"]==decision["candidate_sha256"])
                shadow_rows.append({
                    "case":case,"seed":seed,
                    "proposal_step":decision["proposal_step"],
                    "abstained_candidate_sha256":decision["candidate_sha256"],
                    "matched_same_candidate_in_fixed512":same,
                    "shadow_realized_nonfeasible":(
                        match["realized_nonfeasible"] if same else None),
                    "shadow_outcome_is_not_available_to_abstention_algorithm":True,
                })
    shadow={
        "n_abstentions":len(shadow_rows),
        "same_candidate_shadow_matches":sum(r["matched_same_candidate_in_fixed512"] for r in shadow_rows),
        "shadow_failures":sum(r["shadow_realized_nonfeasible"]==1 for r in shadow_rows
                              if r["matched_same_candidate_in_fixed512"]),
        "shadow_feasible_outcomes":sum(r["shadow_realized_nonfeasible"]==0 for r in shadow_rows
                                       if r["matched_same_candidate_in_fixed512"]),
        "rows":shadow_rows,
        "post_run_counterfactual_only":True,
    }
    return by_case_arm,paired,shadow
