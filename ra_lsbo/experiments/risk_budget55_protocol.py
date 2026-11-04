"""PR55 predeclared independent on-policy risk-budget generalization protocol.

Pure Python: no aircraft, GP/VAE, or physics model imported. Freeze the exact
four mathematical tasks, independent seeds, controls, and outcome accounting
BEFORE the workstation runs any new analytical evaluations.
"""
import math
import statistics

SCHEMA = "ra_lsbo_pr55_independent_math_budget_generalization_v1"
PARENT_PR54_SHA = "fd5492d33b21421ff8f54ad8bba453543caf39fe"
PARENT_PR54_RESULTS = "41db5094f29645cf5afd17b08e65ff62544d3f41"
CASES = ("branin_ordinary", "branin_sparse",
         "rosenbrock_ordinary", "rosenbrock_sparse")
ARMS = ("legacy", "semantic_only", "budgeted")
SEEDS = tuple(range(996101, 996117))
INITIAL = 12
ONLINE = 12
OBSERVED_FAILURE_CAP = 6
POOL_SIZE = 128
SEED_INITIAL_STREAM = 296000000
SEED_ONLINE_STREAM = 297000000
PAIRINGS = (("budgeted", "semantic_only"),
            ("semantic_only", "legacy"),
            ("budgeted", "legacy"))


def registration(profile, source_sha):
    if profile not in ("smoke", "full") or not isinstance(source_sha, str) or len(source_sha) != 40:
        raise ValueError("PR55 requires frozen source SHA and smoke/full profile")
    seeds = list(SEEDS[:1] if profile == "smoke" else SEEDS)
    steps = 2 if profile == "smoke" else ONLINE
    return {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "profile": profile,
        "analytical_cases": list(CASES),
        "arms": list(ARMS),
        "seeds": seeds,
        "shared_initial": INITIAL,
        "online_horizon": ONLINE,
        "online_steps_to_execute": steps,
        "observed_online_nonfeasible_stop": OBSERVED_FAILURE_CAP,
        "candidate_pool_size": POOL_SIZE,
        "initial_stream_base": SEED_INITIAL_STREAM,
        "online_stream_base": SEED_ONLINE_STREAM,
        "shared_initial_evaluated_once_per_case_seed": True,
        "independent_of_pr54_seeds": all(s not in range(995401, 995409) for s in seeds),
        "new_analytical_evaluation_cap": len(CASES) * len(seeds) *
            (INITIAL + len(ARMS) * steps),
        "new_aircraft_evaluations": 0,
        "new_rcaide_evaluations": 0,
        "methods": {
            "legacy": "original PR54/PR45 dual-view selection and record-status factor",
            "semantic_only": "constraint-only posterior risk ranking",
            "budgeted": "same constraint-only ranking plus predicted remaining-failure allowance",
        },
        "fixed_algorithm_source_pr54_sha": PARENT_PR54_SHA,
        "fixed_design_intent": "independent-seed and Rosenbrock transfer of PR54 mechanism",
        "selection_rule_unchanged_from_pr54": True,
        "same_observed_failure_stop_rule_all_arms": True,
        "source_outcomes_used_as_training": "each method only its own previously settled analytical observations",
        "predeclared_science": "exploratory on-policy extension, not independent qualified aircraft evidence",
    }


def _good_finite(value):
    return value is None or (type(value) in (int, float) and math.isfinite(value))


def validate_trial(row, steps=ONLINE):
    if (row.get("case") not in CASES or row.get("arm") not in ARMS or
            row.get("seed") not in SEEDS or
            type(row.get("online_calls")) is not int or
            not 0 <= row["online_calls"] <= steps or
            type(row.get("online_nonfeasible")) is not int or
            not 0 <= row["online_nonfeasible"] <=
            min(OBSERVED_FAILURE_CAP, row["online_calls"])):
        raise ValueError("Wrong PR55 task/seed/trial observed-budget evidence")
    initial, final = row.get("initial_best"), row.get("final_best")
    if not _good_finite(initial) or not _good_finite(final):
        raise ValueError("Nonfinite feasible objective")
    if initial is not None and (final is None or final > initial + 1e-9):
        raise ValueError("Best feasible incumbent cannot vanish or regress")
    curve = row.get("best_feasible_after_each_actual_online_call")
    if not isinstance(curve, list) or len(curve) != row["online_calls"] + 1:
        raise ValueError("Incorrect number of recorded online incumbents")
    if curve[0] != initial or curve[-1] != final:
        raise ValueError("Initial/terminal objective not linked to sequential curve")
    prior = None
    for value in curve:
        if not _good_finite(value):
            raise ValueError("Nonnumeric online incumbent curve")
        if prior is not None and (value is None or value > prior + 1e-9):
            raise ValueError("Online best objective increased")
        prior = value
    first = row.get("first_online_feasible_iteration")
    if first is not None and (type(first) is not int or
                              not 0 <= first < row["online_calls"]):
        raise ValueError("Invalid first online feasible iteration")
    if (initial is None and final is not None and first is None) or (
            first is not None and final is None):
        raise ValueError("Feasible discovery record inconsistent with actual incumbent")
    if row["online_nonfeasible"] < OBSERVED_FAILURE_CAP and row["online_calls"] < steps:
        raise ValueError("Trial stopped without reaching failure cap or horizon")
    if row["online_calls"] < steps and row["online_nonfeasible"] != OBSERVED_FAILURE_CAP:
        raise ValueError("Early stop must have observed six online failures")
    if len(row.get("pre_evaluation_risk_receipts", [])) != row["online_calls"]:
        raise ValueError("Recorded candidate receipts do not cover all online evaluations")
    seen = [r.get("iteration") for r in row["pre_evaluation_risk_receipts"]]
    if seen != list(range(row["online_calls"])):
        raise ValueError("Duplicate or missing sequential risk receipt")
    return row


def summarize(results, profile):
    if profile not in ("smoke", "full"):
        raise ValueError("Unknown PR55 profile")
    seeds = list(SEEDS[:1] if profile == "smoke" else SEEDS)
    steps = 2 if profile == "smoke" else ONLINE
    found = {}
    for row in results:
        validate_trial(row, steps)
        key = row["case"], row["seed"], row["arm"]
        if key in found:
            raise ValueError("Duplicate case/seed/arm trial")
        found[key] = row
    expected = {(case,seed,arm) for case in CASES for seed in seeds for arm in ARMS}
    if set(found) != expected:
        raise ValueError("Missing/foreign trial for the predeclared 4-case 3-arm roster")

    by_arm, comparisons = {}, {}
    for case in CASES:
        for arm in ARMS:
            trials = [found[case, seed, arm] for seed in seeds]
            final = [r["final_best"] for r in trials if r["final_best"] is not None]
            cold = [r for r in trials if r["initial_best"] is None]
            risk = [rcpt for r in trials
                    for rcpt in r["pre_evaluation_risk_receipts"]
                    if rcpt.get("forecast_available")]
            by_arm[f"{case}/{arm}"] = {
                "n_seeds":len(trials),
                "initial_feasible":sum(r["initial_best"] is not None for r in trials),
                "final_feasible":len(final),
                "initially_infeasible_groups":len(cold),
                "new_feasible_discoveries_in_initially_infeasible_groups":
                    sum(r["final_best"] is not None for r in cold),
                "first_new_feasible_online_iterations": [
                    dict(seed=r["seed"],iteration=r["first_online_feasible_iteration"])
                    for r in cold if r["final_best"] is not None],
                "mean_online_calls":statistics.mean(r["online_calls"] for r in trials),
                "mean_online_nonfeasible":statistics.mean(r["online_nonfeasible"] for r in trials),
                "failure_cap_reached":sum(r["online_nonfeasible"] == OBSERVED_FAILURE_CAP for r in trials),
                "feasible_final_mean_conditional":statistics.mean(final) if final else None,
                "feasible_final_median_conditional":statistics.median(final) if final else None,
                "predicted_risk_receipt_coverage":len(risk),
                "predicted_allowance_gate_fallbacks":sum(
                    x.get("budget_gate_fallback") ==
                    "no_candidate_below_allowance_min_predicted_failure" for x in risk),
                "observed_nonfeasible_in_predicted_gate_fallbacks":sum(
                    x.get("realized_nonfeasible") == 1 and
                    x.get("budget_gate_fallback") ==
                    "no_candidate_below_allowance_min_predicted_failure" for x in risk),
                "new_objective_improvement_groups":sum(
                    r["final_best"] is not None and r["initial_best"] is not None and
                    r["final_best"] < r["initial_best"] - 1e-10 for r in trials),
            }
        for arm, control in PAIRINGS:
            paired=[]
            for seed in seeds:
                a,b=found[case,seed,arm],found[case,seed,control]
                if a["initial_best"]!=b["initial_best"]:
                    raise ValueError("Comparators must use the same shared initial data")
                fa,fb=a["final_best"],b["final_best"]
                delta=fa-fb if fa is not None and fb is not None else None
                paired.append({
                    "seed":seed, "candidate_arm":arm, "control_arm":control,
                    "both_final_feasible":delta is not None,
                    "candidate_only_feasible":fa is not None and fb is None,
                    "control_only_feasible":fa is None and fb is not None,
                    "objective_difference_if_both_feasible":delta,
                    "online_calls_candidate_minus_control":
                        a["online_calls"]-b["online_calls"],
                    "online_nonfeasible_candidate_minus_control":
                        a["online_nonfeasible"]-b["online_nonfeasible"],
                    "candidate_reached_cap":a["online_nonfeasible"]==OBSERVED_FAILURE_CAP,
                    "control_reached_cap":b["online_nonfeasible"]==OBSERVED_FAILURE_CAP,
                    "same_full_selected_path_sha256":(
                        a["selected_path_digest"] == b["selected_path_digest"]),
                })
            differences = [r["objective_difference_if_both_feasible"] for r in paired
                           if r["objective_difference_if_both_feasible"] is not None]
            comparisons[f"{case}/{arm}_vs_{control}"] = {
                "independent_seed_groups":len(paired),
                "both_final_feasible":len(differences),
                "candidate_only_feasible":sum(r["candidate_only_feasible"] for r in paired),
                "control_only_feasible":sum(r["control_only_feasible"] for r in paired),
                "mean_objective_delta_if_both_feasible":statistics.mean(differences)
                    if differences else None,
                "median_objective_delta_if_both_feasible":statistics.median(differences)
                    if differences else None,
                "paired_objective_improved":sum(x < -1e-10 for x in differences),
                "paired_objective_worsened":sum(x > 1e-10 for x in differences),
                "paired_objective_tied":sum(abs(x) <= 1e-10 for x in differences),
                "mean_online_calls_delta":statistics.mean(
                    r["online_calls_candidate_minus_control"] for r in paired),
                "mean_online_nonfeasible_delta":statistics.mean(
                    r["online_nonfeasible_candidate_minus_control"] for r in paired),
                "full_selected_paths_differ":sum(
                    not r["same_full_selected_path_sha256"] for r in paired),
                "conditional_objective_comparison_not_equal_work":True,
                "paired_results":paired,
            }
    return by_arm, comparisons
