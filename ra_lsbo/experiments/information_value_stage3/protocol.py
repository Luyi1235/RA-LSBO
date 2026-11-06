"""Frozen, exploratory PR68 pilot; no historical labels and no aircraft calls."""
import math
import statistics

PARENT = "0cefd1f2507c355067f3c4f56d2dd43d919f65ef"
SCHEMA = "ra_lsbo_information_value_stage3_gp_fantasy_v1"
CASES = ("branin_ordinary", "branin_sparse", "rosenbrock_ordinary", "rosenbrock_sparse")
ARMS = ("soft_gp_eligible", "cei_myopic", "cei_two_step", "cei_two_step_no_abort_update")
FANTASIES_PER_BRANCH = 4
FIRST_ACTIONS = 4
POOL = 128
INITIAL = 12


def plan(profile):
    if profile not in ("ci", "smoke", "full"):
        raise ValueError("Registered ci/smoke/full profile required")
    cases = ("rosenbrock_sparse",) if profile == "ci" else CASES
    seeds = (998790,) if profile == "ci" else ((998791,) if profile == "smoke" else tuple(range(998801, 998805)))
    steps = 6 if profile == "full" else 2
    initial = len(cases) * len(seeds) * INITIAL
    online = len(cases) * len(seeds) * len(ARMS) * steps
    return {"schema": SCHEMA, "profile": profile, "cases": list(cases), "seeds": list(seeds),
            "arms": list(ARMS), "initial": INITIAL, "steps": steps, "pool": POOL,
            "initial_calls": initial, "online_calls": online, "total_calls": initial + online,
            "first_action_shortlist": FIRST_ACTIONS, "fantasies_per_region_branch": FANTASIES_PER_BRANCH,
            "new_policy_horizon": steps, "legacy_internal_horizon": 12,
            "primary_pair": ["cei_two_step", "cei_myopic"],
            "mechanism_pair": ["cei_two_step", "cei_two_step_no_abort_update"],
            "actual_feedback_full_in_all_arms": True, "no_stopping_or_failure_cap_before_planned_end": True,
            "fresh_from_pr54_to_pr67": True, "historical_result_reads": False,
            "aircraft_calls": 0, "rcaide_calls": 0,
            "scope": "Exploratory four-seed/task real GP pilot, not confirmatory superiority",
            "fantasy_approximation": "EP refit at current selected kernel, normalization and representation; no fantasy hyperparameter reselection or VAE training",
            "cost_contract": "Equal actual evaluation calls; surrogate/fantasy wall time is reported separately"}


def compare(x, y):
    for value in (x, y):
        if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
            raise ValueError("Invalid objective; missing is not a numeric penalty")
    if x is None:
        return 0 if y is None else -1
    if y is None:
        return 1
    return 1 if x < y - 1e-10 else (-1 if x > y + 1e-10 else 0)


def summarize(rows, spec):
    indexed = {(r["case"], r["seed"], r["arm"]): r for r in rows}
    expected = {(c, s, a) for c in spec["cases"] for s in spec["seeds"] for a in ARMS}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Incomplete or duplicate trial roster")
    output = {}
    for case in spec["cases"]:
        for seed in spec["seeds"]:
            paired = [indexed[case, seed, a] for a in ARMS]
            if len({r["initial_hash"] for r in paired}) != 1:
                raise ValueError("Initial evidence differs between arms")
        result = {"arms": {}, "pairs": {}}
        for arm in ARMS:
            group = [indexed[case, s, arm] for s in spec["seeds"]]
            values = [r["final_best"] for r in group if r["final_best"] is not None]
            result["arms"][arm] = {
                "trials": len(group), "final_feasible": len(values),
                "initial_feasible": sum(r["initial_best"] is not None for r in group),
                "cold_discoveries": sum(r["initial_best"] is None and r["final_best"] is not None for r in group),
                "online_calls": sum(r["online_calls"] for r in group),
                "nonfeasible": sum(r["nonfeasible"] for r in group),
                "conditional_objective_median": statistics.median(values) if values else None,
                "conditional_objective_mean": statistics.mean(values) if values else None,
                "proposal_seconds": sum(r["proposal_seconds"] for r in group),
                "fantasy_head_fits_started": sum(r["fantasy_head_fits_started"] for r in group),
                "fantasy_head_fits_completed": sum(r["fantasy_head_fits_completed"] for r in group),
                "fantasy_fallbacks": sum(r["fantasy_fallbacks"] for r in group),
                "model_free_calls": sum(r["model_free_calls"] for r in group)}
        for left, right in ((ARMS[2], ARMS[1]), (ARMS[2], ARMS[3]), (ARMS[1], ARMS[0]), (ARMS[2], ARMS[0])):
            outcomes, deltas, changed = [], [], 0
            for seed in spec["seeds"]:
                a, b = indexed[case, seed, left], indexed[case, seed, right]
                outcomes.append(compare(a["final_best"], b["final_best"]))
                if a["final_best"] is not None and b["final_best"] is not None:
                    deltas.append(a["final_best"] - b["final_best"])
                changed += a["path_hash"] != b["path_hash"]
            result["pairs"][left + "_vs_" + right] = {
                "wins_losses_ties": [outcomes.count(1), outcomes.count(-1), outcomes.count(0)],
                "both_feasible": len(deltas), "objective_delta_mean": statistics.mean(deltas) if deltas else None,
                "objective_delta_median": statistics.median(deltas) if deltas else None, "changed_paths": changed}
        output[case] = result
    return output
