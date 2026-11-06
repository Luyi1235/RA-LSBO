"""Frozen PR67 protocol and pure support-mask helpers."""
import math

SCHEMA = "ra_lsbo_information_value_stage2_all_support_v1"
CASES = ("branin_ordinary", "branin_sparse", "rosenbrock_ordinary", "rosenbrock_sparse")
ARMS = ("soft_gp_eligible", "soft_gp_all_support")
SEEDS = tuple(range(998701, 998713))
INITIAL = 12
ONLINE_FULL = 6
POOL = 128


def all_candidate_support(original_eligible, candidate_count):
    """Remove only the optimistic-violation support mask, not legal-pool checks."""
    if (type(candidate_count) is not int or candidate_count < 1 or
            not isinstance(original_eligible, list) or
            len(original_eligible) != candidate_count or
            any(type(v) is not bool for v in original_eligible)):
        raise ValueError("Misaligned original support mask")
    return [True] * candidate_count


def plan(profile, source_sha):
    if profile not in ("smoke", "full") or not isinstance(source_sha, str) or len(source_sha) != 40:
        raise ValueError("Registered smoke/full profile and exact source SHA required")
    seeds = list(SEEDS[:1] if profile == "smoke" else SEEDS)
    steps = 2 if profile == "smoke" else ONLINE_FULL
    n_traces = len(CASES) * len(seeds)
    initial_calls = n_traces * INITIAL
    online_calls = n_traces * len(ARMS) * steps
    return {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "profile": profile,
        "cases": list(CASES),
        "seeds": seeds,
        "arms": list(ARMS),
        "initial_per_case_seed_shared_once": INITIAL,
        "online_calls_per_arm": steps,
        "full_online_calls_per_arm": ONLINE_FULL,
        "candidate_pool_size": POOL,
        "initial_calls": initial_calls,
        "online_calls": online_calls,
        "total_mathematical_calls": initial_calls + online_calls,
        "same_actual_online_calls": True,
        "same_candidate_pool_stream": True,
        "arm_histories_refit_independently": True,
        "aircraft_calls": 0,
        "rcaide_calls": 0,
        "primary_contrast": "soft_gp_all_support vs soft_gp_eligible",
        "isolated_change": "remove optimistic_violation eligible mask; retain same GP/VAE, risk proxy, soft utility, phase schedule, candidate pool size and evaluation budget",
        "not_a_probability_gate_ablation": True,
        "no_probability_or_safety_guarantee": True,
        "exploratory_after_PR66": True,
    }


def feasible_first(a, b):
    """Return 1 if a wins, -1 if b wins, else 0; never impute missing objectives."""
    x, y = a.get("final_best"), b.get("final_best")
    if x is None and y is None:
        return 0
    if x is not None and y is None:
        return 1
    if x is None and y is not None:
        return -1
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, y)):
        raise ValueError("Nonfinite feasible objective")
    return 1 if x < y - 1e-10 else (-1 if x > y + 1e-10 else 0)
