"""Run-level inference for paired optimization studies with failed discovery."""

from itertools import product
import math

import numpy as np


def finite_values(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("A nonempty vector of finite run-level outcomes is required")
    return values


def paired_effect(differences, seed=20261025, resamples=20000):
    values = finite_values(differences)
    if type(resamples) is not int or resamples < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    n, effect = len(values), float(np.mean(values))
    rng = np.random.default_rng(seed)
    means = np.mean(values[rng.integers(n, size=(resamples, n))], axis=1)
    interval = np.quantile(means, [.025, .975]).tolist()
    # Exact label swaps assume paired exchangeability under the null.
    # Keep zeros in the sample: they convey ties rather than extra replication.
    p = None
    if n <= 20:
        observed = abs(float(np.sum(values)))
        tail = sum(abs(float(np.dot(signs, values))) >= observed-1e-12
                   for signs in product((-1, 1), repeat=n))
        p = tail/(2**n)
    return {"n_pairs": n, "mean_candidate_minus_reference": effect,
            "median_candidate_minus_reference": float(np.median(values)),
            "paired_bootstrap_percentile_95": interval,
            "exact_two_sided_sign_flip_p": p,
            "bootstrap_unit": "paired optimization seed within one task",
            "bootstrap_seed": seed, "bootstrap_replicates": resamples,
            "small_sample_warning": n < 10,
            "sign_flip_assumption": "paired method labels exchangeable under the null"}


def holm(p_values):
    """Missing planned tests stay in the family with p=1."""
    clean = [1. if p is None else float(p) for p in p_values]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in clean):
        raise ValueError("P values must lie in [0,1]")
    order = sorted(range(len(clean)), key=lambda i: clean[i])
    adjusted, running = [1.]*len(clean), 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(clean)-rank)*clean[index]))
        adjusted[index] = running
    return [None if p is None else adjusted[i] for i, p in enumerate(p_values)]


def ordered_endpoint(candidate, reference):
    """Positive means candidate worse; no artificial target for no-feasible runs."""
    for value in (candidate, reference):
        if value is not None and not math.isfinite(value):
            raise ValueError("Endpoint objective must be finite or explicitly unavailable")
    if candidate is None:
        return 0 if reference is None else 1
    if reference is None:
        return -1
    return int(candidate > reference)-int(candidate < reference)


def run_metrics(initial, observations, horizon):
    if type(horizon) is not int or horizon < 1 or len(observations) > horizon:
        raise ValueError("Invalid common evaluation horizon")
    for row in initial+observations:
        risk = row["risk"]
        if risk["feasible_label"] not in (True, False, None):
            raise ValueError("Invalid feasibility label")
        if risk["feasible_label"] is True and (
                risk["objective"] is None or not math.isfinite(risk["objective"])):
            raise ValueError("Feasible objective must be observed and finite")
    initial_targets = [r["risk"]["objective"] for r in initial
                       if r["risk"]["feasible_label"] is True]
    initial_best = min(initial_targets) if initial_targets else None
    best, first = initial_best, 0 if initial_best is not None else None
    trajectory = [{"online_call": 0, "best_feasible": best, "discovered": first is not None}]
    for call, row in enumerate(observations, 1):
        risk = row["risk"]
        if risk["feasible_label"] is True:
            best = risk["objective"] if best is None else min(best, risk["objective"])
            if first is None:
                first = call
        trajectory.append({"online_call": call, "best_feasible": best, "discovered": first is not None})
    complete = len(observations) == horizon
    known_initial = all(r["risk"]["feasible_label"] is not None for r in initial)
    resolved = known_initial and all(r["risk"]["feasible_label"] is not None for r in observations)
    # Only administrative censoring at a common budget identifies min(T,B).
    # A software stop with no event cannot be silently imputed as B.
    discovery_cost = (min(first, horizon) if first is not None
                      else horizon if complete and resolved else None)
    return {
        "complete_horizon": complete, "all_labels_resolved": resolved,
        "initial_feasible": len(initial_targets), "initial_best": initial_best,
        "online_calls": len(observations),
        "online_feasible_count": sum(r["risk"]["feasible_label"] is True for r in observations),
        "hidden_failure_count": sum(r["risk"].get("hidden_success") == 0 for r in observations),
        "unknown_count": sum(r["risk"]["feasible_label"] is None for r in observations),
        "endpoint_best": best if complete and resolved else None,
        "observed_best": best, "first_feasible_online_call": first,
        "first_feasible_event": first is not None,
        "restricted_first_feasible_cost": discovery_cost if known_initial else None,
        "administratively_censored": first is None and complete and resolved,
        "early_stop_not_administrative_censoring": not complete,
        "trajectory": trajectory,
    }
