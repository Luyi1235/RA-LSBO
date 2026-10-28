"""Metrics with explicit censoring and no invented objective for failed runs."""

from collections import Counter, defaultdict

import numpy as np
from scipy.stats import beta


def binomial_interval(successes, count, confidence=.95):
    if (type(successes) is not int or type(count) is not int or not 0 <= successes <= count
            or not 0 < confidence < 1):
        raise ValueError("Invalid binomial sample")
    if count == 0:
        return [None, None]
    tail = (1-confidence)/2
    return [0. if successes == 0 else float(beta.ppf(tail, successes, count-successes+1)),
            1. if successes == count else float(beta.ppf(1-tail, successes+1, count-successes))]


def cvar(values, alpha=.95):
    """Empirical upper-tail CVaR with fractional mass at the quantile."""
    values = np.sort(np.asarray(values, float))[::-1]
    if not len(values) or not np.isfinite(values).all() or not 0 <= alpha < 1:
        raise ValueError("Finite nonempty CVaR sample and alpha in [0, 1) required")
    mass = (1-alpha)*len(values)
    whole = min(int(np.floor(mass)), len(values))
    weighted = float(values[:whole].sum())
    if whole < len(values):
        weighted += (mass-whole)*values[whole]
    return weighted/mass


def probability_scores(probabilities, labels):
    if len(probabilities) != len(labels):
        raise ValueError("Probability and label counts differ")
    if any(label not in (True, False, None) for label in labels):
        raise ValueError("Boolean or unknown labels required")
    valid = np.array([label is not None for label in labels])
    all_p = np.asarray(probabilities, float)
    if not np.isfinite(all_p).all() or np.any((all_p < 0) | (all_p > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    p, y = all_p[valid], np.array([label for label in labels if label is not None], float)
    if not len(p):
        return {"n": 0, "unknown": len(labels), "brier": None, "log_loss": None,
                "positive_brier": None, "negative_brier": None, "bins": []}
    clipped = np.clip(p, 1e-12, 1-1e-12)
    bins = []
    for index in range(10):
        mask = (p >= index/10) & ((p <= 1) if index == 9 else (p < (index+1)/10))
        bins.append({"lower": index/10, "upper": (index+1)/10, "n": int(mask.sum()),
                     "predicted": float(p[mask].mean()) if mask.any() else None,
                     "observed": float(y[mask].mean()) if mask.any() else None})
    return {"n": len(p), "unknown": int((~valid).sum()), "positives": int(y.sum()),
            "brier": float(np.mean((p-y)**2)),
            "log_loss": float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
            "positive_brier": float(np.mean((p[y == 1]-1)**2)) if np.any(y == 1) else None,
            "negative_brier": float(np.mean(p[y == 0]**2)) if np.any(y == 0) else None,
            "high_confidence_errors": int(np.sum((p >= .9) & (y == 0))), "bins": bins}


def qualification(records):
    usable = sum(bool(row.usable_feasible) for row in records)
    return {"n": len(records), "usable_feasible": usable,
            "usable_fraction": usable/len(records) if records else None,
            "binomial_95": binomial_interval(usable, len(records)),
            "physical_unknown": sum(row.physical_feasible is None for row in records),
            "interpretation": "uniform over discrete configurations, uniform active continuous coordinates, conditional on exact geometry screen"}


def trajectory_metrics(initial, online, horizon, stopped=False):
    if len(online) > horizon:
        raise ValueError("History exceeds declared horizon")
    history = list(initial)+list(online)
    first = next((index for index, row in enumerate(history, 1) if row.usable_feasible), None)
    initial_best = min((row.observation.objective.value for row in initial if row.usable_feasible), default=None)
    best, curve = initial_best, []
    for step in range(horizon+1):
        if 0 < step <= len(online) and online[step-1].usable_feasible:
            value = online[step-1].observation.objective.value
            best = value if best is None else min(best, value)
        curve.append({"online_budget": step, "nominal_total_budget": len(initial)+step,
                      "actual_calls": len(initial)+min(step, len(online)),
                      "best_feasible": best, "has_feasible": best is not None})
    known_failures = [name for row in online for name, value in row.observation.constraints
                      if value.constraint_satisfaction() is False]
    return {"initial_kind": "warm" if initial_best is not None else "cold", "initial_calls": len(initial),
            "online_calls": len(online), "stopped": stopped, "has_feasible": best is not None,
            "first_feasible_total_call": first,
            "first_feasible_online_call": None if first is None else max(0, first-len(initial)),
            "first_feasible_censored": first is None, "best_feasible": best, "initial_best": initial_best,
            "online_usable": sum(bool(row.usable_feasible) for row in online),
            "online_explicit_infeasible": sum(row.physical_feasible is False for row in online),
            "online_physical_unknown": sum(row.physical_feasible is None for row in online),
            "explicit_violations": dict(Counter(known_failures)),
            "hidden_failures": sum(row.observation.status == "hidden_failure" for row in online),
            "numerical_failures": sum(bool(row.diagnostics.get("numerical_failure")) for row in online),
            "model_domain_failures": sum(bool(row.diagnostics.get("model_domain_failure")) for row in online),
            "terminal_unknown": sum(row.observation.status in {"unknown", "infrastructure_error"} for row in online),
            "terminations": dict(Counter(row.diagnostics.get("termination_reason", row.observation.status) for row in online)),
            "curve": curve}


def flight_risk(records, reserve=.2, alpha=.95):
    if not records:
        return {"status": "no_nominal_feasible_incumbent", "n": 0}
    lower, upper, violated, unknown, complete_energy = [], [], 0, 0, []
    for row in records:
        if row.diagnostics.get("final_soc_exact"):
            loss = max(0., reserve-float(np.clip(row.diagnostics["final_soc"], 0., 1.)))
            lower.append(loss)
            upper.append(loss)
            violated += int(loss > 0)
            complete_energy.append(row.observation.objective.value)
        else:
            # Final clipped SOC lies in [0, 1]; numerical failures remain censored.
            lower.append(0.)
            upper.append(reserve)
            unknown += 1
    count = len(records)
    low_ci = binomial_interval(violated, count)
    high_ci = binomial_interval(violated+unknown, count)
    return {"status": "evaluated", "n": count, "soc_violations": violated, "soc_unknown": unknown,
            "soc_failure_fraction_bounds": [violated/count, (violated+unknown)/count],
            "soc_failure_probability_95_conservative": [low_ci[0], high_ci[1]],
            "soc_shortfall_cvar_alpha": alpha, "soc_shortfall_cvar_bounds": [cvar(lower, alpha), cvar(upper, alpha)],
            "completed_mission_count": len(complete_energy),
            "completed_only_mean_energy_Wh": float(np.mean(complete_energy)) if complete_energy else None,
            "usable_fraction": sum(bool(row.usable_feasible) for row in records)/count,
            "hidden_failures": sum(row.observation.status == "hidden_failure" for row in records),
            "interpretation": "iid screening scenario risk; CVaR bounds preserve unknown final SOC; not a certification claim"}


def paired_comparisons(results, reference="ra_lsbo"):
    lookup = {(row["case"], row["seed"], row["initial_policy"], row["method"]): row for row in results}
    groups = defaultdict(list)
    for row in results:
        key = (row["case"], row["seed"], row["initial_policy"], reference)
        if row["method"] != reference and key in lookup:
            groups[(row["case"], row["initial_policy"], row["method"])].append((lookup[key], row))
    comparisons = []
    for (case, policy, method), pairs in sorted(groups.items()):
        both = [(left["best_feasible"], right["best_feasible"]) for left, right in pairs
                if left["has_feasible"] and right["has_feasible"]]
        differences = np.array([left-right for left, right in both])
        confidence = [None, None]
        if len(differences) >= 2:
            random = np.random.default_rng(2601026)
            boot = differences[random.integers(0, len(differences), (2000, len(differences)))].mean(axis=1)
            confidence = np.quantile(boot, [.025, .975]).tolist()
        comparisons.append({"case": case, "initial_policy": policy, "reference": reference, "comparator": method,
            "paired_seeds": len(pairs), "both_feasible": len(both),
            "reference_only_feasible": sum(left["has_feasible"] and not right["has_feasible"] for left, right in pairs),
            "comparator_only_feasible": sum(right["has_feasible"] and not left["has_feasible"] for left, right in pairs),
            "neither_feasible": sum(not left["has_feasible"] and not right["has_feasible"] for left, right in pairs),
            "conditional_objective_difference_mean": float(differences.mean()) if len(differences) else None,
            "conditional_paired_bootstrap_95": confidence,
            "inference": "objective difference conditions on both methods feasible; no replacement for failed runs"})
    return comparisons
