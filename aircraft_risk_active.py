"""Cold-start acquisition and held-out scoring for observable staged feedback."""
from collections import Counter, defaultdict
import math
from random import Random
from statistics import mean

from ra_lsbo.budget import digest

OUTCOMES = ("abort", "violation", "usable")
PRIMARY = ("risk_active", "risk_coverage")
CONTRASTS = (PRIMARY, ("risk_active", "risk_greedy"), ("risk_active", "risk_pointwise"),
             ("risk_active_latent", "risk_active"))


def configuration_weights(candidates):
    configs = [digest(row["discrete"]) for row in candidates]
    counts = Counter(configs)
    return [1/(len(counts)*counts[key]) for key in configs]


def reduction_components(left, right, joint, weights):
    """Mean expected posterior Brier reduction, with equal configuration weights."""
    import numpy as np

    c, f = left["mission"], left["feasible_given_mission"]
    cr, fr = right["mission"], right["feasible_given_mission"]
    cc, ff = (joint[key] for key in ("mission", "feasible_given_mission"))
    # Each branch holds the joint probability of its feedback and a usable y.
    cross = ((cr[None, :]-cc)*fr[None, :], cc*(fr[None, :]-ff), cc*ff)
    probabilities = (1-c, c*(1-f), c*f)
    components = {}
    for name, p, joint_usable in zip(OUTCOMES, probabilities, cross):
        residual = joint_usable-p[:, None]*(cr*fr)[None, :]
        reduction = np.divide(residual**2, p[:, None], out=np.zeros_like(residual), where=p[:, None] > 1e-14)
        components[name] = (reduction @ weights).tolist()
    return components


def information_receipt(heads, query, reference, forecast, weights):
    from aircraft_risk_discovery import stage_forecast, stage_joint_probabilities

    reference_forecast = stage_forecast(heads, reference)
    joint = stage_joint_probabilities(heads, query, reference, forecast, reference_forecast)
    return reduction_components(forecast[0], reference_forecast[0], joint, weights)


def active_scores(rule, receipt, n):
    if set(receipt) != {"mission", "feasible_given_mission", "coverage", "reductions"}:
        raise ValueError("Wrong cold-start acquisition receipt")
    c, f, coverage, parts = (receipt[k] for k in ("mission", "feasible_given_mission", "coverage", "reductions"))
    if (any(len(col) != n for col in (c, f, coverage)) or not all(0 <= x <= 1 for x in (*c, *f))
            or not all(math.isfinite(x) and x >= 0 for x in coverage)):
        raise ValueError("Invalid candidate forecasts")
    p = [a*b for a, b in zip(c, f)]
    if rule == "risk_information":
        if (set(parts) != set(OUTCOMES) or any(
                len(row) != n or not all(0 <= v <= .25+1e-12 for v in row) for row in parts.values())):
            raise ValueError("Invalid integrated Brier reductions")
        scores = [a+b+c for a, b, c in zip(*(parts[k] for k in OUTCOMES))]
        if any(v > .25+1e-12 for v in scores):
            raise ValueError("Integrated reduction exceeds Bernoulli variance bound")
        return scores
    if parts:
        raise ValueError("Unexpected information receipt")
    if rule == "risk_greedy":
        return p
    if rule == "risk_pointwise":
        return [v*(1-v) for v in p]
    if rule == "risk_coverage":
        return coverage
    raise ValueError("Unknown cold-start acquisition rule")


def configuration_brier(rows):
    """Average the Brier score equally across configurations with known labels."""
    groups = defaultdict(list)
    for candidate, label, probability in rows:
        if label is not None:
            groups[digest(candidate["discrete"])].append((probability-label)**2)
    return (mean(mean(group) for group in groups.values()) if groups else None), len(groups)


def learning_summary(metrics, checkpoints):
    groups = defaultdict(dict)
    for row in metrics:
        if row["head"] == "usable_factored" and row["n"] in checkpoints:
            key = (row["seed"], row["arm"])
            groups[key][row["n"]] = row["configuration_brier"]
    for series in groups.values():
        if set(series) != set(checkpoints) or any(value is None for value in series.values()):
            raise ValueError("Early learning endpoint unavailable")
    scores = {key: mean(series.values()) for key, series in groups.items()}
    seeds = sorted({seed for seed, _ in scores})
    result = []
    for arm, comparator in CONTRASTS:
        pairs = [(seed, scores[seed, arm]-scores[seed, comparator]) for seed in seeds]
        values = [delta for _, delta in pairs]
        rng = Random(20261116)
        interval = None
        if len(values) > 1:
            samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(10000))
            interval = [samples[249], samples[9749]]
        result.append({"arm": arm, "comparator": comparator, "mean_delta": mean(values),
                       "paired_bootstrap_95": interval, "seed_deltas": pairs})
    return result
