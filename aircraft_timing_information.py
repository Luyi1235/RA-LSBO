"""Completion-information acquisition under a minimum immediate success score."""

import numpy as np
from scipy.special import ndtr

from aircraft_risk_acquisition import posterior_cross_covariance, sign_variance_reduction
from aircraft_risk_active import configuration_weights
from run_aircraft_initial_timing_diagnostic import fit_completion_heads
from run_aircraft_initial_timing_validation import TIMING_REASONS
from run_aircraft_risk_cold_start import select


PROBABILITY_FLOOR = .9


def score_information(head, query, candidates, interval_edges=None):
    moments = head.predict(query)
    sd = np.sqrt(np.maximum(moments["variance"], 1e-14))
    z = -moments["mean"] / sd
    p = ndtr(z)
    greedy = select(candidates, p)
    feasible = np.flatnonzero(p >= PROBABILITY_FLOOR*p[greedy])
    if len(candidates) == 1:
        return p, greedy, feasible, np.zeros(len(feasible))
    x = query/head.divisor
    cov = posterior_cross_covariance(head.model, x[feasible], x)*head.scale**2
    rho = cov / sd[feasible, None] / sd[None, :]
    if not np.isfinite(rho).all() or np.any(np.abs(rho) > 1.00001):
        raise FloatingPointError("Invalid timing posterior correlation")
    if interval_edges is None:
        reduction = sign_variance_reduction(z[feasible], z, rho)
    else:
        from aircraft_timing_interval_value import interval_variance_reduction

        reduction = np.stack([
            interval_variance_reduction(moments["mean"][i], sd[i], z, row, interval_edges[i])
            for i, row in zip(feasible, rho)
        ])
    weights = np.asarray(configuration_weights(candidates))
    # The queried design leaves the candidate set. Score its information on
    # other designs, and give equal aggregate weight to each configuration.
    reduction[np.arange(len(feasible)), feasible] = 0.
    scores = (reduction @ weights) / (1-weights[feasible])
    return p, greedy, feasible, scores


def select_information(head, query, candidates):
    p, _, feasible, scores = score_information(head, query, candidates)
    best = select([candidates[i] for i in feasible], scores)
    index = int(feasible[best])
    return index, float(p[index]), float(scores[best])


def information_choice(problem, history, candidates, seed):
    heads, query, _ = fit_completion_heads(problem, history, candidates, seed,
                                          methods=("timing_interval",))
    return select_information(heads["timing_interval"], query, candidates)


def supports_timing(record):
    return record.diagnostics.get("termination_reason") in TIMING_REASONS
