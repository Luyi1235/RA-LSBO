"""Two-evaluation completion value from possible timing feedback."""

import numpy as np
from scipy.special import ndtr

from aircraft_risk_acquisition import posterior_cross_covariance, sign_covariance
from aircraft_timing_information import PROBABILITY_FLOOR
from aircraft_timing_interval_value import deficit_edges
from run_aircraft_risk_cold_start import select


DECISION_ARMS = ("greedy", "gated_decision", "decision_value")


def two_call_value(mean, sd, reference_z, correlation, edges):
    """Estimate P(first completes) plus the best follow-up in each abort cell."""
    z = (np.asarray(edges)-mean)/sd
    p_second = ndtr(reference_z)
    correlation = np.broadcast_to(correlation, (len(z), len(reference_z)))
    both = ndtr(z)[:, None]*p_second + sign_covariance(z, reference_z, correlation)
    cells = np.diff(np.vstack((both, p_second)), axis=0)
    survival = ndtr(-np.append(z, np.inf))
    abort_mass = survival[:-1]-survival[1:]
    joint = np.clip(cells, 0., abort_mass[:, None])
    return float(np.clip(ndtr(z[0]) + joint.max(axis=1).sum(), 0., 1.))


def decision_choices(head, query, second, candidates, settings):
    proposed, reference = head.predict(query), head.predict(second)
    sd = np.sqrt(np.maximum(proposed["variance"], 1e-14))
    ref_sd = np.sqrt(np.maximum(reference["variance"], 1e-14))
    ref_z = -reference["mean"]/ref_sd
    completion = ndtr(-proposed["mean"]/sd)
    covariance = posterior_cross_covariance(head.model, query/head.divisor, second/head.divisor)
    correlation = covariance*head.scale**2 / sd[:, None] / ref_sd[None, :]
    if not np.isfinite(correlation).all() or np.any(np.abs(correlation) > 1.00001):
        raise FloatingPointError("Invalid posterior timing correlation")
    value = np.array([two_call_value(mu, scale, ref_z, row, deficit_edges(candidate, settings))
                      for mu, scale, row, candidate in
                      zip(proposed["mean"], sd, correlation, candidates)])
    greedy = select(candidates, completion)
    eligible = np.flatnonzero(completion >= PROBABILITY_FLOOR*completion[greedy])
    gated = eligible[select([candidates[i] for i in eligible], value[eligible])]
    best = select(candidates, value)
    choices = {arm: {"index": int(i), "probability": float(completion[i]),
                     "model_two_call_completion": float(value[i])}
               for arm, i in zip(DECISION_ARMS, (greedy, gated, best))}
    return choices, len(eligible)
