"""Cold-start acquisition targeting feasible designs below a fixed energy threshold."""

from dataclasses import replace

import numpy as np
from scipy.special import ndtr
from scipy.spatial.distance import cdist

from aircraft_timing_usable_value import correlation, forecast, timing_joint_cells
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings, MixedObservationGP
from run_aircraft_risk_cold_start import select


def energy_evidence(record):
    """Keep exact mission energy and trustworthy lower bounds, with distinct labels."""
    packet = record.observation
    objective = packet.objective
    diagnostics = record.diagnostics
    if diagnostics.get("numerical_failure"):
        return None
    if (packet.status == "completed" and diagnostics.get("mission_completed") is True
            and objective.kind == "exact" and objective.value > 0.):
        return objective, record.physical_feasible
    if (packet.status == "hidden_failure" and diagnostics.get("mission_completed") is False
            and diagnostics.get("termination_reason") == "physical_soc_reserve_abort"
            and objective.kind == "interval" and objective.lower is not None
            and objective.lower > 0. and objective.upper is None):
        return objective, None
    return None


def target_marks(references, query, observed, target, seed):
    """Combine planned-energy GP scores with sparse conditional constraint evidence.

    Use log(E/target), with zero GP prior mean and a fixed unit prior SD.
    Inputs are in the completion head's scaled latent/continuous coordinates.
    Treat planned-energy probabilities as a proxy for the distribution given
    completion. This factorization omits posterior completion-energy dependence.
    """
    if not observed:
        return None
    outputs = [replace(energy, value=np.log(energy.value/target)) if energy.kind == "exact"
               else replace(energy, lower=np.log(energy.lower/target)) for energy, _ in observed]
    model = MixedObservationGP(references, outputs, seed=seed,
                               settings=MixedGPSettings(stable_tail_moments=True))
    moments = model.predict(query)
    p_low = ndtr(-moments["mean"]/np.sqrt(np.maximum(moments["variance"], 1e-14)))
    resolved = np.array([energy.kind == "exact" and energy.value <= target and label is not None
                         for energy, label in observed])
    # No resolved low-energy flight outcome gives the neutral Beta(1, 1) mean.
    radius = np.sqrt(5.) * cdist(query, references[resolved]) / model.settings.length_scale
    weights = (1. + radius + radius**2 / 3.) * np.exp(-radius)
    feasible = np.array([label is True for _, label in observed])[resolved]
    p_feasible = (1. + weights[:, feasible].sum(axis=1)) / (2. + weights.sum(axis=1))
    return p_low * p_feasible


def two_call_target(immediate, first_mark, second_mark, joint_cells):
    """Score the first query with adaptive and fixed follow-ups, respectively.

    Each cell is P(timing cell at first design, second design completes).
    Conditional on completion, quality marks are held fixed and assumed
    independent across designs and of the other design's timing feedback.
    """
    if len(second_mark) == 0:
        return float(immediate), float(immediate)
    branches = joint_cells * second_mark
    branches[0] *= 1. - first_mark  # Completion cell, with no target hit on the first call.
    adaptive = immediate + branches.max(axis=1).sum()
    static = immediate + branches.sum(axis=0).max()
    return float(np.clip(adaptive, 0., 1.)), float(np.clip(static, 0., 1.))


def target_choice(head, query, candidates, edges, marks, *, lookahead):
    """Only the common engineering screen restricts the candidate set."""
    mean, sd, z, p_complete = forecast(head, query)
    immediate = p_complete * marks
    if not np.isfinite(immediate).all():
        raise FloatingPointError("Invalid probability for the energy target")
    index = select(candidates, immediate)
    receipt = {"p_complete": float(p_complete[index]), "target_probability": float(immediate[index])}
    if not lookahead or len(query) == 1:
        return index, receipt

    n = len(query)
    rho = correlation(head, np.concatenate((query, query)), n, np.tile(sd, 2))
    index_set = np.arange(n)
    values = []
    for i in index_set:
        other = index_set != i
        cells = timing_joint_cells(mean[i], sd[i], z[other], rho[i, other], edges[i])
        values.append(two_call_target(immediate[i], marks[i], marks[other], cells))
    adaptive, static = np.asarray(values).T
    if not (np.isfinite(adaptive).all() and np.isfinite(static).all()):
        raise FloatingPointError("Invalid two-call probability for the energy target")
    # A two-call objective can be invariant to the order of those two calls.
    # In an exact tie, keep the order with the better immediate probability.
    index = select(candidates, list(zip(adaptive, immediate)))
    return index, {"p_complete": float(p_complete[index]), "target_probability": float(immediate[index]),
                   "predicted_two_call_target": float(adaptive[index]),
                   "predicted_fixed_follow_up_target": float(static[index])}
