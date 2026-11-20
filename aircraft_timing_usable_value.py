"""Two-call usable-design acquisition from interval timing and constraint evidence."""

import numpy as np
from scipy.special import ndtr

from aircraft_risk_acquisition import posterior_cross_covariance, sign_covariance
from aircraft_risk_discovery import fit_stage_head
from aircraft_timing_interval_value import deficit_edges
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.partial_bo.models import PrefixFeatures
from run_aircraft_initial_timing_diagnostic import fit_completion_heads
from run_aircraft_risk_cold_start import discovery_outcome, select


TIMING = "timing_interval"
FEASIBILITY = "feasible_given_mission"
ARMS = ("interval_greedy", "usable_greedy", "usable_lookahead")


def fit_heads(problem, history, candidates, seed):
    features = PrefixFeatures(problem, BASE, history, seed, latent=False)
    heads, query, _ = fit_completion_heads(problem, history, candidates, seed, methods=(TIMING,),
                                           features=features)
    past = features.transform([record.candidate for record in history])
    outcomes = [discovery_outcome(record.to_dict()) for record in history]
    heads[FEASIBILITY] = fit_stage_head(FEASIBILITY, history, past, outcomes, seed,
                                       len(problem.space.continuous))
    return heads, query


def forecast(head, query, *, probit=False):
    prediction = head.predict(query)
    sd = np.sqrt(np.maximum(prediction["variance"] + float(probit), 1e-14))
    z = -prediction["mean"] / sd
    return prediction["mean"], sd, z, ndtr(z)


def correlation(head, inputs, boundary, sd):
    inputs = inputs / head.divisor
    covariance = posterior_cross_covariance(head.model, inputs[:boundary], inputs[boundary:])
    rho = covariance * head.scale**2 / sd[:boundary, None] / sd[None, boundary:]
    if not np.isfinite(rho).all() or np.any(np.abs(rho) > 1.00001):
        raise FloatingPointError("Invalid lookahead posterior correlation")
    return np.clip(rho, -1., 1.)


def timing_joint_cells(mean, sd, second_z, rho, edges):
    """Return P(feedback cell at x, completion at y), for each cell and y."""
    z = (np.asarray(edges)-mean) / sd
    second = ndtr(second_z)
    marginal = ndtr(z)
    joint = marginal[:, None]*second + sign_covariance(
        z, second_z, np.broadcast_to(rho, (len(z), len(second_z))))
    joint = np.clip(joint, np.maximum(0., marginal[:, None]+second-1.),
                    np.minimum(marginal[:, None], second))
    cells = np.diff(np.vstack((np.zeros_like(second), joint, second)), axis=0)
    return np.maximum(0., cells)


def two_call_usable(completion, feasible, second_feasible, joint_feasible, joint_timing_cells):
    """Combine the two heads, treating their posteriors as independent."""
    after_abort = joint_timing_cells[1:] * second_feasible
    after_violation = joint_timing_cells[0] * (second_feasible - joint_feasible)
    branches = np.vstack((after_abort, after_violation))
    immediate = completion * feasible
    static = immediate + branches.sum(axis=0).max()
    adaptive = immediate + branches.max(axis=1).sum()
    return min(adaptive, 1.), min(static, 1.)


def choices(heads, query, candidates, settings):
    n = len(candidates)
    mean, sd, _, completion = forecast(heads[TIMING], query)
    _, feasible_sd, feasible_z, feasible = forecast(heads[FEASIBILITY], query, probit=True)
    timing_rho = correlation(heads[TIMING], query, n, sd)
    feasible_rho = correlation(heads[FEASIBILITY], query, n, feasible_sd)
    first_f, second_f = feasible[:n, None], feasible[None, n:]
    joint_f = first_f*second_f + sign_covariance(feasible_z[:n], feasible_z[n:], feasible_rho)
    joint_f = np.clip(joint_f, np.maximum(0., first_f+second_f-1.), np.minimum(first_f, second_f))
    values = []
    for i, candidate in enumerate(candidates):
        cells = timing_joint_cells(mean[i], sd[i], -mean[n:]/sd[n:], timing_rho[i],
                                   deficit_edges(candidate, settings))
        values.append(two_call_usable(completion[i], feasible[i], second_f[0], joint_f[i], cells))
    lookahead, static = np.asarray(values).T
    usable = completion*feasible
    scores = (completion[:n], usable[:n], lookahead)
    selected = {}
    for arm, score in zip(ARMS, scores):
        index = select(candidates, score)
        selected[arm] = {
            "index": index, "p_complete": float(completion[index]), "p_feasible_given_complete": float(feasible[index]),
            "p_usable": float(usable[index]), "predicted_two_call_usable": float(lookahead[index]),
            "predicted_fixed_follow_up_usable": float(static[index]),
        }
    return selected, (completion[n:], usable[n:])


def follow_up_scores(heads, query):
    completion = forecast(heads[TIMING], query)[-1]
    feasible = forecast(heads[FEASIBILITY], query, probit=True)[-1]
    return completion, completion*feasible
