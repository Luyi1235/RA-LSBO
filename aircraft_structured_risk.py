"""Configuration-aware completion modeling and adaptive two-call acquisition."""

import numpy as np

from aircraft_timing_information import PROBABILITY_FLOOR
from aircraft_timing_usable_value import correlation, forecast, timing_joint_cells
from ra_lsbo.partial_bo.models import EvidenceGP
from run_aircraft_coverage_factorial import FixedTimingHead
from run_aircraft_risk_cold_start import select


PROFILE = "structured-risk-paper"
CONFIRMATION_PROFILE = "structured-risk-confirmation"
STRUCTURED_GREEDY = "ra_structured_greedy"
STRUCTURED_SIGN_GREEDY = "ra_structured_sign_greedy"
STRUCTURED_DECISION = "ra_structured_decision"
LATENT_DECISION = "ra_interval_decision"
STRUCTURED_ARMS = (STRUCTURED_GREEDY, STRUCTURED_DECISION, STRUCTURED_SIGN_GREEDY)
DECISION_ARMS = (LATENT_DECISION, STRUCTURED_DECISION)


def with_configuration(features, candidates):
    """Expose the ordinal battery size and the propulsor hierarchy to the GP."""
    metadata = []
    for candidate in candidates:
        discrete = candidate["discrete"]
        layout = {"wingtip": 0, "distributed": 1}[discrete["prop_layout"]]
        count = discrete["propeller_count"] if layout == 1 else 0
        metadata.append((discrete["parallel_strings"]/8., layout, count))
    return np.column_stack((features, np.asarray(metadata, dtype=float).reshape(-1, 3)))


class HierarchicalTimingGP(EvidenceGP):
    def _kernel(self, left, right):
        latent = super()._kernel(left[:, :-3], right[:, :-3])
        delta = left[:, None, -3:] - right[None, :, -3:]
        capacity = np.exp(-.5*delta[..., 0]**2)
        layout = delta[..., 1] == 0.
        configuration = np.all(delta == 0., axis=-1)
        hierarchy = .5 * (layout.astype(float) + configuration)
        # The weighted sum of positive semidefinite terms leaves prior point
        # variance unchanged. Neither a capacity trend nor a cutoff is imposed.
        return .5*latent + .25*self.settings.signal_variance*(capacity+hierarchy)


class HierarchicalTimingHead(FixedTimingHead):
    model_class = HierarchicalTimingGP
    metadata_dimensions = 3


def choose_completion_with_lookahead(head, query, candidates, edges):
    """Maximize modeled probability of completion now or after one further query."""
    mean, sd, z, p = forecast(head, query)
    best = select(candidates, p)
    eligible = np.flatnonzero(p >= PROBABILITY_FLOOR*p[best])
    if len(query) == 1:
        return best, {"p_complete": float(p[best]), "p_complete_greedy": float(p[best]),
                      "adaptive_two_call_completion": float(p[best]), "fixed_second_completion": float(p[best])}

    boundary = len(eligible)
    rho = correlation(head, np.concatenate((query[eligible], query)), boundary,
                      np.concatenate((sd[eligible], sd)))
    values, fixed = [], []
    positions = np.arange(len(query))
    for i, cross in zip(eligible, rho):
        other = positions != i
        # Row zero is completion. An adaptive second call has value only in
        # termination cells, where the best second candidate can change.
        failed_cells = timing_joint_cells(mean[i], sd[i], z[other], cross[other], edges[i])[1:]
        values.append(float(np.clip(p[i] + failed_cells.max(axis=1).sum(), 0., 1.)))
        fixed.append(float(np.clip(p[i] + failed_cells.sum(axis=0).max(), 0., 1.)))
    if not np.isfinite(values).all() or not np.isfinite(fixed).all():
        raise FloatingPointError("Invalid completion lookahead scores")
    selected = select([candidates[i] for i in eligible], list(zip(values, p[eligible])))
    index = int(eligible[selected])
    return index, {"p_complete": float(p[index]), "p_complete_greedy": float(p[best]),
                   "adaptive_two_call_completion": values[selected], "fixed_second_completion": fixed[selected]}
