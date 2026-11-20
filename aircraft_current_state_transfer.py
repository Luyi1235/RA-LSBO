"""Separate current-margin functions for stopped and completed trajectories."""

from dataclasses import replace

import numpy as np
from scipy.special import ndtr

from ra_lsbo.partial_bo.models import EvidenceGP


STATE_TRANSFER = {"prefix_fixed_full": 1., "prefix_fixed_half": .5}


class CompletionStateEvidenceGP(EvidenceGP):
    """Share a fixed fraction of the design kernel between evidence states."""

    def __init__(self, x, observations, seed=0, settings=None, *, cross_state):
        if not 0. <= cross_state <= 1.:
            raise ValueError("Cross-state correlation must lie in [0, 1]")
        self.cross_state = cross_state
        super().__init__(x, observations, seed=seed, settings=settings)

    def _kernel(self, left, right):
        same = left[:, None, -1] == right[None, :, -1]
        return super()._kernel(left[:, :-1], right[:, :-1]) * np.where(same, 1., self.cross_state)


def transferred_current_probability(exact_head, x, query, observations, seed, cross_state):
    """Predict completed-design feasibility with normalization and kernel locked."""
    center, scale = exact_head.center, exact_head.scale
    normalized = []
    for obs in observations:
        if obs.kind == "exact":
            obs = replace(obs, value=(obs.value-center)/scale)
        elif obs.kind == "interval":
            obs = replace(obs, lower=(obs.lower-center)/scale)
        normalized.append(obs)
    # An interval comes from a stopped trajectory. Prediction is always for
    # the completed state. Missing observations contribute no likelihood.
    inputs = np.column_stack((x/exact_head.divisor, [o.kind == "interval" for o in observations]))
    targets = np.column_stack((query/exact_head.divisor, np.zeros(len(query))))
    model = CompletionStateEvidenceGP(inputs, normalized, seed=seed, settings=exact_head.model.settings,
                                      cross_state=cross_state)
    result = model.predict(targets)
    return ndtr(-(center+scale*result["mean"])/np.sqrt(np.maximum(scale**2*result["variance"], 1e-14)))
