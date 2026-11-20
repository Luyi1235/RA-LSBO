"""Value of feedback for choosing between designs under different sharing models."""

import numpy as np
from scipy.special import ndtr


FEEDBACK_EDGES = (0., .25, .50, .75)


def interval_probability(mean, sd, lower, upper):
    lo, hi = (lower-mean)/sd, (upper-mean)/sd
    return np.maximum(0., np.where(lo >= 0., ndtr(-lo)-ndtr(-hi), ndtr(hi)-ndtr(lo)))


def feedback_probabilities(mean, sd, edges=FEEDBACK_EDGES):
    """Predict completion and positive timing cells under each sharing model."""
    bounds = np.array([-np.inf, *edges, np.inf])
    return interval_probability(mean[..., None], sd[..., None], bounds[:-1], bounds[1:])


def comparison_value(weights, probabilities, feedback):
    """Expected gain in the two-design choice from identifying the sharing model.

    probabilities[m, a] is model m's present completion probability for action
    a. feedback[m, x, z] is its predicted probability of feedback cell z from
    probing x. Hold within-model action values fixed in this calculation.
    """
    weights, probabilities, feedback = map(np.asarray, (weights, probabilities, feedback))
    if (probabilities.shape != (len(weights), 2) or feedback.ndim != 3
            or feedback.shape[0] != len(weights)
            or not np.isfinite(probabilities).all() or not np.isfinite(feedback).all()
            or np.any((probabilities < 0.) | (probabilities > 1.))
            or np.any(feedback < 0.) or not np.allclose(feedback.sum(axis=2), 1.)
            or np.any(weights < 0.) or not np.isclose(weights.sum(), 1.)):
        raise ValueError("Incompatible sharing-model forecasts")
    gap = weights*(probabilities[:, 0]-probabilities[:, 1])
    updated_joint_gap = np.einsum("m,mxz->xz", gap, feedback)
    advantage = (np.abs(updated_joint_gap).sum(axis=1)-abs(gap.sum()))/2.
    return np.maximum(0., advantage)
