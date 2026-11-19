"""Expected Brier reduction from an ordered partition of timing feedback."""

import numpy as np
from scipy.special import ndtr

from aircraft_risk_acquisition import sign_covariance


def deficit_edges(candidate, settings):
    """Compute dimensionless cutpoints from the registered integrator grid.

    Completion occupies the nonpositive cell. Positive values are partitioned
    at mission-step endpoints. The first step absorbs the unphysical GP tail
    beyond a deficit of one. The evaluator's training intervals are unchanged.
    """
    ground_speed = (candidate["continuous"]["cruise_airspeed"]
                    - settings["scenario"]["headwind_m_s"])
    climb = settings["climb_height_m"]
    durations = np.array([climb/2, settings["nominal_cruise_distance_m"]/ground_speed,
                          climb/1.5, settings["reserve_duration_s"]])
    counts = np.maximum(1, np.ceil(durations/settings["integration_step_s"]).astype(int))
    elapsed = np.cumsum(np.repeat(durations/counts, counts))
    return np.concatenate(([0.], (1.-elapsed[:-1]/elapsed[-1])[::-1]))


def interval_variance_reduction(mean, sd, reference_z, correlation, edges):
    """Var(E[completion at y | timing cell at x]) under the fitted joint GP."""
    z = (np.asarray(edges)-mean)/sd
    bounds = np.concatenate(([-np.inf], z, [np.inf]))
    lo, hi = bounds[:-1], bounds[1:]
    masses = np.where(lo >= 0., ndtr(-lo)-ndtr(-hi), ndtr(hi)-ndtr(lo))
    rho = np.broadcast_to(correlation, (len(z), len(reference_z)))
    cov = sign_covariance(z, reference_z, rho)
    cov = np.vstack((np.zeros(len(reference_z)), cov, np.zeros(len(reference_z))))
    numerators = np.diff(cov, axis=0)**2
    per_cell = np.divide(numerators, masses[:, None], out=np.zeros_like(numerators),
                         where=masses[:, None] > 1e-14)
    return np.clip(per_cell.sum(axis=0), 0., ndtr(reference_z)*ndtr(-reference_z))
