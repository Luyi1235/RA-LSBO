"""Small, dependency-light numerical safeguards for comparison experiments."""

from contextlib import contextmanager
import re
import warnings

import numpy as np
from scipy.special import erfcx, log_ndtr


SLSQP_CLIP_MESSAGE = (
    "Values in x were outside bounds during a minimize step, clipping to bounds"
)


class InvalidLocalPoint(ValueError):
    """A local solver supplied an unusable trial point or prediction."""


def bounded_point(values, lower, upper, *, tolerance=1e-10):
    """Project only round-off excursions; reject material ones and nonfinite x.

    All coordinates are normalized to [0, 1] by the caller. The tolerance is
    for numerical bounds handling, never for accepting a physical constraint.
    """
    x = np.asarray(values, dtype=float)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (x.ndim != 1 or x.shape != lo.shape or lo.shape != hi.shape
            or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(lo > hi) or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("Invalid local bounds or coordinate shape")
    if not np.isfinite(x).all():
        raise InvalidLocalPoint("Nonfinite local-search coordinate")
    excess = float(max(0., np.max(lo-x, initial=0.), np.max(x-hi, initial=0.)))
    if excess > tolerance:
        raise InvalidLocalPoint("Local-search coordinate materially outside bounds")
    return np.clip(x, lo, hi), excess


@contextmanager
def record_slsqp_clipping():
    """Record only SciPy's known clipping warning; other warnings stay visible.

    Scoped to one serial minimize call. Do not use as a process-wide filter.
    SciPy reports no excursion magnitude, so the count is not a severity score.
    """
    diagnostics = {"slsqp_clipping_warnings": 0}
    with warnings.catch_warnings():
        original_showwarning = warnings.showwarning

        def showwarning(message, category, filename, lineno, file=None, line=None):
            if issubclass(category, RuntimeWarning) and str(message) == SLSQP_CLIP_MESSAGE:
                diagnostics["slsqp_clipping_warnings"] += 1
            else:
                original_showwarning(message, category, filename, lineno, file=file, line=line)

        warnings.filterwarnings("always", message="^"+re.escape(SLSQP_CLIP_MESSAGE)+"$",
                                category=RuntimeWarning)
        warnings.showwarning = showwarning
        yield diagnostics


def log_expected_improvement(mean, std, best):
    """Stable log E[max(best-Y, 0)] for a univariate Gaussian (minimization).

    For negative standardized improvement use an erfcx expression, then a
    Mills-ratio asymptotic expansion beyond -20. This avoids evaluating tiny
    EI and then flooring it; all finite, distinguishable log-EIs retain rank.
    Zero variance is treated as deterministic improvement, not as noise.
    """
    mean, std = np.broadcast_arrays(np.asarray(mean, float), np.asarray(std, float))
    if not np.isfinite(best) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError("Finite mean/best and nonnegative finite standard deviation required")
    improvement = best-mean
    result = np.full(mean.shape, -np.inf, dtype=float)
    deterministic = (std == 0) & (improvement > 0)
    result[deterministic] = np.log(improvement[deterministic])
    mask = std > 0
    if not np.any(mask):
        return result
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        z = improvement[mask]/std[mask]
        log_phi = -.5*z*z-.5*np.log(2*np.pi)
        log_h = np.empty_like(z)
        positive = z >= 0
        # Separate z=0 to avoid log(0) warnings.
        zp = z[positive]
        log_z = np.full_like(zp, -np.inf)
        np.log(zp, out=log_z, where=zp > 0)
        log_h[positive] = np.logaddexp(log_phi[positive], log_z+log_ndtr(zp))
        negative = (z < 0) & (z >= -20)
        t = -z[negative]
        log_ratio = np.log(t)+.5*np.log(np.pi/2)+np.log(erfcx(t/np.sqrt(2)))
        log_h[negative] = log_phi[negative]+np.log(-np.expm1(log_ratio))
        tail = z < -20
        t = -z[tail]
        inverse_square = (1/t)**2
        # 1 - 3/t^2 + 15/t^4 - ...; at t>20 this is well conditioned.
        polynomial = np.ones_like(t)
        term = np.ones_like(t)
        for k in range(1, 12):
            term *= -(2*k+1)*inverse_square
            polynomial += term
        log_h[tail] = log_phi[tail]-2*np.log(t)+np.log(polynomial)
        values = np.log(std[mask])+log_h
        # Positive infinite z can only be caused by an underflow-small std.
        values = np.where(np.isposinf(z), np.log(improvement[mask]), values)
    if np.isnan(values).any():
        raise FloatingPointError("Nonfinite log-EI intermediate")
    result[mask] = values
    return result
