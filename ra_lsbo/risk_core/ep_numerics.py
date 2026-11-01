"""Stable, explicitly opt-in moments for extreme truncated Gaussian tails.

This module is numerical support for the existing interval EP likelihood.
It does not invent interval midpoints or change statistical observations.
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=2)
def _legendre(order):
    return np.polynomial.legendre.leggauss(order)


def extreme_tail_moments(a, b):
    """Return E[Z|a<Z<b], Var[Z|a<Z<b] when one tail is >=8 sigma.

    Return None for a non-extreme interval; the caller then uses its
    unchanged legacy calculation. For a>=8, calculate moments of
    T=Z-a, whose density is proportional to exp(-a*T-T**2/2).
    Numerically integrate on [0, min(b-a, 35/a)] using 64-point
    Gauss-Legendre quadrature. The discarded unbounded tail has
    relative mass below about exp(-35), and all moments are
    computed in local coordinates to avoid cancellation of a**2.

    Left tails use exact normal symmetry. No parameter tuning is
    performed based on data, and no EP convergence is forced.
    """
    a, b = float(a), float(b)
    if np.isnan(a) or np.isnan(b) or not a < b:
        raise FloatingPointError("Invalid standardized truncated interval")
    if b <= -8.0:
        reflected = extreme_tail_moments(-b, -a)
        if reflected is None:
            raise AssertionError("Tail symmetry lost")
        mean, variance = reflected
        return -mean, variance
    if a < 8.0:
        return None
    if not np.isfinite(a):
        raise FloatingPointError("Extreme-tail lower endpoint must be finite")
    bound = min(b - a, 35.0 / a)
    if not np.isfinite(bound) or bound <= 0:
        raise FloatingPointError("Extreme-tail quadrature interval collapsed")
    nodes, weights = _legendre(64)
    offset = (nodes + 1.0) * (0.5 * bound)
    # The exp(-a**2/2) factor cancels in the conditional distribution.
    unnormalized = weights * np.exp(-a * offset - 0.5 * offset**2)
    total = float(np.sum(unnormalized))
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Extreme-tail conditional mass underflow")
    excess_mean = float(np.dot(unnormalized, offset) / total)
    excess_variance = float(np.dot(
        unnormalized, (offset - excess_mean)**2) / total)
    mean = a + excess_mean
    if not np.isfinite(mean) or not np.isfinite(excess_variance) or excess_variance <= 0:
        raise FloatingPointError("Extreme-tail quadrature moments invalid")
    return mean, excess_variance
