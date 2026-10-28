"""CBOB-inspired mixed-likelihood GP, with no numeric imputation of bounds.

Numeric observations are conditioned analytically. EP approximates the remaining
interval likelihoods by Gaussian *sites*, not by invented measured responses.
All response parameters and predictions use the caller's physical output units.
"""

from dataclasses import asdict, dataclass
from numbers import Integral
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import ndtr
from scipy.stats import truncnorm

from .observations import OutputObservation


@dataclass(frozen=True)
class MixedGPSettings:
    """Fixed prior; length scales use input units, noise fields are output SDs."""

    length_scale: object = 0.5
    signal_variance: float = 1.0
    prior_mean: float = 0.0
    exact_noise: float = 1e-6
    interval_noise: float = 0.0
    jitter: float = 1e-10
    damping: float = 0.7
    tolerance: float = 1e-8
    max_sweeps: int = 60

    def __post_init__(self):
        scales = np.asarray(self.length_scale, dtype=float)
        if (scales.ndim > 1 or scales.size == 0
                or not np.all(np.isfinite(scales)) or np.any(scales <= 0)):
            raise ValueError("length_scale must be positive and finite")
        if scales.ndim == 1:
            object.__setattr__(self, "length_scale", tuple(scales.tolist()))
        else:
            object.__setattr__(self, "length_scale", float(scales))
        for name in ("signal_variance", "jitter", "tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")
        for name in ("exact_noise", "interval_noise"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(name + " must be nonnegative and finite")
        if not np.isfinite(self.prior_mean):
            raise ValueError("prior_mean must be finite")
        if not np.isfinite(self.damping) or not 0 < self.damping <= 1:
            raise ValueError("damping must be in (0, 1]")
        if (isinstance(self.max_sweeps, bool)
                or not isinstance(self.max_sweeps, Integral) or self.max_sweeps < 1):
            raise ValueError("max_sweeps must be a positive integer")


def _truncated_moments(a, b):
    """Standard normal conditional moments; log-tail formulas are in SciPy.

    Analytic CDF-ratio formulas lose variance digits for very narrow intervals.
    Only in that bounded case use centered, log-rescaled Gaussian quadrature.
    This integrates the likelihood, never assigns an interval midpoint target.
    """
    if not a < b:
        raise FloatingPointError("Interval collapsed at floating-point resolution")
    narrow = np.isfinite(a) and np.isfinite(b) and (
        (b - a) * (1 + min(abs(a), abs(b))) < 0.1)
    if not narrow:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean, variance = truncnorm.stats(a, b, moments="mv")
        if (np.isfinite(mean) and np.isfinite(variance)
                and 0 < variance <= 1 + 1e-8):
            return float(mean), float(min(variance, 1.0))
        if not np.isfinite(a) or not np.isfinite(b):
            raise FloatingPointError("Truncated tail moments exceeded numerical precision")
        raise FloatingPointError("Unstable truncated Gaussian moments")
    nodes, weights = np.polynomial.legendre.leggauss(32)
    width = (b - a) / 2
    center = a + width
    offsets = width * nodes
    log_weights = -center * offsets - 0.5 * offsets**2
    weights = weights * np.exp(log_weights - np.max(log_weights))
    weights /= np.sum(weights)
    displacement = weights @ offsets
    return float(center + displacement), float(weights @ (offsets - displacement)**2)


def _interval_tilt(mean, variance, lower, upper, noise):
    """Moments of f given lower < f + N(0, noise**2) < upper."""
    total = variance + noise**2
    sd = np.sqrt(total)
    a, b = (lower - mean) / sd, (upper - mean) / sd
    shift, truncated_variance = _truncated_moments(a, b)
    tilted_mean = mean + variance / sd * shift
    # This form avoids subtracting almost equal variances for a narrow interval.
    tilted_variance = variance * (
        noise**2 / total + variance / total * truncated_variance)
    if not np.isfinite(tilted_mean) or not np.isfinite(tilted_variance) or tilted_variance <= 0:
        raise FloatingPointError("Invalid interval tilted moments")
    return tilted_mean, tilted_variance


class MixedObservationGP:
    """Scalar response GP with numeric, interval, or missing OutputObservations.

    Inputs must be finite arrays of shape (n, d), including rows marked missing.
    Missing rows do not affect inference or the prior. No usable evidence gives
    the configured prior. All-one-sided evidence has a fixed, finite prior;
    it cannot identify an output scale or guarantee calibrated probabilities.
    """

    supports_empty_observations = True

    def __init__(self, x, observations, seed=0, settings=None):
        self.settings = MixedGPSettings() if settings is None else settings
        if not isinstance(self.settings, MixedGPSettings):
            raise ValueError("settings must be MixedGPSettings")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        self.seed = int(seed)  # Reserved for runner identity; inference is deterministic.
        self.x = self._inputs(x).copy()
        self.observations = tuple(observations)
        if len(self.observations) != len(self.x):
            raise ValueError("One OutputObservation is required per input row")
        if not all(isinstance(o, OutputObservation) for o in self.observations):
            raise ValueError("Expected explicit OutputObservation evidence")
        self.n_features = self.x.shape[1]
        scales = np.asarray(self.settings.length_scale, dtype=float)
        if scales.ndim == 1 and len(scales) != self.n_features:
            raise ValueError("ARD length scales must match the input dimension")
        self._scales = scales
        exact = [i for i, o in enumerate(self.observations) if o.kind == "exact"]
        interval = [i for i, o in enumerate(self.observations) if o.kind == "interval"]
        self.exact_x = self.x[exact]
        self._exact_factor = None
        self._exact_alpha = None
        if exact:
            covariance = self._kernel(self.exact_x, self.exact_x)
            covariance.flat[::len(exact) + 1] += (
                self.settings.exact_noise**2 + self.settings.jitter)
            self._exact_factor = cho_factor(covariance, lower=True)
            values = np.array([self.observations[i].value for i in exact])
            self._exact_alpha = cho_solve(
                self._exact_factor, values - self.settings.prior_mean)
        self.interval_x = self.x[interval]
        self._bounds = [
            (-np.inf if self.observations[i].lower is None else self.observations[i].lower,
             np.inf if self.observations[i].upper is None else self.observations[i].upper)
            for i in interval]
        self.diagnostics = {
            "n_exact": len(exact), "n_interval": len(interval),
            "n_missing": len(self.x) - len(exact) - len(interval),
            "converged": True, "sweeps": 0, "last_site_change": 0.0,
            "hyperparameters": "fixed", "seed": self.seed,
            "n_interval_sites": len(interval),
            "prior_only": not exact and not interval,
            "normalization": {"input": "identity", "output": "identity"},
            "length_scale": np.broadcast_to(scales, (self.n_features,)).tolist(),
            "kernel": "matern52",
            "inference": "ep" if interval else ("exact_gaussian" if exact else "prior"),
            "settings": asdict(self.settings),
        }
        if interval:
            self._coalesce_hard_intervals()
            self.diagnostics["n_interval_sites"] = len(self.interval_x)
            self._fit_intervals()

    @staticmethod
    def _inputs(x):
        result = np.asarray(x, dtype=float)
        if (result.ndim != 2 or result.shape[1] == 0
                or not np.all(np.isfinite(result))):
            raise ValueError("Inputs must be finite with shape (n, d), d >= 1")
        return result

    def _kernel(self, left, right):
        delta = (left[:, None, :] - right[None, :, :]) / self._scales
        radius = np.sqrt(5.0) * np.linalg.norm(delta, axis=2)
        if not np.all(np.isfinite(radius)):
            raise ValueError("Input distances overflowed; rescale input coordinates")
        return self.settings.signal_variance * (
            1 + radius + radius**2 / 3) * np.exp(-radius)

    def _baseline_mean(self, x):
        result = np.full(len(x), self.settings.prior_mean, dtype=float)
        if self._exact_factor is not None:
            result += self._kernel(x, self.exact_x) @ self._exact_alpha
        return result

    def _baseline_covariance(self, left, right):
        covariance = self._kernel(left, right)
        if self._exact_factor is not None:
            covariance -= self._kernel(left, self.exact_x) @ cho_solve(
                self._exact_factor, self._kernel(self.exact_x, right))
        return covariance

    def _coalesce_hard_intervals(self):
        if self.settings.interval_noise != 0:
            return
        # Products of hard indicators at one design equal their intersection.
        # Separate EP approximations would double-count duplicate evidence.
        bounds_by_point = {}
        for point, (lower, upper) in zip(self.interval_x, self._bounds):
            key = tuple(point)
            old_lower, old_upper = bounds_by_point.get(key, (-np.inf, np.inf))
            combined = max(lower, old_lower), min(upper, old_upper)
            if combined[0] >= combined[1]:
                raise ValueError("Contradictory hard interval evidence at one input")
            bounds_by_point[key] = combined
        self.interval_x = np.array(list(bounds_by_point), dtype=float)
        self._bounds = list(bounds_by_point.values())

    def _fit_intervals(self):
        n = len(self.interval_x)
        base_mean = self._baseline_mean(self.interval_x)
        covariance = self._baseline_covariance(self.interval_x, self.interval_x)
        covariance = (covariance + covariance.T) / 2
        covariance.flat[::n + 1] += self.settings.jitter
        self._interval_factor = cho_factor(covariance, lower=True)
        root = cholesky(covariance, lower=True)
        tau, nu = np.zeros(n), np.zeros(n)
        mean, posterior = base_mean.copy(), covariance.copy()
        converged = False
        for sweep in range(1, self.settings.max_sweeps + 1):
            old_tau, old_nu = tau.copy(), nu.copy()
            for i, (lower, upper) in enumerate(self._bounds):
                cavity_precision = 1 / posterior[i, i] - tau[i]
                if cavity_precision <= 1e-6 * max(1 / posterior[i, i], tau[i]):
                    # Remove concentrated sites before factorization instead of
                    # subtracting nearly equal posterior/site precisions.
                    cavity_tau, cavity_nu = tau.copy(), nu.copy()
                    cavity_tau[i], cavity_nu[i] = 0., 0.
                    cavity_m, cavity_s = self._site_posterior(
                        root, base_mean, cavity_tau, cavity_nu)
                    cavity_precision = 1 / cavity_s[i, i]
                    cavity_mean = cavity_m[i]
                else:
                    cavity_mean = (
                        mean[i] / posterior[i, i] - nu[i]) / cavity_precision
                if not np.isfinite(cavity_precision) or cavity_precision <= 0:
                    raise FloatingPointError("Nonpositive EP cavity precision")
                cavity_variance = 1 / cavity_precision
                tilted_mean, tilted_variance = _interval_tilt(
                    cavity_mean, cavity_variance, lower, upper,
                    self.settings.interval_noise)
                new_tau = 1 / tilted_variance - cavity_precision
                if new_tau < -1e-8 * max(1.0, cavity_precision):
                    raise FloatingPointError("Negative interval site precision")
                new_tau = max(0.0, new_tau)
                new_nu = tilted_mean / tilted_variance - cavity_mean * cavity_precision
                delta_tau = self.settings.damping * (new_tau - tau[i])
                tau[i] += delta_tau
                nu[i] += self.settings.damping * (new_nu - nu[i])
                column = posterior[:, i].copy()
                denominator = 1 + delta_tau * posterior[i, i]
                if denominator <= 0:
                    raise FloatingPointError("Invalid EP covariance update")
                if denominator > 1e5:
                    mean, posterior = self._site_posterior(root, base_mean, tau, nu)
                else:
                    posterior -= delta_tau / denominator * np.outer(column, column)
                    mean = base_mean + posterior @ (nu - tau * base_mean)
            # Refactor in whitened coordinates to control rank-update roundoff.
            mean, posterior = self._site_posterior(root, base_mean, tau, nu)
            change = max(
                np.max(np.abs(tau - old_tau) / (1 + np.abs(tau))),
                np.max(np.abs(nu - old_nu) / (1 + np.abs(nu))))
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(posterior)):
                raise FloatingPointError("Nonfinite EP posterior")
            if change <= self.settings.tolerance:
                converged = True
                break
        self.diagnostics.update(
            converged=converged, sweeps=sweep, last_site_change=float(change))
        if not converged:
            raise RuntimeError("MixedObservationGP EP did not converge within max_sweeps")
        self._interval_base_mean = base_mean
        self._posterior_mean = mean
        self._posterior_covariance = posterior

    @staticmethod
    def _site_posterior(root, base_mean, tau, nu):
        precision = np.eye(len(tau)) + root.T @ (tau[:, None] * root)
        covariance = root @ cho_solve(cho_factor(precision, lower=True), root.T)
        covariance = (covariance + covariance.T) / 2
        return base_mean + covariance @ (nu - tau * base_mean), covariance

    def predict(self, x):
        """Return 1-D arrays ``mean`` and latent ``variance`` in physical units."""
        x = self._inputs(x)
        if x.shape[1] != self.n_features:
            raise ValueError("Prediction input dimension differs from training")
        mean = self._baseline_mean(x)
        # Avoid allocating a quadratic prediction-prediction covariance matrix.
        variance = np.full(len(x), self.settings.signal_variance, dtype=float)
        if self._exact_factor is not None:
            cross = self._kernel(x, self.exact_x)
            variance -= np.einsum(
                "ij,ji->i", cross, cho_solve(self._exact_factor, cross.T))
        if len(self.interval_x):
            cross = self._baseline_covariance(x, self.interval_x)
            projection = cho_solve(self._interval_factor, cross.T).T
            mean += projection @ (self._posterior_mean - self._interval_base_mean)
            variance -= np.sum(projection * cross, axis=1)
            variance += np.sum(
                (projection @ self._posterior_covariance) * projection, axis=1)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise FloatingPointError("Nonfinite mixed GP prediction")
        if np.any(variance < -1e-8 * self.settings.signal_variance):
            raise FloatingPointError("Materially negative predictive variance")
        return {"mean": mean, "variance": np.maximum(variance, 0.0)}

    def to_dict(self):
        """JSON-compatible inputs/settings for deterministic refit, not a pickle."""
        return {
            "schema": "mixed-observation-gp-v1",
            "x": self.x.tolist(),
            "input_dimension": self.n_features,
            "observations": [asdict(o) for o in self.observations],
            "seed": self.seed, "settings": asdict(self.settings),
        }

    @classmethod
    def from_dict(cls, state):
        if state.get("schema") != "mixed-observation-gp-v1":
            raise ValueError("Unsupported mixed GP serialization schema")
        x = state["x"]
        if len(x) == 0:
            x = np.empty((0, state["input_dimension"]))
        return cls(
            x, [OutputObservation(**o) for o in state["observations"]],
            seed=state["seed"], settings=MixedGPSettings(**state["settings"]))


def eicb_factor(mean, variance, beta=1.96):
    """CBOB equations (6), (10), one constraint with g <= 0.

    At zero variance use deterministic feasibility, including g == 0. This
    convention avoids the path-dependent Gaussian limit at (mean, variance)=0.
    """
    mean, variance = np.broadcast_arrays(
        np.asarray(mean, dtype=float), np.asarray(variance, dtype=float))
    if (not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance))
            or np.any(variance < 0)):
        raise ValueError("mean/variance must be finite and variance nonnegative")
    if not np.isscalar(beta) or not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be a finite nonnegative scalar")
    ratio = np.zeros_like(mean)
    with np.errstate(over="ignore"):
        np.divide(mean, np.sqrt(variance), out=ratio, where=variance > 0)
    p = ndtr(-ratio)
    rho = ndtr(beta - ratio) - ndtr(-beta - ratio)
    factor = np.clip((1 + rho) * p, 0, 1)
    return np.where(variance > 0, factor, (mean <= 0).astype(float))
