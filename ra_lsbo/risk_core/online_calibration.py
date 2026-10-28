"""Prequential upper residual calibration for one exact scalar constraint g <= 0.

This is a literature-inspired adaptation, not a safety guarantee or GP model.
The caller must freeze predictions before revealing outcomes or refitting a GP.
"""

from collections import deque
from dataclasses import asdict, dataclass
import math
from numbers import Real


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("{} must be a finite real scalar".format(name))
    return float(value)


def _key(key):
    if not isinstance(key, str) or not key.strip():
        raise ValueError("key must be a nonempty string")
    return key


def _upper(mean, sd, quantile):
    # The extended prediction set is all/none, even for a zero-scale GP.
    if math.isinf(quantile):
        return quantile
    return _finite(mean + quantile * sd, "computed upper")


def _rank(alpha, n):
    if alpha <= 0:
        return None, "alpha_nonpositive"
    if alpha >= 1:
        return 0, "alpha_at_least_one"
    index = math.ceil((n + 1) * (1 - alpha))
    return index, "finite" if 1 <= index <= n else "insufficient_history"


@dataclass(frozen=True)
class PredictionToken:
    """Immutable snapshot; observe uses its internal copy, never a returned dict."""

    key: str
    mean: float
    sd: float
    scale: float
    upper: float
    quantile: float
    alpha: float
    quantile_index: object
    quantile_status: str
    n_residuals: int
    n_observed: int


class OnlineResidualCalibrator:
    """One-sided signed residuals with an unclipped adaptive alpha controller.

    Call predict(mean, sd, key) before obtaining that key's outcome, then call
    observe(key, exact_g_or_None) once. Keys cannot be reused in this instance.
    Pending predictions may resolve in any order; updates follow arrival order.
    Residual storage is bounded by window; duplicate-prevention keys are retained.
    """

    def __init__(self, target_alpha=.2, learning_rate=.05, window=64, *, epsilon=1e-12):
        target_alpha = _finite(target_alpha, "target_alpha")
        learning_rate = _finite(learning_rate, "learning_rate")
        epsilon = _finite(epsilon, "epsilon")
        if not 0 < target_alpha < 1:
            raise ValueError("target_alpha must be strictly between zero and one")
        if learning_rate <= 0 or epsilon <= 0:
            raise ValueError("learning_rate and epsilon must be positive")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer")
        self._target_alpha = target_alpha
        self._learning_rate = learning_rate
        self._window = window
        self._epsilon = epsilon
        self._alpha = target_alpha
        self._residuals = deque(maxlen=window)
        self._pending = {}
        self._closed = set()
        self._n_observed = 0
        self._n_missing = 0
        self._n_errors = 0

    @property
    def alpha(self):
        return self._alpha

    def current_quantile(self):
        """Read the pool-wide calibration snapshot without reserving any keys.

        Returns a detached dict. Use its quantile for candidate scoring, then
        predict only the selected point before obtaining its true outcome.
        """
        n = len(self._residuals)
        index, status = _rank(self._alpha, n)
        if status == "finite":
            quantile = sorted(self._residuals)[index - 1]
        elif status == "alpha_at_least_one":
            quantile = -math.inf
        else:
            quantile = math.inf
        return {
            "quantile": quantile, "alpha": self._alpha,
            "quantile_index": index, "quantile_status": status,
            "n_residuals": n, "n_observed": self._n_observed,
        }

    def predict(self, mean, sd, key):
        """Freeze the upper bound using only previously observed residuals.

        Returns a detached dict including a frozen PredictionToken under "token".
        Unavailable finite-sample ranks give +inf. Alpha <= 0 gives +inf and
        alpha >= 1 gives -inf; neither the alpha nor signed quantile is clipped.
        """
        key = _key(key)
        if key in self._pending or key in self._closed:
            raise ValueError("key already predicted: {!r}".format(key))
        mean, sd = _finite(mean, "mean"), _finite(sd, "sd")
        if sd < 0:
            raise ValueError("sd must be nonnegative")
        calibration = self.current_quantile()
        token = PredictionToken(
            key=key, mean=mean, sd=sd, scale=max(sd, self._epsilon),
            upper=_upper(mean, sd, calibration["quantile"]), **calibration,
        )
        self._pending[key] = token
        return dict(asdict(token), token=token)

    def observe(self, key, value):
        """Consume a prior prediction with an exact finite g, or missing None.

        Error is 1{observed_g > frozen_upper}; equality is covered. Missing values
        consume the key but change neither residuals nor alpha. Invalid numeric
        feedback raises without consuming the key or changing any state.
        """
        key = _key(key)
        if key in self._closed:
            raise ValueError("key already observed: {!r}".format(key))
        if key not in self._pending:
            raise KeyError("predict must precede observe: {!r}".format(key))
        token = self._pending[key]
        residual = error = None
        alpha_before = self._alpha
        alpha_after = alpha_before
        if value is not None:
            value = _finite(value, "value")
            residual = _finite((value - token.mean) / token.scale, "computed residual")
            error = int(value > token.upper)
            # Start from current alpha, not a stale pending token's alpha.
            alpha_after = _finite(
                alpha_before + self._learning_rate * (self._target_alpha - error),
                "updated alpha",
            )
        del self._pending[key]
        self._closed.add(key)
        if value is None:
            self._n_missing += 1
        else:
            self._residuals.append(residual)
            self._alpha = alpha_after
            self._n_observed += 1
            self._n_errors += error
        return {
            "key": key, "observed": value is not None, "residual": residual,
            "error": error, "alpha_before": alpha_before, "alpha_after": alpha_after,
            "n_observed": self._n_observed, "n_missing": self._n_missing,
            "token": token,
        }

    def state_dict(self):
        """Return a detached Python checkpoint; infinities remain float values."""
        return {
            "version": 1, "target_alpha": self._target_alpha,
            "learning_rate": self._learning_rate, "window": self._window,
            "epsilon": self._epsilon, "alpha": self._alpha,
            "residuals": list(self._residuals),
            "pending": [asdict(token) for token in self._pending.values()],
            "closed_keys": sorted(self._closed), "n_observed": self._n_observed,
            "n_missing": self._n_missing, "n_errors": self._n_errors,
        }

    @classmethod
    def from_state_dict(cls, state):
        """Restore a trusted state_dict, including pending tokens and used keys.

        This checks numeric and structural consistency, not checkpoint provenance.
        Do not merge streams or drop used keys to reduce checkpoint size.
        """
        if state["version"] != 1:
            raise ValueError("unsupported calibration checkpoint version")
        result = cls(state["target_alpha"], state["learning_rate"], state["window"],
                     epsilon=state["epsilon"])
        result._alpha = _finite(state["alpha"], "alpha")
        for name in ("n_observed", "n_missing", "n_errors"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError("invalid checkpoint count: " + name)
        if state["n_errors"] > state["n_observed"]:
            raise ValueError("errors cannot exceed observations")
        residuals = [_finite(value, "residual") for value in state["residuals"]]
        if len(residuals) != min(state["n_observed"], result._window):
            raise ValueError("residual count does not match the observation window")
        closed = [_key(key) for key in state["closed_keys"]]
        if (len(set(closed)) != len(closed)
                or len(closed) != state["n_observed"] + state["n_missing"]):
            raise ValueError("invalid closed keys")
        result._closed = set(closed)
        for item in state["pending"]:
            token = PredictionToken(**item)
            key = _key(token.key)
            mean, sd = _finite(token.mean, "mean"), _finite(token.sd, "sd")
            alpha = _finite(token.alpha, "token alpha")
            if (type(token.n_observed) is not int
                    or not 0 <= token.n_observed <= state["n_observed"]
                    or type(token.n_residuals) is not int
                    or token.n_residuals != min(token.n_observed, result._window)):
                raise ValueError("invalid pending history counts")
            index, status = _rank(alpha, token.n_residuals)
            q = token.quantile
            if (isinstance(q, bool) or not isinstance(q, Real) or math.isnan(q)
                    or (status == "finite" and not math.isfinite(q))
                    or (status == "alpha_at_least_one" and q != -math.inf)
                    or (status in {"alpha_nonpositive", "insufficient_history"}
                        and q != math.inf)):
                raise ValueError("invalid pending quantile")
            if (key in result._closed or key in result._pending or sd < 0
                    or token.scale != max(sd, result._epsilon)
                    or token.quantile_index != index or token.quantile_status != status
                    or token.upper != _upper(mean, sd, q)):
                raise ValueError("inconsistent pending token")
            result._pending[key] = token
        result._residuals.extend(residuals)
        result._n_observed = state["n_observed"]
        result._n_missing = state["n_missing"]
        result._n_errors = state["n_errors"]
        return result
