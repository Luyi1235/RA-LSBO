"""Masked, independently trained probit ensembles inspired by BE-CBO.

These are raw classification probabilities, not calibrated safety bounds.
Numeric objective and constraint surrogates belong to the caller.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .observations import OutputObservation


@dataclass(frozen=True)
class BoundarySettings:
    """Fixed small-budget defaults; no automatic tuning or class balancing."""

    n_networks: int = 5
    hidden_width: int = 16
    steps: int = 80
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    band_scale: float = 1.0

    def __post_init__(self):
        for name, minimum in (("n_networks", 2), ("hidden_width", 1), ("steps", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError("{} must be an integer >= {}".format(name, minimum))
        for name in ("learning_rate", "weight_decay", "band_scale"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError("{} must be finite and nonnegative".format(name))
        if self.learning_rate == 0 or self.band_scale == 0:
            raise ValueError("learning_rate and band_scale must be positive")
        # Normalize NumPy scalars so checkpoints contain only safe builtin metadata.
        for name in ("n_networks", "hidden_width", "steps"):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ("learning_rate", "weight_decay", "band_scale"):
            object.__setattr__(self, name, float(getattr(self, name)))


def constraint_labels(observations):
    """Map explicit g <= 0 evidence to 1/0, leaving ambiguous evidence masked.

    Run status/provenance filtering must precede this output-level conversion.
    In particular, a solver failure alone is not a constraint label.
    """
    labels = []
    for observation in observations:
        if not isinstance(observation, OutputObservation):
            raise ValueError("Expected explicit OutputObservation evidence")
        satisfied = observation.constraint_satisfaction()
        labels.append(None if satisfied is None else int(satisfied))
    return labels


def _features(x, dimensions=None):
    if isinstance(x, torch.Tensor):
        if x.is_complex():
            raise ValueError("Features must be real")
        x = x.detach().to(device="cpu", dtype=torch.float64).numpy()
    raw = np.asarray(x)
    if raw.dtype.kind not in "fiu":
        raise ValueError("Features must be real numeric values")
    values = np.asarray(raw, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] == 0
            or (dimensions is not None and values.shape[1] != dimensions)
            or not np.isfinite(values).all()):
        raise ValueError("Expected a finite (n, d) feature matrix with d > 0")
    return torch.tensor(values, dtype=torch.float64, device="cpu")


def _settings(settings):
    if settings is None:
        return BoundarySettings()
    if isinstance(settings, BoundarySettings):
        return settings
    if not isinstance(settings, Mapping):
        raise ValueError("settings must be BoundarySettings or a settings mapping")
    try:
        return BoundarySettings(**settings)
    except TypeError as error:
        raise ValueError("Unknown boundary ensemble settings") from error


class _SeededLinear(nn.Module):
    def __init__(self, inputs, outputs, generator):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            (outputs, inputs), dtype=torch.float64, device="cpu"))
        self.bias = nn.Parameter(torch.empty(outputs, dtype=torch.float64, device="cpu"))
        # nn.Linear's constructor consumes global RNG; use only a local generator.
        bound = 1.0 / math.sqrt(inputs)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound, generator=generator)
            self.bias.uniform_(-bound, bound, generator=generator)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class BoundaryEnsemble:
    """Fit one boundary head using scalar labels 0, 1, or None.

    Every member fits the same observed rows using an independent initialization
    and Bernoulli(Phi(latent)) negative log likelihood. Predictions are CPU
    float64 NumPy arrays of shape (n,). No RNG, dtype, thread, or deterministic
    algorithm settings in the calling process are changed.
    """

    def __init__(self, x, labels, seed=0, settings=None):
        x = _features(x)
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()
        try:
            labels = list(labels)
        except TypeError as error:
            raise ValueError("Expected one scalar label or None per row") from error
        if len(labels) != len(x):
            raise ValueError("One scalar binary label or None is required per row")
        if any(label is not None and (
                not isinstance(label, Real) or not math.isfinite(label)
                or label not in (0, 1)) for label in labels):
            raise ValueError("Labels must be scalar 0, 1, or None; NaN is not missing")
        observed_mask = torch.tensor(
            [label is not None for label in labels], dtype=torch.bool, device="cpu")
        targets = torch.tensor(
            [float(label) for label in labels if label is not None],
            dtype=torch.float64, device="cpu")
        counts = (int((targets == 0).sum()), int((targets == 1).sum()))
        self._initialize(x.shape[1], seed, settings, observed_mask, counts)
        train_x = x[observed_mask]
        for net in self.nets:
            optimizer = torch.optim.Adam(
                net.parameters(), lr=self.settings.learning_rate,
                weight_decay=self.settings.weight_decay, foreach=False)
            for _ in range(self.settings.steps):
                optimizer.zero_grad(set_to_none=True)
                latent = net(train_x).squeeze(-1)
                # log Phi(signed latent) avoids clipped-probability tail gradients.
                loss = -torch.special.log_ndtr((2 * targets - 1) * latent).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss; check feature scale")
                loss.backward()
                optimizer.step()
            if any(not torch.isfinite(p).all() for p in net.parameters()):
                raise ValueError("Nonfinite fitted weights; check feature scale")
        self.nets.eval()
        self.nets.requires_grad_(False)

    def _initialize(self, dimensions, seed, settings, observed_mask, counts):
        if (isinstance(seed, bool) or not isinstance(seed, Integral)
                or not 0 <= seed < 2**63):
            raise ValueError("seed must be an integer in [0, 2**63)")
        self.seed = int(seed)
        self.dimensions = int(dimensions)
        self.settings = _settings(settings)
        self.observed_mask = observed_mask.clone()
        negative, positive = counts
        observed = negative + positive
        status = ("prior_only_no_labels" if observed == 0 else
                  "fitted_single_class" if min(counts) == 0 else "fitted_two_class")
        self.diagnostics = {
            "status": status, "observed": observed, "positive": positive,
            "negative": negative, "missing": len(observed_mask) - observed,
            "calibrated": False, "seed": self.seed,
            "likelihood": "Bernoulli(Phi(latent))",
            "settings": asdict(self.settings),
        }
        self.nets = nn.ModuleList()
        if not observed:
            return
        for member in range(self.settings.n_networks):
            generator = torch.Generator(device="cpu")
            generator.manual_seed((self.seed + 104729 * member) % (2**63))
            width = self.settings.hidden_width
            self.nets.append(nn.Sequential(
                _SeededLinear(self.dimensions, width, generator), nn.ReLU(),
                _SeededLinear(width, width, generator), nn.ReLU(),
                _SeededLinear(width, 1, generator)))

    def predict(self, x):
        """Return member-mean probability and clipped +/- one-spread bands.

        disagreement is the population standard deviation of member
        probabilities, excluding Bernoulli outcome variance. With no labels,
        probability=0.5, band=[0, 1], disagreement=0: no learned boundary.
        """
        x = _features(x, self.dimensions)
        if not len(x):
            return {key: np.empty(0, dtype=np.float64)
                    for key in ("probability", "lower", "upper", "disagreement")}
        if not len(self.nets):
            return {
                "probability": np.full(len(x), 0.5), "lower": np.zeros(len(x)),
                "upper": np.ones(len(x)), "disagreement": np.zeros(len(x)),
            }
        with torch.no_grad():
            latent = torch.stack([net(x).squeeze(-1) for net in self.nets])
            if not torch.isfinite(latent).all():
                raise ValueError("Nonfinite predictions; check feature scale")
            members = torch.special.ndtr(latent)
            probability = members.mean(dim=0)
            disagreement = members.std(dim=0, unbiased=False)
            spread = self.settings.band_scale * disagreement
            result = {
                "probability": probability, "lower": (probability - spread).clamp(0, 1),
                "upper": (probability + spread).clamp(0, 1),
                "disagreement": disagreement,
            }
        return {key: value.numpy().copy() for key, value in result.items()}

    def state_dict(self):
        """Return a versioned weights-only-serializable inference checkpoint."""
        return {
            "version": 1, "dimensions": self.dimensions, "seed": self.seed,
            "settings": asdict(self.settings),
            "observed_mask": self.observed_mask.clone(),
            "class_counts": (self.diagnostics["negative"], self.diagnostics["positive"]),
            "networks": {key: value.detach().clone()
                         for key, value in self.nets.state_dict().items()},
        }

    @classmethod
    def from_state_dict(cls, state):
        """Restore without training or consuming global random numbers."""
        required = {"version", "dimensions", "seed", "settings", "observed_mask",
                    "class_counts", "networks"}
        if (not isinstance(state, Mapping) or set(state) != required
                or type(state["version"]) is not int or state["version"] != 1):
            raise ValueError("Unsupported boundary ensemble checkpoint")
        if (not isinstance(state["settings"], Mapping)
                or set(state["settings"]) != set(asdict(BoundarySettings()))):
            raise ValueError("Incomplete boundary checkpoint settings")
        dimensions = state.get("dimensions")
        mask = state.get("observed_mask")
        counts = state.get("class_counts")
        if (isinstance(dimensions, bool) or not isinstance(dimensions, Integral)
                or dimensions < 1 or not isinstance(mask, torch.Tensor)
                or mask.ndim != 1 or mask.dtype != torch.bool
                or not isinstance(counts, (tuple, list)) or len(counts) != 2
                or any(type(count) is not int or count < 0 for count in counts)
                or sum(counts) != int(mask.sum())):
            raise ValueError("Invalid boundary checkpoint dimensions, mask, or counts")
        model = cls.__new__(cls)
        model._initialize(dimensions, state.get("seed"), state.get("settings"),
                          mask.cpu(), counts)
        networks = state.get("networks")
        if (not isinstance(networks, Mapping) or any(
                not isinstance(value, torch.Tensor) or value.dtype != torch.float64
                or not torch.isfinite(value).all() for value in networks.values())):
            raise ValueError("Invalid boundary checkpoint weights")
        try:
            model.nets.load_state_dict(networks, strict=True)
        except (RuntimeError, KeyError) as error:
            raise ValueError("Boundary checkpoint weight shapes or keys do not match") from error
        model.nets.eval()
        model.nets.requires_grad_(False)
        return model


def boundary_acquisition(heads, x):
    """Return per-head means/bands and all-head optimistic eligibility.

    Each constraint head must have upper >= 0.5. This does not establish a
    feasible incumbent. Empty heads require an explicit caller decision.
    No product across heads or discovery fallback is imposed here.
    """
    if not isinstance(heads, Mapping) or not heads:
        raise ValueError("Supply a nonempty mapping of boundary heads")
    x = _features(x).numpy()
    predictions = {}
    eligible = np.ones(len(x), dtype=bool)
    for name, head in heads.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Boundary heads need explicit nonempty names")
        prediction = head.predict(x)
        for key in ("probability", "lower", "upper", "disagreement"):
            values = np.asarray(prediction[key])
            if (values.shape != (len(x),) or not np.isfinite(values).all()
                    or np.any(values < 0) or np.any(values > 1)):
                raise ValueError("Invalid boundary head prediction")
        if (np.any(prediction["lower"] > prediction["probability"])
                or np.any(prediction["probability"] > prediction["upper"])):
            raise ValueError("Boundary bands must contain the mean probability")
        predictions[name] = prediction
        eligible &= prediction["upper"] >= 0.5
    return {"eligible": eligible, "heads": predictions}
