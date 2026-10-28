from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import hashlib
import json
import math

import numpy as np


@dataclass(frozen=True)
class Variable:
    name: str
    kind: str
    default: object
    bounds: tuple = ()
    values: tuple = ()
    active_when: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    continuous: dict
    discrete: dict

    def to_dict(self):
        return {"continuous": dict(self.continuous), "discrete": dict(self.discrete)}


class DesignSpace:
    """Physical category values at the boundary; indices stay inside the encoder."""

    def __init__(self, variables):
        self.variables = tuple(variables)
        self.continuous = tuple(v for v in variables if v.kind == "continuous")
        self.discrete = tuple(v for v in variables if v.kind != "continuous")
        names = [v.name for v in variables]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate variable names")
        parents = {}
        for v in self.discrete + self.continuous:
            if v.kind not in {"continuous", "ordinal", "nominal"}:
                raise ValueError(f"Unsupported kind: {v.kind}")
            for parent, values in v.active_when.items():
                if parent not in parents or not set(values).issubset(parents[parent]):
                    raise ValueError(f"Invalid/forward activation rule: {v.name}/{parent}")
            if v.kind == "continuous":
                if len(v.bounds) != 2 or not v.bounds[0] < v.bounds[1]:
                    raise ValueError(f"Invalid bounds: {v.name}")
                if not v.bounds[0] <= float(v.default) <= v.bounds[1]:
                    raise ValueError(f"Default outside bounds: {v.name}")
            else:
                if len(set(v.values)) < 2 or len(set(v.values)) != len(v.values) or v.default not in v.values:
                    raise ValueError(f"Invalid categories: {v.name}")
                parents[v.name] = v.values

    @classmethod
    def from_dict(cls, data):
        return cls([Variable(**{**v, "bounds": tuple(v.get("bounds", ())),
                                "values": tuple(v.get("values", ()))}) for v in data["variables"]])

    def active(self, variable, discrete):
        return all(discrete[parent] in values for parent, values in variable.active_when.items())

    def canonical(self, candidate):
        if set(candidate.discrete) - {v.name for v in self.discrete}:
            raise ValueError("Unknown discrete variable")
        if set(candidate.continuous) - {v.name for v in self.continuous}:
            raise ValueError("Unknown continuous variable")
        discrete = {}
        for v in self.discrete:
            value = candidate.discrete.get(v.name, v.default)
            if value not in v.values:
                raise ValueError(f"Invalid physical category {v.name}={value!r}")
            discrete[v.name] = value if self.active(v, discrete) else v.default
        continuous = {}
        for v in self.continuous:
            value = float(candidate.continuous.get(v.name, v.default))
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {v.name}")
            continuous[v.name] = float(np.clip(value, *v.bounds)) if self.active(v, discrete) else float(v.default)
        return Candidate(continuous, discrete)

    def default(self):
        return self.canonical(Candidate({}, {}))

    def mask(self, discrete):
        d = self.canonical(Candidate({}, discrete)).discrete
        return np.array([float(self.active(v, d)) for v in self.continuous])

    def unit(self, candidate):
        candidate = self.canonical(candidate)
        return np.array([(candidate.continuous[v.name] - v.bounds[0]) / (v.bounds[1] - v.bounds[0])
                         for v in self.continuous])

    def from_unit(self, unit, discrete):
        unit = np.asarray(unit, dtype=float)
        if unit.shape != (len(self.continuous),):
            raise ValueError("Incorrect continuous vector size")
        return self.canonical(Candidate({v.name: float(v.bounds[0] + u * (v.bounds[1] - v.bounds[0]))
                                          for v, u in zip(self.continuous, unit)}, discrete))

    def sample(self, rng, discrete=None):
        if discrete is None:
            discrete = {v.name: v.values[int(rng.integers(len(v.values)))] for v in self.discrete}
        return self.from_unit(rng.random(len(self.continuous)), discrete)

    def branch_key(self, discrete):
        discrete = self.canonical(Candidate({}, discrete)).discrete
        return tuple(discrete[v.name] for v in self.discrete)

    def key(self, candidate):
        c = self.canonical(candidate)
        payload = {"d": self.branch_key(c.discrete), "x": np.round(self.unit(c) * self.mask(c.discrete), 10).tolist()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def configurations(self, limit=100000):
        if math.prod(len(v.values) for v in self.discrete) > limit:
            raise ValueError("Grammar too large to enumerate; configure a sampled grammar instead")
        unique = {}
        for combination in product(*(v.values for v in self.discrete)):
            d = self.canonical(Candidate({}, dict(zip((v.name for v in self.discrete), combination)))).discrete
            unique[self.branch_key(d)] = d
        return list(unique.values())

    def indices(self, discrete):
        d = self.canonical(Candidate({}, discrete)).discrete
        return [v.values.index(d[v.name]) for v in self.discrete]

    def from_indices(self, indices):
        if len(indices) != len(self.discrete):
            raise ValueError("Incorrect category vector size")
        if any(int(i) != i or not 0 <= int(i) < len(v.values) for v, i in zip(self.discrete, indices)):
            raise ValueError("Category index out of range")
        d = {v.name: v.values[int(i)] for v, i in zip(self.discrete, indices)}
        return self.canonical(Candidate({}, d)).discrete
