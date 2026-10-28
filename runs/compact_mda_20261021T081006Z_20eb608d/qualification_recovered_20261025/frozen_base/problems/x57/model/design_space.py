"""Physically active hierarchical design space for the X-57 v49 study.

The v48 study exposed variables whose effects were represented partly through
hand-tuned objective offsets.  v49 keeps only variables that are mapped into
the RCAIDE vehicle/mission model.  Additional architecture variables can be
reintroduced after their geometry, mass, and mission effects are modeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ContinuousSpec:
    name: str
    lower: float
    upper: float
    default: float
    active_when: Tuple[str, object] | None = None


@dataclass(frozen=True)
class DiscreteSpec:
    name: str
    values: Tuple[object, ...]
    kind: str


# Bounds are engineering screening domains, not objective penalties. They are
# deliberately narrower than v48's exploratory ranges where the evaluator was
# regularly rescued by hand-tuned Wh corrections.
CONTINUOUS_SPECS: Tuple[ContinuousSpec, ...] = (
    ContinuousSpec("aspect_ratio", 7.5, 13.5, 10.0),
    ContinuousSpec("wing_area", 11.0, 17.0, 14.76),
    ContinuousSpec("quarter_chord_sweep", 0.0, 4.0, 0.0),
    ContinuousSpec("thickness_to_chord", 0.09, 0.14, 0.12),
    ContinuousSpec("taper_ratio", 0.35, 0.85, 0.63),
    ContinuousSpec("root_twist", 2.0, 4.5, 3.0),
    ContinuousSpec("tip_twist", -2.2, 1.5, 0.0),
    ContinuousSpec("dihedral", 0.2, 3.5, 1.0),
    ContinuousSpec("prop_span_spacing", 0.38, 0.78, 0.50, ("prop_layout", "leading_edge_distributed")),
    ContinuousSpec("prop_diameter_ratio", 0.82, 1.15, 1.00, ("prop_layout", "leading_edge_distributed")),
    ContinuousSpec("tip_prop_diameter", 1.55, 1.90, 1.72, ("prop_layout", "wingtip")),
    ContinuousSpec("tip_prop_rpm", 2350.0, 2900.0, 2700.0, ("prop_layout", "wingtip")),
    ContinuousSpec("winglet_fraction", 0.04, 0.12, 0.06, ("has_winglet", "yes")),
    ContinuousSpec("winglet_cant_angle", 45.0, 80.0, 70.0, ("has_winglet", "yes")),
)

DISCRETE_SPECS: Tuple[DiscreteSpec, ...] = (
    DiscreteSpec("prop_layout", ("wingtip", "leading_edge_distributed"), "nominal"),
    DiscreteSpec("propeller_count", (2, 4, 6, 8, 10, 12), "ordinal"),
    DiscreteSpec("wing_airfoil", ("NACA_63_412", "NACA_4412", "NACA_23012", "Clark_Y", "NACA_65_215"), "nominal"),
    DiscreteSpec("blade_count", (2, 3, 4, 5), "ordinal"),
    DiscreteSpec("has_winglet", ("no", "yes"), "nominal"),
)


def continuous_names() -> List[str]:
    return [spec.name for spec in CONTINUOUS_SPECS]


def continuous_bounds() -> Dict[str, Tuple[float, float]]:
    return {spec.name: (spec.lower, spec.upper) for spec in CONTINUOUS_SPECS}


def default_continuous() -> Dict[str, float]:
    return {spec.name: spec.default for spec in CONTINUOUS_SPECS}


def default_discrete() -> Dict[str, object]:
    return {
        "prop_layout": "wingtip",
        "propeller_count": 2,
        "wing_airfoil": "NACA_63_412",
        "blade_count": 3,
        "has_winglet": "no",
    }


def canonicalize_discrete(discrete: Dict[str, object]) -> Dict[str, object]:
    result = default_discrete()
    result.update(discrete)
    if result["prop_layout"] == "wingtip":
        result["propeller_count"] = 2
    return result


def active_mask(discrete: Dict[str, object]) -> np.ndarray:
    discrete = canonicalize_discrete(discrete)
    values = []
    for spec in CONTINUOUS_SPECS:
        if spec.active_when is None:
            values.append(1.0)
        else:
            parent, expected = spec.active_when
            values.append(float(discrete[parent] == expected))
    return np.asarray(values, dtype=float)


def repair_continuous(continuous: Dict[str, float], discrete: Dict[str, object]) -> Dict[str, float]:
    """Clip only to stated engineering bounds; do not repair toward an optimum."""
    result = default_continuous()
    result.update(continuous)
    for spec in CONTINUOUS_SPECS:
        result[spec.name] = float(np.clip(result[spec.name], spec.lower, spec.upper))
    return result


def normalized_active_continuous(continuous: Dict[str, float], discrete: Dict[str, object]) -> np.ndarray:
    continuous = repair_continuous(continuous, discrete)
    mask = active_mask(discrete)
    values = []
    for spec, is_active in zip(CONTINUOUS_SPECS, mask):
        unit_value = (continuous[spec.name] - spec.lower) / (spec.upper - spec.lower)
        values.append(unit_value * is_active)
    return np.asarray(values, dtype=float)


def discrete_key(discrete: Dict[str, object]) -> Tuple[object, ...]:
    discrete = canonicalize_discrete(discrete)
    return tuple(discrete[spec.name] for spec in DISCRETE_SPECS)


def enumerate_configurations() -> List[Dict[str, object]]:
    configs = []
    for values in product(*(spec.values for spec in DISCRETE_SPECS)):
        candidate = canonicalize_discrete(dict(zip((spec.name for spec in DISCRETE_SPECS), values)))
        if candidate not in configs:
            configs.append(candidate)
    return configs


def sample_continuous(rng: np.random.Generator, discrete: Dict[str, object]) -> Dict[str, float]:
    values = {}
    for spec in CONTINUOUS_SPECS:
        values[spec.name] = float(rng.uniform(spec.lower, spec.upper))
    return repair_continuous(values, discrete)


def perturb_continuous(
    center: Dict[str, float],
    discrete: Dict[str, object],
    rng: np.random.Generator,
    scale: float,
) -> Dict[str, float]:
    values = dict(center)
    for spec in CONTINUOUS_SPECS:
        if spec.active_when is not None:
            parent, expected = spec.active_when
            if canonicalize_discrete(discrete)[parent] != expected:
                continue
        radius = scale * (spec.upper - spec.lower)
        values[spec.name] = float(values.get(spec.name, spec.default) + rng.normal(0.0, radius))
    return repair_continuous(values, discrete)
