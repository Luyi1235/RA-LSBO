"""Explicit optional bounds studies, identical in the optimizer and MDA worker."""

from dataclasses import replace
from functools import wraps
import math

VERSION = "compact_wing_bounds_study_20261012_v1"


def normalize(settings):
    if (not isinstance(settings, dict)
            or set(settings) != {"version", "wing_area_bounds_m2", "scope"}
            or settings["version"] != VERSION
            or settings["scope"] != "conceptual_sensitivity_not_NASA_validated_bounds"):
        raise ValueError("Unsupported design-bounds study")
    bounds = settings["wing_area_bounds_m2"]
    if (not isinstance(bounds, list) or len(bounds) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
            or not 9. <= bounds[0] <= 11. or bounds[1] != 17.):
        raise ValueError("Study only supports wing-area lower bounds from 9 to 11 m^2")
    return {**settings, "wing_area_bounds_m2": list(bounds)}


def apply_schema(variables, settings):
    settings = normalize(settings)
    return [{**v, "bounds": list(settings["wing_area_bounds_m2"])} if v["name"] == "wing_area"
            else dict(v) for v in variables]


def install_legacy(legacy_design_space, settings):
    """The legacy repair reads these specs; changing only the parent would clip."""
    settings = normalize(settings)
    low, high = settings["wing_area_bounds_m2"]
    legacy_design_space.CONTINUOUS_SPECS = tuple(
        replace(spec, lower=low, upper=high) if spec.name == "wing_area" else spec
        for spec in legacy_design_space.CONTINUOUS_SPECS)
    return settings


def install_area_audit(nexus, diagnostics, requested_area):
    """Observe the evaluated deep copy, not the unchanged nexus template."""
    native = nexus.procedure.simple_sizing

    @wraps(native)
    def checked(candidate):
        result = native(candidate)
        realized = float(candidate.vehicle_configurations.base.wings.main_wing.areas.reference)
        diagnostics["design_study_realized_wing_area_m2"] = realized
        if not math.isclose(realized, requested_area, abs_tol=1e-9, rel_tol=0):
            raise ValueError("Study wing area was silently clipped in the physical worker")
        return result

    nexus.procedure.simple_sizing = checked
