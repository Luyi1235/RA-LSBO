"""Audited runtime correction; the bundled v49 source files stay unchanged."""

import math
import numpy as np
from .energy_audit import audit_mission
from .operating_limits import LEGACY_POLICY, normalize_policy


POSTPROCESS_VERSION = "nasa_charge_soc_energy_audit_fail_closed_v1"


def values(container):
    # RCAIDE containers can iterate over values but reject integer indexing.
    return list(container.values()) if hasattr(container, "values") else list(container)


def battery_state(nexus):
    network = nexus.vehicle_configurations.base.networks.electric
    busses = values(network.busses)
    if len(busses) != 1:
        raise ValueError("X57 post-processing requires exactly one electric bus")
    bus = busses[0]
    modules = values(bus.battery_modules)
    if not modules:
        raise ValueError("No battery modules in electric bus")
    energies = [float(module.maximum_energy) for module in modules]
    if any(not math.isfinite(e) or e <= 0 for e in energies):
        raise ValueError("Missing, non-finite or non-positive battery capacity")
    segments = values(nexus.results.base.segments)
    if not segments:
        raise ValueError("No mission segments for final SOC")
    final_segment = segments[-1]
    bus_tag = getattr(bus, "tag", "bus")
    condition_bus = final_segment.conditions.energy[bus_tag]
    battery_conditions = values(condition_bus.battery_modules)
    if len(battery_conditions) != len(modules):
        raise ValueError("Final SOC is not available for every battery module")
    socs = [float(bat.state_of_charge[-1, 0]) for bat in battery_conditions]
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in socs):
        raise ValueError("Missing or out-of-range final battery SOC")
    return sum(energies), min(socs), len(modules), getattr(final_segment, "tag", "last_segment")


def post_process(nexus, diagnostics, operating_limit_policy=LEGACY_POLICY):
    """Read audited stored-energy depletion; preserve the v49 stall formula."""
    summary = nexus.summary
    nexus.total_number_of_iterations += 1
    segments = values(nexus.results.base.segments)
    summary.converged = bool(segments) and all(bool(getattr(seg, "converged", False)) for seg in segments)
    if not summary.converged:
        diagnostics["energy_audit_status"] = "not_run_mission_not_converged"
        return nexus
    report = {}
    diagnostics["_energy_audit_report"] = report
    metrics = audit_mission(nexus, report, operating_limit_policy)
    diagnostics.update(metrics)
    summary.total_energy_consumed = metrics["mission_depletion_Wh"]
    summary.min_soc_margin = metrics["final_soc"]
    base = nexus.vehicle_configurations.base
    mtow = getattr(base.mass_properties, "max_takeoff", 0.0)
    wing = base.wings.main_wing
    area, ar = wing.areas.reference, wing.aspect_ratio
    tc = getattr(wing, "thickness_to_chord", 0.12)
    if not all(math.isfinite(float(v)) and float(v) > 0 for v in (mtow, area, ar, tc)):
        raise ValueError('Stall screening requires finite positive mass and wing geometry')
    cl_max_2d = 2.0 * (tc / 0.12) ** 0.15
    cl_max_3d = cl_max_2d / (1.0 + cl_max_2d / (np.pi * ar * 0.8))
    stall = np.sqrt(2.0 * mtow * 9.81 / (1.225 * area * cl_max_3d))
    loading = mtow * 9.81 / area
    summary.stall_speed = stall
    summary.wing_loading = loading
    summary.cl_max_3d = cl_max_3d
    return nexus


def install_postprocess(nexus, operating_limit_policy=LEGACY_POLICY):
    policy = normalize_policy(operating_limit_policy)
    diagnostics = {"postprocess_version": POSTPROCESS_VERSION, 'operating_limit_policy': policy}

    def checked_post_process(candidate_nexus):
        # Functions retain this diagnostic sink when the legacy evaluator deep-copies nexus.
        try:
            return post_process(candidate_nexus, diagnostics, policy)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            diagnostics["postprocess_error"] = f"{type(error).__name__}: {error}"
            diagnostics["energy_audit_status"] = "failed"
            report = diagnostics.get("_energy_audit_report", {})
            report.update(status="failed", error=diagnostics["postprocess_error"])
            raise

    nexus.procedure.post_process = checked_post_process
    return diagnostics
