"""RCAIDE candidate evaluation for constrained hierarchical BO.

This module deliberately returns mission energy and engineering constraints
separately.  It never adds hand-tuned Wh rewards or penalties to the mission
energy, so the optimization objective remains interpretable.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import traceback
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

import RCAIDE
from RCAIDE.Framework.Core import Data, Units
from RCAIDE.Library.Methods.Powertrain.Propulsors.Electric_Rotor import design_electric_rotor

from design_space import canonicalize_discrete, repair_continuous


TIMEOUT_SECONDS = 12 * 60
_REFERENCE_PROPULSOR = None


@dataclass
class EvaluationResult:
    mission_energy_Wh: float | None
    feasible: bool
    status: str
    constraints: Dict[str, float]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _network(config):
    networks = config.networks
    if isinstance(networks, dict):
        if "electric" in networks:
            return networks["electric"]
        for value in networks.values():
            if "Electric" in str(type(value)):
                return value
    return getattr(networks, "electric", None)


def _reference_propulsor(nexus_template):
    global _REFERENCE_PROPULSOR
    if _REFERENCE_PROPULSOR is None:
        net = _network(nexus_template.vehicle_configurations.base)
        if net is not None and getattr(net, "propulsors", None):
            _REFERENCE_PROPULSOR = copy.deepcopy(next(iter(net.propulsors)))
    return copy.deepcopy(_REFERENCE_PROPULSOR)


def rebuild_propulsors(config, nexus_template, discrete, continuous):
    """Rebuild propulsors and mission throttle assignments from one candidate."""
    discrete = canonicalize_discrete(discrete)
    net = _network(config)
    base_net = _network(nexus_template.vehicle_configurations.base)
    if net is None or base_net is None:
        raise RuntimeError("electric propulsion network is unavailable")

    busses = getattr(net, "busses", getattr(net, "buses", None))
    if not busses:
        raise RuntimeError("electric bus is unavailable")
    bus = next(iter(busses.values())) if isinstance(busses, dict) else busses[0]
    ref = _reference_propulsor(nexus_template)
    if ref is None:
        raise RuntimeError("reference propulsor is unavailable")

    layout = discrete["prop_layout"]
    count = int(discrete["propeller_count"])
    span = float(config.wings.main_wing.spans.projected)
    if count % 2:
        raise ValueError("propeller count must be even")

    total_design_thrust = 0.0
    for propulsor in base_net.propulsors:
        cruise = getattr(getattr(propulsor, "rotor", None), "cruise", None)
        thrust = getattr(cruise, "design_thrust", 0.0) if cruise is not None else 0.0
        total_design_thrust += max(float(thrust or 0.0), 0.0)
    total_design_thrust = total_design_thrust or 4000.0

    if layout == "wingtip":
        y_positions = [-span / 2.0, span / 2.0]
    else:
        half = count // 2
        coverage = float(continuous["prop_span_spacing"]) * span / 2.0
        left = np.linspace(-coverage, -coverage / max(half, 1), half)
        right = -left[::-1]
        y_positions = list(left) + list(right)

    net.propulsors.clear()
    per_propulsor_thrust = total_design_thrust / len(y_positions)
    for index, y_position in enumerate(y_positions):
        propulsor = copy.deepcopy(ref)
        origin = [[2.0, float(y_position), 0.95]]
        propulsor.tag = f"propulsor_{index}"
        propulsor.origin = origin
        for component_name in ("motor", "rotor", "electronic_speed_controller", "nacelle"):
            component = getattr(propulsor, component_name, None)
            if component is not None:
                component.origin = origin

        rotor = propulsor.rotor
        rotor.tag = f"rotor_{index}"
        rotor.number_of_blades = int(discrete["blade_count"])
        rotor.cruise.design_thrust = per_propulsor_thrust
        # A copied reference rotor retains its previous design power. RCAIDE
        # requires exactly one design target for each redesign.
        rotor.cruise.design_power = None
        if layout == "wingtip":
            rotor.tip_radius = float(continuous["tip_prop_diameter"]) / 2.0
            rotor.cruise.design_angular_velocity = float(continuous["tip_prop_rpm"]) * Units.rpm
        else:
            rotor.tip_radius *= float(continuous["prop_diameter_ratio"])
        design_electric_rotor(propulsor)
        net.propulsors.append(propulsor)

    bus.assigned_propulsors = [[propulsor.tag for propulsor in net.propulsors]]


def _apply_wing_airfoil(config, airfoil_name: str):
    from pathlib import Path

    file_name = {"Clark_Y": "Clark_y"}.get(airfoil_name, airfoil_name)
    path = Path(__file__).resolve().parent / "Airfoils" / f"{file_name}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    airfoil = RCAIDE.Library.Components.Airfoils.Airfoil()
    airfoil.tag = airfoil_name
    airfoil.coordinate_file = str(path)
    for segment in config.wings.main_wing.segments:
        if hasattr(segment, "airfoils"):
            segment.airfoils.clear()
            segment.append_airfoil(airfoil)


def _set_mission_propulsors(mission, propulsor_tags):
    for segment in mission.segments.values():
        throttle = getattr(getattr(segment, "assigned_control_variables", None), "throttle", None)
        if throttle is not None and getattr(throttle, "active", False):
            throttle.assigned_propulsors = [list(propulsor_tags)]


def _flutter_margin(continuous):
    ar = float(continuous["aspect_ratio"])
    tc = float(continuous["thickness_to_chord"])
    taper = float(continuous["taper_ratio"])
    sweep = float(continuous["quarter_chord_sweep"])
    k_ar = 1.0 + (5.0 / 15.0) * (1.0 / ar - 0.5)
    k_taper = (np.clip(taper, 1.0e-3, 1.0) / 0.63) ** 0.18
    k_sweep = 1.0 / max(np.cos(np.radians(sweep)), 0.5)
    reference = (0.12 / 8.8) * (1.0 + (5.0 / 15.0) * (1.0 / 8.8 - 0.5))
    return float((tc / ar) * k_ar * k_taper * k_sweep / reference)


def _structural_ratio(config, continuous):
    mtow = float(getattr(config.mass_properties, "max_takeoff", 0.0))
    wing_mass = max((mtow - 1156.0) / 2.0, 1.0)
    span = float(getattr(config.wings.main_wing.spans, "projected", 0.0))
    root_chord = float(getattr(config.wings.main_wing.chords, "root", 0.0))
    thickness = float(continuous["thickness_to_chord"])
    spar_height = 0.8 * thickness * root_chord
    root_moment = 3.75 * mtow * 9.81 * span / (6.0 * np.pi)
    allowable = 4892.0 * wing_mass * max(spar_height, 1.0e-6)
    return float(root_moment / allowable)


def evaluate_candidate(nexus_template, continuous, discrete) -> EvaluationResult:
    continuous = repair_continuous(continuous, discrete)
    discrete = canonicalize_discrete(discrete)
    nexus = copy.deepcopy(nexus_template)
    config = nexus.vehicle_configurations.base
    wing = config.wings.main_wing

    try:
        wing.aspect_ratio = continuous["aspect_ratio"]
        wing.areas.reference = continuous["wing_area"]
        wing.sweeps.quarter_chord = continuous["quarter_chord_sweep"] * Units.deg
        wing.thickness_to_chord = continuous["thickness_to_chord"]
        wing._AR_correct = continuous["aspect_ratio"]
        wing._tc_correct = continuous["thickness_to_chord"]
        wing._taper_correct = continuous["taper_ratio"]
        wing._sweep_correct = continuous["quarter_chord_sweep"]
        wing.taper = continuous["taper_ratio"]
        wing.twists.root = continuous["root_twist"] * Units.deg
        wing.twists.tip = continuous["tip_twist"] * Units.deg
        for segment in wing.segments:
            if getattr(segment, "tag", "") in ("inboard", "outboard"):
                segment.dihedral_outboard = continuous["dihedral"] * Units.deg
        if discrete["has_winglet"] == "yes":
            wing.winglet_fraction = continuous["winglet_fraction"]
            for segment in wing.segments:
                if getattr(segment, "tag", "") == "winglet":
                    segment.dihedral_outboard = continuous["winglet_cant_angle"] * Units.deg
        else:
            wing.winglet_fraction = 0.0

        _apply_wing_airfoil(config, discrete["wing_airfoil"])
        rebuild_propulsors(config, nexus_template, discrete, continuous)
        net = _network(config)
        _set_mission_propulsors(nexus.missions.base_mission, [propulsor.tag for propulsor in net.propulsors])
        nexus.procedure(nexus)
    except Exception as error:
        return EvaluationResult(
            mission_energy_Wh=None,
            feasible=False,
            status=f"evaluation_error:{type(error).__name__}",
            constraints={"solver": 1.0},
            metrics={"exception": type(error).__name__},
        )

    summary = nexus.summary
    if not bool(summary.get("converged", False)):
        return EvaluationResult(None, False, "mission_not_converged", {"solver": 1.0}, {})

    energy = float(summary.total_energy_consumed)
    soc = float(summary.min_soc_margin)
    stall = float(getattr(summary, "stall_speed", np.nan))
    span = float(np.sqrt(continuous["aspect_ratio"] * continuous["wing_area"]))
    flutter = _flutter_margin(continuous)
    structure = _structural_ratio(config, continuous)
    min_span = 10.0 if discrete["prop_layout"] == "wingtip" else 9.2
    constraints = {
        "soc": 0.15 - soc,
        "stall_speed": stall - 31.4,
        "flutter": 0.85 - flutter,
        "structure": structure - 1.0,
        "span_min": min_span - span,
        "span_max": span - 14.5,
    }
    feasible = bool(np.isfinite(energy) and all(value <= 0.0 for value in constraints.values()))
    metrics = {
        "soc_margin": soc,
        "stall_speed_mps": stall,
        "flutter_margin": flutter,
        "structural_ratio": structure,
        "span_m": span,
        "edge_margin": _edge_margin(continuous),
    }
    return EvaluationResult(energy, feasible, "ok" if feasible else "constraint_violation", constraints, metrics)


def _edge_margin(continuous):
    margins = []
    from design_space import CONTINUOUS_SPECS

    for spec in CONTINUOUS_SPECS:
        value = float(continuous[spec.name])
        unit = (value - spec.lower) / (spec.upper - spec.lower)
        margins.append(min(unit, 1.0 - unit))
    return float(min(margins))


def _child_entry(nexus_template, continuous, discrete, queue):
    try:
        queue.put(evaluate_candidate(nexus_template, continuous, discrete).to_dict())
    except Exception as error:
        queue.put(EvaluationResult(None, False, f"worker_error:{type(error).__name__}", {"worker": 1.0}, {}).to_dict())


def evaluate_with_timeout(nexus_template, continuous, discrete, timeout_seconds=TIMEOUT_SECONDS) -> EvaluationResult:
    queue = mp.Queue()
    process = mp.Process(target=_child_entry, args=(nexus_template, continuous, discrete, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return EvaluationResult(None, False, "timeout", {"timeout": 1.0}, {})
    if queue.empty():
        return EvaluationResult(None, False, "worker_no_result", {"worker": 1.0}, {})
    payload = queue.get()
    return EvaluationResult(**payload)
