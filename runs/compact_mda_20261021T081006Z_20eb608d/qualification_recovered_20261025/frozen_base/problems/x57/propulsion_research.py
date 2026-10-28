"""Opt-in physics consistency study; not a calibrated powered-lift model."""

from functools import wraps
import math
from types import SimpleNamespace
import numpy as np

VERSION = 'propulsion_consistency_research_20261002_v1'
DEFAULTS = dict(version=VERSION, synchronize_mission_mass=True,
                correct_nacelle_drag=True, native_wake=False)


def normalize(settings):
    if not isinstance(settings, dict) or set(settings) != set(DEFAULTS):
        raise ValueError('Research physics requires all explicit versioned settings')
    if settings['version'] != VERSION or any(type(settings[k]) is not bool for k in DEFAULTS if k != 'version'):
        raise ValueError('Invalid research physics settings')
    return dict(settings)


def loft_area(length, stations, angular_points=128):
    """Triangulated surface of the existing piecewise elliptical nacelle loft."""
    rows = np.asarray(stations, dtype=float)
    if (rows.ndim != 2 or rows.shape[1] != 3 or len(rows) < 2
            or not np.isfinite(rows).all() or not math.isfinite(length) or length <= 0
            or np.any(np.diff(rows[:, 0]) <= 0) or np.any(rows[:, 1:] < 0)
            or rows[0, 0] < 0 or rows[-1, 0] > 1 or angular_points < 16):
        raise ValueError('Invalid nacelle loft')
    angle = np.arange(angular_points) * 2 * np.pi / angular_points
    rings = np.stack([np.full((len(rows), angular_points), rows[:, 0, None] * length),
                      rows[:, 1, None] * np.cos(angle) / 2,
                      rows[:, 2, None] * np.sin(angle) / 2], axis=-1)
    a, b = rings[:-1], rings[1:]
    c, d = np.roll(b, -1, axis=1), np.roll(a, -1, axis=1)
    area = .5 * (np.linalg.norm(np.cross(b-a, c-a), axis=-1).sum()
                 + np.linalg.norm(np.cross(c-a, d-a), axis=-1).sum())
    # Close finite-area ends; template pointed ends contribute zero.
    area += np.pi / 4 * (rows[0, 1]*rows[0, 2] + rows[-1, 1]*rows[-1, 2])
    if not np.isfinite(area) or area <= 0:
        raise ValueError('Nonpositive nacelle surface area')
    return float(area)


def corrected_parasite_total(native, state, settings, vehicle):
    """Replace only nacelle aggregation; retain other native drag components."""
    parasite = state.conditions.aerodynamics.coefficients.drag.parasite
    records = []
    for network in vehicle.networks:
        for prop in network.propulsors:
            nacelle = getattr(prop, 'nacelle', None)
            if nacelle is None:
                continue
            raw = np.array(parasite[nacelle.tag].total, copy=True)
            pylon = np.array(parasite[nacelle.tag + '_pylon'].total, copy=True) if nacelle.has_pylon else None
            records.append((nacelle, raw, pylon))
    if len({n.tag for n, _, _ in records}) != len(records):
        raise ValueError('Nacelle tags must be unique before drag evaluation')
    native(state, settings, vehicle)
    reduction = 1. - settings.drag_reduction_factors.parasite_drag
    for nacelle, raw, pylon in records:
        # Native nacelle_drag normalizes by frontal area, NOT pi*d*length.
        frontal = np.pi * nacelle.diameter**2 / 4
        legacy = raw * np.pi * nacelle.diameter * nacelle.length / vehicle.reference_area
        corrected = raw * frontal / vehicle.reference_area
        parasite.total += (corrected - legacy) * reduction
        parasite[nacelle.tag].total = corrected
        if pylon is not None:
            # RCAIDE pylon_drag already normalizes by vehicle reference area.
            legacy_pylon = pylon*np.pi*nacelle.diameter*nacelle.length/vehicle.reference_area
            parasite.total += (pylon-legacy_pylon)*reduction
            parasite[nacelle.tag + '_pylon'].total = pylon


def rowwise_axial_wake(native, rotor, conditions, points, count):
    """Apply native radial axial-wake interpolation at every mission point.

    The installed native routine uses row zero only and an obsolete rotation
    attribute. Swirl is explicitly excluded; no rotation sign is guessed.
    """
    radial = np.asarray(conditions.disc_radial_distribution)
    axial = np.asarray(conditions.blade_axial_induced_velocity)
    tangential = np.asarray(conditions.blade_tangential_induced_velocity)
    if min(len(radial), len(axial), len(tangential)) < count:
        raise ValueError('Insufficient rotor wake time histories')
    proxy = SimpleNamespace(tip_radius=rotor.tip_radius, origin=rotor.origin, rotation=1.)
    result = np.zeros((count, len(points.XC), 3))
    for i in range(count):
        local = SimpleNamespace(disc_radial_distribution=radial[i:i+1],
            blade_axial_induced_velocity=axial[i:i+1],
            blade_tangential_induced_velocity=np.zeros_like(tangential[i:i+1]))
        result[i:i+1] = native(proxy, local, points, 1)
    return result


def statistics(value):
    array = np.asarray(value, dtype=float)
    if not array.size or not np.isfinite(array).all():
        return None
    return dict(min=float(array.min()), max=float(array.max()), mean=float(array.mean()))


def install(evaluator, nexus, diagnostics, settings):
    settings = normalize(settings)
    report = diagnostics['propulsion_research'] = dict(settings=settings,
        scope='experimental_consistency_check_not_validated_powered_lift',
        powered_stall_credit_applied=False,
        propulsion_mass_policy='legacy_fixed_nonwing_mass_retained_pending_calibrated_component_model',
        phase_policy='all_installed_cruise_propulsors_one_throttle_group_not_X57_highlift_plus_cruise')
    rebuild = evaluator.rebuild_propulsors

    @wraps(rebuild)
    def rebuild_checked(config, template, discrete, continuous):
        rebuild(config, template, discrete, continuous)
        components = []
        for i, prop in enumerate(evaluator._network(config).propulsors):
            # Independent histories are necessary before separate control groups.
            prop.motor.tag = f'research_motor_{i}'
            prop.electronic_speed_controller.tag = f'research_esc_{i}'
            nacelle = prop.nacelle
            original_area = float(nacelle.areas.wetted)
            area = loft_area(float(nacelle.length), [(float(s.percent_x_location),
                             float(s.width), float(s.height)) for s in nacelle.segments])
            if settings['correct_nacelle_drag']:
                nacelle.tag = f'research_nacelle_{i}'
                nacelle.areas.wetted = area
            components.append(dict(tag=prop.tag, motor_mass_kg=float(prop.motor.mass_properties.mass),
                nacelle_original_wetted_m2=original_area, nacelle_loft_wetted_m2=area,
                motor_design_efficiency=float(prop.motor.efficiency),
                esc_efficiency=float(prop.electronic_speed_controller.efficiency),
                rotor_design_power_W=float(prop.rotor.cruise.design_power),
                rotor_design_thrust_N=float(prop.rotor.cruise.design_thrust)))
        report['components'] = components
        report['motor_mass_sum_kg'] = sum(c['motor_mass_kg'] for c in components)
    evaluator.rebuild_propulsors = rebuild_checked

    native_weight = nexus.procedure.weights

    @wraps(native_weight)
    def weight(candidate):
        result = native_weight(candidate)
        vehicle = candidate.vehicle_configurations.base
        old_mass = float(vehicle.mass_properties.takeoff)
        target = float(vehicle.mass_properties.max_takeoff)
        if not math.isfinite(target) or target <= 0:
            raise ValueError('Invalid candidate takeoff mass')
        if settings['synchronize_mission_mass']:
            for config in candidate.vehicle_configurations:
                config.mass_properties.takeoff = target
            for analysis in candidate.analyses:
                analysis.weights.vehicle.mass_properties.takeoff = target
                analysis.energy.vehicle.mass_properties.takeoff = target
        report['weight'] = dict(template_mission_mass_kg=old_mass,
                               constraint_mass_kg=target,
                               configured_mission_mass_kg=float(vehicle.mass_properties.takeoff))
        return result
    nexus.procedure.weights = weight

    if settings['correct_nacelle_drag']:
        import inspect
        from RCAIDE.Library.Methods.Aerodynamics.Common import Drag
        native_total = Drag.parasite_total
        source = inspect.getsource(native_total)
        if ('nacelle_parasite_drag * nacelle_parasite_drag' not in source
                or 'np.pi * nacelle.diameter * nacelle.length' not in source):
            raise RuntimeError('Installed nacelle aggregation differs; review before applying correction')
        def total(state, aero_settings, vehicle):
            return corrected_parasite_total(native_total, state, aero_settings, vehicle)
        # Worker-local callback only; installed package files stay untouched.
        Drag.parasite_total = total
        for analysis in nexus.analyses:
            analysis.aerodynamics.process.compute.drag.parasite.total = total

    if settings['native_wake']:
        from RCAIDE.Library.Methods.Powertrain.Converters.Rotor.Performance.Blade_Element_Momentum_Theory_Helmholtz_Wake import wake_model
        native_wake = wake_model.compute_wake_induced_velocity
        def wake(rotor, conditions, points, count):
            return rowwise_axial_wake(native_wake, rotor, conditions, points, count)
        wake_model.compute_wake_induced_velocity = wake
        report['wake_scope'] = 'native_radial_axial_wake_all_rows_no_swirl_not_stall_validated'
        for analysis in nexus.analyses:
            analysis.aerodynamics.settings.use_surrogate = False
            analysis.aerodynamics.settings.propeller_wake_model = True

    native_post = nexus.procedure.post_process

    @wraps(native_post)
    def post(candidate):
        from .dep_diagnostics import collect
        collect(candidate, evaluator, report)
        segments = []
        for seg in candidate.results.base.segments.values():
            c = seg.conditions
            entry = dict(segment=seg.tag, mission_mass_kg=statistics(c.weights.total_mass),
                airspeed_mps=statistics(c.freestream.velocity),
                lift_coefficient=statistics(c.aerodynamics.coefficients.lift.total),
                drag_coefficient=statistics(c.aerodynamics.coefficients.drag.total))
            propulsion = []
            for prop in evaluator._network(candidate.vehicle_configurations.base).propulsors:
                p = c.energy.propulsors[prop.tag]
                rotor = c.energy.converters[prop.rotor.tag]
                record = dict(tag=prop.tag, throttle=statistics(p.throttle))
                for field in ('thrust', 'power', 'efficiency', 'omega'):
                    if field in rotor:
                        record[field] = statistics(rotor[field])
                propulsion.append(record)
            entry['propulsors'] = propulsion
            segments.append(entry)
        report['segments'] = segments
        if settings['synchronize_mission_mass']:
            target = report['weight']['constraint_mass_kg']
            for seg in segments:
                mass = seg['mission_mass_kg']
                if mass is None or max(abs(mass['min']-target), abs(mass['max']-target)) > 1e-6:
                    raise ValueError('Mission mass differs from candidate constraint mass')
        return native_post(candidate)
    nexus.procedure.post_process = post
