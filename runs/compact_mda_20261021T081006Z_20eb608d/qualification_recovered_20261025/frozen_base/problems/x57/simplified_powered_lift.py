"""Conceptual powered-stall proxy from native axial-wake lift increments.

Transfers an attached-flow increment to the existing empirical lift limit.
This is an explicit engineering approximation, NOT a viscous CLmax prediction.
"""

from copy import deepcopy
from functools import wraps
import json
import math
from pathlib import Path
import numpy as np

VERSION = 'native_delta_lift_concept_screen_20261004_v1'
DEFAULTS = dict(version=VERSION, enabled=True, reference_alpha_deg=8.,
                spanwise_panels=48, chordwise_panels=5)
SCREEN_SPEED_MPS = 31.4


def normalize(settings):
    if (not isinstance(settings, dict) or set(settings) != set(DEFAULTS)
            or settings['version'] != VERSION or type(settings['enabled']) is not bool):
        raise ValueError('Explicit versioned simplified powered-lift settings required')
    alpha = settings['reference_alpha_deg']
    if type(alpha) not in (float, int) or not math.isfinite(alpha) or not 2 <= alpha <= 12:
        raise ValueError('Reference alpha must describe an attached-flow diagnostic point')
    for key, lo, hi in (('spanwise_panels', 12, 64), ('chordwise_panels', 2, 10)):
        if type(settings[key]) is not int or not lo <= settings[key] <= hi:
            raise ValueError('Invalid powered-lift panel count')
    return dict(settings)


def validate_config(config):
    research = config.get('propulsion_research', {})
    if (not all(research.get(k) for k in ('synchronize_mission_mass', 'correct_nacelle_drag'))
            or research.get('native_wake', True)
            or 'section_consistency' not in config
            or any(k in config for k in ('dep_groups', 'powered_lift_probe'))):
        raise ValueError('Simplified lift requires consistent wake-off mission and no group/probe mode')
    return normalize(config['simplified_powered_lift'])


def corrected_limit(dry_clmax, dry_stall, delta_cl, enabled=True):
    if (not all(math.isfinite(x) for x in (dry_clmax, dry_stall, delta_cl))
            or dry_clmax <= 0 or dry_stall <= 0 or dry_clmax+delta_cl <= 0):
        raise ValueError('Invalid equivalent low-speed lift limit')
    effective = dry_clmax + delta_cl if enabled else dry_clmax
    return effective, dry_stall*math.sqrt(dry_clmax/effective)


def choose_operating_point(evaluate, power_limit_W):
    """Bound a common RPM ratio by actual mission peak power and battery limits."""
    if not math.isfinite(power_limit_W) or power_limit_W <= 0:
        raise ValueError('Missing audited mission power bound')
    records, selected, upper = [], None, 1.

    def trial(ratio):
        point = evaluate(float(ratio))
        electrical = point['electrical']
        power = electrical.get('terminal_power_W')
        valid = (electrical.get('within_reference_envelope') is True and power is not None
                 and math.isfinite(power) and 0 <= power <= power_limit_W)
        records.append(dict(rpm_ratio=float(ratio), admissible=valid, electrical=electrical))
        return dict(point, rpm_ratio=float(ratio)) if valid else None

    for ratio in np.linspace(1., .2, 9):
        point = trial(ratio)
        if point is not None:
            selected = point
            break
        upper = float(ratio)
    if selected is not None and selected['rpm_ratio'] < 1.:
        lower = selected['rpm_ratio']
        for _ in range(7):
            mid = (lower+upper)/2
            point = trial(mid)
            if point is None:
                upper = mid
            else:
                lower, selected = mid, point
    return selected, records


def calculate(candidate, evaluator, diagnostics, settings, report):
    from RCAIDE.Library.Methods.Aerodynamics.Vortex_Lattice_Method.VLM import VLM
    from RCAIDE.Library.Methods.Powertrain.Converters.Rotor.compute_rotor_performance import compute_rotor_performance
    from .powered_lift_probe import flight_conditions, drive_power

    vehicle = deepcopy(candidate.vehicle_configurations.base)
    wing = vehicle.wings.main_wing
    network = evaluator._network(vehicle)
    props = list(network.propulsors)
    bus = next(iter(network.busses.values()))
    if not props or any(not p.active for p in props) or not math.isclose(float(bus.efficiency), 1.):
        raise ValueError('Simplified lift currently requires all-active propulsors and reference bus')
    seed = next(iter(candidate.results.base.segments.values())).conditions
    soc = float(diagnostics['minimum_recorded_soc'])
    power_limit = float(diagnostics['maximum_terminal_power_W'])
    if not 0 < soc <= 1:
        raise ValueError('Invalid audited mission SOC')
    auxiliary = float(bus.avionics.power_draw)+float(bus.payload.power_draw)

    def evaluate(ratio):
        c = flight_conditions(seed, SCREEN_SPEED_MPS, settings['reference_alpha_deg'])
        for p in props:
            rotor = c.energy.converters[p.rotor.tag]
            rotor.omega = np.array([[float(p.rotor.cruise.design_angular_velocity)*ratio]])
            rotor.throttle = np.ones((1, 1))
            rotor.design_flag = False
            rotor.commanded_thrust_vector_angle = np.zeros((1, 1))
            compute_rotor_performance(p.rotor, c)
            if float(np.asarray(rotor.power).reshape(-1)[0]) < 0:
                return dict(electrical=dict(within_reference_envelope=False, reason='windmilling_not_credited'))
        return dict(conditions=c, electrical=drive_power(props, c, soc, auxiliary))

    point, trials = choose_operating_point(evaluate, power_limit)
    dry_clmax, dry_stall = float(candidate.summary.cl_max_3d), float(candidate.summary.stall_speed)
    report.update(minimum_mission_soc=soc, mission_peak_terminal_power_W=power_limit,
        speed_mps=SCREEN_SPEED_MPS, dry_clmax=dry_clmax, dry_stall_speed_mps=dry_stall,
        wing_area_m2=float(wing.areas.reference), vehicle_reference_area_m2=float(vehicle.reference_area),
        operating_point_trials=trials, coefficient_reference='main_wing_area_not_aircraft_reference_area',
        empirical_delta_transfer='same_attached_flow_increment_added_to_dry_lift_limit',
        power_policy='not_above_mission_peak_power_at_minimum_mission_SOC',
        delta_cl=0., applied=False, status='unblown_fallback_no_admissible_powered_point')
    if point is not None:
        c = point['conditions']
        paired = {}
        from RCAIDE.Library.Methods.Powertrain.Converters.Rotor.Performance.Blade_Element_Momentum_Theory_Helmholtz_Wake import wake_model
        from .propulsion_research import rowwise_axial_wake
        native_wake = wake_model.compute_wake_induced_velocity
        def axial(rotor, conditions, points, count):
            return rowwise_axial_wake(native_wake, rotor, conditions, points, count)
        wake_model.compute_wake_induced_velocity = axial
        try:
            for wake in (False, True):
                options = deepcopy(candidate.analyses.base.aerodynamics.settings)
                options.number_of_spanwise_vortices = settings['spanwise_panels']
                options.number_of_chordwise_vortices = settings['chordwise_panels']
                options.propeller_wake_model = wake
                result = VLM(c, options, vehicle)
                if 'mission_slipstream' in diagnostics:
                    from .mission_slipstream import circulation_drag
                    result.CDrag_induced = circulation_drag(result, vehicle)[0]
                cl = float(result.CLift_wings[wing.tag][0, 0])
                if not math.isfinite(cl):
                    raise ValueError('Nonfinite native wing lift')
                paired['on' if wake else 'off'] = dict(wing_CL=cl,
                    aircraft_CL=float(result.CLift[0, 0]), aircraft_CDi=float(result.CDrag_induced[0, 0]))
        finally:
            wake_model.compute_wake_induced_velocity = native_wake
        delta = paired['on']['wing_CL']-paired['off']['wing_CL']
        report.update(paired=paired, delta_cl=delta, rpm_ratio=point['rpm_ratio'],
            electrical=point['electrical'], status='conceptual_estimate', applied=settings['enabled'])
    effective, stall = corrected_limit(dry_clmax, dry_stall, report['delta_cl'], settings['enabled'])
    report.update(equivalent_clmax=effective, equivalent_stall_speed_mps=stall,
        dry_stall_constraint=dry_stall-SCREEN_SPEED_MPS, powered_stall_constraint=stall-SCREEN_SPEED_MPS)
    candidate.summary.cl_max_3d = effective
    candidate.summary.stall_speed = stall
    # Do not alter energy, mission states or any other constraint.


def install(nexus, evaluator, diagnostics, config, artifact_dir=None):
    settings = validate_config(config)
    report = diagnostics['simplified_powered_lift'] = dict(version=VERSION, settings=settings,
        scope='conceptual_power_on_low_speed_proxy_not_validated_stall', status='not_run',
        mission_aerodynamics=('coupled_native_mission_see_mission_slipstream_settings'
            if 'mission_slipstream' in config else
            'existing_wake_off_model_native_wake_used_in_low_speed_constraint_only'),
        validated_CLmax=False, motor_thermal_rating_validated=False,
        limitations=['fixed_alpha_delta_transfer', 'native_axial_wake_no_swirl',
                     'no_stall_separation_or_low_speed_trim', 'existing_fixed_propulsion_mass_policy'])
    from .airfoil_assignment import install as assign_airfoil
    assign_airfoil(evaluator, report)
    native = nexus.procedure.post_process

    @wraps(native)
    def post(candidate):
        result = native(candidate)
        if not candidate.summary.converged or diagnostics.get('energy_audit_status') != 'passed':
            raise ValueError('Powered lift requires converged, energy-audited mission')
        try:
            calculate(candidate, evaluator, diagnostics, settings, report)
            diagnostics['propulsion_research']['powered_stall_credit_applied'] = report['applied']
        except Exception as error:
            report.update(status='failed', failure=f'{type(error).__name__}: {error}')
            raise
        finally:
            if artifact_dir is not None:
                path = Path(artifact_dir)/'simplified_powered_lift.json'
                path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
                diagnostics['simplified_powered_lift_file'] = str(path)
        print('[NATIVE LIFT] '+json.dumps({k: report[k] for k in
              ('status', 'delta_cl', 'dry_stall_speed_mps', 'equivalent_stall_speed_mps', 'applied')}), flush=True)
        return result
    nexus.procedure.post_process = post
