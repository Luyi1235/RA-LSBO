"""Opt-in four-rotor group-control experiment, not a calibrated X57 DEP model."""

from copy import deepcopy
from functools import wraps
import math
import numpy as np

VERSION = 'dep_group_schedule_20261002_v1'
SEGMENTS = ('DER', 'ICA', 'climb_1', 'climb_2', 'cruise', 'decent', 'Downleg',
            'Reserve_Climb', 'Reserve_Cruise', 'Reserve_Descent', 'Baseleg')


def normalize(settings):
    if not isinstance(settings, dict) or set(settings) != {'version', 'scope', 'phases'}:
        raise ValueError('Group schedule requires explicit version, scope and phases')
    if settings['version'] != VERSION or settings['scope'] != 'existing_four_rotor_control_experiment':
        raise ValueError('Unsupported DEP group experiment')
    if set(settings['phases']) != set(SEGMENTS):
        raise ValueError('Specify the auxiliary-group state for all eleven segments')
    for phase in settings['phases'].values():
        if set(phase) != {'state', 'throttle_ratio', 'rotor_drag_area_m2_each', 'source'}:
            raise ValueError('Incomplete phase control/drag provenance')
        ratio, area = phase['throttle_ratio'], phase['rotor_drag_area_m2_each']
        if (type(ratio) not in (int, float) or type(area) not in (int, float)
                or not math.isfinite(ratio) or not math.isfinite(area)
                or not isinstance(phase['source'], str) or not phase['source'].strip()):
            raise ValueError('Invalid control/drag values or source')
        if phase['state'] == 'active':
            if not 0 < ratio <= 1 or area != 0:
                raise ValueError('Active rotor uses BEM drag/power, not an extra stopped-rotor drag')
        elif phase['state'] == 'stopped':
            if ratio != 0 or area <= 0:
                raise ValueError('Stopped group needs zero command and positive explicit drag area')
        else:
            raise ValueError('Only active and stopped states are implemented; no windmilling/folding model')
    return deepcopy(settings)


def split_groups(props):
    """Select outer symmetric pair; all hardware remains installed."""
    ordered = sorted(props, key=lambda p: float(p.rotor.origin[0][1]))
    if len(ordered) != 4 or len({p.tag for p in ordered}) != 4:
        raise ValueError('Control experiment requires exactly four distinct installed propulsors')
    for left, right in ((ordered[0], ordered[3]), (ordered[1], ordered[2])):
        a, b = np.asarray(left.rotor.origin[0]), np.asarray(right.rotor.origin[0])
        if (not np.allclose(a, b * [1, -1, 1], atol=1e-8, rtol=0)
                or a[1] >= 0 or b[1] <= 0
                or not math.isclose(left.rotor.tip_radius, right.rotor.tip_radius)):
            raise ValueError('Group-control experiment requires symmetric paired rotors')
    return [ordered[0].tag, ordered[3].tag], [ordered[1].tag, ordered[2].tag]


def set_commands(segment, cruise, auxiliary, phase, props):
    energy = segment.state.conditions.energy
    command = np.asarray(energy.propulsors[cruise[0]].throttle)
    for tag in auxiliary:
        energy.propulsors[tag].throttle = command.copy() * phase['throttle_ratio']
        if phase['state'] == 'stopped':
            prop = props[tag]
            esc = energy.modulators[prop.electronic_speed_controller.tag]
            esc.throttle = command * 0
            for side in (esc.inputs, esc.outputs):
                side.power = command * 0
                side.current = command * 0
            esc.outputs.voltage = command * 0
            # Stopped components are excluded from network evaluation. Initialize
            # their recorded state explicitly, rather than borrowing previous BEM data.
            rotor = energy.converters[prop.rotor.tag]
            for field in ('power', 'omega', 'torque', 'efficiency'):
                rotor[field] = command * 0
            rotor.thrust = np.zeros((len(command), 3))


def add_stopped_drag(segment, drag_area):
    c = segment.state.conditions
    area = float(segment.analyses.aerodynamics.vehicle.reference_area)
    q = np.asarray(c.freestream.dynamic_pressure)
    if area <= 0 or not np.isfinite(q).all():
        raise ValueError('Invalid stopped-rotor aerodynamic reference')
    increment = np.where(q > 0, drag_area / area, 0.)
    c.aerodynamics.coefficients.drag.total += increment
    c.aerodynamics.coefficients.drag.dep_stopped_rotors = increment
    c.frames.wind.force_vector[:, 0] -= (np.maximum(q, 0) * drag_area)[:, 0]


def _attach_controls(segment, cruise, auxiliary, phase, network):
    unpack = segment.process.iterate.unknowns.network
    def commands(seg):
        unpack(seg)
        set_commands(seg, cruise, auxiliary, phase, network.propulsors)
    segment.process.iterate.unknowns.network = commands
    native_aero = segment.process.iterate.conditions.aerodynamics
    def aero(seg):
        native_aero(seg)
        add_stopped_drag(seg, len(auxiliary) * phase['rotor_drag_area_m2_each'])
    segment.process.iterate.conditions.aerodynamics = aero


def install(nexus, diagnostics, settings):
    settings = normalize(settings)
    physics = diagnostics.get('propulsion_research', {}).get('settings', {})
    if (physics.get('native_wake', True) or not physics.get('synchronize_mission_mass')
            or not physics.get('correct_nacelle_drag')):
        raise ValueError('Group-control prototype requires mass/drag consistency and wake disabled')
    report = dict(version=VERSION, scope=settings['scope'], phases=[], audit_status='not_evaluated',
                  hardware='existing_cruise_designed_rotors_reused_not_NASA_highlift_hardware',
                  control='one_trim_throttle_with_prescribed_auxiliary_ratio_no_extra_trim_DOF',
                  stopped_drag='explicit_synthetic_sensitivity_not_calibrated',
                  mass_policy='all_installed_hardware_retained_no_shutdown_mass_credit')
    diagnostics['dep_group_schedule'] = report
    diagnostics['propulsion_research']['phase_policy'] = VERSION
    mission = nexus.missions.base_mission
    native_energy = mission.process.initialize.energy

    @wraps(native_energy)
    def initialize(candidate_mission):
        if set(s.tag for s in candidate_mission.segments.values()) != set(SEGMENTS):
            raise ValueError('Mission differs from the explicit group schedule')
        for segment in candidate_mission.segments.values():
            if any(getattr(segment.flight_dynamics, key, False) for key in ('moment_x', 'moment_y', 'moment_z')):
                raise ValueError('Stopped-rotor drag adapter supports force trim, not moment trim')
            # Copy after mass/aero preprocessing: per-segment state must not switch
            # a shared vehicle network off in all other mission segments.
            segment.analyses.energy = deepcopy(segment.analyses.energy)
            vehicle = segment.analyses.energy.vehicle
            networks = list(vehicle.networks.values())
            if len(networks) != 1 or len(networks[0].busses) != 1:
                raise ValueError('Group experiment requires one electrical network and bus')
            network = networks[0]
            cruise, auxiliary = split_groups(list(network.propulsors))
            phase = settings['phases'][segment.tag]
            active = phase['state'] == 'active'
            bus = next(iter(network.busses.values()))
            bus.assigned_propulsors = [cruise] + ([auxiliary] if active else [])
            for prop in network.propulsors:
                prop.active = prop.tag in cruise or active
            segment.assigned_control_variables.throttle.assigned_propulsors = [cruise]
            report['phases'].append(dict(segment=segment.tag, cruise=cruise, auxiliary=auxiliary,
                **phase, installed_tags=[p.tag for p in network.propulsors],
                installed_takeoff_mass_kg=float(vehicle.mass_properties.takeoff),
                motor_groups=len(bus.assigned_propulsors)))
        native_energy(candidate_mission)
        for segment in candidate_mission.segments.values():
            network = next(iter(segment.analyses.energy.vehicle.networks.values()))
            cruise, auxiliary = split_groups(list(network.propulsors))
            _attach_controls(segment, cruise, auxiliary, settings['phases'][segment.tag], network)
    mission.process.initialize.energy = initialize

    native_post = nexus.procedure.post_process
    @wraps(native_post)
    def post(candidate):
        records = {p['segment']: p for p in report['phases']}
        for segment in candidate.results.base.segments.values():
            phase = records[segment.tag]
            network = next(iter(segment.analyses.energy.vehicle.networks.values()))
            energy = segment.conditions.energy
            for tag in phase['auxiliary']:
                prop = network.propulsors[tag]
                actual = energy.propulsors[tag].throttle
                expected = energy.propulsors[phase['cruise'][0]].throttle * phase['throttle_ratio']
                if not np.allclose(actual, expected, atol=1e-10, rtol=1e-10):
                    raise ValueError('Auxiliary group command not applied')
                if phase['state'] == 'stopped':
                    rotor = energy.converters[prop.rotor.tag]
                    esc = energy.modulators[prop.electronic_speed_controller.tag]
                    if any(np.max(np.abs(x)) > 1e-8 for x in (rotor.power, rotor.omega, rotor.thrust, esc.inputs.power)):
                        raise ValueError('Stopped rotor still produces power/thrust or consumes electrical power')
            area = len(phase['auxiliary']) * phase['rotor_drag_area_m2_each']
            cd = segment.conditions.aerodynamics.coefficients.drag.dep_stopped_rotors
            if not np.allclose(cd, area / float(segment.analyses.aerodynamics.vehicle.reference_area)):
                raise ValueError('Stopped rotor drag missing from aerodynamic trim')
            phase['audit_passed'] = True
        report['audit_status'] = 'passed'
        diagnostics['propulsion_research']['dep_group_schedule'] = report
        return native_post(candidate)
    nexus.procedure.post_process = post
