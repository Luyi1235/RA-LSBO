"""Versioned conceptual propulsor mapping with explicit static clearances."""

from copy import deepcopy
from functools import wraps
import math
import numpy as np

GEOMETRY_VERSION = 'constant_disk_area_clearance_20260927_v2'
GEOMETRY_CONSTRAINTS = ('prop_disk_clearance', 'prop_fuselage_clearance', 'prop_wing_clearance')
DEFAULTS = dict(version=GEOMETRY_VERSION, reference_diameter_m=1.72, reference_rpm=2700.,
                fuselage_half_width_m=.60, min_disk_clearance_m=.05,
                min_fuselage_clearance_m=.05, min_wing_clearance_m=.05)


def normalize_geometry(settings=None):
    settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - set(DEFAULTS):
        raise ValueError('Unsupported propulsion geometry settings')
    result = dict(DEFAULTS, **settings)
    if result['version'] != GEOMETRY_VERSION:
        raise ValueError('Incompatible propulsion geometry version')
    for key, value in result.items():
        if key != 'version' and (not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError(f'Invalid geometry setting: {key}')
    # Reference dimensions identify the archived physical template, not tuning knobs.
    for key in ('reference_diameter_m', 'reference_rpm'):
        if result[key] != DEFAULTS[key]:
            raise ValueError(f'{key} must match the bundled reference propulsor')
    return result


def clearance_constraints(y_positions, radii, span, layout, settings):
    y, radius = np.asarray(y_positions, dtype=float), np.asarray(radii, dtype=float)
    if y.size < 2 or y.shape != radius.shape or not np.all(np.isfinite(y)) or not np.all(np.isfinite(radius)):
        raise ValueError('Invalid propulsor geometry arrays')
    gaps = [abs(y[i] - y[j]) - radius[i] - radius[j]
            for i in range(len(y)) for j in range(i + 1, len(y))]
    values = dict(prop_disk_clearance=float(settings['min_disk_clearance_m'] - min(gaps)),
                prop_fuselage_clearance=float(settings['fuselage_half_width_m'] +
                    settings['min_fuselage_clearance_m'] - np.min(np.abs(y) - radius)),
                prop_wing_clearance=float(np.max(np.abs(y) + radius) + settings['min_wing_clearance_m'] - span / 2)
                    if layout == 'leading_edge_distributed' else 0.)
    return {key: 0. if abs(value) < 1e-12 else value for key, value in values.items()}


def make_layout(continuous, discrete, settings=None):
    settings = normalize_geometry(settings)
    span = math.sqrt(float(continuous['aspect_ratio']) * float(continuous['wing_area']))
    layout = discrete['prop_layout']
    if layout == 'wingtip':
        count = 2
        diameter = float(continuous['tip_prop_diameter'])
        rpm = float(continuous['tip_prop_rpm'])
        positions = [-span / 2, span / 2]
    elif layout == 'leading_edge_distributed':
        count = discrete['propeller_count']
        if count not in (2, 4, 6, 8, 10, 12):
            raise ValueError('Unsupported propeller count')
        diameter = settings['reference_diameter_m'] * math.sqrt(2. / count) * float(continuous['prop_diameter_ratio'])
        rpm = settings['reference_rpm'] * settings['reference_diameter_m'] / diameter
        inner = settings['fuselage_half_width_m'] + settings['min_fuselage_clearance_m'] + diameter / 2
        outer_limit = span / 2 - settings['min_wing_clearance_m'] - diameter / 2
        coverage = float(continuous['prop_span_spacing'])
        if not 0 <= coverage <= 1:
            raise ValueError('Spanwise spacing ratio must be in [0, 1]')
        outer = inner + coverage * (outer_limit - inner)
        half = count // 2
        positive = np.linspace(inner, outer, half) if half > 1 else np.array([outer])
        positions = np.concatenate([-positive[::-1], positive]).tolist()
    else:
        raise ValueError('Unsupported propulsion layout')
    if not all(math.isfinite(v) and v > 0 for v in (span, diameter, rpm)):
        raise ValueError('Nonpositive or nonfinite layout dimension')
    constraints = clearance_constraints(positions, [diameter / 2] * count, span, layout, settings)
    return dict(version=GEOMETRY_VERSION, layout=layout, count=count, span_m=span,
                diameter_m=diameter, rpm=rpm, y_positions_m=positions, constraints=constraints,
                total_disk_area_m2=count * math.pi * diameter ** 2 / 4,
                rotational_tip_speed_m_s=math.pi * diameter * rpm / 60,
                settings=settings)


def install_propulsion_geometry(evaluator, nexus, diagnostics, settings=None):
    """Replace only the archived propulsor rebuild; retain true rotor/motor design."""
    settings = normalize_geometry(settings)
    diagnostics['propulsion_geometry_version'] = GEOMETRY_VERSION
    diagnostics['propulsion_geometry_settings'] = settings
    diagnostics['nonwing_mass_scope'] = 'legacy_fixed_nonwing_mass; not a variable propulsion mass model'

    def rebuild(config, template, discrete, continuous):
        plan = make_layout(continuous, discrete, settings)
        if any(value > 0 for value in plan['constraints'].values()):
            raise ValueError('Propulsor layout failed static clearance precheck')
        half_width = 0.
        for fuselage in config.fuselages:
            half_width = max(half_width, float(fuselage.width) / 2,
                max((float(s.width) / 2 for s in fuselage.segments), default=0.))
        if half_width > settings['fuselage_half_width_m'] + 1e-8:
            raise ValueError('Declared fuselage envelope is smaller than actual vehicle')
        net, base_net = evaluator._network(config), evaluator._network(template.vehicle_configurations.base)
        ref = evaluator._reference_propulsor(template)
        if net is None or base_net is None or ref is None:
            raise ValueError('Missing electric propulsion template')
        if not math.isclose(2 * float(ref.rotor.tip_radius), settings['reference_diameter_m'], abs_tol=1e-9):
            raise ValueError('Reference propeller diameter mismatch')
        total_thrust = sum(float(p.rotor.cruise.design_thrust) for p in base_net.propulsors)
        if not math.isfinite(total_thrust) or total_thrust <= 0:
            raise ValueError('Invalid total reference design thrust')
        config.wings.main_wing.spans.projected = plan['span_m']
        net.propulsors.clear()
        size_ratio = plan['diameter_m'] / settings['reference_diameter_m']
        for index, y in enumerate(plan['y_positions_m']):
            propulsor = deepcopy(ref)
            propulsor.tag = f'propulsor_{index}'
            origin = [[float(ref.rotor.origin[0][0]), y, float(ref.rotor.origin[0][2])]]
            propulsor.origin = origin
            for name in ('motor', 'rotor', 'electronic_speed_controller', 'nacelle'):
                component = getattr(propulsor, name, None)
                if component is not None:
                    component.origin = deepcopy(origin)
            rotor = propulsor.rotor
            rotor.tag = f'rotor_{index}'
            rotor.tip_radius = plan['diameter_m'] / 2
            rotor.hub_radius = float(ref.rotor.hub_radius) * size_ratio
            rotor.number_of_blades = int(discrete['blade_count'])
            rotor.cruise.design_angular_velocity = plan['rpm'] * evaluator.Units.rpm
            rotor.cruise.design_thrust = total_thrust / plan['count']
            rotor.cruise.design_power = None
            nacelle = getattr(propulsor, 'nacelle', None)
            if nacelle is not None:
                nacelle.length *= size_ratio
                nacelle.diameter *= size_ratio
                nacelle.areas.wetted *= size_ratio ** 2
                for segment in nacelle.segments:
                    segment.width *= size_ratio
                    segment.height *= size_ratio
            evaluator.design_electric_rotor(propulsor)
            net.propulsors.append(propulsor)
        busses = getattr(net, 'busses', getattr(net, 'buses', None))
        bus = next(iter(busses.values())) if isinstance(busses, dict) else busses[0]
        bus.assigned_propulsors = [[p.tag for p in net.propulsors]]
        config._lsbo_propulsion_layout = plan
        audit(config)

    def audit(config):
        from .propulsion_diagnostics import geometry_report
        plan = config._lsbo_propulsion_layout
        propulsors = list(evaluator._network(config).propulsors)
        report = geometry_report(propulsors)
        actual_y = [float(p.rotor.origin[0][1]) for p in propulsors]
        actual_r = [float(p.rotor.tip_radius) for p in propulsors]
        span = float(config.wings.main_wing.spans.projected)
        if abs(span - plan['span_m']) > 1e-8 or not np.allclose(actual_y, plan['y_positions_m'], atol=1e-8, rtol=0):
            raise ValueError('Candidate span/propulsor placement drifted after sizing')
        constraints = clearance_constraints(actual_y, actual_r, span, plan['layout'], settings)
        if any(value > 1e-8 for value in constraints.values()):
            raise ValueError('Actual rebuilt geometry violates clearance contract')
        report.update(scope='enforced_static_clearances_not_full_CAD_or_certification',
                      version=GEOMETRY_VERSION, constraints=constraints, layout_plan=plan,
                      placement_span_m=span, requested_span_m=plan['span_m'], placement_span_error_m=span-plan['span_m'])
        diagnostics['propulsion_geometry'] = report

    evaluator.rebuild_propulsors = rebuild
    native_sizing = nexus.procedure.simple_sizing

    @wraps(native_sizing)
    def checked_sizing(candidate_nexus):
        result = native_sizing(candidate_nexus)
        from RCAIDE.Library.Methods.Geometry.Planform import wing_segmented_planform
        base = candidate_nexus.vehicle_configurations.base
        span = base._lsbo_propulsion_layout['span_m']
        # AR is defined using projected span. Do not apply a second projection
        # to an angle that the archived routine has already converted to radians.
        for config in candidate_nexus.vehicle_configurations:
            wing = config.wings.main_wing
            old_span = float(wing.spans.projected)
            wing.chords.root *= old_span / span
            wing.spans.projected = span
            wing_segmented_planform(wing)
        audit(base)
        return result

    nexus.procedure.simple_sizing = checked_sizing
