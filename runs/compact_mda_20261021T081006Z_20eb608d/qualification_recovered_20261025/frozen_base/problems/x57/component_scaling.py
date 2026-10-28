"""Explicit conceptual component-similarity assumptions, not X57 calibration."""

from functools import wraps
import math

VERSION = 'component_similarity_v1'
DEFAULTS = dict(version=VERSION, scale_motor_no_load=True, reconcile_nacelle_diameter=True,
                leading_edge_exposed_fraction=1.0)


def normalize(settings):
    if (not isinstance(settings, dict) or set(settings) != set(DEFAULTS)
            or settings['version'] != VERSION
            or any(type(settings[k]) is not bool for k in ('scale_motor_no_load','reconcile_nacelle_diameter'))):
        raise ValueError('Explicit versioned component-similarity settings required')
    fraction = settings['leading_edge_exposed_fraction']
    if type(fraction) not in (float,int) or not math.isfinite(fraction) or not .5 <= fraction <= 1.:
        raise ValueError('Concept exposure fraction must be finite and between 0.5 and 1')
    return dict(settings)


def reference_fraction(power, efficiency, voltage, no_load_current):
    values = (power, efficiency, voltage, no_load_current)
    if (not all(math.isfinite(v) for v in values) or min(power, voltage) <= 0
            or not 0 < efficiency < 1 or no_load_current < 0):
        raise ValueError('Invalid reference motor loss inputs')
    fraction = no_load_current / (power / (efficiency * voltage))
    if not 0 <= fraction < 1-efficiency:
        raise ValueError('Reference no-load fraction incompatible with design efficiency')
    return fraction


def scaled_current(voltage, omega, torque, efficiency, gear_ratio, fraction):
    if (not all(math.isfinite(v) for v in (voltage,omega,torque,efficiency,gear_ratio,fraction))
            or min(voltage,omega,torque,gear_ratio) <= 0 or not 0 < efficiency < 1
            or not 0 <= fraction < 1-efficiency):
        raise ValueError('Invalid motor-similarity inputs')
    return fraction * torque * omega * gear_ratio / (efficiency * voltage)


def equivalent_diameter(sections):
    """Circular diameter with the largest elliptical loft-section frontal area."""
    sections = list(sections)
    if (not sections or any(not math.isfinite(v) or v < 0 for row in sections for v in row)
            or any(len(row) != 2 for row in sections)):
        raise ValueError('Invalid nacelle section dimensions')
    area_factor = max(width*height for width,height in sections)
    if area_factor <= 0:
        raise ValueError('Zero nacelle frontal area')
    return math.sqrt(area_factor)


def prepare(evaluator, nexus, diagnostics, settings):
    settings = normalize(settings)
    ref = evaluator._reference_propulsor(nexus)
    if ref is None:
        raise ValueError('Missing reference electric propulsor')
    motor = ref.motor
    power = float(ref.rotor.cruise.design_power)
    voltage, efficiency, idle = map(float, (motor.nominal_voltage,motor.efficiency,motor.no_load_current))
    if not math.isclose(float(motor.gearbox.gear_ratio),1.):
        raise ValueError('Component-similarity reference requires direct-drive motor')
    fraction = reference_fraction(power,efficiency,voltage,idle)
    diagnostics['component_scaling'] = dict(settings=settings,
        scope='declared_conceptual_similarity_assumption_not_measured_X57_calibration',
        motor_rule='constant_no_load_fraction_of_design_input_current',
        reference_power_W=power, reference_voltage_V=voltage, reference_efficiency=efficiency,
        reference_no_load_current_A=idle, reference_no_load_fraction=fraction,
        nacelle_rule='equivalent_diameter_from_maximum_elliptical_loft_section',
        installation_rule='declared_effective_exposed_wetted_fraction_for_leading_edge_only',
        installation_scope='assumed_integration_not_CAD_union_or_measured_interference_drag',
        unchanged=['design_efficiency','ESC_efficiency','rotor_BEM','battery','mission',
                   'design_variables','constraints','fixed_nonwing_mass'])
    return fraction if settings['scale_motor_no_load'] else None


def install_geometry(evaluator, diagnostics, settings):
    settings = normalize(settings)
    original = evaluator.rebuild_propulsors
    @wraps(original)
    def rebuild(config, template, discrete, continuous):
        result = original(config,template,discrete,continuous)
        rows = []
        for prop in evaluator._network(config).propulsors:
            nacelle = prop.nacelle
            old = float(nacelle.diameter)
            equivalent = equivalent_diameter([(float(s.width),float(s.height)) for s in nacelle.segments])
            if settings['reconcile_nacelle_diameter']:
                nacelle.diameter = equivalent
            full_area = float(nacelle.areas.wetted)
            fraction = (settings['leading_edge_exposed_fraction']
                        if discrete['prop_layout'] == 'leading_edge_distributed' else 1.)
            if not math.isfinite(full_area) or full_area <= 0:
                raise ValueError('Corrected nacelle wetted area must precede integration assumption')
            nacelle.areas.wetted = full_area * fraction
            rows.append(dict(tag=prop.tag, inherited_diameter_m=old,
                loft_equivalent_diameter_m=equivalent, applied_diameter_m=float(nacelle.diameter),
                full_loft_wetted_m2=full_area, effective_wetted_m2=float(nacelle.areas.wetted),
                exposed_fraction=fraction,
                no_load_current_A=float(prop.motor.no_load_current),
                motor_design_current_A=float(prop.motor.design_current)))
        diagnostics['component_scaling']['components'] = rows
        return result
    evaluator.rebuild_propulsors = rebuild
