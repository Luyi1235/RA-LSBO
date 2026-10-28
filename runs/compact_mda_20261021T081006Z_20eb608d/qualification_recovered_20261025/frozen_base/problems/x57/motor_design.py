"""Solve the existing RCAIDE DC motor design equalities without slack targets."""

import math
from functools import wraps

MOTOR_DESIGN_VERSION = 'dc_exact_design_20260926'


def parameters(voltage, omega, torque, efficiency, no_load_current, gear_ratio=1.):
    values = (voltage, omega, torque, efficiency, no_load_current, gear_ratio)
    if not all(math.isfinite(v) for v in values) or min(voltage, omega, torque, gear_ratio) <= 0:
        raise ValueError('Invalid DC motor design inputs')
    if not 0 < efficiency < 1 or no_load_current < 0:
        raise ValueError('Invalid DC motor efficiency or no-load current')
    speed = omega * gear_ratio
    # Same torque/efficiency equations as RCAIDE design_optimal_motor:
    # Q=(I-I0)/Kv, eta=Q*omega_internal/(V*I), V=omega_internal/Kv+I*R.
    current = torque * speed / (efficiency * voltage)
    kv = (current - no_load_current) / torque
    if kv <= 0:
        raise ValueError('No positive DC motor speed constant for these design targets')
    resistance = (voltage - speed / kv) / current
    if not .01 <= kv <= 100 or not .001 <= resistance <= 10:
        raise ValueError(f'DC motor hard-design solution outside original bounds: Kv={kv}, R={resistance}')
    checked_current = (voltage - speed / kv) / resistance
    actual_torque = (checked_current - no_load_current) / kv
    actual_efficiency = (1 - no_load_current / checked_current) * speed / (voltage * kv)
    if not math.isclose(actual_torque, torque, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError('DC motor torque equality verification failed')
    if not math.isclose(actual_efficiency, efficiency, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError('DC motor efficiency equality verification failed')
    return {'speed_constant': kv, 'resistance': resistance, 'design_current': current,
            'torque_residual_Nm': actual_torque - torque,
            'efficiency_residual': actual_efficiency - efficiency}


def install_motor_design(evaluator, diagnostics, *, no_load_fraction=None):
    """Scope the callback to candidate propulsor redesign in the isolated worker."""
    from RCAIDE.Library.Components.Powertrain.Converters import DC_Motor
    native = evaluator.design_electric_rotor
    if getattr(native, '_lsbo_exact_motor', False):
        raise ValueError('Exact motor design already installed')
    if 'design_optimal_motor' not in native.__globals__:
        raise ValueError('Unsupported RCAIDE electric-rotor design interface')
    diagnostics['motor_design_version'] = MOTOR_DESIGN_VERSION
    records = diagnostics.setdefault('motor_design_records', [])

    def design(motor):
        if type(motor) is not DC_Motor:
            raise ValueError('Exact motor design supports RCAIDE DC_Motor only')
        loss_record = {}
        if no_load_fraction is not None:
            from .component_scaling import scaled_current
            before = float(motor.no_load_current)
            motor.no_load_current = scaled_current(float(motor.nominal_voltage),
                float(motor.design_angular_velocity),float(motor.design_torque),
                float(motor.efficiency),float(motor.gearbox.gear_ratio),no_load_fraction)
            loss_record = dict(no_load_current_before_A=before,
                no_load_current_A=float(motor.no_load_current),no_load_fraction=no_load_fraction)
        result = parameters(float(motor.nominal_voltage), float(motor.design_angular_velocity),
                            float(motor.design_torque), float(motor.efficiency),
                            float(motor.no_load_current), float(motor.gearbox.gear_ratio))
        for key in ('speed_constant', 'resistance', 'design_current'):
            setattr(motor, key, result[key])
        records.append(dict(result, tag=str(motor.tag), **loss_record))
        return motor

    @wraps(native)
    def checked(*args, **kwargs):
        namespace = native.__globals__
        previous = namespace['design_optimal_motor']
        namespace['design_optimal_motor'] = design
        try:
            return native(*args, **kwargs)
        finally:
            namespace['design_optimal_motor'] = previous

    checked._lsbo_exact_motor = True
    evaluator.design_electric_rotor = checked
