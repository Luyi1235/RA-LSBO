"""Research-only propulsion power-chain checks and low-speed requirements."""

import math
import numpy as np
from .dep_physics import footprint, lift_requirement, conditional_power_requirement, VERSION


def energy_chain(time_s, bus_power_W, rotor_power_W, esc_input_power_W):
    """Same left-endpoint convention as the versioned battery audit.

    ESC input includes motor/ESC losses. Bus minus ESC input retains auxiliary
    and distribution loads; these are not subtracted from the objective.
    """
    t, bus = (np.asarray(v,dtype=float).reshape(-1) for v in (time_s,bus_power_W))
    rotor, esc = (np.asarray(v,dtype=float) for v in (rotor_power_W,esc_input_power_W))
    if (len(t)<2 or bus.shape!=t.shape or rotor.ndim!=2 or esc.shape!=rotor.shape
            or rotor.shape[1]!=len(t) or len(rotor)==0
            or not all(np.isfinite(v).all() for v in (t,bus,rotor,esc))
            or np.any(np.diff(t)<=0)):
        raise ValueError('Invalid propulsion power-chain histories')
    shaft, electrical = rotor.sum(axis=0), esc.sum(axis=0)
    tolerance = .01 + 1e-6*np.maximum(abs(bus),abs(electrical))
    if (np.any(rotor < -tolerance) or np.any(esc-rotor < -tolerance)
            or np.any(bus-electrical < -tolerance)):
        raise ValueError('Nonphysical/unaccounted propulsion power or conversion losses')
    dt = np.diff(t)
    energy = lambda p: float(np.dot(p[:-1],dt)/3600.)
    terminal, mechanical = energy(bus), energy(shaft)
    conversion, auxiliary = energy(electrical-shaft), energy(bus-electrical)
    return dict(integration='left_endpoint_same_as_battery_audit', duration_s=float(t[-1]-t[0]),
                bus_terminal_energy_Wh=terminal, rotor_shaft_energy_Wh=mechanical,
                motor_esc_loss_Wh=conversion, auxiliary_distribution_energy_Wh=auxiliary,
                balance_residual_Wh=terminal-mechanical-conversion-auxiliary,
                rotor_count=len(rotor), minimum_bus_minus_esc_W=float((bus-electrical).min()))


def collect(nexus, evaluator, report):
    vehicle = nexus.vehicle_configurations.base
    wing = vehicle.wings.main_wing
    propulsors = list(evaluator._network(vehicle).propulsors)
    motors = [p.motor.tag for p in propulsors]
    controllers = [p.electronic_speed_controller.tag for p in propulsors]
    if len(set(motors)) != len(motors) or len(set(controllers)) != len(controllers):
        raise ValueError('Independent propulsion energy accounting needs unique component tags')
    wing_z = float(wing.origin[0][2])
    disks = [(float(p.rotor.origin[0][1]),float(p.rotor.tip_radius),
              float(p.rotor.origin[0][2])-wing_z) for p in propulsors]
    area, span, taper = float(wing.areas.reference), float(wing.spans.projected), float(wing.taper)
    fuselage_half_width = float(vehicle._lsbo_propulsion_layout['settings']['fuselage_half_width_m'])
    projection = footprint(area,span,taper,disks,fuselage_half_width=fuselage_half_width)
    # Preserve the inherited unpowered screen as an explicitly named assumption.
    cl2d = 2*(float(wing.thickness_to_chord)/.12)**.15
    cl3d = cl2d/(1+cl2d/(math.pi*float(wing.aspect_ratio)*.8))
    requirement = lift_requirement(float(vehicle.mass_properties.max_takeoff),area,31.4,cl3d)
    conditional = conditional_power_requirement(requirement['required_global_loading_ratio'],
        projection['coverage_fraction'],sum(math.pi*p.rotor.tip_radius**2 for p in propulsors),31.4)
    chain = []
    network = evaluator._network(vehicle)
    busses = list(network.busses.values())
    if len(busses)!=1:
        raise ValueError('Diagnostic currently requires the single-bus X57 architecture')
    for segment in nexus.results.base.segments.values():
        c = segment.conditions
        mechanical = [np.asarray(c.energy.converters[p.rotor.tag].power).reshape(-1) for p in propulsors]
        electrical = [np.asarray(c.energy.modulators[p.electronic_speed_controller.tag].inputs.power).reshape(-1)
                      for p in propulsors]
        entry = energy_chain(c.frames.inertial.time,c.energy[busses[0].tag].power_draw,mechanical,electrical)
        schedule = report.get('dep_group_schedule', {})
        phase = next((p for p in schedule.get('phases', []) if p['segment']==segment.tag), None)
        entry.update(segment=segment.tag,
            roles=['auxiliary_control_prototype' if phase and p.tag in phase['auxiliary'] else 'cruise'
                   for p in propulsors],
            operating_states=[phase['state'] if phase and p.tag in phase['auxiliary'] else 'active'
                              for p in propulsors])
        chain.append(entry)
    report['dep_diagnostics'] = dict(version=VERSION, wing_root_z_m=wing_z,
        realized_wing=dict(area_m2=area,span_m=span,taper=taper,
                           aspect_ratio=float(wing.aspect_ratio),
                           thickness_to_chord=float(wing.thickness_to_chord)),
        projection=projection, low_speed_requirement=requirement,
        conditional_power_requirement=conditional, power_chain=chain,
        power_chain_status='passed',
        powered_stall_credit_applied=False,
        auxiliary_operating_group_verified=report.get('dep_group_schedule', {}).get('audit_status')=='passed',
        high_lift_operating_group_implemented=False,
        variable_installed_propulsion_mass_implemented=False,
        ready_for_architecture_optimization=False)
