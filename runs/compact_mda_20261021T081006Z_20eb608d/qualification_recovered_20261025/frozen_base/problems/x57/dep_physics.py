"""Auditable DEP bookkeeping and conditional actuator-disk diagnostics.

No result here is a calibrated CLmax or permission to relax a constraint.
The planar footprint neglects wake contraction, swirl and separation.
"""

import math
import numpy as np

VERSION = 'dep_conditional_physics_20261002_v1'
MASS_ITEMS = ('motor', 'controller', 'rotor', 'nacelle', 'mount', 'wiring', 'folding_mechanism')


def positive(**values):
    if any(isinstance(v, bool) or not math.isfinite(float(v)) or float(v) <= 0
           for v in values.values()):
        raise ValueError('Expected finite positive inputs: ' + ', '.join(values))


def finite_nonnegative(**values):
    if any(isinstance(v, bool) or not math.isfinite(float(v)) or float(v) < 0
           for v in values.values()):
        raise ValueError('Expected finite nonnegative inputs: ' + ', '.join(values))


def footprint(area, span, taper, disks, fuselage_half_width=0.):
    """Union of coplanar disk projections on a trapezoidal wing, not wake CFD.

    A disk is (spanwise center, radius, vertical hub-to-wing separation).
    Exclude the fuselage strip and merge overlaps, never count a strip twice.
    """
    positive(area=area, span=span, taper=taper)
    finite_nonnegative(fuselage_half_width=fuselage_half_width)
    if taper > 1 or fuselage_half_width >= span/2:
        raise ValueError('Unsupported taper or fuselage envelope')
    root = 2*area/(span*(1+taper))
    slope = root*(taper-1)/(span/2)
    intervals = []
    for y, radius, z in disks:
        positive(radius=radius)
        if not all(math.isfinite(float(v)) for v in (y,z)):
            raise ValueError('Invalid disk position')
        if abs(z) >= radius:
            continue
        reach = math.sqrt(radius*radius-z*z)
        for lo, hi in ((-span/2,-fuselage_half_width),(fuselage_half_width,span/2)):
            left, right = max(lo,y-reach), min(hi,y+reach)
            if right > left:
                intervals.append((left,right))
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1],right)
        else:
            merged.append([left,right])
    def integral(y):
        return root*y + .5*slope*y*abs(y)
    covered = sum(integral(b)-integral(a) for a,b in merged)
    return dict(covered_area_m2=covered, coverage_fraction=covered/area,
                intervals_m=merged, scope='planar_disk_projection_no_wake_contraction')


def ideal_disk(thrust_N, speed_mps, density_kg_m3, disk_area_m2):
    """Axial momentum balance T=2*rho*A*vi*(V+vi), P=T*(V+vi)."""
    finite_nonnegative(thrust_N=thrust_N, speed_mps=speed_mps)
    positive(density_kg_m3=density_kg_m3, disk_area_m2=disk_area_m2)
    c = thrust_N/(2*density_kg_m3*disk_area_m2)
    vi = 0. if c == 0 else 2*c/(math.sqrt(speed_mps**2+4*c)+speed_mps)
    return dict(induced_speed_mps=vi, ideal_shaft_power_W=thrust_N*(speed_mps+vi),
                disk_speed_mps=speed_mps+vi, far_wake_speed_mps=speed_mps+2*vi)


def lift_requirement(mass_kg, area_m2, speed_mps, unpowered_clmax,
                     density_kg_m3=1.225, gravity_m_s2=9.81):
    positive(mass_kg=mass_kg, area_m2=area_m2, speed_mps=speed_mps,
             unpowered_clmax=unpowered_clmax, density_kg_m3=density_kg_m3,
             gravity_m_s2=gravity_m_s2)
    required = 2*mass_kg*gravity_m_s2/(density_kg_m3*speed_mps**2*area_m2)
    return dict(required_cl=required, assumed_unpowered_clmax=unpowered_clmax,
                required_global_loading_ratio=max(1.,required/unpowered_clmax),
                actual_powered_clmax=None, powered_stall_credit_applied=False)


def conditional_power_requirement(loading_ratio, coverage_fraction, disk_area_m2,
                                  speed_mps, density_kg_m3=1.225):
    """Uniform local-q scenario, NOT a stall prediction or certified power bound.

    Hold sectional maximum coefficient fixed; solve 1+f*(q_local/q_inf-1)=R.
    Report ideal disk- and far-wake limits without choosing a favorable value.
    Rotor losses, off-axis forces, separation and wake contraction need validation.
    """
    positive(loading_ratio=loading_ratio, disk_area_m2=disk_area_m2,
             speed_mps=speed_mps, density_kg_m3=density_kg_m3)
    if not math.isfinite(coverage_fraction) or not 0 <= coverage_fraction <= 1:
        raise ValueError('Coverage fraction must be in [0,1]')
    ratio = max(1.,loading_ratio)
    if coverage_fraction == 0 and ratio > 1:
        return dict(status='no_projected_coverage', local_q_ratio=None,
                    ideal_shaft_power_W=None, usable_as_constraint=False)
    local_q = 1. if ratio == 1 else 1+(ratio-1)/coverage_fraction
    delta = speed_mps*(math.sqrt(local_q)-1)
    cases = {}
    for name,k in (('disk_plane',1.),('far_wake',2.)):
        vi = delta/k
        thrust = 2*density_kg_m3*disk_area_m2*vi*(speed_mps+vi)
        cases[name] = dict(ideal_shaft_power_W=thrust*(speed_mps+vi), thrust_N=thrust,
                           induced_speed_mps=vi)
    return dict(status='conditional_scenario_only', local_q_ratio=local_q,
                cases=cases, usable_as_constraint=False,
                assumptions='uniform_loading_fixed_section_CLmax_no_contraction_or_losses')


def installed_mass(records):
    """Require an explicit per-installed-unit entry for every mass category.

    Zero is allowed only when explicitly justified; an omitted item is not zero.
    This checks completeness, not the scientific validity of the supplied source.
    """
    if not records:
        raise ValueError('Missing installed propulsion groups')
    tags, total = set(), 0.
    for group in records:
        tag, count = group['tag'], group['count']
        if not isinstance(tag,str) or not tag.strip() or tag in tags:
            raise ValueError('Invalid/duplicate mass group')
        tags.add(tag)
        if type(count) is not int or count <= 0:
            raise ValueError('Installed count must be a positive integer')
        items = group['items']
        if set(items) != set(MASS_ITEMS):
            raise ValueError('All propulsion mass items must be explicit')
        for name, item in items.items():
            finite_nonnegative(mass_kg=item['mass_kg'])
            if not isinstance(item.get('source'),str) or not item['source'].strip():
                raise ValueError('Missing mass source: ' + name)
            total += count*item['mass_kg']
    return total


def replacement_mass(reference_nonwing_kg, reference_groups, candidate_groups):
    """Replace an already included propulsion system; never add its mass twice."""
    positive(reference_nonwing_kg=reference_nonwing_kg)
    old, new = installed_mass(reference_groups), installed_mass(candidate_groups)
    if old >= reference_nonwing_kg:
        raise ValueError('Reference propulsion cannot exhaust nonwing mass')
    return dict(reference_propulsion_kg=old, candidate_propulsion_kg=new,
                propulsion_delta_kg=new-old,
                candidate_nonwing_kg=reference_nonwing_kg-old+new,
                requires_wing_mass_reiteration=True,
                source_validity_verified=False)


def phase_energy(time_s, speed_mps, density_kg_m3, groups):
    """Integrate explicit group loads; folded does NOT mean zero drag/mass.

    Inputs are per-unit shaft power, motor/ESC efficiency and rotor-only flat
    plate area. Rotor drag is a force contribution to be returned to trim,
    not a guessed electrical-energy addition. Nacelle drag remains separate.
    """
    t, speed, density = (np.asarray(v,dtype=float) for v in (time_s,speed_mps,density_kg_m3))
    if (t.ndim != 1 or len(t)<2 or speed.shape != t.shape or density.shape != t.shape
            or not all(np.isfinite(v).all() for v in (t,speed,density))
            or np.any(np.diff(t)<=0) or np.any(speed<0) or np.any(density<=0)):
        raise ValueError('Invalid phase histories')
    shaft, electrical, drag = (np.zeros_like(t) for _ in range(3))
    tags, cruise_active = set(), False
    for group in groups:
        tag, count, role, state = (group[k] for k in ('tag','count','role','state'))
        if tag in tags or type(count) is not int or count <= 0:
            raise ValueError('Invalid propulsion group count/tag')
        tags.add(tag)
        if role not in ('cruise','high_lift') or state not in ('active','folded','stopped','windmilling'):
            raise ValueError('Unknown role/state')
        p = np.asarray(group['shaft_power_W'],dtype=float)
        area = float(group['rotor_drag_area_m2'])
        if p.shape != t.shape or not np.isfinite(p).all() or np.any(p<0):
            raise ValueError('Missing/invalid per-unit shaft-power history')
        finite_nonnegative(rotor_drag_area_m2=area)
        if not group.get('drag_source'):
            raise ValueError('Rotor off/active-state drag needs an explicit source')
        if state != 'active' and np.any(p != 0):
            raise ValueError('Passive rotor has nonzero commanded shaft power')
        if state == 'active':
            eta = float(group['motor_efficiency'])*float(group['esc_efficiency'])
            if not all(math.isfinite(float(group[k])) and 0<float(group[k])<=1
                       for k in ('motor_efficiency','esc_efficiency')):
                raise ValueError('Invalid conversion efficiency')
            electrical += count*p/eta
            cruise_active |= role == 'cruise'
        shaft += count*p
        drag += count*.5*density*speed**2*area
    if not cruise_active:
        raise ValueError('This cruise-plus-high-lift contract requires an active cruise group')
    integrate = getattr(np,'trapezoid',None) or np.trapz
    return dict(shaft_energy_Wh=float(integrate(shaft,t)/3600),
                propulsion_terminal_energy_Wh=float(integrate(electrical,t)/3600),
                conversion_loss_Wh=float(integrate(electrical-shaft,t)/3600),
                rotor_drag_N=drag.tolist(),
                rotor_drag_work_Wh=float(integrate(drag*speed,t)/3600),
                battery_loss_and_auxiliary_loads_included=False,
                requires_drag_force_feedback_to_mission=True,
                installed_mass_independent_of_operating_state=True)
