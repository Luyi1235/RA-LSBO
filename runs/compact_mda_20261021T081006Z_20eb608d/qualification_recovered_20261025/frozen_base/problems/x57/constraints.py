"""Cheap inherited design screens; not a flutter or certification analysis."""

import math
import numpy as np

from lsbo.records import Evaluation
from .propulsion_geometry import GEOMETRY_CONSTRAINTS, make_layout

POLICY_VERSION = 'legacy_screening_precheck_v1'
CHEAP_DESIGN_CONSTRAINTS = ('flutter', 'span_min', 'span_max')
ANALYTICAL_CONSTRAINTS = GEOMETRY_CONSTRAINTS + CHEAP_DESIGN_CONSTRAINTS
CONSTRAINT_POLICY = {
    'version': POLICY_VERSION,
    'thresholds_changed': False,
    'flutter': 'inherited_dimensionless_geometry_proxy_not_dynamic_flutter_validation',
    'structure': 'inherited_root_bending_capacity_proxy_not_detailed_stress_validation',
    'stall_speed': 'inherited_sea_level_empirical_CLmax_not_airfoil_specific_stall_validation',
    'span': 'inherited_layout_dependent_design_bounds_not_NASA_certification_limits',
    'soc': 'charge_fraction_minimum_0.15',
    'usable_energy': 'terminal_output_cap_47000_Wh_not_stored_energy_depletion',
}


def flutter_proxy(continuous):
    ar, tc, taper, sweep = (float(continuous[k]) for k in
                            ('aspect_ratio', 'thickness_to_chord', 'taper_ratio', 'quarter_chord_sweep'))
    if not all(math.isfinite(v) for v in (ar, tc, taper, sweep)) or ar <= 0 or tc <= 0:
        raise ValueError('Invalid flutter-proxy inputs')
    # Preserve the snapshot's floating-point operation order at the threshold.
    k_ar = 1.0 + (5.0 / 15.0) * (1.0 / ar - 0.5)
    k_taper = (np.clip(taper, 1.0e-3, 1.0) / 0.63) ** 0.18
    k_sweep = 1.0 / max(np.cos(np.radians(sweep)), 0.5)
    reference = (0.12 / 8.8) * (1.0 + (5.0 / 15.0) * (1.0 / 8.8 - 0.5))
    return float((tc / ar) * k_ar * k_taper * k_sweep / reference)


def design_constraints(continuous, discrete):
    area, ar = float(continuous['wing_area']), float(continuous['aspect_ratio'])
    if not all(math.isfinite(v) and v > 0 for v in (area, ar)):
        raise ValueError('Invalid planform inputs')
    span = float(np.sqrt(ar * area))
    return {'flutter': .85 - flutter_proxy(continuous),
            'span_min': (10. if discrete['prop_layout'] == 'wingtip' else 9.2) - span,
            'span_max': span - 14.5}


def analytical_checks(candidate, geometry_settings):
    plan = make_layout(candidate.continuous, candidate.discrete, geometry_settings)
    return {**plan['constraints'], **design_constraints(candidate.continuous, candidate.discrete)}, plan


def rejection_result(checks, plan):
    if not all(math.isfinite(value) for value in checks.values()):
        raise ValueError('Nonfinite analytical constraint')
    rejected = [name for name, value in checks.items() if value > 0]
    if not rejected:
        return None
    status = ('geometry_rejected' if any(name in GEOMETRY_CONSTRAINTS for name in rejected)
              else 'analytical_constraint_rejected')
    return Evaluation(None, False, checks, status, {
        'mda_evaluated': False, 'constraint_policy': CONSTRAINT_POLICY,
        'precheck_violations': rejected,
        'propulsion_geometry': {'layout_plan': plan, 'constraints': plan['constraints']}})
