"""Read-only audit of the actual rebuilt propulsors, not a new constraint."""

from functools import wraps
import numpy as np


def geometry_report(propulsors):
    rows = [{'tag': str(p.tag), 'origin_m': np.asarray(p.rotor.origin, dtype=float).reshape(-1, 3)[0].tolist(),
             'radius_m': float(p.rotor.tip_radius), 'design_thrust_N': float(p.rotor.cruise.design_thrust),
             'design_power_W': float(p.rotor.cruise.design_power)} for p in propulsors]
    clearances = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            a, b = np.array(left['origin_m']), np.array(right['origin_m'])
            # The bundled model installs identical forward-facing rotors in a
            # common x plane. This is disk geometry, not an interference model.
            if abs(a[0] - b[0]) < 1e-8:
                clearance = float(np.linalg.norm(a[1:] - b[1:]) - left['radius_m'] - right['radius_m'])
                clearances.append((left['tag'], right['tag'], clearance))
    return {'propulsors': rows, 'count': len(rows),
            'total_design_thrust_N': sum(r['design_thrust_N'] for r in rows),
            'total_design_power_W': sum(r['design_power_W'] for r in rows),
            'minimum_coplanar_disk_clearance_m': min((r[2] for r in clearances), default=None),
            'overlapping_disk_pairs': [list(r) for r in clearances if r[2] < -1e-8],
            'scope': 'geometry_diagnostic_only_not_added_to_legacy_feasibility'}


def install_propulsion_diagnostics(evaluator, diagnostics):
    native = evaluator.rebuild_propulsors

    @wraps(native)
    def checked(config, *args, **kwargs):
        result = native(config, *args, **kwargs)
        report = geometry_report(list(evaluator._network(config).propulsors))
        continuous = args[2] if len(args) > 2 else kwargs.get('continuous', {})
        report['placement_span_m'] = float(config.wings.main_wing.spans.projected)
        if 'aspect_ratio' in continuous and 'wing_area' in continuous:
            report['requested_span_m'] = float(np.sqrt(continuous['aspect_ratio'] * continuous['wing_area']))
            report['placement_span_error_m'] = report['placement_span_m'] - report['requested_span_m']
        diagnostics['propulsion_geometry'] = report
        if report['overlapping_disk_pairs']:
            print('[RCAIDE] Geometry warning: coplanar propeller disks overlap; '
                  'legacy numerical feasibility does not establish geometric validity', flush=True)
        if abs(report.get('placement_span_error_m', 0.)) > 1e-6:
            print('[RCAIDE] Geometry warning: propulsor placement used the template span, '
                  'not the requested candidate span', flush=True)
        return result

    evaluator.rebuild_propulsors = checked
