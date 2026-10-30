"""Descriptive step sensitivity, including censored outputs; no evaluations.

A lower bound at one stopping time is NOT an exact terminal response. Comparable
completed stage endpoints are separated from the event endpoint. No convergence
order, error bound, safety certificate or physical calibration is inferred.
"""
from collections import Counter
import math
from pathlib import Path
import json

from . import reviewed_qualification as q


def _span(values):
    return max(values)-min(values) if values and all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in values) else None


def sensitivity(records, steps):
    """Inspect the SAME design at declared steps without changing any observation."""
    packets = [r.to_dict() if hasattr(r, 'to_dict') else r for r in records]
    if len(packets) != len(steps) or len(packets) < 2:
        raise ValueError('Two or more paired step records required')
    if len(set(steps)) != len(steps) or any(not math.isfinite(s) or s <= 0 for s in steps):
        raise ValueError('Distinct positive finite steps required')
    if any(r['candidate'] != packets[0]['candidate'] for r in packets):
        raise ValueError('Step comparison cannot change the design')
    constraints = [dict(r['observation']['constraints']) for r in packets]
    if any(set(c) != set(constraints[0]) for c in constraints):
        raise ValueError('Constraint identities changed')
    labels = {n: [q.satisfaction(c[n]) for c in constraints] for n in sorted(constraints[0])}
    exact_ranges = {n: _span([c[n]['value'] for c in constraints]) for n in labels
                    if all(c[n]['kind'] == 'exact' for c in constraints)}
    outputs = [dict(c, objective=r['observation']['objective']) for c,r in zip(constraints, packets)]
    bounds = {}
    for name in outputs[0]:
        values = [r[name] for r in outputs]
        if any(v['kind'] == 'interval' for v in values):
            bounds[name] = {
                'kinds': [v['kind'] for v in values],
                'lower': [v.get('lower') if v['kind']=='interval' else None for v in values],
                'upper': [v.get('upper') if v['kind']=='interval' else None for v in values],
                'not_exact_terminal_values': True,
            }
            for end in ('lower', 'upper'):
                bounds[name][end+'_range'] = _span(bounds[name][end])
    reasons = [r['diagnostics'].get('termination_reason') for r in packets]
    traces = [r['diagnostics'].get('trace', []) for r in packets]
    exact_objectives = [r['observation']['objective']['kind']=='exact' for r in packets]
    completed = []
    for trace, exact in zip(traces, exact_objectives):
        # The last trace of an aborted mission is a partial stage, not its endpoint.
        completed.append({t['stage']:t for t in (trace if exact else trace[:-1])})
    common = sorted(set.intersection(*(set(c) for c in completed)))
    endpoints = {}
    for stage in common:
        values = [c[stage] for c in completed]
        times = [t['elapsed_s'] for t in values]
        same_time = all(math.isclose(t, times[0], rel_tol=1e-10, abs_tol=1e-8) for t in times)
        endpoints[stage] = {'same_time': same_time, 'elapsed_s': times,
            'soc': [t['soc'] for t in values],
            'soc_range': _span([t['soc'] for t in values]) if same_time else None,
            'energy_prefix_Wh': [t['energy_prefix_Wh'] for t in values]}
    terminal = [trace[-1] if trace else None for trace in traces]
    partial = not all(exact_objectives)
    return {'steps_s': list(steps), 'termination_reasons': reasons,
        'termination_agrees': len(set(reasons))==1,
        'physical_states': [q.physical_state(r) for r in packets],
        'constraint_labels': labels,
        'constraint_labels_agree': all(all(x==v[0] for x in v) for v in labels.values()),
        'absolute_exact_constraint_ranges': exact_ranges,
        'terminal_elapsed_s': [t['elapsed_s'] if t else None for t in terminal],
        'terminal_time_range_s': _span([t['elapsed_s'] if t else None for t in terminal]),
        'terminal_soc': [t['soc'] if t else None for t in terminal],
        'completed_stage_endpoints': endpoints, 'interval_bounds': bounds,
        'status': 'partial_outputs_not_certified' if partial else 'complete_outputs_descriptive_check',
        'physical_validation': False, 'full_response_convergence_proven': False,
        'interpretation': 'fixed-stage energy agreement does not certify SOC or event convergence; '
                          'bounds at different stopping times are not equal terminal measurements'}


def qualification_report(folder):
    """Import only pinned sealed data; never run snapshots or physical evaluators."""
    root = Path(folder)
    receipt = q.verify(root)
    plan = json.loads(q.safe_read(root, 'plan.json'))
    events = [json.loads(line) for line in q.safe_read(root, 'calls.jsonl').splitlines() if line.strip()]
    rows = {e['tag']:e['record'] for e in events if e['event']=='settled'}
    cases = {}
    for case, task in plan['tasks'].items():
        checks = {}
        for kind, candidates, steps in [('anchor', task['anchors'], (20,10,5)),
                                         ('branch', task['branch_probes'], (20,5))]:
            for index in range(len(candidates)):
                group = [rows[f'{case}/{kind}/{index}/{step}'] for step in steps]
                checks[f'{kind}/{index}'] = sensitivity(group, steps)
        cases[case] = {'checks': checks, 'status_counts': dict(Counter(x['status'] for x in checks.values()))}
    if q.verify(root) != receipt:
        raise ValueError('Qualification changed during diagnostic import')
    return {'receipt': receipt, 'cases': cases, 'new_model_calls': 0,
            'new_rcaide_mda_calls': 0, 'does_not_promote_partial_convergence': True}
