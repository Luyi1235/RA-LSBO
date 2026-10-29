"""Descriptive diagnostics from recorded decisions, never model evaluations."""
from collections import defaultdict

import numpy as np

from .policies import NEAR_TOLERANCE, nearest_distance


def decision_diagnostics(problem, initial, proposals, online):
    if len(proposals) != len(online):
        raise ValueError('Proposal/observation counts differ')
    result = {name: 0 for name in ('online_calls', 'cold_online_calls', 'near_repeat_calls',
        'cold_near_repeat_calls', 'coverage_calls', 'scheduled_coverage_calls',
        'singleton_fallback_calls', 'model_singleton_steps', 'selected_local_calls',
        'near_rejected_candidates', 'local_solver_reports', 'slsqp_clipping_warnings', 'local_errors')}
    history = list(initial)
    for proposal, record in zip(proposals, online):
        if proposal['candidate'] != record.candidate:
            raise ValueError('Decision does not match settled candidate')
        metadata = proposal['metadata']
        cold = not any(r.usable_feasible for r in history)
        distance = nearest_distance(problem, record.candidate, [r.candidate for r in history])
        near = distance is not None and distance <= NEAR_TOLERANCE
        result['online_calls'] += 1
        result['cold_online_calls'] += int(cold)
        result['near_repeat_calls'] += int(near)
        result['cold_near_repeat_calls'] += int(cold and near)
        result['coverage_calls'] += int(metadata.get('coverage_reason') is not None)
        result['scheduled_coverage_calls'] += int(metadata.get('coverage_reason') == 'scheduled_coverage')
        result['singleton_fallback_calls'] += int(metadata.get('coverage_reason') == 'eligible_at_most_one')
        eligible = metadata.get('eligible')
        result['model_singleton_steps'] += int(eligible is not None and sum(eligible) <= 1)
        result['selected_local_calls'] += int(metadata.get('selected_from_local_augmentation', False))
        result['near_rejected_candidates'] += sum(len(metadata.get(k, [])) for k in
                                                  ('global_near_rejections', 'local_near_rejections'))
        local = metadata.get('local_reports', [])
        result['local_solver_reports'] += len(local)
        result['slsqp_clipping_warnings'] += sum(r.get('slsqp_clipping_warnings', 0) for r in local)
        result['local_errors'] += sum(r.get('local_error') is not None for r in local)
        history.append(record)
    return result


def lexicographic_pairs(rows, reference, rtol=1e-10, atol=1e-12):
    """Feasibility first, objective second; no penalty imputation or p-values.

    Both-unresolved/nonfeasible runs are ties in this declared decision rule,
    not assertions of identical physical outcomes. Unknown counts remain in the
    underlying trial summaries. Report each task separately.
    """
    lookup = {(r['case'], r['seed'], r['initial_policy'], r['method']): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate paired trial identity')
    groups = defaultdict(list)
    for right in rows:
        if right['method'] == reference:
            continue
        key = (right['case'], right['seed'], right['initial_policy'], reference)
        if key not in lookup:
            raise ValueError('Missing paired reference')
        left = lookup[key]
        if left['initial_kind'] != right['initial_kind']:
            raise ValueError('Unmatched cold/warm initialization')
        groups[(right['case'], right['initial_policy'], right['method'])].append((left, right))
    result = []
    for (case, policy, comparator), pairs in sorted(groups.items()):
        counts = {k: 0 for k in ('reference_wins', 'comparator_wins', 'ties',
                  'cold_n', 'cold_reference_wins', 'cold_comparator_wins', 'cold_ties')}
        for left, right in pairs:
            if left['has_feasible'] != right['has_feasible']:
                outcome = 'reference_wins' if left['has_feasible'] else 'comparator_wins'
            elif not left['has_feasible'] or np.isclose(left['best_feasible'], right['best_feasible'], rtol=rtol, atol=atol):
                outcome = 'ties'
            else:
                outcome = 'reference_wins' if left['best_feasible'] < right['best_feasible'] else 'comparator_wins'
            counts[outcome] += 1
            if left['initial_kind'] == 'cold':
                counts['cold_n'] += 1
                counts['cold_'+outcome] += 1
        result.append({'case': case, 'initial_policy': policy, 'reference': reference,
                       'comparator': comparator, 'paired_n': len(pairs), **counts,
                       'rtol': rtol, 'atol': atol, 'inference': 'development-only descriptive decision rule'})
    return result
