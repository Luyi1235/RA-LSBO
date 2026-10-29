"""Phase-aware descriptive endpoints; no evaluators, fitted models or p-values."""
from itertools import combinations
import math
import statistics


def phase_diagnostics(initial, proposals, online):
    if len(proposals) != len(online):
        raise ValueError('Proposal/observation counts differ')
    warm = any(r.usable_feasible for r in initial)
    result = {'initial_kind': 'warm' if warm else 'cold', 'first_feasible_channel': None}
    for phase in ('cold', 'warm'):
        for metric in ('calls', 'coverage', 'scheduled', 'singleton_escape', 'model',
                       'model_boundary', 'feasible', 'posterior_skipped'):
            result[phase+'_'+metric] = 0
    for proposal, row in zip(proposals, online):
        if proposal['candidate'] != row.candidate:
            raise ValueError('Proposal differs from observation')
        phase = 'warm' if warm else 'cold'
        info = proposal['metadata']
        reason = info.get('coverage_reason')
        channel = reason or ('model_boundary' if info.get('boundary_slot') else 'model')
        result[phase+'_calls'] += 1
        result[phase+'_coverage'] += int(reason is not None)
        result[phase+'_scheduled'] += int(reason == 'scheduled_coverage')
        result[phase+'_singleton_escape'] += int(reason == 'eligible_at_most_one')
        result[phase+'_model'] += int(reason is None)
        result[phase+'_model_boundary'] += int(reason is None and bool(info.get('boundary_slot')))
        result[phase+'_posterior_skipped'] += int(bool(info.get('posterior_fit_skipped')))
        result[phase+'_feasible'] += int(row.usable_feasible)
        if row.usable_feasible and not warm:
            result['first_feasible_channel'] = channel
        warm = warm or row.usable_feasible
    return result


def warm_pairs(rows, rtol=1e-10, atol=1e-12):
    """Compare final values at the SAME initially warm seeds, not against initial y.

    Both methods retain the same observed feasible initial incumbent. A broken
    identity/missing pair raises rather than silently changing a denominator.
    Differences are left minus right. No ratios (objectives may be zero/negative).
    """
    lookup = {(r['case'], r['seed'], r['method']): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate warm comparison identity')
    output = []
    for case in sorted({r['case'] for r in rows}):
        selected = [r for r in rows if r['case'] == case]
        methods = sorted({r['method'] for r in selected})
        seeds = sorted({r['seed'] for r in selected})
        for left, right in combinations(methods, 2):
            differences, better, worse, ties = [], 0, 0, 0
            for seed in seeds:
                if (case,seed,left) not in lookup or (case,seed,right) not in lookup:
                    raise ValueError('Missing warm comparison pair')
                a, b = lookup[case,seed,left], lookup[case,seed,right]
                if a['initial_kind'] != b['initial_kind'] or a['initial_best'] != b['initial_best']:
                    raise ValueError('Unmatched shared initialization')
                if a['initial_kind'] != 'warm':
                    continue
                x, y = a['best_feasible'], b['best_feasible']
                if not (a['has_feasible'] and b['has_feasible'] and
                        isinstance(x, (int,float)) and isinstance(y, (int,float)) and
                        math.isfinite(x) and math.isfinite(y)):
                    raise ValueError('Initially warm run lost its recorded incumbent')
                differences.append(x-y)
                if math.isclose(x,y,rel_tol=rtol,abs_tol=atol): ties += 1
                elif x < y: better += 1
                else: worse += 1
            output.append({'case':case, 'left':left, 'right':right, 'n':len(differences),
                'better':better, 'ties':ties, 'worse':worse,
                'median_difference':statistics.median(differences) if differences else None,
                'rtol':rtol, 'atol':atol,
                'inference':'descriptive paired warm final objective; no superiority claim'})
    return output
