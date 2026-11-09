"""Standard-library paired summaries; no evaluation or model imports.

Exact sign tests concern within-task seed pairs, conditional on non-ties.
Holm adjustment is over four predefined primary tasks, never over decisions.
"""
import math
import statistics


def outcome(a, b):
    for x in (a, b):
        if x is not None and (type(x) not in (int, float) or not math.isfinite(x)):
            raise ValueError('Finite objective or None required')
    if a is None or b is None:
        return 0 if a is None and b is None else (1 if b is None else -1)
    tol = 1e-10 * max(1., abs(a), abs(b))
    return 1 if a < b - tol else (-1 if a > b + tol else 0)


def sign_p(wins, losses):
    if any(type(x) is not int or x < 0 for x in (wins, losses)):
        raise ValueError('Nonnegative integer paired counts required')
    n = wins + losses
    if n == 0:
        return 1.
    return min(1., 2. * sum(math.comb(n, j) for j in range(min(wins, losses) + 1)) / 2**n)


def holm(pvalues):
    if any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
           for p in pvalues.values()):
        raise ValueError('Invalid p value')
    ranked = sorted(pvalues, key=lambda k: (pvalues[k], k))
    result, previous = {}, 0.
    for i, k in enumerate(ranked):
        previous = max(previous, min(1., (len(ranked) - i) * pvalues[k]))
        result[k] = previous
    return result


def paired(rows):
    """Rows have repeat,a,b; never replace missing outcomes by zero."""
    if len({r['repeat'] for r in rows}) != len(rows):
        raise ValueError('Duplicate seed pair')
    wlt = [0, 0, 0]
    deltas = []
    output = []
    for row in rows:
        a, b = row['a'], row['b']
        z = outcome(a, b)
        wlt[0 if z > 0 else 1 if z < 0 else 2] += 1
        d = None if a is None or b is None else a-b
        if d is not None:
            deltas.append(d)
        output.append(dict(row, delta=d))
    return dict(n=len(rows), wins_losses_ties=wlt, paired=output,
                mean_delta=math.fsum(deltas)/len(deltas) if deltas else None,
                median_delta=statistics.median(deltas) if deltas else None,
                two_sided_sign_p=sign_p(wlt[0], wlt[1]))


def selected_forecast(rows):
    """Selection-conditioned diagnostic, NOT independent global calibration."""
    if any(type(r['feasible']) is not bool or type(r['p']) not in (int,float)
           or not math.isfinite(r['p']) or not 0 <= r['p'] <= 1 for r in rows):
        raise ValueError('Invalid selected prediction')
    n = len(rows)
    return dict(n=n, mean_predicted_feasibility=math.fsum(r['p'] for r in rows)/n if n else None,
                observed_feasibility=sum(r['feasible'] for r in rows)/n if n else None,
                brier=math.fsum((r['p']-int(r['feasible']))**2 for r in rows)/n if n else None)


def analyze_units(units, spec, *, confirmatory=False):
    arms = spec['arms']; tasks = list(dict.fromkeys(u['task'] for u in spec['units']))
    expected = {(u['task'], u['repeat']) for u in spec['units']}
    if {(u['task'], u['repeat']) for u in units} != expected or len(units) != len(expected):
        raise ValueError('Missing/duplicated unit')
    bytask = {}
    checkpoints = spec['checkpoints']
    for task in tasks:
        us = [u for u in units if u['task'] == task]
        for k in checkpoints:
            for stratum in ('all', 'initial_warm', 'initial_cold'):
                uu = [u for u in us if stratum == 'all' or
                      u['trials'][arms[0]]['initial_warm'] == (stratum == 'initial_warm')]
                aggregates = {}
                for a in arms:
                    rr = [u['trials'][a] for u in uu]
                    probs = [p for u in uu for p in u['selected_forecasts'][a][:k] if p is not None]
                    branch_counts = [len({c['branch'] for c in u['online_candidates'][a][:k]}) for u in uu]
                    aggregates[a] = dict(
                        n=len(rr), improved=sum(outcome(r['best_curve'][k],r['initial_best'])>0 for r in rr
                                               if r['initial_best'] is not None),
                        final_feasible=sum(r['best_curve'][k] is not None for r in rr),
                        online_nonfeasible=sum(k-r['information_curve'][k]['feasible']+r['information_curve'][0]['feasible'] for r in rr),
                        new_exact_objectives=sum(r['information_curve'][k]['exact_objectives']-r['information_curve'][0]['exact_objectives'] for r in rr),
                        selected_forecast=selected_forecast(probs),
                        online_only_branch_counts=branch_counts,
                        model_failure_calls=sum(sum(m=='model_failure_hash' for m in r['modes'][:k]) for r in rr))
                comparisons = {}
                for a,b in spec['contrasts']:
                    pairs=[]
                    for u in uu:
                        x,y=u['trials'][a],u['trials'][b]
                        fx=k-x['information_curve'][k]['feasible']+x['information_curve'][0]['feasible']
                        fy=k-y['information_curve'][k]['feasible']+y['information_curve'][0]['feasible']
                        ex=x['information_curve'][k]['exact_objectives']-x['information_curve'][0]['exact_objectives']
                        ey=y['information_curve'][k]['exact_objectives']-y['information_curve'][0]['exact_objectives']
                        pairs.append(dict(repeat=u['repeat'],a=x['best_curve'][k],b=y['best_curve'][k],
                            failure_delta=fx-fy,exact_label_delta=ex-ey,
                            different_path=x['candidate_path'][:k]!=y['candidate_path'][:k]))
                    comparisons[a+'_vs_'+b]=paired(pairs)
                bytask['%s/k%d/%s'%(task,k,stratum)] = dict(arms=aggregates,comparisons=comparisons)
    primary_name='_vs_'.join(spec['contrasts'][0])
    primary={t:bytask['%s/k%d/initial_warm'%(t,spec['steps'])]['comparisons'][primary_name]
             for t in tasks}
    adjusted=holm({t:r['two_sided_sign_p'] for t,r in primary.items()})
    for t,r in primary.items():
        r['holm_p']=adjusted[t]
        r['reject_equal_direction_probability_at_0_05'] = bool(confirmatory and adjusted[t]<=.05)
    return dict(primary=primary,by_task=bytask,
        inference=dict(confirmatory=confirmatory,alpha=.05,primary_family=tasks,
            method='two-sided exact sign test, non-ties, Holm over four tasks',
            no_cross_task_objective_mean=True,no_pooling_with_pilot=True),
        limits=['Replication on known tasks is not held-out-task generalization.',
                'Selected-point forecast diagnostics are policy-conditioned, not global calibration.',
                'Failure count is not a calibrated engineering loss or safety guarantee.',
                'Objective improvement does not prove more labels or any single mediator caused it.'])
