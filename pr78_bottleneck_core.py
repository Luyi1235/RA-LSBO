"""Posthoc PR78 finite-pool diagnostics. No learner or evaluator imports.

Oracle substitutions are privileged sensitivity checks, not fair algorithms,
not independent confirmation, and not an additive causal decomposition.
"""
import hashlib
import json
import math
import statistics

SCHEMA = 'pr78_posthoc_bottleneck_v1'
PARENT = 'ab12650769e685966661648940301f330a84993e'
EVIDENCE = '5ff896b387a7506991ee356d9bfc981a8e080e6a'
EVIDENCE_PATH = 'results/phase-b1-objective-value-20261107/full'
BUDGETS = (1, 2, 4, 6, 12)
ACTUAL = ('uniform_hash', 'risk_only', 'objective_ei', 'risk_ei')
PRIVILEGED = ('known_feasibility_ei', 'known_objective_learned_risk',
              'known_both', 'known_feasibility_mean')
COMPARISONS = (('risk_ei', 'objective_ei'), ('risk_ei', 'risk_only'),
               ('known_feasibility_ei', 'risk_ei'),
               ('known_objective_learned_risk', 'risk_ei'),
               ('known_both', 'known_feasibility_ei'),
               ('known_feasibility_mean', 'known_feasibility_ei'))
NEW_BILL = dict(new_true_evaluations=0, new_gp_fits=0, new_vae_fits=0,
                new_fantasy_fits=0, aircraft_calls=0, rcaide_calls=0)


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf8')).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def finite(xs):
    require(all(type(x) in (int, float) and math.isfinite(x) for x in xs),
            'Finite non-Boolean numbers required')


def tol(x):
    return 1e-10 * max(1., abs(x))


def winner(a, b):
    if a is None or b is None:
        return 0 if a is None and b is None else (1 if b is None else -1)
    finite((a, b))
    eps = 1e-10 * max(1., abs(a), abs(b))
    return 1 if a < b-eps else (-1 if a > b+eps else 0)


def privileged_orders(ids, ei, probability, mean, best, truth, salt):
    n = len(ids)
    require(n > 0 and len(set(ids)) == n, 'Distinct query IDs required')
    require(all(isinstance(cid, str) for cid in ids), 'String IDs required')
    require(all(len(v) == n for v in (ei, probability, mean, truth)), 'Misaligned inputs')
    finite([best] + list(ei) + list(probability) + list(mean))
    require(all(e >= 0 for e in ei) and all(0 <= p <= 1 for p in probability), 'Invalid scores')
    require(all(type(t['feasible']) is bool for t in truth), 'Boolean feasibility required')
    finite([t['objective'] for t in truth])
    tie = [digest(['phase_b1_tie_v1', salt, cid]) for cid in ids]
    gain = [max(0., best-t['objective']) for t in truth]
    # A Boolean support mask is a lexicographic priority, not multiplication by
    # zero: infeasible points must not win an all-zero EI tie. Remaining points
    # are included only as budget padding when fewer than k feasible points exist.
    def exact_support(values):
        return sorted(range(n), key=lambda i: (not truth[i]['feasible'], -values[i], tie[i], ids[i]))
    return dict(known_feasibility_ei=exact_support(ei),
                known_objective_learned_risk=sorted(range(n),
                    key=lambda i: (-gain[i]*probability[i], tie[i], ids[i])),
                known_both=exact_support(gain),
                known_feasibility_mean=exact_support([-m for m in mean]))


def outcome(order, pool, k, best, probability):
    n = len(pool)
    require(type(k) is int and 1 <= k <= n, 'Invalid budget')
    require(len(order) == n and all(type(i) is int for i in order)
            and sorted(order) == list(range(n)), 'Full permutation required')
    chosen = order[:k]
    values = [pool[i]['truth']['objective'] for i in chosen if pool[i]['truth']['feasible']]
    all_values = values + ([] if best is None else [best])
    final = min(all_values) if all_values else None
    return dict(selected_ids=[pool[i]['candidate_id'] for i in chosen],
                selected_feasible=len(values), selected_nonfeasible=k-len(values),
                best_after=final, improvement=None if best is None else max(0., best-final),
                improved=None if best is None else best-final > tol(best),
                mean_predicted_feasibility=math.fsum(probability[i] for i in chosen)/k,
                observed_feasible_fraction=len(values)/k)


def tie_category(a, b, opportunity, incumbent):
    if incumbent is None:
        return 'cold_shared_fallback'
    w = winner(a['best_after'], b['best_after'])
    if w:
        return 'risk_wins' if w > 0 else 'risk_loses'
    if opportunity <= tol(incumbent):
        return 'tie_no_pool_opportunity'
    if not a['improved'] and not b['improved']:
        return 'tie_neither_improves_despite_opportunity'
    if a['improved'] and b['improved']:
        return 'tie_both_improve'
    return 'tie_within_comparison_tolerance'


def analyze_state(state, pool, budgets=BUDGETS):
    out = {k: state[k] for k in ('task', 'repeat', 'n', 'status', 'labels')}
    out['source_state_sha256'] = digest(state)
    out['source_pool_sha256'] = digest(pool)
    if state['status'] != 'OK':
        out.update(initial_warm=None, audit_status='FAILED_MODEL_RETAINED', by_budget={})
        return out
    d = state['decision']; ids = d['candidate_ids']; best = d['incumbent']
    require(ids == [r['candidate_id'] for r in pool], 'Pool identity or order mismatch')
    p = state['predictions']['risk']['probabilities']['joint']
    finite(p); require(len(p) == len(pool) and all(0 <= x <= 1 for x in p), 'Invalid probabilities')
    ranks = {name: list(d['rankings'][name]) for name in ACTUAL}
    out.update(initial_warm=best is not None, initial_best=best,
               audit_status='POSTHOC_PRIVILEGED' if best is not None else 'COLD_ACTUAL_ONLY',
               observable_objective_labels=state['labels']['objective']['exact'],
               known_joint_feasible_labels=state['labels']['joint_feasible'])
    feasible = [i for i, r in enumerate(pool) if r['truth']['feasible']]
    opportunity = None
    if best is not None:
        finite([best]); require(state['predictions']['objective'] is not None, 'Warm missing objective')
        mean = state['predictions']['objective']['mean']
        ranks.update(privileged_orders(ids, d['scores']['objective_ei'], p, mean,
            best, [r['truth'] for r in pool], '%s/r%d' % (state['task'], state['repeat'])))
        pool_best = min([best] + [pool[i]['truth']['objective'] for i in feasible])
        opportunity = max(0., best-pool_best)
        rmse = (math.sqrt(math.fsum((mean[i]-pool[i]['truth']['objective'])**2
                                   for i in feasible)/len(feasible)) if feasible else None)
        out.update(pool_best_including_incumbent=pool_best,
                   feasible_objective_rmse=rmse,
                   objective_rmse_scaled=None if rmse is None else rmse/state['objective_scale'])
    else:
        require(ranks['objective_ei'] == ranks['risk_ei'] == ranks['uniform_hash'],
                'Cold shared fallback changed')
    out.update(pool_feasible=len(feasible), pool_opportunity=opportunity, rankings=ranks, by_budget={})
    for k in budgets:
        outcomes = {name: outcome(order, pool, k, best, p) for name, order in ranks.items()}
        a, b = outcomes['risk_ei'], outcomes['objective_ei']
        result = dict(outcomes=outcomes, tie_category=tie_category(a, b, opportunity, best),
                      selection_set_changed=set(a['selected_ids']) != set(b['selected_ids']))
        if best is not None:
            oracle = outcomes['known_both']['improvement']
            require(abs(oracle-opportunity) <= tol(best), 'Known-both must reach pool opportunity')
            learned = a['improvement']
            r = outcomes['known_feasibility_ei']['improvement']
            f = outcomes['known_objective_learned_risk']['improvement']
            result['signed_information_swaps'] = dict(
                known_feasibility_minus_learned=r-learned,
                known_objective_minus_learned=f-learned,
                joint_interaction=oracle-r-f+learned,
                total_remaining_gap=oracle-learned,
                not_causal_error_shares=True)
        out['by_budget'][str(k)] = result
    return out


def distribution(values):
    return dict(n=len(values), minimum=min(values) if values else None,
                median=statistics.median(values) if values else None,
                maximum=max(values) if values else None)


def counts(rows, k):
    good = [r for r in rows if r['status'] == 'OK']
    warm = [r for r in good if r['initial_warm']]
    result = dict(planned_states=len(rows), failed_states=len(rows)-len(good),
                  warm=len(warm), cold=len(good)-len(warm), tie_categories={}, comparisons={},
                  training_exact_objectives=distribution([r['observable_objective_labels'] for r in good]),
                  warm_training_exact_objectives=distribution([r['observable_objective_labels'] for r in warm]),
                  outcomes={})
    for r in warm:
        name = r['by_budget'][str(k)]['tie_category']
        result['tie_categories'][name] = result['tie_categories'].get(name, 0)+1
    for a, b in COMPARISONS:
        wlt = [0, 0, 0]
        for r in warm:
            o = r['by_budget'][str(k)]['outcomes']
            w = winner(o[a]['best_after'], o[b]['best_after'])
            wlt[0 if w > 0 else 1 if w < 0 else 2] += 1
        result['comparisons'][a+'_vs_'+b] = wlt
    for name in ACTUAL + PRIVILEGED:
        vals = [r['by_budget'][str(k)]['outcomes'][name] for r in warm]
        result['outcomes'][name] = dict(improved=sum(v['improved'] is True for v in vals),
            selected_feasible=sum(v['selected_feasible'] for v in vals), selected_total=k*len(vals))
    result['opportunity_positive'] = sum(r['pool_opportunity'] > tol(r['initial_best']) for r in warm)
    result['no_raw_cross_task_objective_mean'] = True
    return result


def summarize(rows):
    keys = sorted({(r['task'], r['n']) for r in rows})
    groups = {}
    for task, n in keys:
        group = [r for r in rows if (r['task'], r['n']) == (task, n)]
        for k in BUDGETS:
            item = counts(group, k)
            item['paired_final_objectives'] = [dict(repeat=r['repeat'], initial_best=r['initial_best'],
                outcomes={name:o['best_after'] for name,o in r['by_budget'][str(k)]['outcomes'].items()})
                for r in group if r['status']=='OK']
            groups['%s/n%d/k%d' % (task, n, k)] = item
    descriptive = {}
    for n in sorted({r['n'] for r in rows}):
        descriptive[str(n)] = counts([r for r in rows if r['task'].startswith('tradeoff_') and r['n']==n], 6)
    return dict(per_task=groups, tradeoff_k6_descriptive=descriptive,
                primary_key='tradeoff_k6_descriptive/48',
                posthoc_not_confirmatory=True, original_primary_setting_retained=True)
