"""Evaluate the SAME Q2 functional at preregistered nested Sobol resolutions.

This does not select an online action. Numerical failures invalidate the whole
rule, and no successful-node renormalization or fake one-step result is used.
"""
import math
import time
import numpy as np
from scipy.stats import qmc
from quadrature_audit_protocol import require, close, ATOL, RTOL


def points(seed, iteration, n, replicate):
    require(type(n) is int and n >= 4 and n & (n-1) == 0, 'Power-of-two node count >=4 required')
    require(type(replicate) is int and replicate in (0, 1), 'Registered scramble required')
    stream = (int(seed)+730000000+1009*int(iteration)+104729*replicate) % (2**32)
    return np.clip(qmc.Sobol(d=3, scramble=True, seed=stream).random_base2(int(math.log2(n))), 1e-12, 1-1e-12)


def default_kernel():
    from ra_lsbo.experiments.information_value_stage3 import planner
    return planner


def compute_rule(forecast, best, factory, *, seed, iteration, n, replicate, first_indices, kernel=None):
    k = default_kernel() if kernel is None else kernel
    start = time.perf_counter()
    base = np.asarray(k.one_step(forecast, best), float)
    require(np.isfinite(base).all() and (base >= 0).all(), 'Invalid one-step values')
    firsts = list(first_indices)
    require(firsts == k.shortlist(forecast, base), 'Root shortlist changed')
    require(int(np.argmax(base)) in firsts, 'Myopic root absent')
    u = points(seed, iteration, n, replicate)
    if replicate == 0:
        require(np.array_equal(u[:4], k.nodes(seed, iteration)), 'Nested nodes differ from original four nodes')
    result = {'nodes': n, 'replicate': replicate, 'unit_nodes': u.tolist(),
              'first_indices': firsts, 'base_scores': base.tolist(), 'scores': {},
              'selected_index': None, 'status': 'PENDING', 'outcomes': [],
              'numeric_error': None, 'estimated_gain_not_actual_performance': True}
    try:
        for i in firsts:
            continuation = 0.
            support = factory.support(i)['model_candidates']
            require(bool(support), 'Empty continuation support')
            mass = 0.
            for j, (weight, o) in enumerate(k.outcomes(forecast, i, u, best is not None)):
                row = {'first_index': i, 'ordinal': j, 'weight': float(weight),
                       'observation': o.packet(), 'conditioned': False,
                       'second_index': None, 'continuation': 0.}
                if not (best is None and o.feasible):
                    updated = factory.condition(i, o, use_abort_interval=True)
                    row['conditioned'] = True
                    new_best = min(best, o.objective) if best is not None and o.feasible else best
                    future = np.asarray(k.one_step(updated, new_best), float)
                    require(len(future) == len(support), 'Continuation array/support mismatch')
                    if not np.isfinite(future).all() or np.any(future < 0):
                        raise k.FantasyFitError('Invalid numerical continuation')
                    second = int(np.argmax(future))
                    row.update(second_index=second, continuation=float(future[second]))
                result['outcomes'].append(row)
                continuation += weight * row['continuation']
                mass += weight
            close(mass, 1., 'quadrature mass')
            result['scores'][str(i)] = float(base[i]+continuation)
        winner = max(firsts, key=lambda i: (result['scores'][str(i)], -i))
        ranking = sorted(firsts, key=lambda i: (-result['scores'][str(i)], i))
        scale = max(1e-12, max(abs(v) for v in result['scores'].values()))
        gap = result['scores'][str(ranking[0])]-result['scores'][str(ranking[1])]
        myopic = int(np.argmax(base))
        result.update(status='OK', selected_index=winner, ranking=ranking,
                      winner_gap=gap, normalized_winner_gap=gap/scale,
                      winner_near_tie_at_replay_tolerance=gap <= ATOL+RTOL*scale,
                      original_myopic_index=myopic,
                      estimated_advantage_over_myopic=result['scores'][str(winner)]-result['scores'][str(myopic)])
    except k.FantasyFitError as error:
        result.update(status='NUMERICAL_FAILURE', numeric_error=str(error), selected_index=None,
                      partial_scores_not_used=result['scores'], scores={})
    result['elapsed_seconds'] = time.perf_counter()-start
    result['fantasy_fits'] = factory.diagnostics()
    return result


def verify_rule(rule, best, firsts, support_sizes):
    """No refits. Reconcile all saved observation packets and Q2 sums."""
    require(rule['first_indices'] == firsts, 'First-action identity mismatch')
    u = rule['unit_nodes']
    require(len(u) == rule['nodes'] and all(len(v) == 3 and all(math.isfinite(x) and 0 < x < 1 for x in v) for v in u), 'Invalid saved quadrature nodes')
    require(rule['status'] in ('OK', 'NUMERICAL_FAILURE'), 'Unfinished rule')
    if rule['status'] != 'OK':
        require(rule['selected_index'] is None and rule['scores'] == {}, 'Failed rule has a usable winner')
        return
    sums = {i: 0. for i in firsts}
    masses = {i: 0. for i in firsts}
    for row in rule['outcomes']:
        i, obs = row['first_index'], row['observation']
        require(i in sums and 0 < row['weight'] <= 1, 'Invalid root/weight')
        require(math.isfinite(row['continuation']) and row['continuation'] >= 0, 'Invalid saved continuation')
        if obs['aborted']:
            require(obs['region'] == {'kind': 'interval', 'lower': 0., 'lower_inclusive': False}
                    and obs['objective'] == {'kind': 'missing'}, 'Hidden abort observation leaked')
        else:
            require(obs['region']['kind'] == 'exact' and obs['region']['value'] <= 0, 'Invalid completed region')
        require(obs['resource']['kind'] == 'exact' and math.isfinite(obs['resource']['value']), 'Invalid resource')
        achieved = best is None and not obs['aborted'] and obs['resource']['value'] <= 0
        if achieved:
            require(not row['conditioned'] and row['second_index'] is None and row['continuation'] == 0,
                    'Cold success must have zero incremental continuation')
        else:
            require(row['conditioned'] and type(row['second_index']) is int
                    and 0 <= row['second_index'] < support_sizes[str(i)], 'Invalid second action')
        masses[i] += row['weight']
        sums[i] += row['weight'] * row['continuation']
    for i in firsts:
        close(masses[i], 1., 'saved mass')
        close(rule['scores'][str(i)], rule['base_scores'][i]+sums[i], 'saved Q2')
    winner = max(firsts, key=lambda i: (rule['scores'][str(i)], -i))
    require(rule['selected_index'] == winner, 'Saved winner is not Q2 argmax')
