"""Only numerical integration varies; existing PR68 observable-outcome law stays fixed."""
import math
import time
import numpy as np
from scipy.stats import qmc
from ra_lsbo.experiments.information_value_stage3.planner import (
    FantasyFitError, one_step, shortlist, outcomes)
from pr71_protocol import RTOL, ATOL


def node_rule(seed, iteration, n, replicate=0):
    if type(n) is not int or n < 4 or n & (n-1) or type(replicate) is not int or replicate not in (0, 1):
        raise ValueError('Registered power-of-two count and replicate required')
    key = (int(seed) + 730000000 + 1009*int(iteration) + 104729*replicate) % (2**32)
    return np.clip(qmc.Sobol(d=3, scramble=True, seed=key).random_base2(int(math.log2(n))), 1e-12, 1-1e-12)


def compare_forecasts(actual, saved):
    errors = {}
    for name in ('completion', 'objective', 'region', 'resource'):
        a = actual['constraints'][name] if name in ('region', 'resource') else actual.get(name)
        b = saved['constraints'][name] if name in ('region', 'resource') else saved.get(name)
        if a is None or b is None:
            if a is not None or b is not None:
                raise ValueError('Missing head mismatch: '+name)
            continue
        for field in ('mean', 'variance'):
            x, y = np.asarray(a[field], float), np.asarray(b[field], float)
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.allclose(x, y, rtol=RTOL, atol=ATOL):
                raise ValueError('Frozen posterior forecast mismatch: '+name+'/'+field)
            errors[name+'/'+field] = float(np.max(np.abs(x-y)))
    return errors


def score(forecast, best, factory, *, seed, iteration, n, replicate=0):
    t0 = time.perf_counter()
    base = one_step(forecast, best)
    firsts = shortlist(forecast, base)
    supports = {i: factory.support(i) for i in firsts}
    result = {'status': 'ok', 'n': n, 'replicate': replicate,
              'selected_index': None, 'scores': {}, 'fantasies': [], 'cold_mass_error': {}}
    try:
        for i in firsts:
            continuation, successful_mass = 0., 0.
            for weight, o in outcomes(forecast, i, node_rule(seed, iteration, n, replicate), best is not None):
                if o.feasible:
                    successful_mass += weight
                if best is None and o.feasible:
                    value, second = 0., None
                else:
                    updated = factory.condition(i, o, use_abort_interval=True)
                    next_best = min(best, o.objective) if best is not None and o.feasible else best
                    future = one_step(updated, next_best)
                    if len(future) != len(supports[i]['model_candidates']):
                        raise ValueError('Future support length mismatch')
                    second = int(np.argmax(future))
                    value = float(future[second])
                    if not math.isfinite(value) or value < 0:
                        raise FantasyFitError('Invalid continuation')
                result['fantasies'].append({'first_index': i, 'weight': float(weight),
                    'observation': o.packet(), 'second_index': second, 'continuation': value})
                continuation += weight*value
            result['scores'][str(i)] = float(base[i]+continuation)
            if best is None:
                result['cold_mass_error'][str(i)] = float(successful_mass-base[i])
        result['selected_index'] = max(firsts, key=lambda i: (result['scores'][str(i)], -i))
    except FantasyFitError as error:
        # No selected action or successful-only mean from an incomplete rule.
        result.update(status='numerical_failure', error=str(error), selected_index=None)
    result['planner_seconds'] = time.perf_counter()-t0
    result['fantasy_fits'] = factory.diagnostics()
    return result


def parity(result, saved):
    if result['status'] != 'ok' or saved.get('fallback') is not None:
        raise ValueError('Cannot establish registered four-node parity')
    if result['selected_index'] != saved['selected_index']:
        raise ValueError('Four-node selected action did not reproduce')
    for i, v in result['scores'].items():
        if not math.isclose(v, saved['scores'][int(i)], rel_tol=RTOL, abs_tol=ATOL):
            raise ValueError('Four-node Q2 did not reproduce')
    fs = saved['fantasies']
    if len(fs) != len(result['fantasies']):
        raise ValueError('Four-node outcome roster differs')
    for a, b in zip(result['fantasies'], fs):
        if a['first_index'] != b['first_index'] or a['second_index'] != b['second_index'] or a['observation'] != b['observation']:
            raise ValueError('Four-node branch/action contract differs')
        for key in ('weight', 'continuation'):
            if not math.isclose(a[key], b[key], rel_tol=RTOL, abs_tol=ATOL):
                raise ValueError('Four-node continuation differs')
    return True
