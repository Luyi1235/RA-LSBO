"""Read-only diagnostics from sealed forecasts. No fitting or evaluator imports.

These routines reconstruct score arithmetic on a fixed recorded history/pool.
They do not simulate outcomes at unobserved candidates or replay optimization.
They apply to the frozen single-kernel, cold-phase transfer v1 protocol only.
"""
from copy import deepcopy
import math

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr


def normalized_weights(log_weights):
    values = np.asarray(log_weights, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError('Two finite expert log weights are required')
    return np.exp(values - logsumexp(values))


def log_interval_mass(lower, upper):
    if np.isnan([lower, upper]).any() or not lower < upper:
        raise ValueError('An ordered nonempty standardized interval is required')
    if lower == -np.inf:
        return float(log_ndtr(upper))
    if upper == np.inf:
        return float(log_ndtr(-lower))
    big, small = ((log_ndtr(-lower), log_ndtr(-upper)) if lower >= 0
                  else (log_ndtr(upper), log_ndtr(lower)))
    return float(big + np.log(-np.expm1(small - big)))


def observed_log_score(forecast, observation):
    """Transfer-v1 predictive density/mass, never a fabricated measured value."""
    if observation['kind'] == 'missing':
        return None
    mean, variance, scale = (float(forecast[k]) for k in ('mean', 'variance', 'scale'))
    if not np.isfinite([mean, variance, scale]).all() or variance < 0 or scale <= 0:
        raise ValueError('Invalid saved forecast')
    variance = max(variance, 1e-12 * scale**2)
    if observation['kind'] == 'exact':
        value = observation['value']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid exact value')
        variance += (1e-3 * scale)**2  # frozen LearnedMixedHead observation noise
        result = -.5 * (np.log(2*np.pi) + np.log(variance) + (value-mean)**2/variance)
    elif observation['kind'] == 'interval':
        sd = math.sqrt(variance)
        lower = -np.inf if observation['lower'] is None else (observation['lower']-mean)/sd
        upper = np.inf if observation['upper'] is None else (observation['upper']-mean)/sd
        result = log_interval_mass(lower, upper)
    else:
        raise ValueError('Unknown observation kind')
    if not np.isfinite(result):
        raise FloatingPointError('Nonfinite saved predictive evidence score')
    return float(result)


def update_saved_weights(log_weights, forecasts, record, mode):
    """Recompute post-settlement weights from pre-settlement forecasts only."""
    if mode not in {'equal', 'prequential'} or set(log_weights) != set(forecasts):
        raise ValueError('Expert mode or constraint keys changed')
    updated, scores = deepcopy(log_weights), {}
    obs = record['observation']
    constraints = dict(obs['constraints'])
    for name, pair in forecasts.items():
        normalized_weights(log_weights[name])
        if len(pair) != 2:
            raise ValueError('Exactly two expert forecasts required')
        if obs['status'] in {'unknown', 'infrastructure_error'}:
            scores[name] = None
            continue
        if name not in constraints:
            raise ValueError('Missing explicit constraint observation packet')
        values = [observed_log_score(p, constraints[name]) for p in pair]
        if values[0] is None:
            scores[name] = None
            continue
        scores[name] = values
        if mode == 'prequential':
            value = np.asarray(log_weights[name], float) + values
            updated[name] = (value-logsumexp(value)).tolist()
    return updated, scores


def reweight_constraints(prediction, by_constraint, scales):
    """Counterfactual weights on the SAME saved expert means/variances.

    No new expert fitting. Completion/objective distributions are unchanged.
    scales stores the two physical head scales, not an interval midpoint.
    """
    result = deepcopy(prediction)
    if set(by_constraint) != set(result['constraints']) or set(scales) != set(by_constraint):
        raise ValueError('Constraint identities do not match')
    for name, forecast in result['constraints'].items():
        w = np.asarray(by_constraint[name], float)
        means = np.asarray(forecast['components']['mean'], float)
        variances = np.asarray(forecast['components']['variance'], float)
        scale = np.asarray(scales[name], float)
        if (w.shape != (2,) or not np.isfinite(w).all() or np.any(w < 0)
                or not np.isclose(w.sum(), 1., atol=1e-12, rtol=0)
                or means.ndim != 2 or means.shape[0] != 2 or variances.shape != means.shape
                or not np.isfinite(means).all() or not np.isfinite(variances).all()
                or np.any(variances < 0) or scale.shape != (2,)
                or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError('Invalid saved mixture')
        mean = w@means
        forecast.update(mean=mean.tolist(), variance=np.sum(
            w[:, None]*(variances+(means-mean)**2), axis=0).tolist(), scale=float(w@scale))
        forecast['components']['weights'] = w.tolist()
    return result


def _parts(forecast):
    if 'components' in forecast:
        c = forecast['components']
        w, mean, var = (np.asarray(c[k], float) for k in ('weights', 'mean', 'variance'))
    else:
        w = np.ones(1)
        mean, var = (np.asarray(forecast[k], float)[None, :] for k in ('mean', 'variance'))
    if (w.ndim != 1 or mean.ndim != 2 or mean.shape != var.shape or len(w) != len(mean)
            or not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1.)
            or not np.isfinite(mean).all() or not np.isfinite(var).all() or np.any(var < 0)):
        raise ValueError('Invalid saved predictive components')
    return w, mean, var


def cold_model_choice(prediction, iteration, horizon=30, boundary_fraction=.2, threshold=.5):
    """Independent reconstruction of frozen v1 cold acquisition and gates.

    Returning a model index does NOT mean that index was sent to the evaluator:
    a singleton support gate may subsequently invoke the coverage policy.
    """
    if type(iteration) is not int or type(horizon) is not int or not 0 <= iteration < horizon:
        raise ValueError('Invalid iteration/horizon')
    if not 0 <= boundary_fraction < 1 or not 0 < threshold < 1:
        raise ValueError('Invalid selector settings')
    kappa = 2. - iteration/max(1, horizon-1)
    w, mean, var = _parts(prediction['completion'])
    success = w@ndtr(-mean/np.sqrt(1+var))
    upper = w@ndtr(-mean+kappa*np.sqrt(var))
    supported = upper >= threshold
    joint, uncertainty = success.copy(), np.zeros(len(success))
    optimistic = np.ones(len(success), dtype=bool)
    violation = np.zeros(len(success))
    score = np.log(np.maximum(success, 1e-300))
    for forecast in prediction['constraints'].values():
        w, means, variances = _parts(forecast)
        # Gates use the frozen mixture moments; probabilities stay componentwise.
        mu, va = (np.asarray(forecast[k], float) for k in ('mean', 'variance'))
        scale = float(forecast['scale'])
        if (mu.shape != success.shape or va.shape != success.shape or not np.isfinite(mu).all()
                or not np.isfinite(va).all() or np.any(va < 0) or not np.isfinite(scale) or scale <= 0):
            raise ValueError('Invalid saved mixture moments')
        margin = (kappa*np.sqrt(va)-mu)/scale
        optimistic &= margin >= 0
        violation += np.maximum(-margin, 0)
        uncertainty += np.sqrt(va)/scale
        probabilities = np.where(variances <= 1e-16, (means <= 0).astype(float),
                                 ndtr(-means/np.sqrt(np.maximum(variances, 1e-24))))
        ratio = np.divide(means, np.sqrt(variances), out=np.zeros_like(means), where=variances > 0)
        p = ndtr(-ratio)
        rho = ndtr(1.96-ratio)-ndtr(-1.96-ratio)
        factor = np.where(variances > 0, np.clip((1+rho)*p, 0, 1), (means <= 0).astype(float))
        score += np.log(np.maximum(w@factor, 1e-300))
        joint *= w@probabilities
    if not supported.any():
        supported = np.isclose(upper, upper.max(), rtol=1e-10, atol=1e-12)
    eligible = supported & optimistic
    if not eligible.any():
        eligible = supported & np.isclose(violation, violation[supported].min(), rtol=1e-10, atol=1e-12)
    boundary = math.floor((iteration+1)*boundary_fraction) > math.floor(iteration*boundary_fraction)
    if boundary:
        p = np.clip(joint, 1e-12, 1-1e-12)
        score = -p*np.log(p)-(1-p)*np.log(1-p)+.01*uncertainty/(1+uncertainty)
    masked = np.where(eligible, score, -np.inf)
    return {'model_index': int(np.argmax(masked)), 'eligible': eligible.tolist(),
            'eligible_count': int(eligible.sum()), 'boundary_slot': boundary,
            'scores': [float(x) if np.isfinite(x) else None for x in masked]}


def compare_saved_choices(left, right):
    """Separate internal model ranking changes from realized evaluated decisions."""
    if left['iteration'] != right['iteration'] or left['prefix_digest'] != right['prefix_digest']:
        raise ValueError('Only matched histories/iterations support a weight-only comparison')
    lm, rm = left['metadata'], right['metadata']
    if lm['pool'] != rm['pool']:
        raise ValueError('Only matched candidate pools support a weight-only comparison')
    lc = lm.get('coverage_reason')
    rc = rm.get('coverage_reason')
    actual_changed = left['candidate'] != right['candidate']
    model_comparable = all('model_selected_index' in m for m in (lm, rm))
    model_changed = model_comparable and lm['model_selected_index'] != rm['model_selected_index']
    return {'actual_candidate_changed': actual_changed, 'model_comparable': model_comparable,
            'model_candidate_changed': bool(model_changed),
            'gate_changed': model_comparable and lm['eligible'] != rm['eligible'],
            'model_change_overridden_by_coverage': bool(model_changed and not actual_changed and lc and rc),
            'left_channel': lc or 'model', 'right_channel': rc or 'model'}
