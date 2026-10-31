"""Opt-in structure pretraining and scale controls; no optimizer defaults change.

All configurations are schema-known, not physically evaluated. This module never
calls an evaluator. Prediction validation is separate from reconstruction checks.
"""
from copy import copy, deepcopy
from dataclasses import asdict, replace

import numpy as np
import torch
from scipy.special import ndtr

from .budget import digest
from .partial_bo.models import log_interval_probability
from .structured_encoder import EncoderSettings, StructuredRepresentation, state_digest

PRETRAIN_UPDATES = 600  # Fixed before the first new fit; not a tuned stopping rule.
PREFIX_UPDATES = 30
VARIANTS = ('legacy', 'legacy_schema_scale', 'joint_630',
            'pretrained', 'pretrained_schema_scale', 'onehot')


def schema_candidates(features):
    """Enumerate syntax, not feasibility; no random or evaluated designs added."""
    cls = features.modules['space'].Candidate
    return [features.space.canonical(cls({}, d)).to_dict()
            for d in features.space.configurations()]


def reconstruction(representation):
    """Mean-code and sampled-z training checks; neither is generalization."""
    with torch.no_grad():
        indices = representation.grammar
        mu, logvar = representation.model.encode(indices)
        active = representation.model.discrete_mask(indices).bool()
        decoded = torch.stack([layer(mu).argmax(-1)
                               for layer in representation.model.decoder], -1)
        correct = (decoded == indices) | ~active
        # Dedicated diagnostic generator: never advance the training generator.
        generator = torch.Generator().manual_seed(421031)
        totals = []
        for _ in range(16):
            z = mu + (logvar*.5).exp()*torch.randn(mu.shape, generator=generator, dtype=mu.dtype)
            sample = torch.stack([layer(z).argmax(-1)
                                  for layer in representation.model.decoder], -1)
            totals.append(((sample == indices) | ~active).all(-1).double().mean().item())
        return {'mean_code_exact_fraction': correct.all(-1).double().mean().item(),
                'variable_accuracy': {v.name: correct[:, i][active[:, i]].double().mean().item()
                                      for i, v in enumerate(representation.space.discrete)},
                'sampled_exact_fraction': float(np.mean(totals)),
                'sampled_draws_per_configuration': 16,
                'codebook': mu.numpy().tolist(), 'log_variance': logvar.numpy().tolist(),
                'training_grammar_only': True}


def pretrain_grammar(template, seed):
    """An independent label-free checkpoint, reusable only for identical schema."""
    settings = replace(template.encoder.settings, performance_weight=0.,
                       initial_epochs=PRETRAIN_UPDATES)
    checkpoint = StructuredRepresentation(template.space, settings, seed)
    if len(checkpoint.grammar) > settings.batch_size:
        raise ValueError('This bounded study requires one complete grammar batch per update')
    checkpoint.fit([], epochs=PRETRAIN_UPDATES)
    checkpoint._grammar_schema = digest([asdict(v) for v in template.space.variables])
    checkpoint._grammar_seed = seed
    checkpoint._grammar_digest = state_digest(checkpoint)
    return checkpoint


def _observations(features, records):
    result = []
    for i, record in enumerate(records, 1):
        run = record.observation
        usable = run.status not in {'unknown', 'infrastructure_error'}
        exact = run.objective.kind == 'exact' and usable
        evaluation = features.modules['records'].Evaluation(
            run.objective.value if exact else None, exact,
            {n: v.value if usable and v.kind == 'exact' else None
             for n, v in run.constraints}, run.status)
        result.append(features.modules['records'].Observation(
            i, features.candidate(record.candidate), evaluation, 'current_prefix'))
    return result


def reset_normalization(features, records, mode):
    if mode not in {'legacy', 'schema_range'}:
        raise ValueError('Unknown scale control')
    if not features.latent:
        raise ValueError('Scale controls apply only to latent features')
    raw = features._raw([r.candidate for r in records])
    features.offset, features.scale = np.zeros(raw.shape[1]), np.ones(raw.shape[1])
    if mode == 'legacy':
        lo, hi = raw[:, -4:].min(0)-.5, raw[:, -4:].max(0)+.5
    else:
        # Only known configurations, at canonical defaults; no physical labels.
        codes = features._raw(schema_candidates(features))[:, -4:]
        lo, hi = codes.min(0), codes.max(0)
    width = hi-lo
    # A constant/negligible coordinate is not amplified and is explicitly logged.
    features.inactive_latent_coordinates = np.flatnonzero(width <= 1e-10).tolist()
    features.offset[-4:] = lo
    features.scale[-4:] = np.where(width <= 1e-10, 1., width)
    return features


def from_checkpoint(template, records, checkpoint, seed):
    """Clone the pure-grammar state; current-prefix micro-training never leaks back."""
    schema = digest([asdict(v) for v in template.space.variables])
    if (schema != checkpoint._grammar_schema or seed != checkpoint._grammar_seed
            or state_digest(checkpoint) != checkpoint._grammar_digest):
        raise ValueError('Checkpoint schema, seed or sealed state differs')
    result = copy(template)
    result.encoder = deepcopy(checkpoint)
    result.encoder.settings = replace(template.encoder.settings, initial_epochs=PREFIX_UPDATES)
    result.encoder.model.settings = result.encoder.settings
    # Same 30-update loss schedule and learning rate as legacy; only initialization differs.
    result.encoder.fit(_observations(result, records), epochs=PREFIX_UPDATES, incremental=False)
    if state_digest(checkpoint) != checkpoint._grammar_digest:
        raise RuntimeError('Prefix adaptation mutated the shared grammar checkpoint')
    result.state_hash = state_digest(result.encoder)
    reset_normalization(result, records, 'legacy')
    result.identity = digest({'policy': 'grammar600_prefix30', 'checkpoint': checkpoint._grammar_digest,
                              'history': [r.to_dict() for r in records],
                              'encoder': result.state_hash, 'offset': result.offset.tolist(),
                              'scale': result.scale.tolist()})
    return result


def scaled_copy(features, records):
    result = copy(features)
    reset_normalization(result, records, 'schema_range')
    result.identity = digest({'source': features.identity, 'scale_policy': 'schema_range_v1',
                              'offset': result.offset.tolist(), 'scale': result.scale.tolist()})
    return result


def geometry(features):
    raw = features.transform(schema_candidates(features))
    discrete = raw[:, 2*len(features.space.continuous):]
    distance = np.sqrt(np.mean((discrete[:, None]-discrete[None, :])**2, axis=-1))
    return {'discrete_rms_distances': distance.tolist(),
            'offset': features.offset.tolist(), 'scale': features.scale.tolist(),
            'feature_identity': features.identity,
            'reconstruction': reconstruction(features.encoder) if features.latent else None,
            'inactive_latent_coordinates': getattr(features, 'inactive_latent_coordinates', [])}


def score_observations(observations, mean, variance, *, constraint=False, exact_noise=0.):
    """Keep exact density, interval mass and sign scores separate and denominated.

    Interval score is compatibility with recorded censoring, not exact prediction
    error. Unknown intervals spanning zero do not receive invented sign labels.
    """
    mean, variance = np.asarray(mean, float), np.asarray(variance, float)
    if (mean.ndim != 1 or mean.shape != variance.shape or len(mean) != len(observations)
            or not np.isfinite(mean).all() or not np.isfinite(variance).all()
            or np.any(variance < 0) or not np.isfinite(exact_noise) or exact_noise < 0):
        raise ValueError('Finite matched means and nonnegative variances required')
    exact, interval, brier, missing = [], [], [], 0
    floors = int(np.sum(variance < 1e-24))
    sd = np.sqrt(np.maximum(variance, 1e-24))
    for i, obs in enumerate(observations):
        if obs.kind == 'missing':
            missing += 1
            continue
        if obs.kind == 'exact':
            var = max(variance[i] + exact_noise**2, 1e-24)
            error = float(mean[i]-obs.value)
            exact.append((error**2, .5*(np.log(2*np.pi*var)+error**2/var)))
        else:
            lo = -np.inf if obs.lower is None else (obs.lower-mean[i])/sd[i]
            hi = np.inf if obs.upper is None else (obs.upper-mean[i])/sd[i]
            interval.append(-log_interval_probability(lo, hi))
        label = obs.constraint_satisfaction() if constraint else None
        if label is not None:
            brier.append((ndtr(-mean[i]/sd[i])-float(label))**2)
    return {'n_total': len(observations), 'n_missing': missing, 'n_exact': len(exact),
            'n_interval': len(interval), 'n_sign': len(brier), 'variance_floor_count': floors,
            'exact_rmse': float(np.sqrt(np.mean([v[0] for v in exact]))) if exact else None,
            'exact_nll': float(np.mean([v[1] for v in exact])) if exact else None,
            'interval_nlpm': float(np.mean(interval)) if interval else None,
            'sign_brier': float(np.mean(brier)) if brier else None}
