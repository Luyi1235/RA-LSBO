"""Use the UNMODIFIED PR74 StableGroupedHead and its 4x2 grid, no VAE fits."""
import math
import time
from unittest.mock import patch

from .core import MODELS, HEADS, digest, features, known_label


def training_payload(rows, method, privileged=None):
    if method not in MODELS:
        raise ValueError('Unregistered model')
    if method == 'privileged_exact':
        if privileged is None or len(privileged) != len(rows):
            raise ValueError('Explicit aligned privileged training truth required')
    elif privileged is not None:
        raise ValueError('Test/oracle truth forbidden for observable models')
    out = []
    for i, row in enumerate(rows):
        item = {'candidate': row['candidate'], 'candidate_id': row['candidate_id']}
        for head in HEADS:
            if method == 'privileged_exact':
                item[head] = dict(kind='exact', value=privileged[i][head])
            elif method == 'native_sign':
                item[head] = (dict(kind='interval', upper=0.) if known_label(row, head)
                              else dict(kind='interval', lower=0., lower_inclusive=False))
            else:
                item[head] = dict(row[head])
        out.append(item)
    return out


def predict(rows, queries, method, seed, bill, privileged=None):
    payload = training_payload(rows, method, privileged)
    receipt = dict(method=method, training_input_sha256=digest(payload), training_rows=len(rows),
                   query_ids=[digest(c) for c in queries], objective_labels_used=False,
                   privileged_training=(method == 'privileged_exact'), vae_fits=0, heads={})
    start = time.perf_counter()
    if method == 'joint_prevalence':
        probabilities = {h: [(sum(known_label(r,h) for r in rows)+1.)/(len(rows)+2.)]*len(queries) for h in HEADS}
        probabilities['joint'] = [(sum(all(known_label(r,h) for h in HEADS) for r in rows)+1.)/(len(rows)+2.)]*len(queries)
        return dict(candidate_ids=receipt['query_ids'], probabilities=probabilities, margin_mean={}), receipt
    import numpy as np
    from scipy.special import ndtr
    from ra_lsbo.experiments.simplified_aircraft import dualview_optimizer as native
    from ra_lsbo.risk_core.observations import OutputObservation
    X, Q = np.asarray([features(r['candidate']) for r in rows]), np.asarray([features(c) for c in queries])
    original = native.EvidenceGP

    def counted(*args, **kwargs):
        bill['gp_kernel_started'] += 1
        try:
            fitted = original(*args, **kwargs)
        except Exception:
            bill['gp_kernel_errors'] += 1
            raise
        bill['gp_kernel_completed'] += 1
        return fitted

    probabilities, means, gaussian = {}, {}, {}
    for number, head in enumerate(HEADS):
        observations = [OutputObservation(evidence_id=digest([method, head, r['candidate_id']]), **r[head]) for r in payload]
        bill['gp_head_started'] += 1
        try:
            with patch.object(native, 'EvidenceGP', counted):
                fitted = native.StableGroupedHead(X, observations, continuous_count=3,
                             seed=seed+number, name=head, completion=method == 'native_sign')
            guess = fitted.predict(Q)
        except native.ModelNumericalFailure as error:
            bill['gp_head_errors'] += 1
            receipt.update(status='MODEL_NUMERICAL_FAILURE', failing_head=head,
                           error=str(error), kernel_failures=error.failures,
                           elapsed_seconds=time.perf_counter()-start)
            return None, receipt
        bill['gp_head_completed'] += 1
        diag = fitted.diagnostics()
        # StableGroupedHead may reject candidates whose EP evidence is invalid
        # even when the constructor returned. Keep that distinct from constructor errors.
        bill['kernel_candidates_rejected'] += sum(r['status'] != 'ok' for r in diag['candidates'])
        variance = np.asarray(guess['variance']) + (1. if method == 'native_sign' else 0.)
        mean = np.asarray(guess['mean'])
        gaussian[head] = dict(mean=mean.tolist(), variance=np.asarray(guess['variance']).tolist(),
                              observation_noise_variance=1. if method == 'native_sign' else 0.)
        p = ndtr(np.divide(-mean, np.sqrt(variance), out=np.zeros_like(mean), where=variance > 0))
        p = np.where(variance > 0, p, (mean <= 0).astype(float))
        probabilities[head] = p.tolist()
        if method != 'native_sign':
            means[head] = mean.tolist()
        receipt['heads'][head] = diag
    probabilities['joint'] = [a*b for a,b in zip(probabilities['region'], probabilities['resource'])]
    receipt.update(status='OK', joint_probability_assumption='independent_constraint_heads_not_safety_guarantee',
                   elapsed_seconds=time.perf_counter()-start)
    if not all(math.isfinite(p) for values in probabilities.values() for p in values):
        raise FloatingPointError('Nonfinite native prediction')
    return dict(candidate_ids=receipt['query_ids'], probabilities=probabilities, margin_mean=means, gaussian=gaussian), receipt
