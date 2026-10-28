"""Finite-pool advisory screening, not calibrated safety guarantees."""

import numpy as np


def select_candidate(scores, probabilities, thresholds):
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    if not finite.any():
        raise RuntimeError('All candidate acquisition scores are non-finite')
    eligible = finite.copy()
    log_probability = np.zeros(len(scores))
    for name, values in probabilities.items():
        values = np.asarray(values, dtype=float)
        if values.shape != scores.shape or not np.isfinite(values).all():
            raise ValueError(f'Invalid candidate probabilities: {name}')
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f'Out-of-range candidate probabilities: {name}')
        eligible &= values >= thresholds[name]
        log_probability += np.log(np.clip(values, 1e-12, 1.))
    if eligible.any():
        index = int(np.argmax(np.where(eligible, scores, -np.inf)))
        mode = 'screened_acquisition'
    else:
        # No candidate meets the heuristic thresholds: prioritize predicted
        # solver success, using acquisition only to break equal-risk ties.
        choices = np.flatnonzero(finite)
        order = np.lexsort((scores[choices], log_probability[choices]))
        index = int(choices[order[-1]])
        mode = 'maximum_predicted_success_fallback'
    return index, {'mode': mode, 'passing_candidates': int(eligible.sum()),
                   'finite_candidates': int(finite.sum()),
                   'thresholds': {k: float(thresholds[k]) for k in probabilities},
                   'selected_probabilities': {k: float(v[index]) for k, v in probabilities.items()},
                   'interpretation': 'uncalibrated_advisory_not_physical_feasibility'}
