"""Decision-only guards. They never change physical observations or identities."""
from collections import Counter
from dataclasses import replace

import numpy as np

from ...budget import digest
from ...partial_bo.records import DesignRecord
from ...risk_core.observations import OutputObservation

NEAR_TOLERANCE = 1e-8  # normalized decision resolution, NOT physical equivalence
COVERAGE_PERIOD = 5


def active_distance(problem, left, right):
    """L-infinity distance on active unit coordinates, within one canonical branch."""
    if left['discrete'] != right['discrete']:
        return None
    values = [abs(left['continuous'][v.name]-right['continuous'][v.name])/(v.bounds[1]-v.bounds[0])
              for v in problem.space.continuous
              if problem.space.active(v, left['discrete'])]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite decision coordinate')
    return max(values, default=0.)


def nearest_distance(problem, candidate, previous):
    values = [active_distance(problem, candidate, raw) for raw in previous]
    return min((d for d in values if d is not None), default=None)


def filter_near(problem, candidates, history, tolerance=NEAR_TOLERANCE):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid decision resolution')
    previous = [r.candidate for r in history]
    kept, indices, rejected = [], [], []
    for index, candidate in enumerate(candidates):
        distance = nearest_distance(problem, candidate, previous+kept)
        if distance is not None and distance <= tolerance:
            rejected.append({'index': index, 'design_id': digest(candidate), 'distance': distance})
        else:
            indices.append(index)
            kept.append(candidate)
    return kept, indices, rejected


def coverage_pick(problem, pool, history, seed, iteration):
    """Balance visits among available branches, then sample uniformly in that pool.

    No posterior gate, latent decoding, maximization to a box corner or true-value
    screening. This is pool coverage, not a uniform guarantee over the full domain.
    """
    if not pool:
        raise RuntimeError('No novel coverage candidate; stop without a model call')
    counts = Counter(digest(row.candidate['discrete']) for row in history)
    groups = {}
    for i, raw in enumerate(pool):
        groups.setdefault(digest(raw['discrete']), []).append(i)
    least = min(counts[key] for key in groups)
    branches = sorted(key for key in groups if counts[key] == least)
    rng = np.random.default_rng(int(seed)*10000+int(iteration)+910000000)
    branch = branches[int(rng.integers(len(branches)))]
    index = int(rng.choice(groups[branch]))
    return index, {'available_branches': len(groups), 'selected_branch_visits': int(counts[branch]),
                   'rule': 'least_visited_available_branch_then_uniform_pool'}


def feedback_view(records, mode):
    """Coarsen a MODEL view only: exact / exact+sign / full intervals.

    Exact constraint values stay exact in every arm. Sign mode coarsens only
    intervals with resolved g<=0 evidence. Objective intervals have no universal
    sign threshold, and are omitted in sign mode. Unknown is never made negative.
    """
    if mode not in {'exact', 'sign', 'interval'}:
        raise ValueError('Unsupported feedback view')
    if mode == 'interval':
        return list(records)
    def convert(value, objective=False):
        if value.kind != 'interval':
            return value
        label = None if objective else value.constraint_satisfaction()
        if mode == 'sign' and label is not None:
            return (OutputObservation('interval', value.evidence_id, upper=0.) if label else
                    OutputObservation('interval', value.evidence_id, lower=0., lower_inclusive=False))
        return OutputObservation('missing', value.evidence_id, reason='feedback_view_'+mode)
    return [DesignRecord(row.candidate, replace(row.observation,
                objective=convert(row.observation.objective, True),
                constraints=tuple((name, convert(value)) for name, value in row.observation.constraints)),
                row.diagnostics) for row in records]


def optimistic_bound(mean, std, beta=1.96):
    """For g<=0, SEGO-UTB exploration uses mu-beta*sigma <= 0 (not UCB)."""
    mean, std = np.broadcast_arrays(np.asarray(mean, float), np.asarray(std, float))
    if (not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0)
            or not np.isfinite(beta) or beta < 0):
        raise ValueError('Invalid constraint prediction')
    return mean-beta*std
