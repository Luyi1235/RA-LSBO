"""Shared finite-pool proposal machinery; never calls the engineering evaluator."""
import numpy as np

SEARCH_POLICY_VERSION = 'exact_precheck_local_search_20261012_v2'


def rank_observations(observations, constraints):
    scales = {}
    for name in constraints:
        values = [abs(o.result.constraints[name]) for o in observations
                  if o.result.constraints.get(name) is not None
                  and np.isfinite(o.result.constraints[name])]
        scales[name] = max(float(np.median(values)) if values else 1., 1e-8)

    def key(o):
        if o.result.feasible(constraints):
            return (0, o.result.objective)
        if not o.result.has_objective():
            return (2, 0.)
        violation = 0.
        for name in constraints:
            value = o.result.constraints.get(name)
            violation += (max(0., value) / scales[name] if value is not None
                          and np.isfinite(value) else 1e6)
        return (1, violation)
    return sorted(observations, key=key)


def mixed_candidate_pool(optimizer, size, target=None, excluded=()):
    """Half global draws, half multiscale elite moves with categorical mutation.

    This is the existing baseline pool, now also available to latent-space BO.
    All candidates use the same canonicalization, novelty and engineering checks.
    """
    space, rng = optimizer.space, optimizer.rng
    anchors = rank_observations(optimizer.observations, optimizer.problem.required_constraints)[:10]
    configurations = ([d for d in space.configurations()
                       if all(d[k] == v for k, v in target.items())] if target else None)
    pool, keys = [], set(excluded)
    for attempt in range(optimizer.settings.max_proposal_attempts):
        if len(pool) >= size:
            break
        if attempt % 2 and anchors:
            anchor = anchors[int(rng.integers(len(anchors)))].candidate
            radius = (.02, .1, .3)[int(rng.integers(3))]
            d = dict(anchor.discrete)
            if space.discrete and rng.random() < .25:
                v = space.discrete[int(rng.integers(len(space.discrete)))]
                d[v.name] = v.values[int(rng.integers(len(v.values)))]
            c = space.from_unit(space.unit(anchor) + rng.normal(0., radius, len(space.continuous)), d)
        else:
            d = configurations[int(rng.integers(len(configurations)))] if configurations else None
            c = space.sample(rng, d)
        if target and not all(c.discrete[k] == v for k, v in target.items()):
            continue
        key = space.key(c)
        if key not in keys and optimizer.accepted(c):
            pool.append(c)
            keys.add(key)
    return pool
