"""Instrument the frozen optimizer at its raw-pool and final-ranking boundaries."""

import inspect
from types import MethodType
from textwrap import dedent

import numpy as np
import torch


def acquisition_parts(surrogate, features, analytical):
    completion = getattr(surrogate, "completion", None)
    engineering = surrogate.native if completion is not None else surrogate
    energy = getattr(engineering, "energy", None)
    raw = engineering.native if energy is not None else engineering
    with torch.no_grad():
        augmented = features.unsqueeze(-2)
        normalized = (augmented - raw.offset) / raw.scale
        n = len(features)
        objective = (raw.improvement(normalized).reshape(-1).tolist()
                     if raw.improvement is not None else [None] * n)
        probabilities = {name: [1.] * n for name in analytical}
        for name in raw.required_constraints:
            pair = raw.constraints.get(name)
            if pair is None:
                probabilities[name] = [.5] * n
            else:
                model, threshold = pair
                post = model.posterior(normalized)
                z = (threshold - post.mean) / post.variance.clamp_min(1e-12).sqrt()
                probabilities[name] = torch.special.ndtr(z).reshape(-1).tolist()
        if energy is not None:
            probabilities["usable_energy"] = energy.log_probability(features).exp().reshape(-1).tolist()
        p_complete = (completion.log_probability(features).exp().reshape(-1).tolist()
                      if completion is not None else [None] * n)
        scores = surrogate.score(features).reshape(-1)
    return [{"log_expected_improvement": objective[i],
             "expected_improvement_normalized": None if objective[i] is None else float(np.exp(objective[i])),
             "p_complete": p_complete[i],
             "engineering_probabilities": {name: values[i] for name, values in probabilities.items()},
             "joint_log_score": float(scores[i]) if np.isfinite(scores[i]) else None}
            for i in range(n)]


def ordered_branches(candidates, scores, space, centers, quota, reserve):
    leaders = {}
    for index in np.argsort(-scores):
        index = int(index)
        key = space.branch_key(candidates[index].discrete)
        leaders.setdefault(key, index)
    order = list(leaders)
    selected = order[:quota]
    if reserve and selected and all(key in centers for key in selected):
        outsider = next((key for key in order if key not in centers
                         and np.isfinite(scores[leaders[key]])), None)
        if outsider is not None:
            selected[-1] = outsider
    # Keep the unselected tail for parity with the native stopping rule.
    return [leaders[key] for key in selected + [key for key in order if key not in selected]]


def posterior_screen(native, candidates, scores, surrogate, rep, *, screen_after):
    original = int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))
    if len(native.observations) < screen_after:
        return original, {"mode": "joint_score_only"}
    # The two audited marginals are evaluated on the actual final pool.
    # Reuse the frozen joint ranking among designs passing both medians.
    engineering = surrogate.native
    energy = getattr(engineering, "energy", None)
    raw = engineering.native if energy is not None else engineering
    stall = raw.constraints.get("stall_speed")
    if energy is None or stall is None:
        return original, {"mode": "missing_screen_model"}
    with torch.no_grad():
        x = rep.features(candidates)
        energy_ok = (energy.log_probability(x).reshape(-1) >= np.log(0.5)).numpy()
        model, threshold = stall
        post = model.posterior(((x - raw.offset) / raw.scale).unsqueeze(-2))
        stall_ok = (post.mean.reshape(-1) <= threshold).numpy()
    mask = np.isfinite(scores) & energy_ok & stall_ok
    selected = int(np.argmax(np.where(mask, scores, -np.inf))) if mask.any() else original
    return selected, {"mode": "posterior_median_screen", "passing": int(mask.sum()),
                      "original_candidate": native.space.key(candidates[original]),
                      "changed": original != selected}


def configure_native_search(optimizer, *, reserve, joint_final=False, audit=False, force_weighted=False, screen_after=None):
    """Adapt only the pinned method; do not patch the physical module or other arms."""
    original = type(optimizer).propose
    source = dedent(inspect.getsource(original))
    changes = {
        "if elites and self.solver_failures >= 2:":
            "if elites and self.solver_failures >= 2 and not _force_weighted():",
        "and self.rng.random() < self.settings.exploration_probability:":
            "and not _force_weighted() and self.rng.random() < self.settings.exploration_probability:",
        "for index in np.argsort(-scores):":
            "for index in _branch_order(self, candidates, scores, centers):",
        '    screening = {\'mode\': \'disabled\'}':
            '    _raw_scores(self, candidates, scores, raw_pool_count, surrogate, rep)\n'
            '    screening = {\'mode\': \'disabled\'}',
        '    if self.settings.screen_solver_risk:\n        index, screening = select_candidate':
            '    if _joint_final():\n'
            '        index = int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))\n'
            '        screening = {"mode": "joint_score_only"}\n'
            '        challenger = {"mode": "joint_score_only"}\n'
            '    elif self.settings.screen_solver_risk:\n        index, screening = select_candidate',
    }
    if screen_after is not None:
        key = '    if self.settings.screen_solver_risk:\n        index, screening = select_candidate'
        changes[key] = changes[key].replace(
            'index = int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))',
            'index, audit_screen = _posterior_screen(self, candidates, scores, surrogate, rep)')
        changes[key] = changes[key].replace('screening = {"mode": "joint_score_only"}', 'screening = audit_screen')
    for before, after in changes.items():
        if source.count(before) != 1:
            raise ValueError("Frozen native propose layout differs from the audited hook")
        source = source.replace(before, after, 1)
    trace = {}

    def branch_order(native, candidates, scores, centers):
        if not trace.get("weighted", False) and not audit:
            return np.argsort(-scores)
        order = ordered_branches(candidates, scores, native.space, centers, native.settings.local_branches,
                                 reserve and trace.get("weighted", False))
        if audit:
            trace["raw_rank_order"] = [native.space.key(candidates[int(i)]) for i in np.argsort(-scores)]
            trace["local_branch_keys"] = [
                native.space.branch_key(candidates[i].discrete) for i in order[:native.settings.local_branches]]
            trace["elite_branch_keys"] = list(centers)
        return order

    def raw_scores(native, candidates, scores, raw_count, surrogate, rep):
        if not audit:
            return
        parts = acquisition_parts(surrogate, rep.features(candidates), native.problem.analytical_constraints)
        trace["pool"] = [{"candidate_id": native.space.key(point), "discrete": point.discrete,
                          "from_local_search": i >= raw_count,
                          "joint_log_score": float(scores[i]) if np.isfinite(scores[i]) else None}
                         for i, point in enumerate(candidates)]
        trace["parts"] = {point["candidate_id"]: value for point, value in zip(trace["pool"], parts)}

    globals_with_hooks = dict(original.__globals__, _branch_order=branch_order, _raw_scores=raw_scores,
                              _joint_final=lambda: joint_final and trace.get("weighted", False),
                              _force_weighted=lambda: force_weighted and trace.get("has_refinement", False))
    if screen_after is not None:
        from functools import partial
        globals_with_hooks["_posterior_screen"] = partial(posterior_screen, screen_after=screen_after)
    namespace = {}
    exec(compile(source, inspect.getfile(original), "exec"), globals_with_hooks, namespace)
    adapted = namespace["propose"]

    def propose(native, phase, *, refine_surrogate=None):
        trace.clear()
        trace["has_refinement"] = refine_surrogate is not None

        def capture(surrogate, rep):
            result = refine_surrogate(surrogate, rep)
            trace["weighted"] = getattr(result, "completion", None) is not None
            return result

        point, info = adapted(native, phase, refine_surrogate=capture if refine_surrogate else None)
        if audit:
            info["native_trace"] = {k: v for k, v in trace.items() if k != "parts"}
            info["acquisition_components"] = trace.get("parts", {}).get(native.space.key(point))
        return point, info

    optimizer.propose = MethodType(propose, optimizer)
