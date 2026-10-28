"""SCBO-inspired posterior selection with observed, censored completion risk."""

from collections import Counter

import numpy as np
import torch

from .budget import digest
from .selection import incumbent


def choose_sample(objective, constraints, completed, scales):
    objective = np.asarray(objective, dtype=float)
    constraints = np.asarray(constraints, dtype=float)
    completed = np.asarray(completed, dtype=bool)
    scales = np.asarray(scales, dtype=float)
    if (objective.ndim != 1 or constraints.ndim != 2
            or constraints.shape[0] != len(objective) or completed.shape != objective.shape
            or scales.shape != (constraints.shape[1],) or (scales <= 0).any()
            or not all(np.isfinite(a).all() for a in (objective, constraints, scales))
            or len(objective) == 0):
        raise ValueError("Finite compatible posterior samples and positive scales required")
    feasible = completed & (constraints <= 0).all(1)
    if feasible.any():
        index = int(np.argmin(np.where(feasible, objective, np.inf)))
        return index, "sampled_feasible_improvement", np.zeros(len(objective))
    violation = (np.maximum(constraints, 0.)/scales).sum(1)+(~completed).astype(float)
    # Lexicographic tie-break preserves the physical objective's scale.
    index = int(np.lexsort((objective, violation))[0])
    return index, "sampled_feasibility_restoration", violation


def select_posterior(pool, rows, responses, completion, analytical_constraints,
                     seed, use_completion=True):
    if not pool:
        return None, {"reason": "empty_pool"}
    if responses.features.snapshot_id != completion.snapshot_id:
        raise ValueError("Response and completion representations differ")
    candidates = [item["candidate"] for item in pool]
    names = sorted(name for name in responses.diagnostics
                   if name.startswith("constraint:")
                   and name.split(":", 1)[1] not in analytical_constraints)
    unsupported = [name for name in names if name not in responses.models]
    if not names or unsupported or (use_completion and "completion" not in completion.models):
        # Missing expensive observations are not modeled as satisfied constraints.
        return dict(pool[0], mode="baseline", evidence={"basis": "unavailable"}), {
            "reason": "unsupported_expensive_heads_bootstrap", "unsupported": unsupported}
    x = (responses.features.transform(candidates)-responses.offset)/responses.scale
    draws, scales = [], []
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(seed)
        for name in names:
            gp, center, scale, sign = responses.models[name]
            draw = gp.posterior(x).rsample(torch.Size([1])).reshape(-1)
            draws.append((sign*(draw*scale+center)).numpy())
            scales.append(scale)
        if "objective" in responses.models:
            gp, center, scale, sign = responses.models["objective"]
            draw = gp.posterior(x).rsample(torch.Size([1])).reshape(-1)
            objective = (sign*(draw*scale+center)).numpy()
        else:
            objective = np.zeros(len(pool))
    if use_completion:
        completed, probability = completion.sample_completion(candidates, seed+1000003)
    else:
        completed, probability = np.ones(len(pool), bool), np.ones(len(pool))
    constraints = np.column_stack(draws)
    index, mode, violation = choose_sample(objective, constraints, completed, scales)
    # Marginal product is a logging diagnostic under independent head models;
    # the actual acquisition uses joint-across-candidate posterior draws.
    predictions = responses.predict(candidates)
    risk = completion.predict(candidates)["heads"].get("completion")
    p = (np.full(len(pool), .5) if risk is None else np.asarray(risk["probability"]))
    final = p.copy() if use_completion else np.ones(len(pool))
    for name in names:
        final *= np.asarray(predictions[name]["p_satisfied_raw"])
    item = dict(pool[index], mode="baseline", evidence={
        "basis": "raw", "probability": float(final[index]),
        "candidate_key": digest(candidates[index]), "model_id": completion.model_id})
    return item, {
        "reason": "conditional_completion_constrained_posterior_sample", "mode": mode,
        "index": index, "sample_seed": seed, "model_id": completion.model_id,
        "representation_snapshot": completion.snapshot_id,
        "candidate_key": digest(candidates[index]),
        "completion_head_enabled": use_completion, "raw_completion": float(p[index]),
        "sampled_completion_probability": float(probability[index]),
        "sampled_completion": bool(completed[index]),
        "constraint_names": names, "constraint_samples": constraints[index].tolist(),
        "constraint_scales": scales, "objective_sample": float(objective[index]),
        "sampled_violation": float(violation[index]),
        "independent_head_feasibility_raw": float(final[index]),
        "probability_guarantee": False, "calibration_status": "not_applied",
        "observed_incumbent": incumbent(rows)}


def discovery_anchor(rows, analytical_constraints):
    feasible = [r for r in rows if r["risk"]["feasible_label"] is True]
    if feasible:
        return min(feasible, key=lambda r: r["risk"]["objective"])["candidate"], "incumbent"
    names = sorted({n for r in rows for n in r["risk"]["constraints"]
                    if n not in analytical_constraints})
    complete = [r for r in rows if r["risk"]["objective_observed"]
                and all(r["risk"]["constraint_observed"].get(n, False) for n in names)]
    if not complete:
        return None, "global_without_observed_margins"
    values = np.asarray([[r["risk"]["constraints"][n] for n in names] for r in complete])
    scales = np.maximum(values.std(0), 1e-6)
    violation = (np.maximum(values, 0.)/scales).sum(1)
    index = int(np.lexsort(([r["risk"]["objective"] for r in complete], violation))[0])
    return complete[index]["candidate"], "least_observed_violation"


def posterior_pool(problem, rng, count, seen, rows, width=.4):
    anchor, reason = discovery_anchor(rows, problem.analytical_constraints)
    result, keys, rejected = [], set(seen), Counter()
    for attempt in range(count*150):
        if len(result) == count:
            break
        if anchor is not None and attempt % 2:
            fixed = problem.candidate_class(**anchor)
            unit = problem.space.unit(fixed)
            discrete = problem.space.sample(rng).discrete if attempt % 7 == 0 else fixed.discrete
            candidate = problem.space.from_unit(
                rng.uniform(np.maximum(0., unit-width/2), np.minimum(1., unit+width/2)), discrete)
            source = "local_"+reason
        else:
            candidate, source = problem.space.sample(rng), "global_coverage"
        raw = problem.canonical(candidate.to_dict())
        key = digest(raw)
        if key in keys:
            rejected["duplicate"] += 1
            continue
        if any(not np.isfinite(v) or v > 0 for v in problem.precheck(raw).values()):
            rejected["cheap_constraint"] += 1
            continue
        result.append({"candidate": raw, "source": source})
        keys.add(key)
    return result, dict(rejected)
