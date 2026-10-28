"""A conventional feasibility-first control for diagnosing cold-start policy."""

import numpy as np

from .budget import digest
from .posterior_selection import select_posterior
from .selection import incumbent


def log_feasibility(completion, constraint_probabilities):
    factors = np.asarray([completion, *constraint_probabilities], dtype=float)
    if (factors.ndim != 2 or not factors.shape[1] or not np.isfinite(factors).all()
            or (factors < 0).any() or (factors > 1).any()):
        raise ValueError("Compatible finite marginal probabilities in [0,1] required")
    # Preserve an exact zero as impossible under this model, without underflow.
    with np.errstate(divide="ignore"):
        return np.log(factors).sum(axis=0)


def select_feasibility_first(pool, rows, responses, completion, analytical_constraints, seed):
    if incumbent(rows) is not None:
        return select_posterior(pool, rows, responses, completion, analytical_constraints, seed)
    if not pool:
        return None, {"reason": "empty_pool"}
    if responses.features.snapshot_id != completion.snapshot_id:
        raise ValueError("Response and completion representations differ")
    names = sorted(name for name in responses.diagnostics if name.startswith("constraint:")
                   and name.split(":", 1)[1] not in analytical_constraints)
    unsupported = [name for name in names if name not in responses.models]
    if not names or unsupported or "completion" not in completion.models:
        return dict(pool[0], mode="baseline", evidence={"basis": "unavailable"}), {
            "reason": "unsupported_expensive_heads_bootstrap", "unsupported": unsupported}
    candidates = [p["candidate"] for p in pool]
    prediction = responses.predict(candidates)
    probability = completion.predict(candidates)["heads"]["completion"]["probability"]
    scores = log_feasibility(probability, [prediction[name]["p_satisfied_raw"] for name in names])
    index = int(np.argmax(scores))
    selected = dict(pool[index], mode="baseline", evidence={
        "basis": "raw", "probability": float(np.exp(scores[index])),
        "candidate_key": digest(candidates[index]), "model_id": completion.model_id})
    return selected, {
        "reason": "feasibility_first_before_observed_incumbent", "index": index,
        "log_independent_head_feasibility": float(scores[index]) if np.isfinite(scores[index]) else None,
        "all_candidates_zero_under_model": bool(np.isneginf(scores).all()),
        "constraint_names": names, "model_id": completion.model_id,
        "representation_snapshot": completion.snapshot_id,
        "calibration_status": "not_applied", "probability_guarantee": False,
        "assumption": "independent constraint heads conditional on completion",
        "scope": "standard diagnostic control; not a novelty claim",
    }
