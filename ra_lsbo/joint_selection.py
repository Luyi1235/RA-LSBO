"""Constraint posterior selection with ICM covariance across output heads."""

import numpy as np
import torch

from .budget import digest
from .posterior_selection import choose_sample
from .selection import incumbent


def select_joint_posterior(pool, rows, responses, completion, joint, seed):
    if not pool:
        return None, {"reason": "empty_pool"}
    identities = {responses.features.snapshot_id, completion.snapshot_id, joint.snapshot_id}
    if len(identities) != 1:
        raise ValueError("Joint acquisition model representations differ")
    if not joint.names or joint.unsupported or "completion" not in completion.models:
        return dict(pool[0], mode="baseline", evidence={"basis": "unavailable"}), {
            "reason": "unsupported_expensive_heads_bootstrap", "unsupported": joint.unsupported}
    candidates = [item["candidate"] for item in pool]
    constraints = joint.sample(candidates, seed+1, 1)[0].numpy()
    objective = np.zeros(len(pool))
    if "objective" in responses.models:
        gp, center, scale, sign = responses.models["objective"]
        x = (responses.features.transform(candidates)-responses.offset)/responses.scale
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(seed)
            objective = (sign*(gp.posterior(x).rsample(torch.Size([1])).reshape(-1)*scale+center)).numpy()
    completed, probability = completion.sample_completion(candidates, seed+1000003)
    index, mode, violation = choose_sample(objective, constraints, completed, joint.scales)
    item = dict(pool[index], mode="baseline", evidence={"basis": "unavailable"})
    return item, {
        "reason": "joint_constraint_completion_posterior_sample", "mode": mode,
        "candidate_key": digest(candidates[index]), "index": index,
        "joint_model_id": joint.model_id, "completion_model_id": completion.model_id,
        "representation_snapshot": joint.snapshot_id, "sample_seed": seed,
        "constraint_names": joint.names, "constraint_samples": constraints[index].tolist(),
        "constraint_scales": joint.scales, "objective_sample": float(objective[index]),
        "sampled_completion": bool(completed[index]),
        "sampled_completion_probability": float(probability[index]),
        "sampled_violation": float(violation[index]), "observed_incumbent": incumbent(rows),
        "cross_output_model": "ICM", "icm_rank": joint.settings.rank,
        "calibration_status": "not_applied", "probability_guarantee": False,
        "risk_debit": "full unit charge; no posterior probability used as a certificate",
    }
