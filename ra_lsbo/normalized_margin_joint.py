"""Source-separated numerical repair; preserves frozen first bridge experiments."""

import hashlib
from pathlib import Path

import numpy as np

from .adaptive_risk import entropy_decomposition
from .budget import digest


def normalized_members(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 3 or not np.isfinite(p).all() or (p < 0).any()
            or (p.sum(-1) <= 0).any() or not np.allclose(p.sum(-1), 1., atol=2e-6, rtol=0.)):
        raise ValueError("Expected nearly normalized nonnegative member distributions")
    return p/p.sum(-1, keepdims=True)


class NormalizedMarginRisk:
    """Normalize float32 softmax roundoff before exposing or calibrating probabilities."""

    def __init__(self, bridge):
        self.bridge = bridge
        for name in ("features", "snapshot_id", "training_indices", "training_keys",
                     "history_id", "events"):
            setattr(self, name, getattr(bridge, name))
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.model_id = digest({"bridge_id": bridge.model_id, "rule": "float64_member_simplex_normalization",
                                "source_sha256": self.source_sha256})
        self.diagnostics = dict(bridge.diagnostics, numerical_repair="member simplex normalization",
                                unnormalized_bridge_id=bridge.model_id)

    def assert_frozen(self):
        self.bridge.assert_frozen()

    def predict(self, candidates):
        result = self.bridge.predict(candidates)
        members = normalized_members(result["members"])
        total, conditional, information = entropy_decomposition(members)
        probability = members.mean(0)
        # Keep normalization in float64; enforce the finite interval even at exact completion.
        probability = np.clip(probability, 0., 1.)
        probability /= probability.sum(1, keepdims=True)
        return dict(
            result, probabilities=probability.tolist(), members=members.tolist(),
            final_feasibility=probability[:, 0].tolist(), model_id=self.model_id,
            total_entropy=total.tolist(), conditional_entropy=conditional.tolist(),
            information=information.tolist(),
            maximum_member_normalization_delta=float(np.max(np.abs(members-np.asarray(result["members"]))))
            if len(candidates) else 0.)
