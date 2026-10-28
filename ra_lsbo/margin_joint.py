"""Conditional terminal/constraint bridge for the second development revision."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .adaptive_risk import (JointRiskEnsemble, ReferenceTemperature, entropy_decomposition,
                            weights_digest)
from .adaptive_system import AdaptiveLearning
from .budget import digest
from .structured_models import StructuredResponseHeads


def margin_joint_probabilities(member_events, constraint_satisfaction):
    """Chain rule with a union-bound surrogate within the completion branch."""
    members = np.asarray(member_events, dtype=float)
    marginals = np.asarray(constraint_satisfaction, dtype=float)
    if (members.ndim != 3 or marginals.ndim != 2 or marginals.shape[0] != members.shape[1]
            or marginals.shape[1] < 1 or not np.isfinite(marginals).all()
            or (marginals < 0).any() or (marginals > 1).any()
            or not np.isfinite(members).all() or (members < 0).any()
            or not np.allclose(members.sum(-1), 1., atol=1e-6)):
        raise ValueError("Normalized terminal events and valid constraint marginals required")
    feasible_given_completion = np.maximum(0., 1.-(1.-marginals).sum(1))
    completion = members[:, :, :2].sum(-1)
    result = members.copy()
    result[:, :, 0] = completion*feasible_given_completion[None, :]
    result[:, :, 1] = completion*(1.-feasible_given_completion[None, :])
    return result, feasible_given_completion


class MarginJointRisk:
    def __init__(self, terminal, responses):
        if terminal.snapshot_id != responses.snapshot_id:
            raise ValueError("Terminal and margin models must share their encoder")
        self.terminal, self.responses = terminal, responses
        self.features, self.snapshot_id = terminal.features, terminal.snapshot_id
        self.training_indices, self.training_keys = terminal.training_indices, terminal.training_keys
        self.history_id, self.events = terminal.history_id, terminal.events
        self.required_constraints = sorted(n for n in responses.diagnostics if n.startswith("constraint:"))
        if not self.required_constraints:
            raise ValueError("At least one explicit constraint GP is required")
        self.gp_id = self._gp_identity()
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.model_id = digest({"rule": "completion_times_conditional_margin_union_surrogate_v1",
                                "terminal_id": terminal.model_id, "gp_id": self.gp_id,
                                "source_sha256": self.source_sha256})
        self.diagnostics = deepcopy(terminal.diagnostics)
        self.diagnostics.update(
            composition="P(completion) times conditional explicit-feasibility surrogate",
            constraint_joint="max(0, 1-sum(1-p_j)); no independence product across constraints",
            constraint_names=self.required_constraints,
            unavailable_constraints=[n for n in self.required_constraints if n not in responses.models],
            guarantee="Algebraic union expression on GP estimates, not a certified risk bound")

    def _gp_identity(self):
        names = sorted(self.responses.models)
        return digest({
            "weights": weights_digest([self.responses.models[n][0] for n in names]),
            "normalization": {n: list(self.responses.models[n][1:]) for n in names},
            "offset": self.responses.offset.tolist(), "scale": self.responses.scale.tolist(),
            "snapshot": self.responses.snapshot_id})

    def assert_frozen(self):
        self.terminal.assert_frozen()
        if self._gp_identity() != self.gp_id:
            raise RuntimeError("Constraint GP changed after binding the joint score")

    def predict(self, candidates):
        self.assert_frozen()
        original = self.terminal.predict(candidates)
        predictions = self.responses.predict(candidates)
        marginal = np.column_stack([
            np.zeros(len(candidates)) if predictions.get(name) is None else
            predictions[name]["p_satisfied_raw"] for name in self.required_constraints])
        members, conditional = margin_joint_probabilities(original["members"], marginal)
        total, entropy, information = entropy_decomposition(members)
        return {
            "events": list(self.events), "probabilities": members.mean(0).tolist(),
            "members": members.tolist(), "final_feasibility": members.mean(0)[:, 0].tolist(),
            "total_entropy": total.tolist(), "conditional_entropy": entropy.tolist(),
            "information": information.tolist(), "model_id": self.model_id,
            "unsupported_events": self.diagnostics["unsupported_events"],
            "completion_probability": np.asarray(original["probabilities"])[:, :2].sum(1).tolist(),
            "explicit_feasibility_surrogate_given_completion": conditional.tolist(),
            "constraint_marginals": marginal.tolist()}

    def save_checkpoint(self, path):
        self.assert_frozen()
        path = Path(path)
        self.terminal.save_checkpoint(path.with_suffix(".terminal.pt"))
        states = {name: {"state": model.state_dict(), "normalization": [center, scale, sign]}
                  for name, (model, center, scale, sign) in self.responses.models.items()}
        with path.open("xb") as stream:
            torch.save({"models": states, "metadata_json": json.dumps({
                "schema": "ra_margin_joint_v1", "model_id": self.model_id, "gp_id": self.gp_id,
                "source_sha256": self.source_sha256, "diagnostics": self.responses.diagnostics},
                sort_keys=True, allow_nan=False)}, stream)

    @classmethod
    def load_checkpoint(cls, path, features, rows):
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = json.loads(payload["metadata_json"])
        if (metadata["schema"] != "ra_margin_joint_v1"
                or metadata["source_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            raise ValueError("Bridge checkpoint source changed")
        terminal = JointRiskEnsemble.load_checkpoint(path.with_suffix(".terminal.pt"), features)
        responses = StructuredResponseHeads(rows, features, terminal.seed, steps=1)
        if set(payload["models"]) != set(responses.models):
            raise ValueError("Bridge GP set differs")
        for name, saved in payload["models"].items():
            model, center, scale, sign = responses.models[name]
            if [center, scale, sign] != saved["normalization"]:
                raise ValueError("Bridge GP normalization differs")
            model.train()
            model.load_state_dict(saved["state"], strict=True)
            model.eval()
        responses.diagnostics = metadata["diagnostics"]
        result = cls(terminal, responses)
        if result.model_id != metadata["model_id"] or result.gp_id != metadata["gp_id"]:
            raise ValueError("Restored bridge model identity differs")
        return result


class MarginLearning(AdaptiveLearning):
    def fit(self, rows, variables, base_project):
        (terminal, responses, _), metadata = super().fit(rows, variables, base_project)
        model = MarginJointRisk(terminal, responses)
        reference = [r for r in rows if r["evaluation"] in self.reference_ids]
        calibration = ReferenceTemperature(model, reference)
        metadata.update(
            model_id=model.model_id, terminal_model_id=terminal.model_id,
            risk_diagnostics=model.diagnostics, calibration=calibration.to_dict(),
            calibration_id=calibration.calibration_id, method_family="conditional_margin_joint_v1")
        return (model, responses, calibration), metadata
