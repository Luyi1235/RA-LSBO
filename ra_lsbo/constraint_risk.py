"""A constraint-aware raw score followed by the existing held-out calibration."""

from copy import deepcopy
import hashlib

import numpy as np

from .budget import digest


class ConstraintRiskHeads:
    """Minimum of RF feasibility and marginal GP scores; not a joint probability."""

    def __init__(self, risks, responses, required_constraints):
        if (risks.snapshot_id != responses.snapshot_id
                or risks.features.history_id != responses.features.history_id):
            raise ValueError("Risk and constraint models must share training history and representation")
        self.risks, self.responses = risks, responses
        self.features, self.snapshot_id = risks.features, risks.snapshot_id
        self.source_identity, self.heads = risks.source_identity, risks.heads
        self.required_heads = tuple(sorted("constraint:" + name for name in required_constraints))
        if not self.required_heads or len(set(self.required_heads)) != len(self.required_heads):
            raise ValueError("Require distinct expensive constraints")
        self.diagnostics = deepcopy(risks.diagnostics)
        missing = [name for name in self.required_heads if name not in responses.models]
        self.diagnostics["final_feasibility"].update(
            raw_score_rule="minimum_RF_and_constraint_marginals",
            raw_score_is_joint_probability=False, required_constraint_heads=list(self.required_heads),
            missing_constraint_heads=missing)
        if missing:
            self.diagnostics["final_feasibility"]["status"] = "unavailable_constraint_models"
        self.base_model_id = risks.model_id
        self.response_state_id = self._response_identity()
        self.model_id = digest({
            "rule": "minimum_RF_and_constraint_marginals_v1",
            "raw_risk_model": risks.model_id, "response_models": self.response_state_id,
            "required_constraint_heads": self.required_heads,
        })

    def _response_identity(self):
        hashes = {}
        for name in self.required_heads:
            if name not in self.responses.models:
                hashes[name] = None
                continue
            model, center, scale, sign = self.responses.models[name]
            hasher = hashlib.sha256()
            hasher.update(digest({"center": center, "scale": scale, "sign": sign}).encode())
            for key, value in sorted(model.state_dict().items()):
                tensor = value.detach().cpu().contiguous().numpy()
                hasher.update(digest({"name": key, "shape": tensor.shape, "dtype": str(tensor.dtype)}).encode())
                hasher.update(tensor.tobytes())
            hashes[name] = hasher.hexdigest()
        return digest(hashes)

    def predict(self, candidates):
        if self.risks.model_id != self.base_model_id or self._response_identity() != self.response_state_id:
            raise RuntimeError("Raw risk or constraint model changed; refit calibration")
        raw = self.risks.predict(candidates)
        predicted = self.responses.predict(candidates)
        components = [np.asarray(raw["final_feasibility"])]
        for name in self.required_heads:
            values = predicted.get(name)
            if values is not None:
                components.append(np.asarray(values["p_satisfied_raw"]))
        raw["final_feasibility"] = np.minimum.reduce(components).tolist()
        return raw


def fit_frozen_models(plan):
    """Rebuild the pre-pilot models from their unchanged historical split."""
    from .audit import build_snapshot
    from .calibration import CalibratedRiskHeads
    from .features import LatentFeatures
    from .responses import ResponseHeads
    from .risk_model import RiskHeads
    from .transfer_validation import candidate_key, unseen_rows

    rows, _, audit, contract = build_snapshot(plan["historical_source"])
    if audit["source_sha256"] != plan["historical_source_sha256"]:
        raise ValueError("Historical identity changed")
    features = LatentFeatures(plan["base_project"], contract["design_variables"], seed=0).fit(rows[:180], epochs=60)
    risks = RiskHeads(rows[:180], features, seed=0)
    responses = ResponseHeads(rows[:180], features, seed=0, steps=45)
    calibration, _ = unseen_rows(rows[180:240], features,
                                {candidate_key(features, row["candidate"]) for row in rows[:180]})
    old_calibrated = CalibratedRiskHeads(risks, calibration)
    if risks.model_id != plan["model_id"] or old_calibrated.calibration_id != plan["calibration_id"]:
        raise ValueError("Original model identity changed")
    expensive = sorted(set(contract["required_constraints"])-set(contract["analytical_constraints"]))
    combined = ConstraintRiskHeads(risks, responses, expensive)
    new_calibrated = CalibratedRiskHeads(combined, calibration)
    return rows, features, risks, responses, old_calibrated, combined, new_calibrated
