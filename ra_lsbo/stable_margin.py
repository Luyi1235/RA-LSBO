"""Opt-in normalized bridge factory for future runs; frozen studies stay unchanged."""

import hashlib
import json
from pathlib import Path

from .adaptive_risk import ReferenceTemperature
from .budget import digest
from .margin_joint import MarginJointRisk, MarginLearning
from .normalized_margin_joint import NormalizedMarginRisk


class StableMarginRisk(NormalizedMarginRisk):
    def __init__(self, bridge):
        super().__init__(bridge)
        self.factory_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.model_id = digest({"normalization_id": self.model_id,
                                "factory_source_sha256": self.factory_source_sha256})

    def save_checkpoint(self, path):
        self.assert_frozen()
        self.bridge.save_checkpoint(path)
        with Path(path).with_suffix(".normalization.json").open("x", encoding="utf-8") as stream:
            json.dump({"model_id": self.model_id, "bridge_id": self.bridge.model_id,
                       "factory_source_sha256": self.factory_source_sha256,
                       "normalization_source_sha256": self.source_sha256}, stream, allow_nan=False)

    @classmethod
    def load_checkpoint(cls, path, features, rows):
        metadata = json.loads(Path(path).with_suffix(".normalization.json").read_text(encoding="utf-8"))
        result = cls(MarginJointRisk.load_checkpoint(path, features, rows))
        if (result.model_id != metadata["model_id"] or result.bridge.model_id != metadata["bridge_id"]
                or result.factory_source_sha256 != metadata["factory_source_sha256"]
                or result.source_sha256 != metadata["normalization_source_sha256"]):
            raise ValueError("Stable bridge normalization identity differs")
        return result


class StableMarginLearning(MarginLearning):
    def fit(self, rows, variables, base_project):
        (bridge, responses, _), metadata = super().fit(rows, variables, base_project)
        risk = StableMarginRisk(bridge)
        reference = [r for r in rows if r["evaluation"] in self.reference_ids]
        calibration = ReferenceTemperature(risk, reference)
        metadata.update(
            model_id=risk.model_id, unnormalized_bridge_id=bridge.model_id,
            inference_normalization=True, method_family="normalized_conditional_margin_v2",
            risk_diagnostics=risk.diagnostics, calibration=calibration.to_dict(),
            calibration_id=calibration.calibration_id)
        return (risk, responses, calibration), metadata
