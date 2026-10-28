"""Use fresh observations for discovery when chronological calibration is unavailable."""

from collections import Counter
from copy import deepcopy

from .calibration import CalibrationPolicy, CalibratedRiskHeads, observed_target
from .features import LatentFeatures
from .responses import ResponseHeads
from .risk_model import RiskHeads


class FreshDiscoveryLearning:
    """No probability upgrade: full-history discovery remains uncalibrated."""

    def __init__(self):
        self.last_fit_count = None
        self.discovery_only = False

    def should_refresh(self, rows):
        return self.discovery_only and self.last_fit_count != len(rows)

    def fit(self, rows, variables, base_project, settings):
        boundary = len(rows)-settings.calibration_window
        training, calibration = rows[:boundary], rows[boundary:]
        policy = CalibrationPolicy()
        train_labels = [observed_target(row["risk"], "final_feasibility") for row in training]
        calibration_labels = [observed_target(row["risk"], "final_feasibility") for row in calibration]
        train_counts, calibration_counts = Counter(train_labels), Counter(calibration_labels)
        reasons = []
        if min(train_counts[0], train_counts[1]) < 3:
            reasons.append("training_classes_insufficient")
        if calibration_counts[0]+calibration_counts[1] < policy.min_samples:
            reasons.append("calibration_samples_insufficient")
        if min(calibration_counts[0], calibration_counts[1]) < policy.min_per_class:
            reasons.append("calibration_classes_insufficient")
        attempted_diagnostics = None
        calibrated = None
        if not reasons:
            features = LatentFeatures(base_project, variables, settings.seed).fit(training, settings.epochs)
            risks = RiskHeads(training, features, settings.seed)
            calibrated = CalibratedRiskHeads(risks, calibration, policy)
            attempted_diagnostics = deepcopy(calibrated.diagnostics)
            if calibrated.diagnostics["final_feasibility"]["status"] != "fitted_empirical":
                reasons.append(calibrated.diagnostics["final_feasibility"]["status"])
                calibrated = None
        self.discovery_only = calibrated is None
        if self.discovery_only:
            # Held-out labels can train a new raw discovery model, but never
            # calibrate that same model. Its probability remains explicitly raw.
            training = rows
            features = LatentFeatures(base_project, variables, settings.seed).fit(training, settings.epochs)
            risks = RiskHeads(training, features, settings.seed)
        responses = ResponseHeads(training, features, settings.seed, settings.gp_steps)
        self.last_fit_count = len(rows)
        metadata = {
            "model_policy": "fresh_discovery_v1",
            "active_regime": "raw_full_history_discovery" if self.discovery_only else "chronological_calibrated",
            "fallback_reasons": reasons,
            "attempted_training_indices": [row["evaluation"] for row in rows[:boundary]],
            "attempted_calibration_indices": [row["evaluation"] for row in calibration],
            "attempted_calibration_diagnostics": attempted_diagnostics,
            "raw_final_feasibility": deepcopy(risks.diagnostics["final_feasibility"]),
            "refresh_on_next_observation": self.discovery_only,
        }
        return (risks, responses, calibrated), metadata
