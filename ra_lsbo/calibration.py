"""Frozen-model chronological calibration; empirical scores, not safety bounds."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .features import history_identity, validate_history
from .records import finite_number


@dataclass(frozen=True)
class CalibrationPolicy:
    min_samples: int = 30
    min_per_class: int = 5
    min_distinct_scores: int = 3
    regularization_c: float = 1.0
    probability_clip: float = 1e-6

    def __post_init__(self):
        for name in ("min_samples", "min_per_class", "min_distinct_scores"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Calibration thresholds must be positive integers")
        if (not finite_number(self.regularization_c) or self.regularization_c <= 0
                or not finite_number(self.probability_clip)
                or not 0 < self.probability_clip < .5):
            raise ValueError("Invalid calibration regularization or clipping")


def probability_vector(values):
    values = list(values)
    if any(not finite_number(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Probabilities must be finite numbers in [0,1]")
    return np.asarray(values, dtype=float)


def observed_target(risk, name):
    if risk["state"] == "precheck_rejected":
        return None
    if name == "any_hidden_failure":
        value = risk["hidden_success"]
        return None if value is None else 1 - int(value)
    if name == "hidden_success":
        value = risk["hidden_success"]
    elif name == "final_feasibility":
        value = risk["feasible_label"]
    else:
        value = risk["hidden_labels"][name]
    return None if value is None else int(value)


class SigmoidCalibrator:
    """Regularized sigmoid on log-odds, fitted only on held-out predictions."""

    def __init__(self, probabilities, labels, policy=None, base_fitted=True):
        self.policy = policy or CalibrationPolicy()
        probabilities = probability_vector(probabilities)
        labels = list(labels)
        if len(labels) != len(probabilities) or any(type(y) is not int or y not in (0, 1) for y in labels):
            raise ValueError("Calibration labels must be matching binary integers")
        if type(base_fitted) is not bool:
            raise ValueError("base_fitted must be boolean")
        self.model = None
        positive = sum(labels)
        clipped = np.clip(probabilities, self.policy.probability_clip, 1-self.policy.probability_clip)
        distinct = len(np.unique(clipped))
        reasons = []
        if not base_fitted:
            reasons.append("base_unfitted")
        if len(labels) < self.policy.min_samples:
            reasons.append("insufficient_samples")
        if min(positive, len(labels)-positive) < self.policy.min_per_class:
            reasons.append("insufficient_classes")
        if distinct < self.policy.min_distinct_scores:
            reasons.append("insufficient_score_variation")
        self.diagnostics = {
            "method": "regularized_logit_sigmoid",
            "policy": asdict(self.policy), "samples": len(labels),
            "positive": positive, "negative": len(labels)-positive, "distinct_scores": distinct,
            "calibrator_fitted": False, "reliability_verified": False,
            "is_confidence_bound": False, "fallback_reasons": reasons,
            "status": "fallback_" + reasons[0] if reasons else "pending",
            "parameters": None,
        }
        if reasons:
            return
        model = LogisticRegression(
            C=self.policy.regularization_c, solver="lbfgs", max_iter=1000,
            tol=1e-8, class_weight=None, random_state=0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(self._logits(probabilities), labels)
            slope, intercept = float(model.coef_[0, 0]), float(model.intercept_[0])
            if not np.isfinite([slope, intercept]).all():
                raise FloatingPointError("Nonfinite calibration parameters")
        except (ConvergenceWarning, FloatingPointError, np.linalg.LinAlgError) as error:
            self.diagnostics.update(
                status="fallback_fit_failed", fallback_reasons=["fit_failed"],
                error=f"{type(error).__name__}: {error}")
            return
        # Do not silently reverse a risk ordering on a small calibration block.
        if slope <= 0:
            self.diagnostics.update(
                status="fallback_nonpositive_slope", fallback_reasons=["nonpositive_slope"])
            return
        self.model = model
        self.diagnostics.update(
            status="fitted_empirical", calibrator_fitted=True,
            parameters={"slope": slope, "intercept": intercept})

    def _logits(self, values):
        clipped = np.clip(values, self.policy.probability_clip, 1-self.policy.probability_clip)
        return (np.log(clipped) - np.log1p(-clipped)).reshape(-1, 1)

    def predict(self, probabilities):
        probabilities = probability_vector(probabilities)
        if self.model is None:
            return None
        if not len(probabilities):
            return []
        result = self.model.predict_proba(self._logits(probabilities))[:, 1]
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite calibrated probabilities")
        return result.tolist()


class CalibratedRiskHeads:
    """Bind calibrators to one frozen representation and one set of raw heads."""

    def __init__(self, risks, calibration_rows, policy=None):
        validate_history(calibration_rows)
        training_indices = risks.features.training_indices
        if not training_indices or not calibration_rows:
            raise ValueError("Nonempty training and calibration blocks required")
        if calibration_rows[0]["evaluation"] <= max(training_indices):
            raise ValueError("Calibration must be strictly after training, without overlap")
        for key, identity in risks.source_identity.items():
            if any(row.get(key) != identity for row in calibration_rows):
                raise ValueError("Calibration source identity differs from training: " + key)
        self.risks = risks
        self.model_id = risks.model_id
        self.snapshot_id = risks.snapshot_id
        self.policy = policy or CalibrationPolicy()
        self.calibration_indices = [row["evaluation"] for row in calibration_rows]
        self.calibration_history_id = history_identity(calibration_rows)
        self.calibrators, self.diagnostics = {}, {}
        raw = self._raw([row["candidate"] for row in calibration_rows])
        self.calibration_records = []
        for position, row in enumerate(calibration_rows):
            self.calibration_records.append({
                "evaluation": row["evaluation"],
                "risk_probabilities_raw": {name: values[position] for name, values in raw.items()},
                "labels": {name: observed_target(row["risk"], name) for name in raw},
            })
        for name, values in raw.items():
            valid = [(row["evaluation"], p, observed_target(row["risk"], name))
                     for row, p in zip(calibration_rows, values)
                     if observed_target(row["risk"], name) is not None]
            base_name = "hidden_success" if name == "any_hidden_failure" else name
            calibrator = SigmoidCalibrator(
                [p for _, p, y in valid], [y for _, p, y in valid], self.policy,
                base_fitted=risks.diagnostics[base_name]["status"] == "fitted_raw")
            self.calibrators[name] = calibrator
            self.diagnostics[name] = dict(
                deepcopy(calibrator.diagnostics),
                calibration_indices=[index for index, p, y in valid],
                excluded_unknown_or_precheck=len(calibration_rows)-len(valid),
                base_status=risks.diagnostics[base_name]["status"],
                interpretation="P(label=1 | label observed, cheap screening passed)",
                raw_source="1 - hidden_success" if name == "any_hidden_failure" else name)
        self.calibration_id = hashlib.sha256(json.dumps({
            "raw_model_id": self.model_id, "history": self.calibration_history_id,
            "policy": asdict(self.policy), "heads": self.diagnostics,
        }, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _raw(self, candidates):
        if (self.risks.model_id != self.model_id
                or self.risks.features.snapshot_id != self.snapshot_id):
            raise RuntimeError("Raw model or representation changed; refit calibration")
        raw = self.risks.predict(candidates)
        raw["any_hidden_failure"] = [1-p for p in raw["hidden_success"]]
        return raw

    def predict(self, candidates):
        raw = self._raw(candidates)
        calibrated = {name: self.calibrators[name].predict(values) for name, values in raw.items()}
        return {
            "risk_probabilities_raw": raw,
            "risk_probabilities_calibrated": calibrated,
            "risk_probabilities_effective": {
                name: raw[name] if values is None else values
                for name, values in calibrated.items()},
        }
