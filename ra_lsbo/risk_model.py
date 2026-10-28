"""Raw risk heads; no probability calibration or safety guarantee."""

import hashlib
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .records import HIDDEN_MODES
from .training import training_rows
from .features import history_identity, validate_history


class BinaryRiskHead:
    def __init__(self, x, labels, seed=0, min_per_class=3, trees=64):
        if type(min_per_class) is not int or min_per_class < 1 or type(trees) is not int or trees < 1:
            raise ValueError("Positive integer class threshold and tree count required")
        x = np.asarray(x, dtype=float)
        labels = list(labels)
        if x.ndim != 2 or len(x) != len(labels) or not np.isfinite(x).all():
            raise ValueError("Invalid binary-head features")
        if any(type(v) is not int or v not in (0, 1) for v in labels):
            raise ValueError("Binary labels must be integer 0 or 1")
        positives, negatives = sum(labels), len(labels) - sum(labels)
        self.model = None
        self.dimensions = x.shape[1]
        self.prior_probability = (positives + 1.0) / (len(labels) + 2.0)
        self.diagnostics = {
            "samples": len(labels), "positive": positives, "negative": negatives,
            "minimum_per_class": min_per_class, "calibrated": False,
            "status": "prior_only_no_observations" if not labels else
                      "prior_only_insufficient_classes",
            "prior": "Beta(1,1) posterior mean; not calibrated",
        }
        if min(positives, negatives) >= min_per_class:
            self.model = RandomForestClassifier(
                n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=1)
            self.model.fit(x, labels)
            self.diagnostics["status"] = "fitted_raw"

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.dimensions or not np.isfinite(x).all():
            raise ValueError("Invalid prediction features")
        if self.model is None:
            return np.full(len(x), self.prior_probability)
        index = list(self.model.classes_).index(1)
        return self.model.predict_proba(x)[:, index]


class RiskHeads:
    """Mechanism predictions are conditional on observed labels, not multiplied."""

    def __init__(self, rows, features, seed=0, min_per_class=3, trees=64):
        validate_history(rows)
        if (features.training_indices != [row["evaluation"] for row in rows]
                or features.history_id != history_identity(rows)):
            raise ValueError("Representation and risk heads must use the same history prefix")
        self.features, self.snapshot_id = features, features.snapshot_id
        self.source_identity = {
            key: rows[0].get(key) if rows else None
            for key in ("source_journal_sha256", "evaluator_contract_sha256")
        }
        self.model_id = hashlib.sha256(json.dumps({
            "representation": self.snapshot_id, "history": features.history_id,
            "seed": seed, "min_per_class": min_per_class, "trees": trees,
            "method": "RandomForestClassifier_min_samples_leaf_2",
        }, sort_keys=True).encode()).hexdigest()
        tables = training_rows(rows)
        datasets = {
            "hidden_success": tables["hidden_success"],
            "final_feasibility": tables["final_feasibility"],
            **{name: tables["hidden_modes"][name] for name in HIDDEN_MODES},
        }
        self.heads = {}
        self.diagnostics = {}
        for name, table in datasets.items():
            x = features.transform([r["candidate"] for r in table]).numpy()
            head = BinaryRiskHead(x, [r["target"] for r in table], seed, min_per_class, trees)
            self.heads[name] = head
            self.diagnostics[name] = dict(
                head.diagnostics, training_indices=[r["evaluation"] for r in table],
                interpretation="P(label=1 | label observed, cheap screening passed)",
                representation_snapshot=self.snapshot_id)

    def predict(self, candidates):
        if self.features.snapshot_id != self.snapshot_id:
            raise RuntimeError("Representation changed; refit all risk heads")
        x = self.features.transform(candidates).numpy()
        return {name: head.predict(x).tolist() for name, head in self.heads.items()}
