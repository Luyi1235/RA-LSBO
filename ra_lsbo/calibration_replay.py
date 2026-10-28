"""Chronological train/calibration/test diagnostics with a frozen base model."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import uuid

from .audit import build_snapshot
from .calibration import CalibrationPolicy, CalibratedRiskHeads, observed_target
from .features import LatentFeatures, validate_history
from .metrics import binary_metrics
from .risk_model import RiskHeads


def blocked_calibration(rows, contract, base_project, train_count, calibration_count,
                        seed=0, epochs=60, min_per_class=3, trees=64,
                        policy=None, on_prediction=None):
    validate_history(rows)
    if (type(train_count) is not int or type(calibration_count) is not int
            or min(train_count, calibration_count) < 1
            or train_count + calibration_count >= len(rows)):
        raise ValueError("Require nonempty chronological train/calibration/test blocks")
    boundary = train_count + calibration_count
    training, calibration, testing = rows[:train_count], rows[train_count:boundary], rows[boundary:]
    features = LatentFeatures(base_project, contract["design_variables"], seed).fit(training, epochs)
    risks = RiskHeads(training, features, seed, min_per_class, trees)
    calibrated = CalibratedRiskHeads(risks, calibration, policy)
    metadata = {
        "split": {"training_indices": [r["evaluation"] for r in training],
                  "calibration_indices": [r["evaluation"] for r in calibration],
                  "test_indices": [r["evaluation"] for r in testing]},
        "representation_snapshot": features.snapshot_id, "raw_model_id": risks.model_id,
        "calibration_id": calibrated.calibration_id,
        "calibration_policy": asdict(calibrated.policy),
        "base_source_sha256": features.base_hashes,
        "risk_diagnostics": deepcopy(risks.diagnostics),
        "calibration_diagnostics": deepcopy(calibrated.diagnostics),
    }
    predictions = []
    for row in testing:
        candidate = deepcopy(row["candidate"])
        output = calibrated.predict([candidate])
        forecast = {
            "evaluation": row["evaluation"], "candidate": candidate,
            "training_indices": metadata["split"]["training_indices"],
            "calibration_indices": metadata["split"]["calibration_indices"],
            "representation_snapshot": features.snapshot_id,
            "raw_model_id": risks.model_id, "calibration_id": calibrated.calibration_id,
            "calibration_status": {name: info["status"] for name, info in calibrated.diagnostics.items()},
            **{kind: {name: None if values is None else values[0]
                      for name, values in heads.items()} for kind, heads in output.items()},
        }
        if on_prediction is not None:
            on_prediction(deepcopy(forecast))
        forecast["observed"] = deepcopy(row["risk"])
        predictions.append(forecast)
    return predictions, metadata, calibrated.calibration_records


def calibration_scores(predictions):
    heads = list(predictions[0]["risk_probabilities_raw"]) if predictions else []
    result = {}
    for name in heads:
        known = [(row, observed_target(row["observed"], name)) for row in predictions
                 if observed_target(row["observed"], name) is not None]
        paired = [(row, target) for row, target in known
                  if row["risk_probabilities_calibrated"][name] is not None]
        def score(pairs, key):
            return binary_metrics([row[key][name] for row, target in pairs],
                                  [target for row, target in pairs])
        raw = score(paired, "risk_probabilities_raw")
        calibrated = score(paired, "risk_probabilities_calibrated")
        result[name] = {
            "raw_all": score(known, "risk_probabilities_raw"),
            "effective_all": score(known, "risk_probabilities_effective"),
            "raw_on_calibrated_cases": raw,
            "calibrated_only": calibrated,
            "calibration_coverage": len(paired)/len(known) if known else None,
            "calibrated_minus_raw_brier": calibrated["brier"]-raw["brier"] if paired else None,
            "calibrated_minus_raw_log_loss": calibrated["log_loss"]-raw["log_loss"] if paired else None,
        }
    return result


def run_calibration(source, base_project, output_root, train_count, calibration_count,
                    seed=0, epochs=60, min_per_class=3, trees=64, policy=None):
    source, base_project, root = map(lambda p: Path(p).resolve(), (source, base_project, output_root))
    if root == source or source in root.parents:
        raise ValueError("Calibration output must be outside source run")
    rows, _, source_report, contract = build_snapshot(source)
    policy = policy or CalibrationPolicy()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / ("calibration_" + stamp + "_" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    module_root = Path(__file__).parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(module_root.glob("*.py"))}
    settings = dict(train_count=train_count, calibration_count=calibration_count, seed=seed,
                    epochs=epochs, min_per_class=min_per_class, trees=trees)
    report = {
        "status": "running", "mode": "frozen_chronological_calibration_not_optimization",
        "source_run": str(source), "source_sha256": source_report["source_sha256"],
        "evaluator_contract_sha256": source_report["evaluator_contract_sha256"],
        "base_project": str(base_project), "ra_source_sha256": hashes,
        "settings": settings, "calibration_policy": asdict(policy),
        "python_executable": sys.executable, "python": sys.version,
        "dependencies": {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "torch", "scikit-learn", "botorch", "gpytorch")},
        "scope": [
            "Historical chronological holdout, not a prospective independent run; no new MDA.",
            "Representation and raw heads use training only; frozen during calibration and test.",
            "Test labels never fit calibrators, select thresholds or update models.",
            "Raw fallback is not calibrated; fitted sigmoid is not a confidence bound.",
            "Pooled hidden failure uses 1-hidden_success, never a product or sum of mechanism risks.",
            "Pooled probability does not replace a missing per-mechanism calibrated probability.",
            "Adaptive historical sampling may shift distributions; no reliability or optimization guarantee.",
        ],
    }
    def write_report():
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_report()
    try:
        with (destination / "test_forecasts.jsonl").open("x", encoding="utf-8") as stream:
            def save_forecast(record):
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                print(f"[RA calibration] predicted evaluation {record['evaluation']}", flush=True)
            predictions, metadata, samples = blocked_calibration(
                rows, contract, base_project, **settings, policy=policy, on_prediction=save_forecast)
        report.update(metadata)
        for directory, expected in (
                (source, report["source_sha256"]),
                (base_project / "lsbo", report["base_source_sha256"]),
                (module_root, hashes)):
            if any(hashlib.sha256((directory / name).read_bytes()).hexdigest() != sha
                   for name, sha in expected.items()):
                raise RuntimeError("Source changed during calibration; result not accepted")
        report.update(predictions=len(predictions), test_scores=calibration_scores(predictions))
        for filename, records in (("calibration_samples.jsonl", samples),
                                  ("scored_test_predictions.jsonl", predictions)):
            (destination / filename).write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in records), encoding="utf-8")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report()
        print("Calibration report:", destination / "report.json", flush=True)
    return destination, report
