"""Chronological risk/response validation on one immutable historical run."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import uuid

import torch

from .audit import build_snapshot
from .features import BASE_FILES, LatentFeatures, validate_history
from .metrics import binary_metrics
from .responses import ResponseHeads
from .risk_model import RiskHeads


def prequential(rows, contract, base_project, initial=10, seed=0, epochs=60,
                gp_steps=45, min_per_class=3, trees=64, on_prediction=None):
    validate_history(rows)
    if type(initial) is not int or not 1 <= initial < len(rows):
        raise ValueError("Require 1 <= initial < available observations")
    predictions, base_hashes = [], None
    for position in range(initial, len(rows)):
        prefix = rows[:position]
        current = rows[position]
        features = LatentFeatures(base_project, contract["design_variables"], seed)
        if base_hashes is not None and features.base_hashes != base_hashes:
            raise RuntimeError("Base source changed between prefixes")
        base_hashes = features.base_hashes
        features.fit(prefix, epochs)
        risks = RiskHeads(prefix, features, seed, min_per_class, trees)
        responses = ResponseHeads(prefix, features, seed, gp_steps)
        # Only the candidate, never its outcome, is passed to prediction methods.
        candidate = deepcopy(current["candidate"])
        probability = risks.predict([candidate])
        regression = responses.predict([candidate])
        record = {
            "evaluation": current["evaluation"],
            "training_indices": [r["evaluation"] for r in prefix],
            "representation_snapshot": features.snapshot_id,
            "candidate": candidate,
            "risk_probabilities_raw": {k: values[0] for k, values in probability.items()},
            "risk_diagnostics": risks.diagnostics,
            "response_predictions": {
                k: None if values is None else {key: val[0] for key, val in values.items()}
                for k, values in regression.items()},
            "response_diagnostics": responses.diagnostics,
        }
        if on_prediction is not None:
            on_prediction(deepcopy(record))
        # Reveal the outcome after prediction has been saved to the forecast log.
        record["observed"] = deepcopy(current["risk"])
        predictions.append(record)
    return predictions, base_hashes


def summarize(predictions):
    heads = list(predictions[0]["risk_probabilities_raw"]) if predictions else []
    scores = {}
    for name in heads:
        groups = {"all_available": [], "fitted_raw": [], "prior_only": []}
        for row in predictions:
            observed = row["observed"]
            target = (observed["hidden_success"] if name == "hidden_success" else
                      observed["feasible_label"] if name == "final_feasibility" else
                      observed["hidden_labels"][name])
            if observed["state"] == "precheck_rejected" or target is None:
                continue
            pair = (row["risk_probabilities_raw"][name], int(target))
            groups["all_available"].append(pair)
            status = row["risk_diagnostics"][name]["status"]
            groups["fitted_raw" if status == "fitted_raw" else "prior_only"].append(pair)
        scores[name] = {
            group: binary_metrics([p for p, y in pairs], [y for p, y in pairs])
            for group, pairs in groups.items()
        }
    regression = {}
    for name in predictions[0]["response_predictions"] if predictions else []:
        errors = []
        for row in predictions:
            prediction, observed = row["response_predictions"][name], row["observed"]
            if prediction is None:
                continue
            if name == "objective":
                actual = observed["objective"] if observed["objective_observed"] else None
            else:
                actual = observed["constraints"][name.split(":", 1)[1]]
            if actual is not None:
                errors.append(prediction["mean"] - actual)
        regression[name] = {
            "samples": len(errors),
            "rmse": math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None,
            "mae": sum(abs(e) for e in errors)/len(errors) if errors else None,
        }
    return {
        "predictions": len(predictions), "binary_scores_raw": scores,
        "regression_scores_original_units": regression,
        "risk_status_counts": {name: dict(Counter(r["risk_diagnostics"][name]["status"]
                                                for r in predictions)) for name in heads},
    }


def run_replay(source, base_project, output_root, **settings):
    source, base_project, root = map(lambda p: Path(p).resolve(), (source, base_project, output_root))
    if root == source or source in root.parents:
        raise ValueError("Replay output must be outside source run")
    rows, _, source_report, contract = build_snapshot(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / ("replay_" + stamp + "_" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    ra_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(Path(__file__).parent.glob("*.py"))}
    report = {
        "status": "running", "mode": "prequential_diagnostic_not_optimization",
        "source_run": str(source), "source_sha256": source_report["source_sha256"],
        "evaluator_contract_sha256": source_report["evaluator_contract_sha256"],
        "base_project": str(base_project), "ra_source_sha256": ra_hashes,
        "settings": settings, "python_executable": sys.executable, "python": sys.version,
        "dependencies": {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "torch", "scikit-learn", "botorch", "gpytorch")},
        "scope": [
            "Historical candidates, not RA-LSBO selected candidates; no new MDA.",
            "All representations and heads retrained from scratch on each prefix.",
            "Known full design grammar allowed; no future observed objective or labels used.",
            "Mechanism outputs condition on observed labels; never multiplied as a joint probability.",
            "Probabilities are raw; no calibration, confidence bounds or safety guarantees.",
            "This single small run cannot demonstrate optimization gains or reliable calibration.",
        ],
    }
    def write_report():
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_report()
    try:
        with (destination / "forecasts.jsonl").open("x", encoding="utf-8") as stream:
            def save_forecast(record):
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                print(f"[RA] predicted evaluation {record['evaluation']} using "
                      f"{len(record['training_indices'])} earlier observations", flush=True)
            predictions, base_hashes = prequential(
                rows, contract, base_project, on_prediction=save_forecast, **settings)
        report["base_source_sha256"] = base_hashes
        if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != sha
               for name, sha in report["source_sha256"].items()):
            raise RuntimeError("Source run changed; replay not accepted")
        if any(hashlib.sha256((base_project / "lsbo" / name).read_bytes()).hexdigest() != base_hashes[name]
               for name in BASE_FILES):
            raise RuntimeError("Base source changed; replay not accepted")
        if any(hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != sha
               for name, sha in ra_hashes.items()):
            raise RuntimeError("RA source changed; replay not accepted")
        report.update(summarize(predictions))
        (destination / "scored_predictions.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in predictions), encoding="utf-8")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report()
        print("Replay report:", destination / "report.json", flush=True)
    return destination, report
