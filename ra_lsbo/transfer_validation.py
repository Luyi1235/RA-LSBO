"""Frozen cross-policy validation with exact-contract and design-overlap guards."""

from collections import Counter
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
from .calibration_replay import calibration_scores
from .features import LatentFeatures, validate_history
from .risk_model import RiskHeads


def candidate_key(features, candidate):
    canonical = features.candidate(candidate).to_dict()
    canonical["continuous"] = {name: float(value) if value else 0.
                               for name, value in canonical["continuous"].items()}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, allow_nan=False).encode()).hexdigest()


def unseen_rows(rows, features, excluded):
    seen, accepted, removed = set(excluded), [], []
    for row in rows:
        key = candidate_key(features, row["candidate"])
        if key in seen:
            removed.append({"evaluation": row["evaluation"],
                            "reason": "seen_in_fit" if key in excluded else "repeated_in_block"})
        else:
            accepted.append(row)
            seen.add(key)
    return accepted, removed


def gate_diagnostics(predictions, threshold=.8):
    accepted = [p for p in predictions
                if p["risk_probabilities_calibrated"]["final_feasibility"] is not None
                and p["risk_probabilities_calibrated"]["final_feasibility"] >= threshold]
    labels = [observed_target(p["observed"], "final_feasibility") for p in accepted]
    known = [label for label in labels if label is not None]
    return {
        "threshold": threshold, "predictions": len(predictions), "accepted": len(accepted),
        "known_accepted": len(known), "unknown_accepted": len(labels)-len(known),
        "observed_feasible_fraction": sum(known)/len(known) if known else None,
        "observed_nonfeasible": len(known)-sum(known),
        "interpretation": "Retrospective empirical selection only; no reliability or risk bound.",
    }


def transfer_calibration(source_rows, target_histories, contract, base_project,
                         train_count=180, calibration_count=60, seed=0, epochs=60,
                         trees=64, policy=None, on_prediction=None):
    validate_history(source_rows)
    if (type(train_count) is not int or type(calibration_count) is not int
            or min(train_count, calibration_count) < 1
            or train_count+calibration_count >= len(source_rows)):
        raise ValueError("Nonempty train/calibration/holdout blocks required")
    identity = source_rows[0].get("evaluator_contract_sha256")
    if not identity:
        raise ValueError("Frozen evaluator identity required")
    for rows in target_histories.values():
        validate_history(rows)
        if not rows or any(row.get("evaluator_contract_sha256") != identity for row in rows):
            raise ValueError("Cross-run evaluator contract differs")
        if rows[0].get("source_journal_sha256") == source_rows[0].get("source_journal_sha256"):
            raise ValueError("Target is a copy of the fitting journal")
    boundary = train_count+calibration_count
    training = source_rows[:train_count]
    features = LatentFeatures(base_project, contract["design_variables"], seed).fit(training, epochs)
    training_keys = {candidate_key(features, row["candidate"]) for row in training}
    calibration, calibration_removed = unseen_rows(
        source_rows[train_count:boundary], features, training_keys)
    if not calibration:
        raise ValueError("No unseen calibration candidates remain")
    risks = RiskHeads(training, features, seed=seed, trees=trees)
    calibrated = CalibratedRiskHeads(risks, calibration, policy)
    known_keys = {candidate_key(features, row["candidate"]) for row in source_rows[:boundary]}
    cohorts = {"source_holdout": source_rows[boundary:], **target_histories}
    if "source_holdout" in target_histories:
        raise ValueError("Reserved cohort name: source_holdout")
    predictions, metadata = {}, {
        "training_indices": features.training_indices,
        "calibration_indices": calibrated.calibration_indices,
        "calibration_excluded": calibration_removed,
        "representation_snapshot": features.snapshot_id, "raw_model_id": risks.model_id,
        "calibration_id": calibrated.calibration_id, "base_source_sha256": features.base_hashes,
        "risk_diagnostics": deepcopy(risks.diagnostics),
        "calibration_diagnostics": deepcopy(calibrated.diagnostics),
        "calibration_policy": asdict(calibrated.policy), "cohorts": {},
    }
    all_test_keys = set()
    for name, rows in cohorts.items():
        selected, removed = unseen_rows(rows, features, known_keys)
        cohort_keys = {candidate_key(features, row["candidate"]) for row in selected}
        metadata["cohorts"][name] = {
            "available": len(rows), "retained": len(selected), "excluded": removed,
            "overlap_with_previous_test_cohorts": len(cohort_keys & all_test_keys),
            "source_journal_sha256": rows[0].get("source_journal_sha256") if rows else None,
        }
        all_test_keys.update(cohort_keys)
        predictions[name] = []
        if not selected:
            continue
        output = calibrated.predict([row["candidate"] for row in selected])
        for position, row in enumerate(selected):
            forecast = {
                "cohort": name, "evaluation": row["evaluation"],
                "candidate": deepcopy(row["candidate"]),
                "source_journal_sha256": row.get("source_journal_sha256"),
                "evaluator_contract_sha256": identity,
                "representation_snapshot": features.snapshot_id,
                "raw_model_id": risks.model_id, "calibration_id": calibrated.calibration_id,
                **{kind: {head: None if values is None else values[position]
                          for head, values in heads.items()} for kind, heads in output.items()},
            }
            if on_prediction:
                on_prediction(deepcopy(forecast))
            forecast["observed"] = deepcopy(row["risk"])
            predictions[name].append(forecast)
    metadata["unique_test_designs_across_cohorts"] = len(all_test_keys)
    return predictions, metadata, calibrated.calibration_records


def run_transfer(source, targets, base_project, output_root, **settings):
    source, base_project, output_root = [Path(p).resolve() for p in (source, base_project, output_root)]
    targets = [Path(p).resolve() for p in targets]
    if len(set(targets)) != len(targets) or source in targets:
        raise ValueError("Distinct fitting and target source paths required")
    if any(output_root == p or p in output_root.parents for p in [source]+targets):
        raise ValueError("Output must be outside source histories")
    loaded = {str(path): build_snapshot(path) for path in [source]+targets}
    source_rows, _, source_report, contract = loaded[str(source)]
    if any(item[2]["evaluator_contract_sha256"] != source_report["evaluator_contract_sha256"]
           for item in loaded.values()):
        raise ValueError("Cross-run evaluator contract differs; no model fitted")
    if any(not item[2]["evaluator_hashes_present"] for item in loaded.values()):
        raise ValueError("Saved evaluator source hashes required")
    destination = output_root / ("transfer_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                 + "_" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    module_root = Path(__file__).parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in module_root.glob("*.py")}
    report = {
        "status": "running", "mode": "frozen_cross_policy_diagnostic_no_MDA",
        "source_run": str(source), "target_runs": [str(p) for p in targets],
        "settings": settings, "evaluator_contract_sha256": source_report["evaluator_contract_sha256"],
        "source_sha256": {name: item[2]["source_sha256"] for name, item in loaded.items()},
        "base_project": str(base_project), "ra_source_sha256": hashes,
        "python": sys.version, "python_executable": sys.executable,
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "scipy", "torch", "scikit-learn", "botorch", "gpytorch")},
        "scope": [
            "One frozen model; target labels never train the VAE, classifier or calibrators.",
            "Canonical design overlap with training/calibration is excluded, including inactive coordinates.",
            "Repeated designs within each test block are scored once.",
            "Across-cohort test overlap is reported; cohorts are not independent replicates or pooled.",
            "Exact design deduplication does not remove near-neighbor dependence.",
            "Same seed and shared initial designs preclude claims of independent randomized repetitions.",
            "Current LSBO math implementation is hash-recorded; this is not reproduction of paper results.",
            "No test-based threshold selection, no physical evaluator changes and no MDA.",
        ],
    }
    def save_report():
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save_report()
    try:
        with (destination / "test_forecasts.jsonl").open("x", encoding="utf-8") as stream:
            def save_forecast(forecast):
                stream.write(json.dumps(forecast, allow_nan=False)+"\n")
                stream.flush()
            predictions, metadata, samples = transfer_calibration(
                source_rows, {str(path): loaded[str(path)][0] for path in targets},
                contract, base_project, on_prediction=save_forecast, **settings)
        report.update(metadata)
        report["scores"] = {
            name: {"heads": calibration_scores(values), "gate": gate_diagnostics(values),
                   "states": dict(Counter(p["observed"]["state"] for p in values))}
            for name, values in predictions.items()
        }
        for directory, expected in (
            *[(Path(path), values) for path, values in report["source_sha256"].items()],
            (base_project / "lsbo", report["base_source_sha256"]), (module_root, hashes),
        ):
            if any(hashlib.sha256((directory / name).read_bytes()).hexdigest() != sha
                   for name, sha in expected.items()):
                raise RuntimeError("Source changed during transfer validation")
        for name, rows in (("calibration_samples.jsonl", samples),
                           ("scored_test_predictions.jsonl",
                            [p for values in predictions.values() for p in values])):
            (destination / name).write_text(
                "".join(json.dumps(row, allow_nan=False)+"\n" for row in rows), encoding="utf-8")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_report()
        print("Transfer report:", destination / "report.json", flush=True)
    return destination, report
