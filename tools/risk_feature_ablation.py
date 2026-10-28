"""Compare fixed risk feature pipelines on existing histories; never invokes MDA."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import build_snapshot, load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.calibration import SigmoidCalibrator
from ra_lsbo.features import LatentFeatures
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.risk_model import BinaryRiskHead, RiskHeads
from ra_lsbo.transfer_validation import candidate_key, unseen_rows


MODES = ("latent", "hierarchical_onehot", "plain_onehot", "continuous_mask")
SEEDS = tuple(range(5))


def feature_matrix(features, candidates, mode):
    if mode not in MODES:
        raise ValueError("Unknown feature pipeline")
    if mode == "latent":
        return features.transform(candidates).numpy()
    space = features.space
    count = len(space.continuous)
    width = (count if mode == "plain_onehot" else 2*count)
    if mode != "continuous_mask":
        width += sum(len(v.values) for v in space.discrete)
        if mode == "hierarchical_onehot":
            width += len(space.discrete)
    rows = []
    for raw in candidates:
        candidate = features.candidate(raw)
        units, masks = space.unit(candidate), space.mask(candidate.discrete)
        parts = [units] if mode == "plain_onehot" else [units*masks, masks]
        if mode != "continuous_mask":
            for variable in space.discrete:
                active = float(space.active(variable, candidate.discrete))
                vector = np.zeros(len(variable.values))
                vector[variable.values.index(candidate.discrete[variable.name])] = 1.
                if mode == "hierarchical_onehot":
                    parts.extend([vector*active, np.asarray([active])])
                else:
                    parts.append(vector)
        rows.append(np.concatenate(parts))
    return np.asarray(rows).reshape(len(rows), width)


def score_records(records):
    output = {}
    for field in ("raw", "calibrated"):
        predicted = [row for row in records if row[field] is not None]
        known = [row for row in predicted if row["label"] is not None]
        accepted = [row for row in predicted if row[field] >= .8]
        output[field] = {
            **binary_metrics([row[field] for row in known], [int(row["label"]) for row in known]),
            "predictions": len(predicted), "accepted_at_0_8": len(accepted),
            "known_accepted": sum(row["label"] is not None for row in accepted),
            "unknown_accepted": sum(row["label"] is None for row in accepted),
            "infeasible_among_accepted": sum(row["label"] is False for row in accepted),
        }
    return output


def run(pilot, transfer, output_root):
    plan_path, transfer_path = pilot / "plan.json", transfer
    plan, transfer_report = load_json(plan_path.read_bytes()), load_json(transfer_path.read_bytes())
    history, _, audit, contract = build_snapshot(plan["historical_source"])
    if audit["source_sha256"] != plan["historical_source_sha256"]:
        raise ValueError("Historical identity differs from reviewed split")
    features = LatentFeatures(plan["base_project"], contract["design_variables"], seed=0).fit(
        history[:180], epochs=60)
    reference = RiskHeads(history[:180], features, seed=0)
    if reference.model_id != plan["model_id"]:
        raise ValueError("Latent reference model identity differs")
    training = [row for row in history[:180] if row["risk"]["feasible_label"] is not None]
    training_keys = {candidate_key(features, row["candidate"]) for row in history[:180]}
    calibration, calibration_excluded = unseen_rows(history[180:240], features, training_keys)
    known_keys = {candidate_key(features, row["candidate"]) for row in history[:240]}
    available = {"source_holdout": history[240:]}
    sources = {str(Path(plan["historical_source"]) / filename): sha
               for filename, sha in audit["source_sha256"].items()}
    for name in transfer_report["cohorts"]:
        if name == "source_holdout":
            continue
        rows, _, target_audit, _ = build_snapshot(name)
        if target_audit["evaluator_contract_sha256"] != audit["evaluator_contract_sha256"]:
            raise ValueError("Cross-contract data excluded")
        available[name] = rows
        sources.update({str(Path(name) / filename): sha
                        for filename, sha in target_audit["source_sha256"].items()})
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "docs" / "FEATURE_ABLATION_PROTOCOL_20261020.md",
             plan_path, transfer_path, *list((root / "ra_lsbo").glob("*.py"))]
    sources.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    sources.update({str(Path(plan["base_project"]) / "lsbo" / filename): sha
                    for filename, sha in features.base_hashes.items()})
    verify_sources(sources)
    cohorts, metadata, all_keys = {}, {}, set()
    for name, rows in available.items():
        selected, removed = unseen_rows(rows, features, known_keys)
        keys = {candidate_key(features, row["candidate"]) for row in selected}
        cohorts[name] = selected
        metadata[name] = {
            "available": len(rows), "retained": len(selected), "excluded": removed,
            "overlap_with_previous_test_cohorts": len(keys & all_keys),
            "label_counts": dict(Counter(str(row["risk"]["feasible_label"]) for row in selected))}
        all_keys.update(keys)
    folder = output_root / ("feature_ablation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                            + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "mode": "exploratory_historical_no_MDA", "new_mda_calls": 0,
        "training_indices": features.training_indices,
        "calibration_indices": [row["evaluation"] for row in calibration],
        "calibration_excluded": calibration_excluded, "representation_snapshot": features.snapshot_id,
        "seeds": SEEDS, "feature_modes": MODES, "cohorts": metadata, "models": {},
        "source_sha256": sources, "unique_test_designs": len(all_keys),
        "scope": [
            "Previously inspected histories; exploratory feature comparison, not new confirmation.",
            "VAE uses only first 180 training outcomes, never calibration or test labels.",
            "Five RF seeds share one dataset and one frozen VAE; not independent aircraft runs.",
            "Input dimension changes RF feature subsampling; pipelines are compared as implemented.",
            "No probabilities are promoted to a safety bound or used for live dispatch.",
            "Current 120-point campaign is not read by this tool.",
        ],
    }
    def save_report():
        (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save_report()
    try:
        with (folder / "forecasts.jsonl").open("x", encoding="utf-8") as predictions, (
                folder / "scored_predictions.jsonl").open("x", encoding="utf-8") as scored:
            for mode in MODES:
                train_x = feature_matrix(features, [row["candidate"] for row in training], mode)
                cal_rows = [row for row in calibration if row["risk"]["feasible_label"] is not None]
                cal_x = feature_matrix(features, [row["candidate"] for row in cal_rows], mode)
                for seed in SEEDS:
                    model_name = mode + "/seed_" + str(seed)
                    head = BinaryRiskHead(train_x, [int(row["risk"]["feasible_label"]) for row in training], seed)
                    if mode == "latent" and seed == 0:
                        np.testing.assert_array_equal(
                            head.predict(cal_x),
                            reference.predict([row["candidate"] for row in cal_rows])["final_feasibility"])
                    calibrator = SigmoidCalibrator(
                        head.predict(cal_x).tolist(), [int(row["risk"]["feasible_label"]) for row in cal_rows],
                        base_fitted=head.diagnostics["status"] == "fitted_raw")
                    model_report = {
                        "dimensions": train_x.shape[1], "raw_diagnostics": head.diagnostics,
                        "calibration": calibrator.diagnostics, "cohorts": {}}
                    for name, rows in cohorts.items():
                        matrix = feature_matrix(features, [row["candidate"] for row in rows], mode)
                        raw, calibrated = head.predict(matrix), calibrator.predict(head.predict(matrix).tolist())
                        records = []
                        for position, row in enumerate(rows):
                            forecast = {
                                "model": model_name, "cohort": name, "evaluation": row["evaluation"],
                                "candidate_key": candidate_key(features, row["candidate"]),
                                "raw": float(raw[position]),
                                "calibrated": None if calibrated is None else float(calibrated[position])}
                            predictions.write(json.dumps(forecast, allow_nan=False)+"\n")
                            predictions.flush()
                            forecast["label"] = row["risk"]["feasible_label"]
                            records.append(forecast)
                            scored.write(json.dumps(forecast, allow_nan=False)+"\n")
                        model_report["cohorts"][name] = score_records(records)
                    report["models"][model_name] = model_report
                    print("[feature ablation]", model_name, "finished", flush=True)
        verify_sources(sources)
        snapshot = folder / "source_snapshot"
        snapshot.mkdir()
        for filename, sha in sources.items():
            destination = snapshot / (sha+".source")
            if not destination.exists():
                destination.write_bytes(Path(filename).read_bytes())
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=type(error).__name__+": "+str(error))
        raise
    finally:
        save_report()
        print("Feature ablation report:", folder / "report.json", flush=True)
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.pilot.resolve(), args.transfer.resolve(), args.output.resolve())
