"""Rebuild encoder risk forecasts and scores independently from saved raw sources."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import build_snapshot, load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.calibration import SigmoidCalibrator
from ra_lsbo.features import LatentFeatures
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.risk_model import BinaryRiskHead, RiskHeads
from ra_lsbo.structured_encoder import StructuredFeatures
from tools.risk_feature_ablation import feature_matrix


def audit(folder):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    report = load_json((folder/"report.json").read_bytes())
    verification = load_json((folder/"verification.json").read_bytes())
    verify_sources(plan["source_sha256"])
    verify_sources(verification["input_sha256"])
    for sha in set(plan["source_sha256"].values()):
        if hashlib.sha256((folder/"source_snapshot"/(sha+".source")).read_bytes()).hexdigest() != sha:
            raise ValueError("Source snapshot mismatch")
    parent = load_json((Path(plan["boundary_source"])/"plan.json").read_bytes())
    result = load_json((Path(plan["boundary_source"])/"result.json").read_bytes())
    history, _, _, contract = build_snapshot(parent["historical_source"])
    target = [dict(evaluation=r["index"], candidate=r["proposal"]["candidate"], risk=r["risk"])
              for r in result["results"]]
    data = {"training": history[:180], "global_calibration": history[180:240],
            "historical_test": history[240:], "local_calibration": target[:60],
            "boundary_test": target[60:]}
    forecasts = [load_json(line) for line in (folder/"forecasts.jsonl").read_bytes().splitlines()]
    scored = [load_json(line) for line in (folder/"scored_predictions.jsonl").read_bytes().splitlines()]
    if (len(forecasts) != 2400 or len(scored) != len(forecasts)
            or len(report["models"]) != 20 or report["distinct_models"] != 20
            or len({m["model_id"] for m in report["models"]}) != 20):
        raise ValueError("Unexpected experiment/model/prediction counts")
    cached, matched, checkpoints = {}, 0, 0
    for model in report["models"]:
        key = model["variant"], model["seed"]
        if model["shared_baseline"]:
            key = "legacy", 0
        if key not in cached:
            if key[0] == "legacy":
                features = LatentFeatures(plan["base_project"], contract["design_variables"], key[1]).fit(
                    data["training"], epochs=60)
                if key[1] == 0 and RiskHeads(data["training"], features, seed=0).model_id != parent["global_rf_model_id"]:
                    raise ValueError("Legacy model no longer reproduces")
            else:
                representation = next(r for r in report["representations"]
                                      if (r["variant"], r["seed"]) == key)
                features = StructuredFeatures.load_checkpoint(
                    folder/representation["checkpoint"], plan["base_project"], data["training"])
                if representation["snapshot_id"] != features.snapshot_id:
                    raise ValueError("Checkpoint snapshot differs from report")
                checkpoints += 1
            cached[key] = features
        features = cached[key]
        matrices = {
            name: (features.transform_view([r["candidate"] for r in rows], model["feature_view"]).numpy()
                   if isinstance(features, StructuredFeatures) else
                   feature_matrix(features, [r["candidate"] for r in rows], model["feature_view"]))
            for name, rows in data.items()}
        head = BinaryRiskHead(matrices["training"],
                              [int(r["risk"]["feasible_label"]) for r in data["training"]], seed=0)
        raw = {name: head.predict(value).tolist() for name, value in matrices.items()}
        calibrators = {
            name: SigmoidCalibrator(raw[name], [int(r["risk"]["feasible_label"]) for r in data[name]],
                                    base_fitted=head.diagnostics["status"] == "fitted_raw")
            for name in ("global_calibration", "local_calibration")}
        for name, calibrator in calibrators.items():
            if calibrator.diagnostics != model["calibration"][name]:
                raise ValueError("Calibration was not reconstructed from its own source block")
        for cohort in ("historical_test", "boundary_test"):
            subset = [r for r in forecasts if r["model_id"] == model["model_id"] and r["cohort"] == cohort]
            actual = [r for r in scored if r["model_id"] == model["model_id"] and r["cohort"] == cohort]
            if len(subset) != 60 or len(actual) != 60:
                raise ValueError("Missing or extra model/cohort records")
            predicted = {
                "raw": raw[cohort], "global": calibrators["global_calibration"].predict(raw[cohort]),
                "local": calibrators["local_calibration"].predict(raw[cohort])
                if cohort == "boundary_test" else None}
            for i, row in enumerate(data[cohort]):
                before, after = subset[i], actual[i]
                if (before != {k: v for k, v in after.items() if k != "label"}
                        or after["label"] != row["risk"]["feasible_label"]
                        or before["evaluation"] != row["evaluation"]
                        or before["candidate_key"] != features.space.key(features.candidate(row["candidate"]))):
                    raise ValueError("Candidate, source label or forecast/scored pairing differs")
                for field, values in predicted.items():
                    expected = None if values is None else values[i]
                    if expected is None:
                        if before[field] is not None:
                            raise ValueError("Unavailable calibration was filled")
                    elif not np.isclose(expected, before[field], rtol=1e-12, atol=1e-12):
                        raise ValueError("Forecast cannot be reproduced")
                matched += 1
            for field in predicted:
                records = [r for r in actual if r[field] is not None]
                known = [r for r in records if r["label"] is not None]
                accepted = [r for r in records if r[field] >= .8]
                expected = dict(binary_metrics([r[field] for r in known], [int(r["label"]) for r in known]),
                                predictions=len(records), accepted_at_0_8=len(accepted),
                                known_accepted=sum(r["label"] is not None for r in accepted),
                                unknown_accepted=sum(r["label"] is None for r in accepted),
                                infeasible_among_accepted=sum(r["label"] is False for r in accepted))
                if model["metrics"][cohort][field] != expected:
                    raise ValueError("Metrics do not match raw-source labels")
    paths = list(plan["source_sha256"])
    paths.extend(str(folder/name) for name in (
        "plan.json", "report.json", "verification.json", "forecasts.jsonl", "scored_predictions.jsonl",
        "encoder_diagnostics.png"))
    paths.extend(str(path) for path in sorted(folder.glob("*.pt")))
    return {"status": "passed", "models_rebuilt": len(report["models"]),
            "raw_source_predictions_reproduced": matched, "checkpoints_loaded": checkpoints,
            "source_files": len(plan["source_sha256"]),
            "input_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths},
            "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    report = audit(args.folder.resolve())
    with (args.folder/"independent_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in report.items() if key != "input_sha256"}, indent=2))
