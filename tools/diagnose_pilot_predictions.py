"""Reconstruct frozen pre-pilot predictions without fitting on pilot outcomes."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import build_snapshot, load_json
from ra_lsbo.calibration import CalibratedRiskHeads
from ra_lsbo.features import LatentFeatures
from ra_lsbo.responses import ResponseHeads
from ra_lsbo.risk_model import RiskHeads
from ra_lsbo.transfer_validation import candidate_key, unseen_rows


def diagnose(folder):
    plan = load_json((folder / "plan.json").read_bytes())
    rows, _, audit, contract = build_snapshot(plan["historical_source"])
    if audit["source_sha256"] != plan["historical_source_sha256"]:
        raise ValueError("Historical data changed")
    features = LatentFeatures(plan["base_project"], contract["design_variables"], seed=0).fit(rows[:180], epochs=60)
    known = {candidate_key(features, row["candidate"]) for row in rows[:180]}
    calibration, _ = unseen_rows(rows[180:240], features, known)
    risks = RiskHeads(rows[:180], features, seed=0)
    calibrated = CalibratedRiskHeads(risks, calibration)
    if (features.snapshot_id != plan["representation_snapshot"] or risks.model_id != plan["model_id"]
            or calibrated.calibration_id != plan["calibration_id"]):
        raise ValueError("Reconstructed model identity differs")
    responses = ResponseHeads(rows[:180], features, seed=0, steps=45)
    candidates = [item["proposal"]["candidate"] for item in plan["proposals"]]
    response = responses.predict(candidates)
    probabilities = calibrated.predict(candidates)["risk_probabilities_calibrated"]["final_feasibility"]
    records = []
    for index, (item, probability) in enumerate(zip(plan["proposals"], probabilities), 1):
        if abs(probability-item["proposal"]["evidence"]["probability"]) > 1e-12:
            raise ValueError("Frozen feasibility forecast differs")
        records.append({
            "index": index, "mode": item["proposal"]["mode"], "probability": probability,
            "responses": {name: None if values is None else
                          {key: numbers[index-1] for key, numbers in values.items()}
                          for name, values in response.items()},
        })
    return {
        "schema": "posthoc_reconstruction_from_frozen_history_v1",
        "model_identity_matches": True, "pilot_outcomes_used_for_fit": False,
        "training_indices": features.training_indices,
        "calibration_indices": calibrated.calibration_indices,
        "records": records,
        "interpretation": "Diagnostic reconstruction after the pilot, not a prospectively validated new policy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = diagnose(args.folder.resolve())
    path = args.folder / "reconstructed_predictions.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    for row in report["records"]:
        print(row["index"], row["mode"], round(row["probability"], 4),
              row["responses"].get("constraint:stall_speed"))
    print("Diagnostic:", path)
