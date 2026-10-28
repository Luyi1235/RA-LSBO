"""Exploratory diagnosis of a fixed constraint-aware risk score; no MDA."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import torch

from ra_lsbo.audit import build_snapshot, load_json
from ra_lsbo.constraint_risk import fit_frozen_models
from ra_lsbo.transfer_validation import candidate_key, unseen_rows


def score(rows):
    known = [row for row in rows if row["feasible"] is not None]
    result = {}
    for name in ("old", "constraint_aware"):
        probabilities = [row[name] for row in known]
        if any(value is None for value in probabilities):
            result[name] = {"status": "calibration_unavailable"}
            continue
        accepted = [row for row in known if row[name] >= .8]
        result[name] = {
            "samples": len(known),
            "brier": sum((row[name]-int(row["feasible"]))**2 for row in known)/len(known) if known else None,
            "predicted_nonfeasible_sum": sum(1-value for value in probabilities),
            "accepted_at_0_8": len(accepted),
            "infeasible_among_accepted": sum(not row["feasible"] for row in accepted),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = load_json((args.pilot / "plan.json").read_bytes())
    history, features, _, _, old_cal, combined, new_cal = fit_frozen_models(plan)
    transfer = load_json(args.transfer.read_bytes())
    cohorts = {"source_holdout": history[240:]}
    seen = {candidate_key(features, row["candidate"]) for row in history[:240]}
    for name in transfer["cohorts"]:
        if name == "source_holdout":
            continue
        rows, _, audit, _ = build_snapshot(name)
        if audit["evaluator_contract_sha256"] != plan["historical_contract_sha256"]:
            raise ValueError("External physical contract differs")
        cohorts[name], _ = unseen_rows(rows, features, seen)
    pilot_results = load_json((args.pilot / "result.json").read_bytes())
    cohorts["previous_pilot_posthoc"] = [
        {"evaluation": index, "candidate": item["proposal"]["candidate"], "risk": item["risk"]}
        for index, item in enumerate(pilot_results["results"], 1)]
    output = args.pilot.parent / ("constraint_risk_diagnostic_" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    report = {
        "status": "exploratory_completed", "independent_validation": False,
        "training_indices": features.training_indices, "calibration_indices": new_cal.calibration_indices,
        "new_model_id": combined.model_id, "new_calibration_id": new_cal.calibration_id,
        "new_calibration": new_cal.diagnostics["final_feasibility"], "cohorts": {},
        "scope": "Fixed new rule motivated by previous pilot; all datasets already inspected; no new MDA.",
    }
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for name, rows in cohorts.items():
            candidates = [row["candidate"] for row in rows]
            old = old_cal.predict(candidates)["risk_probabilities_calibrated"]["final_feasibility"]
            new = new_cal.predict(candidates)["risk_probabilities_calibrated"]["final_feasibility"]
            records = [{
                "cohort": name, "evaluation": row["evaluation"], "feasible": row["risk"]["feasible_label"],
                "old": old[index] if old is not None else None,
                "constraint_aware": new[index] if new is not None else None,
            } for index, row in enumerate(rows)]
            for record in records:
                stream.write(json.dumps(record, allow_nan=False)+"\n")
            report["cohorts"][name] = score(records)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["cohorts"], indent=2))
    print("Diagnostic:", output / "report.json")


if __name__ == "__main__":
    main()
