"""Prespecified secondary feature validation on the existing boundary campaign."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import build_snapshot, load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import digest
from ra_lsbo.calibration import SigmoidCalibrator
from ra_lsbo.features import LatentFeatures
from ra_lsbo.risk_model import BinaryRiskHead, RiskHeads
from ra_lsbo.transfer_validation import candidate_key, unseen_rows
from tools.analyze_boundary_validation import paired_difference
from tools.risk_feature_ablation import MODES, SEEDS, feature_matrix, score_records
from tools.summarize_risk_feature_ablation import aggregate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def save_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        stream.flush()


def read_rows(path):
    return [load_json(line) for line in Path(path).read_bytes().splitlines()]


def label_blind_result(campaign):
    for attempt in range(3):
        content = (campaign / "result.json").read_bytes()
        try:
            report = load_json(content)
            break
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.05)
    if any(item["index"] > 60 for item in report["results"]):
        raise ValueError("Held-out outcomes already available; do not create forecasts post hoc")
    return content, report


def prepare(campaign, pilot, ablation):
    before_bytes, before = label_blind_result(campaign)
    if before["status"] != "running":
        raise ValueError("Register during the active calibration phase")
    parent_bytes = (campaign / "plan.json").read_bytes()
    parent, old = load_json(parent_bytes), load_json((pilot / "plan.json").read_bytes())
    existing = load_json(ablation.read_bytes())
    if existing["status"] != "complete":
        raise ValueError("Require the fixed historical feature diagnostic")
    request = load_json((campaign / "request.json").read_bytes())
    if request["operator_approved"] is not True or sha(campaign / "plan.json") != request["plan_sha256"]:
        raise ValueError("Parent plan lacks matching approval")
    sources = dict(parent["source_sha256"])
    sources.update(existing["source_sha256"])
    root = Path(__file__).resolve().parents[1]
    paths = [campaign / "plan.json", campaign / "request.json", ablation, Path(__file__),
             root / "docs" / "FEATURE_BOUNDARY_PROTOCOL_20261020.md",
             root / "tools" / "analyze_boundary_validation.py",
             root / "tools" / "summarize_risk_feature_ablation.py"]
    sources.update({str(path.resolve()): sha(path) for path in paths})
    verify_sources(sources)
    history, _, audit, contract = build_snapshot(old["historical_source"])
    if audit["source_sha256"] != old["historical_source_sha256"]:
        raise ValueError("Original historical split changed")
    features = LatentFeatures(old["base_project"], contract["design_variables"], seed=0).fit(
        history[:180], epochs=60)
    reference = RiskHeads(history[:180], features, seed=0)
    if reference.model_id != old["model_id"] or features.snapshot_id != existing["representation_snapshot"]:
        raise ValueError("Original frozen representation/model differs")
    calibration, removed = unseen_rows(
        history[180:240], features, {candidate_key(features, row["candidate"]) for row in history[:180]})
    if removed:
        raise ValueError("Expected the original unchanged 60-point calibration split")
    training = [row for row in history[:180] if row["risk"]["feasible_label"] is not None]
    cal = [row for row in calibration if row["risk"]["feasible_label"] is not None]
    candidates = [item["candidate"] for item in parent["proposals"]]
    models, rows = {}, []
    for mode in MODES:
        train_x = feature_matrix(features, [row["candidate"] for row in training], mode)
        cal_x = feature_matrix(features, [row["candidate"] for row in cal], mode)
        target_x = feature_matrix(features, candidates, mode)
        for seed in SEEDS:
            name = mode+"/seed_"+str(seed)
            head = BinaryRiskHead(train_x, [int(row["risk"]["feasible_label"]) for row in training], seed)
            calibrated = SigmoidCalibrator(
                head.predict(cal_x).tolist(), [int(row["risk"]["feasible_label"]) for row in cal],
                base_fitted=head.diagnostics["status"] == "fitted_raw")
            if calibrated.diagnostics != existing["models"][name]["calibration"]:
                raise ValueError("Historical feature calibration failed reproduction")
            raw = head.predict(target_x).tolist()
            global_p = calibrated.predict(raw)
            if mode == "latent" and seed == 0:
                np.testing.assert_allclose(raw, [item["raw_rf"] for item in parent["proposals"]], rtol=0, atol=1e-12)
                np.testing.assert_allclose(global_p, [item["global_rf"] for item in parent["proposals"]],
                                           rtol=0, atol=1e-12)
            models[name] = {
                "dimensions": train_x.shape[1], "raw_diagnostics": head.diagnostics,
                "global_calibration": calibrated.diagnostics}
            for position, item in enumerate(parent["proposals"]):
                rows.append({
                    "model": name, "index": item["index"], "band": item["band"],
                    "candidate_key": digest(item["candidate"]), "raw": raw[position],
                    "global": None if global_p is None else float(global_p[position])})
    after_bytes, after = label_blind_result(campaign)
    folder = campaign / "feature_validation"
    folder.mkdir(exist_ok=False)
    save_rows(folder / "raw_forecasts.jsonl", rows)
    (folder / "registration_before.json").write_bytes(before_bytes)
    (folder / "registration_after.json").write_bytes(after_bytes)
    save(folder / "plan.json", {
        "schema": "ra_boundary_feature_validation_v1", "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": str(campaign), "parent_plan_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "historical_training_indices": list(range(1, 181)),
        "historical_calibration_indices": [row["evaluation"] for row in calibration],
        "new_calibration_indices": list(range(1, 61)), "new_test_indices": list(range(61, 121)),
        "models": models, "raw_forecasts_sha256": sha(folder / "raw_forecasts.jsonl"),
        "source_sha256": sources,
        "registration_result_counts": [len(before["results"]), len(after["results"])],
        "registration_witness_sha256": {name: sha(folder / name)
                                       for name in ("registration_before.json", "registration_after.json")},
        "scope": "Secondary feature comparison; zero extra MDA; no parent experiment changes."})
    verify_sources(sources)
    snapshot = folder / "source_snapshot"
    snapshot.mkdir()
    for filename, expected in sources.items():
        destination = snapshot / (expected+".source")
        if not destination.exists():
            destination.write_bytes(Path(filename).read_bytes())
    return folder


def local_fit(plan, raw_rows, calibration_results):
    if [item["index"] for item in calibration_results] != list(range(1, 61)):
        raise ValueError("Require the unchanged first 60 calibration outcomes")
    if any(item["risk"] is None for item in calibration_results):
        raise ValueError("Calibration has unsubmitted points")
    labels = {item["index"]: item["risk"]["feasible_label"] for item in calibration_results}
    diagnostics, forecasts = {}, []
    for name, model in plan["models"].items():
        rows = [row for row in raw_rows if row["model"] == name]
        if [row["index"] for row in rows] != list(range(1, 121)):
            raise ValueError("Missing or reordered frozen raw forecasts")
        valid = [row for row in rows[:60] if labels[row["index"]] is not None]
        fitted = SigmoidCalibrator(
            [row["raw"] for row in valid], [int(labels[row["index"]]) for row in valid],
            base_fitted=model["raw_diagnostics"]["status"] == "fitted_raw")
        diagnostics[name] = fitted.diagnostics
        local = fitted.predict([row["raw"] for row in rows[60:]])
        forecasts.extend(dict(row, local=None if local is None else float(local[position]))
                         for position, row in enumerate(rows[60:]))
    return diagnostics, forecasts


def calibrate(folder):
    plan = load_json((folder / "plan.json").read_bytes())
    campaign = Path(plan["campaign"])
    before_bytes, report = label_blind_result(campaign)
    if len(report["results"]) != 60:
        raise ValueError("Wait for all 60 calibration outcomes")
    verify_sources(plan["source_sha256"])
    if sha(folder / "raw_forecasts.jsonl") != plan["raw_forecasts_sha256"]:
        raise ValueError("Raw feature forecasts changed")
    diagnostics, forecasts = local_fit(plan, read_rows(folder / "raw_forecasts.jsonl"), report["results"])
    label_blind_result(campaign)
    save_rows(folder / "test_forecasts.jsonl", forecasts)
    after_bytes, after = label_blind_result(campaign)
    (folder / "calibration_before.json").write_bytes(before_bytes)
    (folder / "calibration_after.json").write_bytes(after_bytes)
    save(folder / "local_calibration.json", {
        "status": "saved_before_heldout_outcomes", "diagnostics": diagnostics,
        "calibration_indices": list(range(1, 61)), "test_forecasts_sha256": sha(folder / "test_forecasts.jsonl"),
        "result_counts_before_after": [len(report["results"]), len(after["results"])],
        "witness_sha256": {name: sha(folder / name)
                          for name in ("calibration_before.json", "calibration_after.json")}})


def watch(folder):
    plan = load_json((folder / "plan.json").read_bytes())
    while True:
        try:
            _, result = label_blind_result(Path(plan["campaign"]))
        except json.JSONDecodeError:
            time.sleep(15)
            continue
        except ValueError as error:
            save(folder / "local_calibration.json", {
                "status": "missed_heldout_forecast_window", "reason": str(error), "new_mda_calls": 0})
            print("Secondary forecast window missed; no retrospective forecasts created.", flush=True)
            return
        if len(result["results"]) == 60:
            if any(item["risk"] is None for item in result["results"]):
                save(folder / "local_calibration.json", {
                    "status": "parent_calibration_incomplete", "new_mda_calls": 0})
                return
            try:
                calibrate(folder)
            except ValueError as error:
                if not str(error).startswith("Held-out outcomes already available"):
                    raise
                save(folder / "local_calibration.json", {
                    "status": "missed_heldout_forecast_window", "reason": str(error), "new_mda_calls": 0})
                print("Held-out outcome arrived before sealing; secondary forecasts excluded.", flush=True)
                return
            print("Secondary forecasts saved before any held-out outcome.", flush=True)
            return
        if result["status"] != "running":
            save(folder / "local_calibration.json", {
                "status": "parent_stopped_before_calibration", "parent_status": result["status"],
                "calibration_outcomes": len(result["results"]), "new_mda_calls": 0})
            print("Parent stopped before the calibration block completed.", flush=True)
            return
        time.sleep(15)


def analyze(folder):
    plan = load_json((folder / "plan.json").read_bytes())
    campaign = Path(plan["campaign"])
    parent, audit = [load_json((campaign / name).read_bytes()) for name in ("result.json", "audit.json")]
    if parent["status"] == "running" or audit["status"] != "passed":
        raise ValueError("Wait for the terminal parent audit")
    verify_sources({**plan["source_sha256"], **audit["input_sha256"]})
    for expected in set(plan["source_sha256"].values()):
        if sha(folder / "source_snapshot" / (expected+".source")) != expected:
            raise ValueError("Secondary source snapshot changed")
    if sha(campaign / "plan.json") != plan["parent_plan_sha256"]:
        raise ValueError("Parent plan changed")
    for name, expected in plan["registration_witness_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError("Registration witness changed")
        witness = load_json((folder / name).read_bytes())
        if (any(item["index"] > 60 for item in witness["results"])
                or witness["results"] != parent["results"][:len(witness["results"])]):
            raise ValueError("Registration used held-out outcomes")
    if sha(folder / "raw_forecasts.jsonl") != plan["raw_forecasts_sha256"]:
        raise ValueError("Raw forecasts changed")
    local = load_json((folder / "local_calibration.json").read_bytes())
    if local["status"] != "saved_before_heldout_outcomes":
        save(folder / "analysis.json", {
            "status": "no_heldout_validation", "reason": local["status"], "new_mda_calls": 0})
        return
    for name, expected in local["witness_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError("Calibration witness changed")
        witness = load_json((folder / name).read_bytes())
        if [item["index"] for item in witness["results"]] != list(range(1, 61)):
            raise ValueError("Forecasts were not sealed before held-out outcomes")
        if witness["results"] != parent["results"][:60]:
            raise ValueError("Calibration witness differs from actual parent outcomes")
    diagnostics, reconstructed = local_fit(plan, read_rows(folder / "raw_forecasts.jsonl"), parent["results"][:60])
    forecasts = read_rows(folder / "test_forecasts.jsonl")
    if (sha(folder / "test_forecasts.jsonl") != local["test_forecasts_sha256"]
            or forecasts != reconstructed or diagnostics != local["diagnostics"]):
        raise ValueError("Local feature predictions differ from calibration-only reconstruction")
    outcomes = {item["index"]: item["risk"]["feasible_label"] for item in parent["results"]
                if item["index"] > 60 and item["risk"] is not None}
    scored, metrics = [], {}
    for name in plan["models"]:
        rows = [dict(row, label=outcomes[row["index"]]) for row in forecasts
                if row["model"] == name and row["index"] in outcomes]
        scored.extend(rows)
        metrics[name] = {}
        for band in ("all", "outer", "near", "inner"):
            group = [row for row in rows if band == "all" or row["band"] == band]
            metrics[name][band] = {
                field: score_records([dict(row, calibrated=row[field]) for row in group])["calibrated"]
                for field in ("raw", "global", "local")}
    summary = []
    for mode in MODES:
        for band in ("all", "outer", "near", "inner"):
            summary.append({
                "mode": mode, "band": band,
                **{field: aggregate([metrics[mode+"/seed_"+str(seed)][band][field]["brier"]
                                     for seed in SEEDS]) for field in ("raw", "global", "local")}})
    pairs = []
    by_model = {name: {row["index"]: row for row in scored if row["model"] == name} for name in plan["models"]}
    for seed in SEEDS:
        reference = by_model["latent/seed_"+str(seed)]
        for mode in MODES[1:]:
            comparison = by_model[mode+"/seed_"+str(seed)]
            rows = [{"band": row["band"], "feasible": row["label"], "latent": row["local"],
                     mode: comparison[index]["local"]} for index, row in reference.items()]
            pairs.append(dict(paired_difference(rows, "latent", mode), rf_seed=seed))
    save_rows(folder / "scored_test_predictions.jsonl", scored)
    save(folder / "analysis.json", {
        "status": "completed_secondary_validation", "new_mda_calls": 0, "metrics": metrics,
        "test_calls": len(outcomes), "known_test_labels": sum(value is not None for value in outcomes.values()),
        "unknown_test_labels": sum(value is None for value in outcomes.values()),
        "unexecuted_test": 60-len(outcomes), "summary": summary, "paired_local_brier": pairs,
        "local_diagnostics": diagnostics,
        "source_sha256": {str(path): sha(path) for path in
                         (folder / "plan.json", folder / "raw_forecasts.jsonl", folder / "local_calibration.json",
                          folder / "test_forecasts.jsonl", campaign / "result.json", campaign / "audit.json",
                          Path(__file__))},
        "limitations": [
            "Secondary comparisons; multiple exploratory intervals, with no multiplicity-adjusted claim.",
            "Five RF seeds share the same held-out labels and one frozen VAE.",
            "Candidate distribution was selected by the original latent/constraint model.",
            "No optimizer superiority or global safety guarantee follows from these scores."]})
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    bands = ("all", "outer", "near", "inner")
    for axis, field in zip(axes, ("global", "local")):
        for offset, (mode, color) in enumerate(zip(MODES, ("#555555", "#008775", "#467cac", "#b66b19"))):
            values = [next(row[field] for row in summary if row["mode"] == mode and row["band"] == band)
                      for band in bands]
            means = [np.nan if row["mean"] is None else row["mean"] for row in values]
            spread = [[np.nan if row["mean"] is None else row["mean"]-row["minimum"] for row in values],
                      [np.nan if row["mean"] is None else row["maximum"]-row["mean"] for row in values]]
            axis.bar(np.arange(4)+(offset-1.5)*.18, means, width=.17, yerr=spread, capsize=2,
                     color=color, label=mode)
        axis.set(xticks=np.arange(4), xticklabels=bands, ylabel="Brier loss",
                 title=field.capitalize()+" calibration; mean and range over RF seeds 0-4")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(ncols=4, fontsize=8)
    figure.suptitle(f"Secondary risk feature validation: {len(outcomes)}/60 held-out calls\n"
                   "Same frozen candidates; no additional MDA; RF ranges are not confidence intervals", fontsize=12)
    figure.savefig(folder / "feature_validation.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "watch", "analyze"))
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--ablation", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    campaign = args.campaign.resolve()
    if args.mode == "prepare":
        folder = prepare(campaign, args.pilot.resolve(), args.ablation.resolve())
        print(json.dumps({"registered": str(folder), "plan_sha256": sha(folder / "plan.json"),
                          "raw_forecasts_sha256": sha(folder / "raw_forecasts.jsonl")}), flush=True)
    elif args.mode == "watch":
        watch(campaign / "feature_validation")
    else:
        analyze(campaign / "feature_validation")
