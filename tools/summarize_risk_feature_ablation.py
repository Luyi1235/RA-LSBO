"""Independently check feature-ablation records and summarize fixed RF seeds."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import build_snapshot, load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.features import LatentFeatures
from ra_lsbo.transfer_validation import candidate_key, unseen_rows
from tools.risk_feature_ablation import MODES, SEEDS, score_records


def aggregate(values):
    known = [value for value in values if value is not None]
    return {
        "available": len(known), "planned": len(values),
        "mean": float(np.mean(known)) if len(known) == len(values) and known else None,
        "minimum": min(known) if known else None, "maximum": max(known) if known else None}


def audit_and_summarize(folder):
    report = load_json((folder / "report.json").read_bytes())
    if report["status"] != "complete" or report["new_mda_calls"] != 0:
        raise ValueError("Require a completed offline diagnostic")
    verify_sources(report["source_sha256"])
    for expected in set(report["source_sha256"].values()):
        path = folder / "source_snapshot" / (expected+".source")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Source snapshot changed")
    plans = [Path(path) for path in report["source_sha256"] if Path(path).name == "plan.json"]
    if len(plans) != 1:
        raise ValueError("Require a unique original pilot plan")
    plan = load_json(plans[0].read_bytes())
    history, _, source_audit, contract = build_snapshot(plan["historical_source"])
    if source_audit["source_sha256"] != plan["historical_source_sha256"]:
        raise ValueError("Historical source differs from original pilot plan")
    features = LatentFeatures(plan["base_project"], contract["design_variables"], seed=0)
    excluded = {candidate_key(features, row["candidate"]) for row in history[:240]}
    truth, cohorts = {}, {}
    for name in report["cohorts"]:
        if name == "source_holdout":
            rows = history[240:]
        else:
            rows, _, target_audit, _ = build_snapshot(name)
            if target_audit["evaluator_contract_sha256"] != source_audit["evaluator_contract_sha256"]:
                raise ValueError("Cross-contract source")
        selected, removed = unseen_rows(rows, features, excluded)
        if (len(selected) != report["cohorts"][name]["retained"]
                or removed != report["cohorts"][name]["excluded"]):
            raise ValueError("Cohort selection changed")
        cohorts[name] = {row["evaluation"]: candidate_key(features, row["candidate"]) for row in selected}
        truth.update({(name, row["evaluation"], candidate_key(features, row["candidate"])):
                      row["risk"]["feasible_label"] for row in selected})
    if len({key[2] for key in truth}) != report["unique_test_designs"]:
        raise ValueError("Unique test design count differs")
    paths = [folder / name for name in ("report.json", "forecasts.jsonl", "scored_predictions.jsonl")]
    loaded = [[load_json(line) for line in path.read_bytes().splitlines()] for path in paths[1:]]
    forecasts, scored = loaded
    if len(forecasts) != len(scored):
        raise ValueError("Forecast/scored counts differ")
    expected_models = {mode+"/seed_"+str(seed) for mode in MODES for seed in SEEDS}
    if set(report["models"]) != expected_models:
        raise ValueError("Fixed models/seeds changed")
    grouped = {(model, cohort): [] for model in expected_models for cohort in cohorts}
    identities = set()
    for forecast, observed in zip(forecasts, scored):
        without_label = {key: value for key, value in observed.items() if key != "label"}
        if "label" in forecast or forecast != without_label:
            raise ValueError("Forecast changed or includes a label")
        key = (forecast["cohort"], forecast["evaluation"], forecast["candidate_key"])
        if key not in truth or observed["label"] is not truth[key]:
            raise ValueError("Candidate/label differs from original historical source")
        identity = (forecast["model"], *key)
        if identity in identities:
            raise ValueError("Duplicate prediction")
        identities.add(identity)
        grouped[forecast["model"], forecast["cohort"]].append(observed)
    for (model, cohort), rows in grouped.items():
        if len(rows) != len(cohorts[cohort]) or score_records(rows) != report["models"][model]["cohorts"][cohort]:
            raise ValueError("Reported metrics differ from original predictions")
    rows = []
    for cohort in cohorts:
        for mode in MODES:
            models = [report["models"][mode+"/seed_"+str(seed)] for seed in SEEDS]
            row = {
                "cohort": cohort, "mode": mode, "test_samples": len(cohorts[cohort]),
                "dimensions": models[0]["dimensions"],
                "calibration_statuses": dict(Counter(model["calibration"]["status"] for model in models))}
            for field in ("raw", "calibrated"):
                row[field] = {
                    metric: aggregate([model["cohorts"][cohort][field][metric] for model in models])
                    for metric in ("brier", "log_loss", "ece", "accepted_at_0_8", "infeasible_among_accepted")}
            rows.append(row)
    paired = []
    for cohort in cohorts:
        for mode in MODES[1:]:
            differences = []
            for seed in SEEDS:
                comparison = report["models"][mode+"/seed_"+str(seed)]["cohorts"][cohort]["calibrated"]["brier"]
                reference = report["models"]["latent/seed_"+str(seed)]["cohorts"][cohort]["calibrated"]["brier"]
                differences.append(None if comparison is None or reference is None else comparison-reference)
            paired.append({
                "cohort": cohort, "comparison": mode, "reference": "latent",
                "brier_differences_by_rf_seed": differences, "difference": aggregate(differences),
                "lower_loss": sum(value is not None and value < -1e-12 for value in differences),
                "higher_loss": sum(value is not None and value > 1e-12 for value in differences)})
    verification = {
        "status": "passed", "models": len(expected_models), "matched_predictions": len(scored),
        "historical_test_designs": report["unique_test_designs"], "new_mda_calls": 0,
        "frozen_sources_verified": len(report["source_sha256"]),
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return verification, {
        "rows": rows, "paired": paired,
        "interpretation": "Means and min/max over five RF seeds on the same histories, not confidence intervals.",
        "scope": report["scope"]}


def plot(summary, destination):
    figure, axes = plt.subplots(2, 1, figsize=(12, 8.4), constrained_layout=True)
    cohorts = list(dict.fromkeys(row["cohort"] for row in summary["rows"]))
    labels = ["Own holdout" if name == "source_holdout" else Path(name).parent.name for name in cohorts]
    for axis, field in zip(axes, ("raw", "calibrated")):
        for offset, (mode, color) in enumerate(zip(MODES, ("#555555", "#008775", "#467cac", "#b66b19"))):
            rows = [next(row for row in summary["rows"] if row["mode"] == mode and row["cohort"] == name)
                    for name in cohorts]
            values = [row[field]["brier"] for row in rows]
            means = np.asarray([np.nan if value["mean"] is None else value["mean"] for value in values])
            spread = np.asarray([[np.nan if value["mean"] is None else value["mean"]-value["minimum"]
                                  for value in values],
                                 [np.nan if value["mean"] is None else value["maximum"]-value["mean"]
                                  for value in values]])
            axis.bar(np.arange(len(cohorts))+(offset-1.5)*.18, means, width=.17,
                     yerr=spread, capsize=2, color=color, label=mode)
        axis.set(xticks=np.arange(len(cohorts)), xticklabels=labels, ylabel="Brier loss",
                 title=("Raw" if field == "raw" else "Historical sigmoid calibration")+
                       ": mean and range over RF seeds 0-4")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
        axis.legend(fontsize=8, ncols=4)
    figure.suptitle("Risk feature ablation on previously inspected histories\n"
                   "One frozen VAE, same train/calibration/test split; no new MDA", fontsize=13)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = args.folder.resolve()
    verification, summary = audit_and_summarize(folder)
    for filename, value in (("verification.json", verification), ("summary.json", summary)):
        with (folder / filename).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    plot(summary, folder / "feature_ablation.png")
    print(json.dumps(verification, indent=2))
