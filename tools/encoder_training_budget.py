"""Training-budget diagnostics of structural recovery only; no new MDA or risk tuning."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import build_snapshot, load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.features import history_identity
from ra_lsbo.structured_encoder import EncoderSettings, StructuredFeatures
from tools.encoder_framework_experiment import geometry
from tools.snapshot_experiment_sources import snapshot


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for dimensions, color in ((4, "#547caa"), (8, "#007f73")):
        for axis, field, threshold, title in (
                (axes[0], "canonical_configuration_recovery", .90, "Complete configuration recovery"),
                (axes[1], "active_head_accuracy", .97, "Active decoder head accuracy")):
            means, lower, upper = [], [], []
            for epochs in (60, 180, 600):
                values = [r["geometry"][field] for r in report["runs"]
                          if r["latent_dim"] == dimensions and r["epochs"] == epochs]
                mean = float(np.mean(values))
                means.append(mean)
                lower.append(mean-min(values))
                upper.append(max(values)-mean)
            axis.errorbar((60, 180, 600), means, yerr=[lower, upper], marker="o",
                          capsize=4, label="z dimensions = "+str(dimensions), color=color)
            axis.axhline(threshold, color="#777777", linestyle=":", linewidth=1)
            axis.set_xscale("log")
            axis.set_xticks((60, 180, 600), ("60", "180", "600"))
            axis.set_ylim(0., 1.03)
            axis.set_xlabel("Training epochs")
            axis.set_title(title)
            axis.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle("Structural training diagnostics; 0 new MDA calls\n"
                 "Training grammar only; ranges over encoder seeds 0-2 are not confidence intervals")
    fig.savefig(output/"training_budget.png", dpi=150)
    plt.close(fig)


def run(parent_folder, output_root):
    torch.set_num_threads(1)
    parent = load_json((parent_folder/"plan.json").read_bytes())
    audited = load_json((parent_folder/"independent_audit.json").read_bytes())
    if audited["status"] != "passed":
        raise ValueError("Require the audited initial framework experiment")
    verify_sources(parent["source_sha256"])
    verify_sources(audited["input_sha256"])
    boundary_plan = load_json((Path(parent["boundary_source"])/"plan.json").read_bytes())
    history, _, _, contract = build_snapshot(boundary_plan["historical_source"])
    training = history[:180]
    if history_identity(training) != parent["training_history_id"]:
        raise ValueError("Training prefix changed")
    project = Path(__file__).resolve().parents[1]
    sources = dict(parent["source_sha256"])
    paths = [Path(__file__).resolve(), parent_folder/"plan.json", parent_folder/"report.json",
             parent_folder/"independent_audit.json",
             project/"docs"/"ENCODER_TRAINING_BUDGET_PROTOCOL_20261020.md"]
    sources.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    verify_sources(sources)
    folder = output_root/("encoder_training_budget_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                          +"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": "ra_encoder_training_budget_v1", "new_mda_calls": 0,
        "training_history_id": history_identity(training), "source_sha256": sources,
        "base_settings": asdict(EncoderSettings()), "latent_dimensions": [4, 8],
        "epochs": [60, 180, 600], "seeds": [0, 1, 2],
        "structural_flags": {"configuration_recovery": .90, "active_head_accuracy": .97},
        "scope": "Training-grammar recovery only, not held-out label/model selection."}
    save(folder/"plan.json", plan)
    snapshot(folder/"plan.json")
    report = {"status": "running", "new_mda_calls": 0, "runs": []}
    for dimensions in plan["latent_dimensions"]:
        for epochs in plan["epochs"]:
            for seed in plan["seeds"]:
                config = replace(EncoderSettings(), latent_dim=dimensions, initial_epochs=epochs)
                started = time.perf_counter()
                features = StructuredFeatures(
                    parent["base_project"], contract["design_variables"], seed, config).fit(training)
                measured = geometry(features)
                checkpoint = "z"+str(dimensions)+"_e"+str(epochs)+"_s"+str(seed)+".pt"
                features.save_checkpoint(folder/checkpoint)
                loaded = StructuredFeatures.load_checkpoint(
                    folder/checkpoint, parent["base_project"], training)
                if geometry(loaded) != measured:
                    raise ValueError("Reloaded structural diagnostics changed")
                record = {
                    "latent_dim": dimensions, "epochs": epochs, "seed": seed,
                    "geometry": measured, "wall_seconds": time.perf_counter()-started,
                    "snapshot_id": features.snapshot_id, "checkpoint": checkpoint,
                    "training_trace": features.representation.training_trace,
                    "structural_flags_met": measured["canonical_configuration_recovery"] >= .90
                    and measured["active_head_accuracy"] >= .97}
                report["runs"].append(record)
                print("z", dimensions, "epochs", epochs, "seed", seed,
                      "recovery", round(measured["canonical_configuration_recovery"], 4),
                      "head_accuracy", round(measured["active_head_accuracy"], 4), flush=True)
    verify_sources(sources)
    report["summary"] = []
    for dimensions in plan["latent_dimensions"]:
        for epochs in plan["epochs"]:
            records = [r for r in report["runs"] if r["latent_dim"] == dimensions and r["epochs"] == epochs]
            row = {"latent_dim": dimensions, "epochs": epochs,
                   "flags_met_seeds": sum(r["structural_flags_met"] for r in records)}
            for field in ("canonical_configuration_recovery", "active_head_accuracy"):
                values = [r["geometry"][field] for r in records]
                row[field] = {"mean": float(np.mean(values)), "minimum": min(values), "maximum": max(values)}
            report["summary"].append(row)
    report.update(status="complete", checkpoint_reconstructions=18,
                  source_files_verified=len(sources), heldout_labels_used_for_selection=False)
    plot(report, folder)
    save(folder/"report.json", report)
    paths = [folder/"plan.json", folder/"report.json", folder/"training_budget.png",
             *sorted(folder.glob("*.pt"))]
    save(folder/"verification.json", {
        "status": "passed", "checkpoint_reconstructions": 18,
        "source_files_verified": len(sources),
        "output_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}})
    print("Report:", folder/"report.json", flush=True)
    return folder, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"runs")
    args = parser.parse_args()
    run(args.parent.resolve(), args.output.resolve())
