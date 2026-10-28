"""Retrospective risk and append-only diagnostics of a structurally qualified encoder."""

import argparse
from dataclasses import asdict
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
from ra_lsbo.budget import digest
from ra_lsbo.calibration import SigmoidCalibrator
from ra_lsbo.features import history_identity
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.risk_model import BinaryRiskHead
from ra_lsbo.structured_encoder import EncoderSettings, StructuredFeatures
from tools.encoder_framework_experiment import evaluate_head, geometry, load_data, matrix
from tools.snapshot_experiment_sources import snapshot


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def model_identity(features, view):
    return digest({"encoder_snapshot": features.snapshot_id if view == "latent" else None,
                   "training": features.history_id, "view": view, "rf_seed": 0})


def risk_summary(models):
    summary = {}
    for key, cohort, calibration in (("historical_global", "historical_test", "global"),
                                    ("boundary_global", "boundary_test", "global"),
                                    ("boundary_local", "boundary_test", "local")):
        values = [m["metrics"][cohort][calibration]["brier"] for m in models]
        available = [v for v in values if v is not None]
        summary[key] = {"brier_values": values, "available": len(available), "planned": len(models),
                        "mean_brier": float(np.mean(available)) if available else None,
                        "accepted_counts": [m["metrics"][cohort][calibration]["accepted_at_0_8"]
                                            for m in models]}
    return summary


def measure(features, base_project, folder, checkpoint, started, previous_latent=None):
    measured = geometry(features)
    with torch.no_grad():
        latent, _ = features.representation.model.encode(features.representation.grammar)
    features.save_checkpoint(folder/checkpoint)
    restored = StructuredFeatures.load_checkpoint(folder/checkpoint, base_project)
    if geometry(restored) != measured or restored.snapshot_id != features.snapshot_id:
        raise ValueError("Checkpoint geometry or identity changed")
    return {
        "seed": features.seed, "prefix": len(features.training_indices),
        "history_id": features.history_id, "snapshot_id": features.snapshot_id,
        "checkpoint": checkpoint, "objective_indices": features.objective_indices,
        "geometry": measured, "wall_seconds": time.perf_counter()-started,
        "latent_step_rms": (None if previous_latent is None else
                            float((latent-previous_latent).square().mean().sqrt())),
        "structural_flags_met": measured["canonical_configuration_recovery"] >= .90
        and measured["active_head_accuracy"] >= .97}, latent


def plot(report, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    names = ("latent", "hierarchical_onehot", "plain_onehot")
    labels = ("Qualified z8", "Hierarchy one-hot", "Plain one-hot")
    for field, color, offset, label in (
            ("historical_global", "#547caa", -.16, "Historical / global"),
            ("boundary_local", "#007f73", .16, "Boundary / local")):
        for i, name in enumerate(names):
            values = [v for v in report["risk_summary"][name][field]["brier_values"] if v is not None]
            if not values:
                continue
            mean = float(np.mean(values))
            axes[0].bar(i+offset, mean, .30, color=color, label=label if i == 0 else None)
            axes[0].errorbar(i+offset, mean, yerr=[[mean-min(values)], [max(values)-mean]],
                             color="black", capsize=3, fmt="none")
    axes[0].set_xticks(range(3), labels)
    axes[0].set_ylabel("Brier loss (lower is better)")
    axes[0].set_title("Fixed split; one-hot models shared")
    axes[0].legend()
    for seed, color in ((0, "#007f73"), (1, "#547caa"), (2, "#ad594b")):
        rows = [r for r in report["updates"] if r["seed"] == seed]
        axes[1].plot([180]+[r["prefix"] for r in rows],
                     [1.]+[r["geometry"]["canonical_configuration_recovery"] for r in rows],
                     marker="o", color=color, label="Incremental seed "+str(seed))
        reset = next(r for r in report["resets"] if r["seed"] == seed)
        axes[1].scatter([240], [reset["geometry"]["canonical_configuration_recovery"]],
                        marker="x", color=color, s=70)
    axes[1].axhline(.9, linestyle=":", color="#777777")
    axes[1].set_ylim(.75, 1.02)
    axes[1].set_xticks((180, 200, 220, 240))
    axes[1].set_xlabel("Historical training prefix (x = reset at 240)")
    axes[1].set_ylabel("Complete configuration recovery")
    axes[1].set_title("Training grammar retention; not long-term stability")
    axes[1].legend(loc="lower left")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Qualified encoder diagnostics; retrospective, 0 new MDA calls\n"
                 "Ranges over encoder seeds are not confidence intervals")
    fig.savefig(folder/"candidate_validation.png", dpi=150)
    plt.close(fig)


def run(parent_folder, budget_folder, output_root):
    torch.set_num_threads(1)
    parent = load_json((parent_folder/"plan.json").read_bytes())
    budget_plan = load_json((budget_folder/"plan.json").read_bytes())
    budget_report = load_json((budget_folder/"report.json").read_bytes())
    for path, field in ((parent_folder/"independent_audit.json", "input_sha256"),
                        (budget_folder/"verification.json", "output_sha256")):
        verification = load_json(path.read_bytes())
        if verification["status"] != "passed":
            raise ValueError("Require the two completed, verified parent studies")
        verify_sources(verification[field])
    verify_sources(budget_plan["source_sha256"])
    boundary, contract, data, sources = load_data(Path(parent["boundary_source"]))
    if history_identity(data["training"]) != parent["training_history_id"]:
        raise ValueError("Training prefix differs from parent")
    project = Path(__file__).resolve().parents[1]
    config_path = project/"configs"/"structured_encoder_structural_candidate.json"
    settings = EncoderSettings(**load_json(config_path.read_bytes()))
    selected = [r for r in budget_report["runs"] if r["latent_dim"] == 8 and r["epochs"] == 600]
    if (len(selected) != 3 or sorted(r["seed"] for r in selected) != [0, 1, 2]
            or not all(r["structural_flags_met"] for r in selected)):
        raise ValueError("Require all three preselected structural checkpoints")
    sources.update(budget_plan["source_sha256"])
    source_paths = [Path(__file__).resolve(), config_path,
                    project/"docs"/"ENCODER_CANDIDATE_PROTOCOL_20261020.md",
                    parent_folder/"plan.json", parent_folder/"report.json",
                    parent_folder/"independent_audit.json", parent_folder/"forecasts.jsonl",
                    budget_folder/"plan.json", budget_folder/"report.json",
                    budget_folder/"verification.json",
                    *(budget_folder/r["checkpoint"] for r in selected)]
    sources.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths})
    verify_sources(sources)
    folder = output_root/("encoder_candidate_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                          +"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    save(folder/"plan.json", {
        "schema": "ra_encoder_candidate_validation_v1", "new_mda_calls": 0,
        "base_project": parent["base_project"], "boundary_source": parent["boundary_source"],
        "parent_source": str(parent_folder), "budget_source": str(budget_folder),
        "physical_contract": boundary["contract_id"], "source_sha256": sources,
        "settings": asdict(settings), "training_history_id": history_identity(data["training"]),
        "seeds": [0, 1, 2], "rf_seed": 0, "incremental_prefixes": [200, 220, 240],
        "reset_prefix": 240, "risk_models": 5, "risk_predictions": 600,
        "scope": "Retrospective; previously inspected labels, training grammar retention only."})
    snapshot(folder/"plan.json")
    report = {"status": "running", "new_mda_calls": 0, "models": [], "representations": [],
              "updates": [], "resets": [], "onehot_reference_predictions": 0}
    parent_report = load_json((parent_folder/"report.json").read_bytes())
    parent_forecasts = [load_json(line) for line in (parent_folder/"forecasts.jsonl").read_bytes().splitlines()]
    shared, restored = {}, {}
    for item in sorted(selected, key=lambda r: r["seed"]):
        seed = item["seed"]
        features = StructuredFeatures.load_checkpoint(
            budget_folder/item["checkpoint"], parent["base_project"], data["training"])
        if asdict(features.settings) != asdict(settings) or features.snapshot_id != item["snapshot_id"]:
            raise ValueError("Candidate checkpoint settings or identity differ")
        restored[seed] = features
        if seed == 0:
            seen = set()
            for rows in data.values():
                for row in rows:
                    candidate = features.space.key(features.candidate(row["candidate"]))
                    if candidate in seen:
                        raise ValueError("Duplicate physical candidate in source split")
                    seen.add(candidate)
            for view in ("hierarchical_onehot", "plain_onehot"):
                model, shared[view], _ = evaluate_head(
                    features, data, view, model_identity(features, view), folder)
                report["models"].append(dict(model, variant=view, seed=None, shared_baseline=True))
                old_model = next(m for m in parent_report["models"] if m["variant"] == view)
                if model["metrics"] != old_model["metrics"]:
                    raise ValueError("One-hot metrics differ from reference")
                old = [r for r in parent_forecasts if r["model_id"] == old_model["model_id"]]
                new = [load_json(line) for line in (folder/"forecasts.jsonl").read_bytes().splitlines()
                       if load_json(line)["model_id"] == model["model_id"]]
                if ([{k: v for k, v in r.items() if k != "model_id"} for r in old] !=
                        [{k: v for k, v in r.items() if k != "model_id"} for r in new]):
                    raise ValueError("One-hot forecasts differ from reference")
                report["onehot_reference_predictions"] += len(old)
        for view, expected in shared.items():
            for cohort, rows in data.items():
                np.testing.assert_array_equal(
                    matrix(features, [r["candidate"] for r in rows], view), expected[cohort])
        model, _, _ = evaluate_head(features, data, "latent", model_identity(features, "latent"), folder)
        report["models"].append(dict(model, variant="latent", seed=seed, shared_baseline=False))
        report["representations"].append({
            "seed": seed, "checkpoint": str(budget_folder/item["checkpoint"]),
            "snapshot_id": features.snapshot_id, "geometry": geometry(features)})
        print("qualified risk seed", seed, "completed", flush=True)
    report["risk_summary"] = {}
    for view in ("latent", "hierarchical_onehot", "plain_onehot"):
        models = [m for m in report["models"] if m["variant"] == view]
        report["risk_summary"][view] = risk_summary(models)
    history, _, _, _ = build_snapshot(boundary["historical_source"])
    for seed, features in restored.items():
        with torch.no_grad():
            previous, _ = features.representation.model.encode(features.representation.grammar)
        for prefix in (200, 220, 240):
            started = time.perf_counter()
            features.update(history[:prefix])
            record, previous = measure(
                features, parent["base_project"], folder, "increment_s"+str(seed)+"_n"+str(prefix)+".pt",
                started, previous)
            if record["history_id"] != history_identity(history[:prefix]):
                raise ValueError("Incremental history mismatch")
            report["updates"].append(record)
            print("increment seed", seed, "prefix", prefix, "geometry", record["geometry"][
                "canonical_configuration_recovery"], flush=True)
        started = time.perf_counter()
        reset = StructuredFeatures(parent["base_project"], contract["design_variables"], seed, settings).fit(
            history[:240])
        record, _ = measure(reset, parent["base_project"], folder, "reset_s"+str(seed)+"_n240.pt", started)
        report["resets"].append(record)
        print("reset seed", seed, "completed", flush=True)
    verify_sources(sources)
    report.update(status="complete", static_predictions=600, checkpoint_round_trips=12,
                  incremental_flags_met=sum(r["structural_flags_met"] for r in report["updates"]),
                  reset_flags_met=sum(r["structural_flags_met"] for r in report["resets"]),
                  source_files_verified=len(sources))
    plot(report, folder)
    save(folder/"report.json", report)
    paths = [folder/"plan.json", folder/"report.json", folder/"forecasts.jsonl",
             folder/"scored_predictions.jsonl", folder/"candidate_validation.png",
             *sorted(folder.glob("*.pt"))]
    save(folder/"verification.json", {
        "status": "passed",
        "output_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}})
    print("Report:", folder/"report.json", flush=True)
    return folder


def audit(folder):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    report = load_json((folder/"report.json").read_bytes())
    verification = load_json((folder/"verification.json").read_bytes())
    verify_sources(plan["source_sha256"])
    verify_sources(verification["output_sha256"])
    for sha in set(plan["source_sha256"].values()):
        if hashlib.sha256((folder/"source_snapshot"/(sha+".source")).read_bytes()).hexdigest() != sha:
            raise ValueError("Source snapshot mismatch")
    _, _, data, _ = load_data(Path(plan["boundary_source"]))
    scored = [load_json(line) for line in (folder/"scored_predictions.jsonl").read_bytes().splitlines()]
    forecasts = [load_json(line) for line in (folder/"forecasts.jsonl").read_bytes().splitlines()]
    if (len(scored) != 600 or len(forecasts) != 600 or len(report["models"]) != 5
            or len({m["model_id"] for m in report["models"]}) != 5
            or len(report["updates"]) != 9 or len(report["resets"]) != 3):
        raise ValueError("Unexpected models or prediction counts")
    if forecasts != [{k: v for k, v in r.items() if k != "label"} for r in scored]:
        raise ValueError("Forecasts differ from scored predictions")
    rebuilt, count = {}, 0
    for model in report["models"]:
        seed = 0 if model["seed"] is None else model["seed"]
        if seed not in rebuilt:
            checkpoint = next(r["checkpoint"] for r in report["representations"] if r["seed"] == seed)
            rebuilt[seed] = StructuredFeatures.load_checkpoint(
                checkpoint, plan["base_project"], data["training"])
        features = rebuilt[seed]
        if model_identity(features, model["feature_view"]) != model["model_id"]:
            raise ValueError("Risk model identity differs")
        matrices = {name: matrix(features, [r["candidate"] for r in rows], model["feature_view"])
                    for name, rows in data.items()}
        head = BinaryRiskHead(matrices["training"],
                              [int(r["risk"]["feasible_label"]) for r in data["training"]], seed=0)
        raw = {name: head.predict(value).tolist() for name, value in matrices.items()}
        calibrators = {
            name: SigmoidCalibrator(raw[name], [int(r["risk"]["feasible_label"]) for r in data[name]],
                                    base_fitted=head.diagnostics["status"] == "fitted_raw")
            for name in ("global_calibration", "local_calibration")}
        for name, value in calibrators.items():
            if value.diagnostics != model["calibration"][name]:
                raise ValueError("Calibration reconstruction differs")
        for cohort in ("historical_test", "boundary_test"):
            subset = [r for r in scored if r["model_id"] == model["model_id"] and r["cohort"] == cohort]
            values = {"raw": raw[cohort],
                      "global": calibrators["global_calibration"].predict(raw[cohort]),
                      "local": (calibrators["local_calibration"].predict(raw[cohort])
                                if cohort == "boundary_test" else None)}
            if len(subset) != 60:
                raise ValueError("Incomplete cohort")
            for i, (saved, row) in enumerate(zip(subset, data[cohort])):
                if (saved["label"] != row["risk"]["feasible_label"]
                        or saved["evaluation"] != row["evaluation"]
                        or saved["candidate_key"] != features.space.key(features.candidate(row["candidate"]))):
                    raise ValueError("Source label or candidate differs")
                for field, array in values.items():
                    expected = None if array is None else array[i]
                    if expected is None:
                        if saved[field] is not None:
                            raise ValueError("Unavailable calibration was filled")
                    elif not np.isclose(expected, saved[field], rtol=1e-12, atol=1e-12):
                        raise ValueError("Prediction reconstruction differs")
                count += 1
            for field in values:
                available = [r for r in subset if r[field] is not None]
                known = [r for r in available if r["label"] is not None]
                accepted = [r for r in available if r[field] >= .8]
                score = dict(binary_metrics([r[field] for r in known], [int(r["label"]) for r in known]),
                             predictions=len(available), accepted_at_0_8=len(accepted),
                             known_accepted=sum(r["label"] is not None for r in accepted),
                             unknown_accepted=sum(r["label"] is None for r in accepted),
                             infeasible_among_accepted=sum(r["label"] is False for r in accepted))
                if score != model["metrics"][cohort][field]:
                    raise ValueError("Independent metric reconstruction differs")
    boundary = load_json((Path(plan["boundary_source"])/"plan.json").read_bytes())
    history, _, _, _ = build_snapshot(boundary["historical_source"])
    for record in report["updates"]+report["resets"]:
        restored = StructuredFeatures.load_checkpoint(
            folder/record["checkpoint"], plan["base_project"], history[:record["prefix"]])
        if restored.snapshot_id != record["snapshot_id"] or geometry(restored) != record["geometry"]:
            raise ValueError("Append/reset structural result differs")
        if restored.objective_indices != record["objective_indices"]:
            raise ValueError("Objective training prefix differs")
    for view in ("latent", "hierarchical_onehot", "plain_onehot"):
        if report["risk_summary"][view] != risk_summary([m for m in report["models"] if m["variant"] == view]):
            raise ValueError("Risk summary differs")
    return {"status": "passed", "models_rebuilt": 5, "source_predictions_reproduced": count,
            "structural_checkpoints_recomputed": 12, "static_checkpoints_loaded": len(rebuilt),
            "input_sha256": {str(folder/"verification.json"): hashlib.sha256(
                (folder/"verification.json").read_bytes()).hexdigest()},
            "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"runs")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit is not None:
        result = audit(args.audit.resolve())
        save(args.audit.resolve()/"independent_audit.json", result)
        print(json.dumps(result, indent=2))
    else:
        if args.parent is None or args.budget is None:
            parser.error("--parent and --budget required for execution")
        run(args.parent.resolve(), args.budget.resolve(), args.output.resolve())
