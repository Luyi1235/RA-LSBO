"""Retrospective encoder diagnostics after framework checks; no physical evaluator."""

import argparse
from dataclasses import asdict, replace
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
from ra_lsbo.budget import digest
from ra_lsbo.calibration import SigmoidCalibrator
from ra_lsbo.features import LatentFeatures, history_identity
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.optimizer import append_json
from ra_lsbo.risk_model import BinaryRiskHead, RiskHeads
from ra_lsbo.structured_encoder import EncoderSettings, StructuredFeatures
from tools.risk_feature_ablation import feature_matrix
from tools.snapshot_experiment_sources import snapshot


SEEDS = (0, 1, 2)
LABELS = {
    "legacy": "Legacy", "paper": "Paper-aligned", "no_performance": "No performance",
    "no_entropy": "No entropy", "constant_gamma": "Fixed perf. weight",
    "no_mmd": "No MMD", "hierarchical_onehot": "Hierarchy one-hot",
    "plain_onehot": "Plain one-hot"}


def variants():
    base = EncoderSettings()
    return {
        "legacy": None, "paper": base,
        "no_performance": replace(base, performance_weight=0.),
        "no_entropy": replace(base, entropy_weight=0.),
        "constant_gamma": replace(base, performance_schedule="constant"),
        "no_mmd": replace(base, mmd_weight=0.)}


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def metrics(probabilities, labels):
    if len(probabilities) != len(labels):
        raise ValueError("Prediction and label counts differ")
    predicted = [(p, y) for p, y in zip(probabilities, labels) if p is not None]
    known = [(p, y) for p, y in predicted if y is not None]
    accepted = [(p, y) for p, y in predicted if p >= .8]
    return dict(
        binary_metrics([p for p, y in known], [int(y) for p, y in known]),
        predictions=len(predicted), accepted_at_0_8=len(accepted),
        known_accepted=sum(y is not None for p, y in accepted),
        unknown_accepted=sum(y is None for p, y in accepted),
        infeasible_among_accepted=sum(y is False for p, y in accepted))


def load_data(boundary):
    plan = load_json((boundary/"plan.json").read_bytes())
    audit = load_json((boundary/"audit.json").read_bytes())
    result = load_json((boundary/"result.json").read_bytes())
    if (audit["status"] != "passed" or result["status"] != "complete"
            or len(result["results"]) != 120 or result["contract_id"] != plan["contract_id"]):
        raise ValueError("Require the completed, audited 120-call boundary dataset")
    verify_sources(audit["input_sha256"])
    verify_sources(plan["source_sha256"])
    rows, _, historical_audit, contract = build_snapshot(plan["historical_source"])
    if len(rows) != 300:
        raise ValueError("Expected the fixed 180/60/60 historical split")
    if any(plan["source_sha256"].get(str(Path(plan["historical_source"])/name)) != sha
           for name, sha in historical_audit["source_sha256"].items()):
        raise ValueError("Historical source differs from the completed boundary protocol")
    target = [dict(evaluation=item["index"], candidate=item["proposal"]["candidate"],
                   risk=item["risk"]) for item in result["results"]]
    source = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
              (boundary/"plan.json", boundary/"audit.json", boundary/"result.json")}
    source.update(plan["source_sha256"])
    return plan, contract, {
        "training": rows[:180], "global_calibration": rows[180:240],
        "historical_test": rows[240:], "local_calibration": target[:60],
        "boundary_test": target[60:]}, source


def matrix(features, candidates, view):
    return (features.transform_view(candidates, view).numpy() if isinstance(features, StructuredFeatures)
            else feature_matrix(features, candidates, view))


def geometry(features):
    rep = features.representation
    with torch.no_grad():
        latent, _ = rep.model.encode(rep.grammar)
        masks = rep.model.discrete_mask(rep.grammar)
        predictions = torch.stack([layer(latent).argmax(-1) for layer in rep.model.decoder], -1)
        accuracy = float(((predictions == rep.grammar).double()*masks).sum()/masks.sum())
        entropy = torch.stack([
            -(torch.softmax(layer(latent), -1)*torch.log_softmax(layer(latent), -1)).sum(-1)*masks[:, i]
            for i, layer in enumerate(rep.model.decoder)], -1).sum(-1).mean()
        recovered = [
            rep.space.branch_key(rep.decode(z, np.random.default_rng(0))) ==
            rep.space.branch_key(rep.space.from_indices(indices.tolist()))
            for z, indices in zip(latent, rep.grammar)]
    return {
        "grammar_samples": len(rep.grammar), "active_head_accuracy": accuracy,
        "canonical_configuration_recovery": sum(recovered)/len(recovered),
        "active_decoder_entropy": float(entropy),
        "latent_variance_per_dimension": latent.var(0, unbiased=False).tolist(),
        "scope": "Training grammar reconstruction, not unseen-configuration generalization."}


def prior_rmse(features, rows):
    rep = features.representation
    valid = [r for r in rows if r["risk"]["objective_observed"]]
    if not valid or not rep.performance_fitted:
        return {"samples": len(valid), "rmse_wh": None, "reason": "performance_head_unavailable"}
    with torch.no_grad():
        candidates = [features.candidate(r["candidate"]) for r in valid]
        estimate = rep.model.performance(rep.features(candidates)).squeeze(-1).numpy()
        estimate = estimate*rep.y_std+rep.y_mean
    actual = np.asarray([r["risk"]["objective"] for r in valid])
    return {"samples": len(valid), "rmse_wh": float(np.sqrt(np.mean((estimate-actual)**2))),
            "reason": None}


def evaluate_head(features, data, view, model_id, output):
    matrices = {name: matrix(features, [r["candidate"] for r in rows], view)
                for name, rows in data.items()}
    train = data["training"]
    known = [i for i, r in enumerate(train) if r["risk"]["feasible_label"] is not None]
    head = BinaryRiskHead(matrices["training"][known],
                          [int(train[i]["risk"]["feasible_label"]) for i in known], seed=0)
    raw = {name: head.predict(value).tolist() for name, value in matrices.items()}
    calibrators = {}
    for name in ("global_calibration", "local_calibration"):
        selected = [i for i, r in enumerate(data[name]) if r["risk"]["feasible_label"] is not None]
        calibrators[name] = SigmoidCalibrator(
            [raw[name][i] for i in selected],
            [int(data[name][i]["risk"]["feasible_label"]) for i in selected],
            base_fitted=head.diagnostics["status"] == "fitted_raw")
    model = {"model_id": model_id, "feature_view": view,
             "input_dimensions": matrices["training"].shape[1],
             "raw_head": head.diagnostics,
             "calibration": {name: value.diagnostics for name, value in calibrators.items()},
             "metrics": {}}
    for cohort in ("historical_test", "boundary_test"):
        rows, probabilities = data[cohort], raw[cohort]
        forecasts = {
            "raw": probabilities,
            "global": calibrators["global_calibration"].predict(probabilities),
            "local": (calibrators["local_calibration"].predict(probabilities)
                      if cohort == "boundary_test" else None)}
        labels = [r["risk"]["feasible_label"] for r in rows]
        model["metrics"][cohort] = {}
        for field, values in forecasts.items():
            model["metrics"][cohort][field] = metrics(
                [None]*len(rows) if values is None else values, labels)
        for i, row in enumerate(rows):
            forecast = {
                "model_id": model_id, "cohort": cohort, "evaluation": row["evaluation"],
                "candidate_key": features.space.key(features.candidate(row["candidate"])),
                **{field: None if values is None else values[i] for field, values in forecasts.items()}}
            append_json(output/"forecasts.jsonl", forecast)
            append_json(output/"scored_predictions.jsonl", dict(forecast, label=labels[i]))
    return model, matrices, raw


def verify_predictions(output, data, reference):
    forecasts = [load_json(line) for line in (output/"forecasts.jsonl").read_bytes().splitlines()]
    scored = [load_json(line) for line in (output/"scored_predictions.jsonl").read_bytes().splitlines()]
    if len(forecasts) != len(scored) or len(forecasts) != 20*120:
        raise ValueError("Forecast/scored counts differ")
    lookup = {
        (name, row["evaluation"]): row for name in ("historical_test", "boundary_test")
        for row in data[name]}
    seen = set()
    for forecast, score in zip(forecasts, scored):
        key = forecast["model_id"], forecast["cohort"], forecast["evaluation"]
        if key in seen or forecast != {k: v for k, v in score.items() if k != "label"}:
            raise ValueError("Duplicated or changed forecast")
        seen.add(key)
        row = lookup[forecast["cohort"], forecast["evaluation"]]
        if (score["label"] != row["risk"]["feasible_label"]
                or forecast["candidate_key"] != reference.space.key(reference.candidate(row["candidate"]))):
            raise ValueError("Prediction truth or physical candidate differs")
    return scored


def summarize(report):
    result = []
    for name in LABELS:
        models = [m for m in report["models"] if m["variant"] == name]
        if not models:
            continue
        row = {"variant": name, "models": len(models)}
        for key, cohort, field in (
                ("historical_global", "historical_test", "global"),
                ("boundary_global", "boundary_test", "global"),
                ("boundary_local", "boundary_test", "local")):
            values = [m["metrics"][cohort][field]["brier"] for m in models]
            available = [v for v in values if v is not None]
            row[key] = {
                "available": len(available), "planned": len(models),
                "mean_brier": float(np.mean(available)) if available else None,
                "minimum": min(available) if available else None,
                "maximum": max(available) if available else None,
                "accepted_counts": [m["metrics"][cohort][field]["accepted_at_0_8"] for m in models],
                "accepted_infeasible_counts": [
                    m["metrics"][cohort][field]["infeasible_among_accepted"] for m in models]}
        result.append(row)
    return result


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    names = [item["variant"] for item in report["summary"]]
    colors = ["#555555", "#007f73", "#547caa", "#bd7318", "#8d649c", "#9b4c59", "#5f8f37", "#4b8e9f"]
    for ax, field, title in (
            (axes[0, 0], "historical_global", "Historical test: global calibration"),
            (axes[0, 1], "boundary_local", "Boundary test: local calibration")):
        values = [item[field]["mean_brier"] for item in report["summary"]]
        bars = ax.bar(range(len(names)), [np.nan if v is None else v for v in values], color=colors)
        for bar, value, item in zip(bars, values, report["summary"]):
            if value is not None:
                ax.errorbar(bar.get_x()+.5*bar.get_width(), value,
                            yerr=[[value-item[field]["minimum"]], [item[field]["maximum"]-value]],
                            fmt="none", color="black", capsize=3)
        ax.set_xticks(range(len(names)), [LABELS[n] for n in names], rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Brier loss")
        ax.grid(axis="y", alpha=.2)
    ax = axes[1, 0]
    representations = report["representations"]
    for name, color in zip(list(variants()), colors):
        values = [r["performance_rmse"]["boundary_test"]["rmse_wh"]
                  for r in representations if r["variant"] == name]
        valid = [v for v in values if v is not None]
        position = list(variants()).index(name)
        if valid:
            ax.scatter([position]*len(valid), valid, color=color)
    ax.set_xticks(range(6), [LABELS[n] for n in variants()], rotation=25, ha="right")
    ax.set_title("Performance head only: boundary objective error")
    ax.set_ylabel("RMSE (Wh); missing if head untrained")
    ax.grid(axis="y", alpha=.2)
    ax = axes[1, 1]
    trace = next(r["training_trace"] for r in representations if r["variant"] == "paper" and r["seed"] == 0)
    for field in ("reconstruction", "mmd", "performance", "entropy"):
        ax.plot([r["epoch"] for r in trace], [r["losses"][field] for r in trace], label=field)
    ax.set_title("Paper-aligned seed 0: unweighted training terms")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss term")
    ax.legend()
    ax.grid(alpha=.2)
    fig.suptitle("Retrospective encoder diagnostics; 0 new MDA calls\n"
                 "Encoder ranges share the same labels; one-hot controls are shared models", fontsize=14)
    fig.savefig(output/"encoder_diagnostics.png", dpi=150)
    plt.close(fig)


def run(boundary, root):
    torch.set_num_threads(1)
    parent, contract, data, sources = load_data(boundary)
    base = Path(parent["base_project"])
    project = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), project/"docs"/"ENCODER_EXPERIMENT_PROTOCOL_20261020.md",
             project/"tools"/"risk_feature_ablation.py",
             project/"tools"/"snapshot_experiment_sources.py",
             *sorted((project/"ra_lsbo").glob("*.py"))]
    paths.extend(base/"lsbo"/name for name in
                 ("__init__.py", "space.py", "records.py", "representation.py", "surrogate.py"))
    sources.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    verify_sources(sources)
    output = root/("encoder_framework_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                   +"_"+uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": "ra_encoder_framework_experiment_v1", "scope": "retrospective_not_prospective",
        "new_mda_calls": 0, "base_project": str(base), "boundary_source": str(boundary),
        "physical_contract": parent["contract_id"], "source_sha256": sources,
        "variants": {name: None if value is None else asdict(value) for name, value in variants().items()},
        "encoder_seeds": SEEDS, "rf_seed": 0, "epochs": 60,
        "cohort_counts": {name: len(rows) for name, rows in data.items()},
        "training_history_id": history_identity(data["training"]),
        "historical_cost": {"training": 180, "global_calibration": 60,
                            "historical_test": 60, "previous_boundary": 120}}
    save(output/"plan.json", plan)
    snapshot(output/"plan.json")
    report = {"status": "running", "new_mda_calls": 0,
              "scope": "Retrospective, previously inspected labels; not optimization superiority.",
              "models": [], "representations": [], "legacy_reference_verified": False,
              "onehot_consistency_checks": 0}
    reference = LatentFeatures(base, contract["design_variables"], seed=0).fit(data["training"], epochs=60)
    if RiskHeads(data["training"], reference, seed=0).model_id != parent["global_rf_model_id"]:
        raise ValueError("Original encoder/model identity no longer reproduces")
    known = set()
    for name, rows in data.items():
        keys = [reference.space.key(reference.candidate(r["candidate"])) for r in rows]
        if len(set(keys)) != len(keys) or set(keys) & known:
            raise ValueError("Physical duplicates across the fixed training/calibration/test blocks")
        known.update(keys)
    shared = {}
    for view in ("hierarchical_onehot", "plain_onehot"):
        model_id = digest({"training": plan["training_history_id"], "view": view, "rf_seed": 0})
        model, shared[view], _ = evaluate_head(reference, data, view, model_id, output)
        report["models"].append(dict(model, variant=view, seed=None, shared_baseline=True))
    for name, config in variants().items():
        for seed in SEEDS:
            features = (LatentFeatures(base, contract["design_variables"], seed=seed).fit(
                data["training"], epochs=60) if config is None else
                StructuredFeatures(base, contract["design_variables"], seed, config).fit(data["training"]))
            for view, expected in shared.items():
                for cohort, rows in data.items():
                    np.testing.assert_array_equal(
                        matrix(features, [r["candidate"] for r in rows], view), expected[cohort])
                    report["onehot_consistency_checks"] += 1
            model_id = digest({"encoder_snapshot": features.snapshot_id, "view": "latent", "rf_seed": 0})
            model, _, raw = evaluate_head(features, data, "latent", model_id, output)
            if name == "legacy" and seed == 0:
                all_target = raw["local_calibration"]+raw["boundary_test"]
                np.testing.assert_allclose(all_target, [p["raw_rf"] for p in parent["proposals"]],
                                           rtol=1e-12, atol=1e-12)
                report["legacy_reference_verified"] = True
            report["models"].append(dict(model, variant=name, seed=seed, shared_baseline=False))
            checkpoint = None
            if isinstance(features, StructuredFeatures):
                checkpoint = name+"_seed_"+str(seed)+".pt"
                features.save_checkpoint(output/checkpoint)
                restored = StructuredFeatures.load_checkpoint(
                    output/checkpoint, base, expected_history=data["training"])
                torch.testing.assert_close(
                    features.transform_view([r["candidate"] for r in data["boundary_test"]]),
                    restored.transform_view([r["candidate"] for r in data["boundary_test"]]), rtol=0, atol=0)
            report["representations"].append({
                "variant": name, "seed": seed, "snapshot_id": features.snapshot_id,
                "geometry": geometry(features), "checkpoint": checkpoint,
                "performance_rmse": {cohort: prior_rmse(features, data[cohort])
                                     for cohort in ("historical_test", "boundary_test")},
                "training_trace": getattr(features.representation, "training_trace", None)})
            print(name, "seed", seed, "completed", flush=True)
    scored = verify_predictions(output, data, reference)
    for model in report["models"]:
        for cohort in ("historical_test", "boundary_test"):
            rows = [r for r in scored if r["model_id"] == model["model_id"] and r["cohort"] == cohort]
            if len(rows) != len(data[cohort]):
                raise ValueError("Incomplete per-model scored cohort")
            for field in ("raw", "global", "local"):
                if model["metrics"][cohort][field] != metrics([r[field] for r in rows], [r["label"] for r in rows]):
                    raise ValueError("Scored forecast metrics differ")
    verify_sources(sources)
    report["summary"] = summarize(report)
    report.update(status="complete", distinct_models=len({m["model_id"] for m in report["models"]}),
                  verification={"source_files": len(sources), "prediction_records": len(scored),
                                "checkpoint_round_trips": 15, "physical_split_overlap": 0,
                                "metric_reconstruction": "passed"})
    plot(report, output)
    save(output/"report.json", report)
    save(output/"verification.json", {
        "status": "passed", **report["verification"],
        "input_sha256": {str(output/name): hashlib.sha256((output/name).read_bytes()).hexdigest()
                         for name in ("plan.json", "report.json", "forecasts.jsonl", "scored_predictions.jsonl")}})
    print("Report:", output/"report.json", flush=True)
    return output, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"runs")
    args = parser.parse_args()
    run(args.boundary.resolve(), args.output.resolve())
