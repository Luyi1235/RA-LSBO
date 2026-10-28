"""Evaluate new posterior uncertainty on old, strictly held-out failure families."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.budget import digest
from ra_lsbo.features import history_identity
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.posterior_risk import CompletionPosterior, PosteriorSettings
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.system_experiment import save_new
from tools.release_system_training_data import load_model_bundle
from tools.run_system_validation import combine_history, read_lines
from tools.validate_mechanism_transfer import split_groups, row as probe_row
from ra_lsbo.x57_live import X57LiveProblem


TARGET = "mission_solver_nonconvergence"


def main():
    torch.set_num_threads(1)
    old = ROOT/"runs/system_validation_v5_20261020T192312Z_5f6531eb"
    previous = ROOT/"runs/adaptive_research_20261021T071809Z_6fba5696/physical_retrospective"
    reference = ROOT/"runs/system_shared_reference_v5_20261021"
    output = ROOT/"runs/posterior_transfer_20261021"
    output.mkdir(exist_ok=False)
    plan = load_json((old/"plan.json").read_bytes())
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                             output/"disabled_worker", False)
    initial = load_json((old/"initial_history.json").read_bytes())
    probes = read_lines(old/"probe_results.jsonl")
    folds = []
    settings = PosteriorSettings(steps=120, inducing=64)
    for anchor in (5, 19):
        train_probes, held = split_groups(plan, probes, anchor)
        training, duplicates = combine_history(problem, [initial, [probe_row(r) for r in train_probes]])
        if duplicates:
            raise ValueError("Duplicate training rows")
        keys = {digest(r["candidate"]) for r in training}
        known = [r for r in held if r["risk"]["hidden_labels"][TARGET] is not None]
        if any(digest(r["candidate"]) in keys for r in known):
            raise ValueError("Held family leaked into training")
        features = StructuredFeatures.load_checkpoint(
            previous/("without_anchor_"+str(anchor)+".pt"),
            plan["physical"]["base_project"], training)
        labels = [r["risk"]["hidden_labels"][TARGET] for r in known]
        for view in ("paper_gp", "hierarchical_onehot"):
            model = CompletionPosterior(training, features, 0,
                                        PosteriorSettings(**dict(asdict(settings), view=view)))
            prediction = model.predict([r["candidate"] for r in known])
            head = prediction["heads"][TARGET]
            if head is None:
                raise ValueError("Expected observed mission-failure contrast")
            probability = np.asarray(head["probability"])
            variance = np.asarray(head["latent_variance"])
            metric = binary_metrics(probability.tolist(), labels)
            metric["recall_at_half"] = float(((probability >= .5) & (np.asarray(labels) == 1)).sum()/sum(labels))
            item = {
                "anchor": anchor, "view": view, "training_history_id": history_identity(training),
                "test_indices": [r["index"] for r in known], "labels": labels,
                "predictions": prediction, "mission_failure_metrics": metric,
                "mean_latent_variance_at_failures": float(variance[np.asarray(labels) == 1].mean()),
                "mean_latent_variance_at_successes": (float(variance[np.asarray(labels) == 0].mean())
                                                     if 0 in labels else None)}
            folds.append(item)
            model.save_checkpoint(output/("anchor_"+str(anchor)+"_"+view+".pt"))
            print("Transfer", anchor, view, metric["recall_at_half"],
                  metric["brier"], item["mean_latent_variance_at_failures"], flush=True)
    _, responses, _ = load_model_bundle(reference)
    all_rows = load_json((reference/"observations.json").read_bytes())
    spec = load_json((reference/"bundle_spec.json").read_bytes())
    training = [r for r in all_rows if r["evaluation"] in set(spec["training_indices"])]
    test = load_json((previous/"joint_full_forecasts.json").read_bytes())["test"]
    model = CompletionPosterior(training, responses.features, 0, settings)
    candidates = [r["candidate"] for r in test]
    completion = model.predict(candidates)
    probability = np.asarray(completion["heads"]["completion"]["probability"])
    for name, values in responses.predict(candidates).items():
        if name.startswith("constraint:") and values is not None:
            probability *= np.asarray(values["p_satisfied_raw"])
    labels = [r["label"] for r in test]
    unique = list({digest(r["candidate"]): i for i, r in enumerate(test)}.values())
    metrics = {
        "all_forecasts": binary_metrics(probability.tolist(), labels),
        "unique_designs": binary_metrics(probability[unique].tolist(), [labels[i] for i in unique])}
    model.save_checkpoint(output/"historical_completion.pt")
    save_new(output/"diagnostics.json", {
        "status": "completed", "additional_real_mda_calls": 0, "mda_enabled": problem.allow_mda,
        "whole_family_folds": folds, "historical_metrics": metrics,
        "historical_test": test, "historical_final_probability_raw": probability.tolist(),
        "completion_forecast": completion, "settings": asdict(settings),
        "calibration": "not_applied; no use of the 72 held-out reference labels",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__).resolve(), ROOT/"ra_lsbo/posterior_risk.py")},
        "limitations": [
            "Post-study development, not prospective physical optimization",
            "Only two known failure families; latent variance is not a calibrated error bar",
            "Independent constraint heads; no identified cross-constraint physical dependence",
            "Four unsupported physical failure mechanisms remain unsupported"]})
    print("Historical", metrics, flush=True)


if __name__ == "__main__":
    main()
