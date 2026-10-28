"""Retrospective fixed-design and whole-area diagnostics; physical MDA disabled."""

import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.adaptive_risk import JointRiskEnsemble, JointRiskSettings, ReferenceTemperature
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.features import history_identity
from ra_lsbo.mechanism_system import SystemSettings
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.structured_models import StructuredRiskHeads
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem
from tools.release_system_training_data import load_model_bundle
from tools.run_system_validation import combine_history, read_lines
from tools.validate_mechanism_transfer import split_groups, row as probe_row


TARGET = "mission_solver_nonconvergence"


def evaluate(folder):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    if plan["schema"] != "ra_adaptive_research_v1" or plan["additional_real_mda_calls"] != 0:
        raise ValueError("Require frozen zero-real-MDA development protocol")
    verify_sources(plan["source_sha256"])
    parent, reference = Path(plan["physical_parent"]), Path(plan["retrospective"]["reference_bundle"])
    previous = load_json((parent/"plan.json").read_bytes())
    verify_sources(previous["source_sha256"])
    output = folder/"physical_retrospective"
    output.mkdir(exist_ok=False)
    problem = X57LiveProblem(plan["base_project"], previous["physical"]["config_path"],
                            output/"disabled_workers", allow_mda=False)
    if problem.contract_id != previous["physical"]["contract_id"]:
        raise ValueError("Physical contract changed")
    old_risk, responses, old_calibration = load_model_bundle(reference)
    features = responses.features
    spec = load_json((reference/"bundle_spec.json").read_bytes())
    shared = load_json((reference/"observations.json").read_bytes())
    training = [r for r in shared if r["evaluation"] in set(spec["training_indices"])]
    calibration = [r for r in shared if r["evaluation"] in set(spec["calibration_indices"])]
    report = load_json((parent/"report.json").read_bytes())
    observed = []
    for trial in report["physical"]:
        observed.extend(dict(r, historical_method=trial["method"], historical_seed=trial["seed"])
                        for r in read_lines(Path(trial["directory"])/"observations.jsonl"))
    shared_keys = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in shared}
    test = [r for r in observed if problem.space.key(problem.candidate_class(**r["candidate"])) not in shared_keys]
    candidates = [r["candidate"] for r in test]
    labels = [int(r["risk"]["feasible_label"]) for r in test]
    if len(test) < 200:
        raise ValueError("Too few eligible historical optimizer outcomes")
    old_raw = old_risk.predict(candidates)["final_feasibility"]
    old_cal = old_calibration.predict(candidates)["risk_probabilities_calibrated"]["final_feasibility"]
    results = [{"method": "old_common_reference", "raw": binary_metrics(old_raw, labels),
                "calibrated": binary_metrics(old_cal, labels)}]
    full = JointRiskSettings()
    variants = {"joint_full": full, "joint_aggregate": replace(full, aggregate=True),
                "joint_onehot": replace(full, latent=False), "joint_no_auxiliary": replace(full, auxiliary=False)}
    for name, settings in variants.items():
        risk = JointRiskEnsemble(training, features, seed=0, settings=settings)
        temperature = ReferenceTemperature(risk, calibration)
        raw = risk.predict(candidates)
        transformed = temperature.transform(raw["probabilities"], risk.model_id)
        risk.save_checkpoint(output/(name+".pt"))
        reloaded = JointRiskEnsemble.load_checkpoint(output/(name+".pt"), features)
        np.testing.assert_allclose(reloaded.predict(candidates)["probabilities"], raw["probabilities"],
                                   atol=1e-7, rtol=1e-6)
        item = {
            "method": name, "model_id": risk.model_id, "diagnostics": risk.diagnostics,
            "settings": asdict(settings), "calibration": temperature.to_dict(),
            "raw": binary_metrics(raw["final_feasibility"], labels),
            "calibrated": binary_metrics(transformed[:, 0].tolist(), labels)}
        results.append(item)
        save_new(output/(name+"_forecasts.json"), {
            "model": item, "raw": raw, "calibrated": transformed.tolist(),
            "test": [{"historical_method": r["historical_method"], "historical_seed": r["historical_seed"],
                      "evaluation": r["evaluation"], "candidate": r["candidate"], "label": y}
                     for r, y in zip(test, labels)]})
        print("Historical fixed-design diagnostic", name, item["raw"]["brier"],
              item["calibrated"]["brier"], flush=True)
    initial = load_json((parent/"initial_history.json").read_bytes())
    probes = read_lines(parent/"probe_results.jsonl")
    folds = []
    variables = [asdict(v) for v in problem.space.variables]
    for anchor in plan["retrospective"]["held_out_anchor_families"]:
        train_probes, held = split_groups(previous, probes, anchor)
        fold_training, duplicates = combine_history(problem, [initial, [probe_row(r) for r in train_probes]])
        if duplicates:
            raise ValueError("Unexpected duplicate fold training")
        keys = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in fold_training}
        if any(problem.space.key(problem.candidate_class(**r["candidate"])) in keys for r in held):
            raise ValueError("Held-out anchor family leaked into training")
        fold_features = StructuredFeatures(
            plan["base_project"], variables, seed=0, settings=SystemSettings().encoder()).fit(fold_training)
        fold_features.save_checkpoint(output/("without_anchor_"+str(anchor)+".pt"))
        known = [r for r in held if r["risk"]["hidden_labels"][TARGET] is not None]
        labels = [r["risk"]["hidden_labels"][TARGET] for r in known]
        for name, settings in variants.items():
            if name == "joint_aggregate":
                continue
            model = JointRiskEnsemble(fold_training, fold_features, seed=0, settings=settings)
            probability = np.asarray(model.predict([r["candidate"] for r in known])["probabilities"])
            # The native stage label is conditional on reaching the mission stage.
            stage = probability[:, 5]/np.maximum(probability[:, [0, 1, 5, 6]].sum(1), 1e-12)
            metric = binary_metrics(stage.tolist(), labels)
            metric["recall_at_half"] = sum(p >= .5 and y == 1 for p, y in zip(stage, labels))/sum(labels)
            model.save_checkpoint(output/(name+"_without_anchor_"+str(anchor)+".pt"))
            folds.append({"anchor": anchor, "method": name, "training_history_id": history_identity(fold_training),
                          "test_indices": [r["index"] for r in known], "metrics": metric,
                          "probabilities": stage.tolist(), "labels": labels,
                          "interpretation": "Uncalibrated conditional mission-stage probabilities"})
        for view in ("latent", "hierarchical_onehot"):
            old = StructuredRiskHeads(fold_training, fold_features, seed=0, view=view)
            p = old.predict([r["candidate"] for r in known])[TARGET]
            metric = binary_metrics(p, labels)
            metric["recall_at_half"] = sum(q >= .5 and y == 1 for q, y in zip(p, labels))/sum(labels)
            folds.append({"anchor": anchor, "method": "old_RF_"+view, "metrics": metric,
                          "probabilities": p, "labels": labels})
        print("Whole-anchor diagnostic", anchor, [f["metrics"]["recall_at_half"] for f in folds
                                                 if f["anchor"] == anchor], flush=True)
    result = {
        "status": "completed", "additional_real_mda_calls": 0, "mda_enabled": problem.allow_mda,
        "fixed_design_rows": len(test), "excluded_shared_design_rows": len(observed)-len(test),
        "fixed_design_unique": len({problem.space.key(problem.candidate_class(**r["candidate"])) for r in test}),
        "training_count": len(training), "reference_count": len(calibration),
        "training_history_id": history_identity(training), "fixed_design_results": results,
        "whole_area_folds": folds, "source_verified": True,
        "limitations": [
            "Post-study method development using already inspected data; no prospective efficiency result",
            "Historical optimizer selections are dependent and share incumbents; no iid confidence claims",
            "Whole-area folds are only two predefined targeted neighborhoods",
            "Four real hidden failure types still have zero positive training examples",
            "Reference calibration is not a safety guarantee; all unfavorable variants retained"]}
    save_new(output/"diagnostics.json", result)
    verify_sources(plan["source_sha256"])
    print("Retrospective diagnostics completed", output, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    evaluate(args.folder.resolve())
