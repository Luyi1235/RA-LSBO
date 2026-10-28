"""Exploratory leave-one-failure-neighborhood-out prediction; zero new MDA."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.features import history_identity
from ra_lsbo.mechanism_system import SystemSettings, mechanism_support
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.structured_models import StructuredRiskHeads
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem
from tools.run_system_validation import combine_history, read_lines


TARGET = "mission_solver_nonconvergence"


def split_groups(plan, results, held_out):
    """Remove the anchor and every generated neighbor from the training group."""
    members = {p["index"] for p in plan["probes"]
               if p["index"] == held_out or p.get("anchor_index") == held_out}
    training, test = [], []
    for row in results:
        if row["risk"] is not None:
            (test if row["index"] in members else training).append(row)
    if not test or not any(r["index"] == held_out for r in test):
        raise ValueError("Held-out anchor was not actually observed")
    return training, test


def row(record):
    return {"evaluation": record["index"], "candidate": record["candidate"], "risk": record["risk"]}


def validate(folder, output):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    report = load_json((folder/"report.json").read_bytes())
    audit = load_json((folder/"independent_audit.json").read_bytes())
    if report["status"] != "terminal" or audit["status"] != "passed":
        raise ValueError("Require completed, audited study")
    verify_sources(plan["source_sha256"])
    verify_sources(audit["input_sha256"])
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                            output/"unused", allow_mda=False)
    if problem.contract_id != plan["physical"]["contract_id"]:
        raise ValueError("Physical contract differs")
    initial = load_json((folder/"initial_history.json").read_bytes())
    probes = read_lines(folder/"probe_results.jsonl")
    anchors = sorted({p["anchor_index"] for p in plan["probes"] if "anchor_index" in p})
    output.mkdir(parents=True, exist_ok=False)
    diagnostics = []
    for anchor in anchors:
        train_probes, test = split_groups(plan, probes, anchor)
        training, duplicates = combine_history(problem, [initial, [row(r) for r in train_probes]])
        if duplicates:
            raise ValueError("Unexpected duplicate training design")
        test_keys = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in test}
        train_keys = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in training}
        if train_keys & test_keys:
            raise ValueError("Held-out design leaked into training")
        features = StructuredFeatures(
            plan["physical"]["base_project"], [asdict(v) for v in problem.space.variables],
            seed=0, settings=SystemSettings(**plan["settings"]).encoder()).fit(training)
        checkpoint = output/("without_anchor_"+str(anchor)+".pt")
        features.save_checkpoint(checkpoint)
        for view in ("latent", "hierarchical_onehot"):
            risks = StructuredRiskHeads(training, features, seed=0, view=view)
            predictions = risks.predict([r["candidate"] for r in test])[TARGET]
            observed = [(p, r["risk"]["hidden_labels"][TARGET])
                        for p, r in zip(predictions, test) if r["risk"]["hidden_labels"][TARGET] is not None]
            labels = [y for p, y in observed]
            probs = [p for p, y in observed]
            scores = binary_metrics(probs, labels)
            scores.update(
                recall_at_half=(sum(p >= .5 and y == 1 for p, y in observed)/sum(labels)
                                if sum(labels) else None),
                specificity_at_half=(sum(p < .5 and y == 0 for p, y in observed)/(len(labels)-sum(labels))
                                     if len(labels) > sum(labels) else None))
            diagnostics.append({
                "held_out_anchor": anchor, "view": view, "training_history_id": history_identity(training),
                "training_probe_indices": [r["index"] for r in train_probes],
                "test_probe_indices": [r["index"] for r in test],
                "training_support": mechanism_support(training)[TARGET],
                "test_support": mechanism_support([row(r) for r in test])[TARGET],
                "risk_head_status": risks.diagnostics[TARGET]["status"], "scores": scores,
                "representation_snapshot": features.snapshot_id, "model_id": risks.model_id,
                "predictions": [{"probe": r["index"], "probability": p,
                                 "label": r["risk"]["hidden_labels"][TARGET]}
                                for r, p in zip(test, predictions)],
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    result = {
        "status": "completed", "additional_mda_calls": 0, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Exploratory post-study two-area transfer, not preregistered efficacy evidence",
        "protocol": "Remove entire held-out anchor family; initial history plus other probes only; seed 0",
        "excluded_from_all_training": "Fresh calibration and every optimizer observation",
        "limitations": [
            "Only two predefined probe anchor families; fresh-calibration failure configurations excluded; "
            "no broad mechanism or domain generalization proof",
            "Correlated targeted neighbors; class prevalence and metrics are not population estimates",
            "Raw mechanism probabilities; no calibration on held-out area and no safety guarantee",
            "This diagnostic does not change frozen optimizer settings or comparative results"],
        "anchor_families": anchors, "folds": diagnostics,
        "input_audit_sha256": hashlib.sha256((folder/"independent_audit.json").read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_new(output/"transfer.json", result)
    (output/"transfer.source.py").write_bytes(Path(__file__).read_bytes())
    print("Failure-area transfer completed", output/"transfer.json", flush=True)
    for item in diagnostics:
        print(item["held_out_anchor"], item["view"], item["scores"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validate(args.folder.resolve(), args.output.resolve())
