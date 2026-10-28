"""Inspect the released model's component contributions, without new MDA."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.system_experiment import save_new
from tools.release_system_training_data import load_model_bundle


def forest_complement_check(left, right):
    """Prove complemented class scores only if every RF tree has that form."""
    if left is None or right is None:
        return {"available": False}
    if (left.classes_.tolist() != [0, 1] or right.classes_.tolist() != [0, 1]
            or left.n_features_in_ != right.n_features_in_
            or len(left.estimators_) != len(right.estimators_)):
        return {"available": True, "complement_for_all_inputs_within_tolerance": False}
    for first, second in zip(left.estimators_, right.estimators_):
        a, b = first.tree_, second.tree_
        if any(not np.array_equal(getattr(a, field), getattr(b, field)) for field in
               ("feature", "threshold", "children_left", "children_right")):
            return {"available": True, "complement_for_all_inputs_within_tolerance": False}
        total_a, total_b = a.value.sum(axis=2, keepdims=True), b.value.sum(axis=2, keepdims=True)
        if not np.allclose(a.value/total_a, (b.value/total_b)[:, :, ::-1], rtol=0, atol=1e-14):
            return {"available": True, "complement_for_all_inputs_within_tolerance": False}
    return {"available": True, "complement_for_all_inputs_within_tolerance": True,
            "leaf_fraction_absolute_tolerance": 1e-14,
            "verified_trees": len(left.estimators_),
            "proof_scope": "Identical tree partitions and reversed leaf class fractions; raw RF scores only"}


def diagnose(bundle, output):
    torch.set_num_threads(1)
    model, responses, calibrated = load_model_bundle(bundle)
    spec = load_json((bundle/"bundle_spec.json").read_bytes())
    observations = load_json((bundle/"observations.json").read_bytes())
    train_indices, cal_indices = set(spec["training_indices"]), set(spec["calibration_indices"])
    training = [r for r in observations if r["evaluation"] in train_indices]
    rows = [r for r in observations if r["evaluation"] in cal_indices]
    candidates = [r["candidate"] for r in rows]
    rf = model.risks.predict(candidates)
    gps = responses.predict(candidates)
    components = {"RF_final": np.asarray(rf["final_feasibility"])}
    components.update({name: np.asarray(gps[name]["p_satisfied_raw"])
                       for name in model.required_heads if gps[name] is not None})
    base = np.minimum.reduce(list(components.values()))
    aggregate = base.copy()
    if model.hidden_success_available:
        components["aggregate_hidden_success"] = np.asarray(rf["hidden_success"])
        aggregate = np.minimum(aggregate, components["aggregate_hidden_success"])
    mechanisms = {name: 1.-np.asarray(rf[name]) for name in model.active_mechanisms}
    all_components = dict(components, **mechanisms)
    full = np.minimum.reduce(list(all_components.values()))
    if not np.allclose(full, model.predict(candidates)["final_feasibility"], rtol=1e-10, atol=1e-12):
        raise ValueError("Component reconstruction differs from actual model score")
    labels = [(i, int(r["risk"]["feasible_label"])) for i, r in enumerate(rows)
              if r["risk"]["feasible_label"] is not None]
    scores = {name: binary_metrics([float(values[i]) for i, y in labels], [y for i, y in labels])
              for name, values in (("RF_GP_only", base), ("plus_aggregate_hidden", aggregate),
                                   ("full_mechanisms", full))}
    redundancy = {}
    for name in mechanisms:
        pairs = [(r["risk"]["hidden_success"], r["risk"]["hidden_labels"][name]) for r in training
                 if r["risk"]["hidden_success"] is not None and r["risk"]["hidden_labels"][name] is not None]
        redundancy[name] = {
            "coobserved_training_labels": len(pairs),
            "aggregate_and_mechanism_label_disagreements": sum(success != 1-failure for success, failure in pairs),
            "max_survival_score_difference_from_aggregate": (
                float(np.max(np.abs(mechanisms[name]-components["aggregate_hidden_success"])))
                if model.hidden_success_available else None),
            "forest_structure_complement": forest_complement_check(
                model.risks.heads["hidden_success"].model, model.risks.heads[name].model)}
    summary = {
        "scope": "Released model raw-score diagnostics on its held-out calibration block; no optimization evidence",
        "additional_mda_calls": 0, "observations": len(rows),
        "model_id": model.model_id, "active_mechanisms": list(model.active_mechanisms),
        "raw_scores_are_not_joint_probabilities": True,
        "calibrated_scores_not_scored_on_their_own_fit_block": True,
        "aggregate_changes_base_count": int(np.sum(np.abs(aggregate-base) > 1e-12)),
        "separate_mechanisms_change_aggregate_count": int(np.sum(np.abs(full-aggregate) > 1e-12)),
        "maximum_additional_mechanism_reduction": float(np.max(aggregate-full)),
        "minimum_component_counts_with_ties": {
            name: int(np.sum(np.isclose(value, full, rtol=0, atol=1e-12)))
            for name, value in all_components.items()},
        "raw_calibration_block_metrics": scores,
        "mechanism_aggregate_redundancy": redundancy,
        "observed_calibration_states": dict(Counter(r["risk"]["state"] for r in rows)),
        "bundle_spec_sha256": hashlib.sha256((bundle/"bundle_spec.json").read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=False)
    save_new(output/"components.json", summary)
    (output/"components.source.py").write_bytes(Path(__file__).read_bytes())
    print("Component diagnostic", output/"components.json", flush=True)
    print("Additional separate-mechanism score changes",
          summary["separate_mechanisms_change_aggregate_count"], "/", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.bundle.resolve(), args.output.resolve())
