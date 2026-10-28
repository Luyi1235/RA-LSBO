"""Release same-contract unique observations and an explicitly post-study model."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import sys

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.calibration import CalibratedRiskHeads
from ra_lsbo.features import history_identity
from ra_lsbo.mechanism_system import FullSystemLearning, MechanismRiskHeads, SystemSettings, mechanism_support
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.structured_models import StructuredResponseHeads, StructuredRiskHeads
from ra_lsbo.system_experiment import save_new
from ra_lsbo.training import training_rows
from ra_lsbo.x57_live import X57LiveProblem
from tools.run_system_validation import combine_history, read_lines


def equivalent_observation(left, right):
    for key in ("state", "feasible_label", "objective_observed", "constraint_observed",
                "hidden_labels", "hidden_success"):
        if left[key] != right[key]:
            return False
    for a, b in [(left["objective"], right["objective"]),
                 *[(left["constraints"][name], right["constraints"][name])
                   for name in left["constraints"]]]:
        if a is None or b is None:
            if a is not b:
                return False
        elif not np.isclose(a, b, rtol=1e-8, atol=1e-6):
            return False
    return True


def save_model_weights(output, models, metadata, base, observations, sources):
    risks, responses, calibrated = models
    joblib.dump({name: head.model for name, head in risks.risks.heads.items()},
                output/"risk_forests.joblib", compress=3)
    torch.save({name: {"state_dict": model.state_dict(), "center": center,
                      "scale": scale, "sign": sign}
                for name, (model, center, scale, sign) in responses.models.items()},
               output/"gp_states.pt")
    save_new(output/"bundle_spec.json", {
        "schema": "ra_poststudy_model_bundle_v1", "base_project": base, "seed": 0,
        "python_version": sys.version,
        "runtime_versions": {name: importlib.metadata.version(name) for name in
                             ("numpy", "scipy", "torch", "scikit-learn", "botorch", "gpytorch", "joblib")},
        "source_sha256": dict(sources, **{
            str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}),
        "training_indices": metadata["training_indices"],
        "calibration_indices": metadata["calibration_indices"],
        "model_id": risks.model_id,
        "calibration_id": None if calibrated is None else calibrated.calibration_id,
        "metadata_path": "model_metadata.json",
        "file_sha256": {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in
                        ("observations.json", "model_metadata.json", "encoder.pt",
                         "risk_forests.joblib", "gp_states.pt")}})
    restored = load_model_bundle(output)
    candidates = [row["candidate"] for row in observations[-16:]]
    expected_raw, actual_raw = risks.predict(candidates), restored[0].predict(candidates)
    for name in expected_raw:
        if not np.allclose(expected_raw[name], actual_raw[name], rtol=1e-8, atol=1e-10):
            raise ValueError("Saved forests/GP score round trip differs")
    expected, actual = responses.predict(candidates), restored[1].predict(candidates)
    for name in expected:
        if expected[name] is None:
            if actual[name] is not None:
                raise ValueError("Saved GP availability differs")
        else:
            for key in expected[name]:
                if not np.allclose(expected[name][key], actual[name][key], rtol=1e-8, atol=1e-10):
                    raise ValueError("Saved GP forecast round trip differs")
    if calibrated is not None:
        a = calibrated.predict(candidates)["risk_probabilities_calibrated"]
        b = restored[2].predict(candidates)["risk_probabilities_calibrated"]
        for name in a:
            if a[name] is None:
                if b[name] is not None:
                    raise ValueError("Saved calibration availability differs")
            elif not np.allclose(a[name], b[name], rtol=1e-8, atol=1e-10):
                raise ValueError("Saved calibrated forecast round trip differs")


def load_model_bundle(folder):
    """Load a source-bound local bundle, restoring RF and GP learned parameters."""
    folder = Path(folder)
    spec = load_json((folder/"bundle_spec.json").read_bytes())
    if spec["schema"] != "ra_poststudy_model_bundle_v1":
        raise ValueError("Unrecognized model bundle")
    if (sys.version != spec["python_version"] or any(
            importlib.metadata.version(name) != version for name, version in spec["runtime_versions"].items())):
        raise ValueError("Model-bundle runtime changed")
    verify_sources(spec["source_sha256"])
    verify_sources({str(folder/name): sha for name, sha in spec["file_sha256"].items()})
    rows = load_json((folder/"observations.json").read_bytes())
    training = [r for r in rows if r["evaluation"] in set(spec["training_indices"])]
    calibration = [r for r in rows if r["evaluation"] in set(spec["calibration_indices"])]
    features = StructuredFeatures.load_checkpoint(folder/"encoder.pt", spec["base_project"], training)
    risks = StructuredRiskHeads(training, features, seed=spec["seed"])
    forests = joblib.load(folder/"risk_forests.joblib")
    if forests.keys() != risks.heads.keys():
        raise ValueError("Risk forest heads differ")
    for name in forests:
        risks.heads[name].model = forests[name]
    responses = StructuredResponseHeads(training, features, seed=spec["seed"], steps=1)
    states = torch.load(folder/"gp_states.pt", map_location="cpu", weights_only=True)
    if states.keys() != responses.models.keys():
        raise ValueError("GP head set differs")
    for name, saved in states.items():
        gp, center, scale, sign = responses.models[name]
        if [center, scale, sign] != [saved["center"], saved["scale"], saved["sign"]]:
            raise ValueError("GP normalization differs")
        gp.train()
        gp.load_state_dict(saved["state_dict"], strict=True)
        gp.eval()
    metadata = load_json((folder/spec["metadata_path"]).read_bytes())
    responses.diagnostics = metadata["response_diagnostics"]
    required = [name.removeprefix("constraint:") for name in
                metadata["risk_diagnostics"]["final_feasibility"]["required_constraint_heads"]]
    combined = MechanismRiskHeads(risks, responses, required)
    calibrated = CalibratedRiskHeads(combined, calibration) if calibration else None
    if (combined.model_id != spec["model_id"]
            or (None if calibrated is None else calibrated.calibration_id) != spec["calibration_id"]):
        raise ValueError("Restored model/calibration identity differs")
    return combined, responses, calibrated


def release(folder, output):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    report = load_json((folder/"report.json").read_bytes())
    audit = load_json((folder/"independent_audit.json").read_bytes())
    if report["status"] != "terminal" or audit["status"] != "passed":
        raise ValueError("Only terminal, audited physical data can be released")
    verify_sources(plan["source_sha256"])
    verify_sources(audit["input_sha256"])
    base = plan["physical"]["base_project"]
    problem = X57LiveProblem(base, plan["physical"]["config_path"], output/"unused", allow_mda=False)
    if problem.contract_id != plan["physical"]["contract_id"]:
        raise ValueError("Physical contract changed")
    shared = load_json((folder/"shared_history.json").read_bytes())
    new = []
    for run in report["physical"]:
        for row in read_lines(Path(run["directory"])/"observations.jsonl"):
            worker = row["risk"]["evidence"].get("worker_dir")
            response = None if worker is None else Path(worker)/"response.json"
            timestamp = response.stat().st_mtime if response is not None and response.exists() else float("inf")
            new.append((timestamp, run["seed"], run["method"], row["evaluation"],
                        dict(row, release_source={
                            "kind": "optimizer", "method": run["method"], "seed": run["seed"],
                            "evaluation": row["evaluation"], "directory": run["directory"],
                            "response_completion_utc": (datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
                                                        if np.isfinite(timestamp) else None)})))
    ordered = [item[-1] for item in sorted(new, key=lambda x: x[:4])]
    rows = [dict(r, release_source={"kind": "shared_acquisition", "evaluation": r["evaluation"]})
            for r in shared] + ordered
    groups = {}
    for row in rows:
        canonical = problem.canonical(row["candidate"])
        key = problem.space.key(problem.candidate_class(**canonical))
        groups.setdefault(key, []).append(row)
    conflicts = {key for key, replicas in groups.items()
                 if any(not equivalent_observation(replicas[0]["risk"], r["risk"]) for r in replicas[1:])}
    filtered = [r for r in rows
                if problem.space.key(problem.candidate_class(**r["candidate"])) not in conflicts]
    combined, duplicates = combine_history(problem, [filtered])
    if len(combined) <= 75:
        raise ValueError("Insufficient clean chronological observations for a post-study fit")
    output.mkdir(parents=True, exist_ok=False)
    save_new(output/"observations.json", combined)
    replica_report = [{
        "candidate_key": key, "observations": len(replicas),
        "conflicting": key in conflicts, "sources": [r["release_source"] for r in replicas]}
        for key, replicas in groups.items() if len(replicas) > 1]
    save_new(output/"replicas.json", replica_report)
    settings = SystemSettings(calibration_window=72)
    learner = FullSystemLearning("ra_full", settings, seed=0)
    models, metadata = learner.fit(
        combined, [v.__dict__ for v in problem.space.variables], base,
        sorted(set(problem.required_constraints)-set(problem.analytical_constraints)))
    learner.features.save_checkpoint(output/"encoder.pt")
    save_new(output/"model_metadata.json", metadata)
    save_model_weights(output, models, metadata, base, combined, plan["source_sha256"])
    training, calibration = combined[:-72], combined[-72:]
    tables = training_rows(training)
    save_new(output/"training_tables.json", tables)
    restored = StructuredFeatures.load_checkpoint(output/"encoder.pt", base, training)
    if restored.snapshot_id != learner.features.snapshot_id:
        raise ValueError("Released checkpoint reconstruction differs")
    summary = {
        "schema": "ra_same_contract_poststudy_training_release_v1",
        "interpretation": "Post-study model, not used for prequential evaluation or comparative claims",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": problem.contract_id,
        "historical_mda_cost": 420, "complete_new_mda_cost": report["complete_new_mda_calls"],
        "additional_mda_calls": 0,
        "available_observation_records": len(rows), "unique_designs_before_conflict_filter": len(groups),
        "conflicting_unique_designs_excluded": len(conflicts),
        "consistent_repeated_records_removed": len(duplicates),
        "unique_released_observations": len(combined), "training_observations": len(training),
        "calibration_observations": 72,
        "history_id": history_identity(combined),
        "chronology": "Shared acquisition first; optimizer responses ordered by audited local completion timestamps",
        "timestamp_interpretation": "Local ordering consistency, not external temporal witnessing",
        "training_mechanism_support": mechanism_support(training),
        "calibration_mechanism_support": mechanism_support(calibration),
        "all_available_mechanism_support": mechanism_support(combined),
        "training_support_flags": {name: c["positive"] >= 20 and c["negative"] >= 20
                                   for name, c in mechanism_support(training).items()},
        "active_mechanisms": list(models[0].active_mechanisms),
        "objective_observations": len(tables["objective"]),
        "constraint_training_support": {
            name: {"observed": len(table), "violations": sum(r["target"] > 0 for r in table),
                   "satisfied": sum(r["target"] <= 0 for r in table),
                   "minimum_margin": min((r["target"] for r in table), default=None),
                   "maximum_margin": max((r["target"] for r in table), default=None)}
            for name, table in tables["constraints"].items()},
        "constraint_support_interpretation": "Regression observations; counts do not prove boundary coverage",
        "risk_diagnostics": metadata["risk_diagnostics"],
        "response_diagnostics": metadata["response_diagnostics"],
        "calibration_diagnostics": metadata["calibration"],
        "checkpoint_recovery": "passed",
        "full_model_bundle_roundtrip": "passed",
        "input_audit_sha256": hashlib.sha256((folder/"independent_audit.json").read_bytes()).hexdigest(),
        "release_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in output.iterdir() if path.is_file()},
        "label_policy": "Unknown labels retained; no old-contract or controlled data; no imputation",
    }
    save_new(output/"release.json", summary)
    print("Released", len(combined), "unique observations;", len(training), "training;",
          "active mechanisms", list(models[0].active_mechanisms), flush=True)
    print("Training mechanism support", summary["training_mechanism_support"], flush=True)
    print("New MDA cost", report["complete_new_mda_calls"], "additional release cost", 0, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release(args.folder.resolve(), args.output.resolve())
