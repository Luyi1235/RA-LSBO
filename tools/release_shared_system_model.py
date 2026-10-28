"""Release the common pre-optimization model with its fixed held-out block."""

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
from ra_lsbo.calibration import observed_target
from ra_lsbo.features import history_identity
from ra_lsbo.mechanism_system import FullSystemLearning, SystemSettings, mechanism_support
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.system_experiment import save_new
from ra_lsbo.training import training_rows
from ra_lsbo.x57_live import X57LiveProblem
from tools.release_system_training_data import save_model_weights


def release(folder, output):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    report = load_json((folder/"report.json").read_bytes())
    audit = load_json((folder/"independent_audit.json").read_bytes())
    if report["status"] != "terminal" or audit["status"] != "passed":
        raise ValueError("Require terminal independently audited study")
    verify_sources(plan["source_sha256"])
    verify_sources(audit["input_sha256"])
    shared = load_json((folder/"shared_history.json").read_bytes())
    training = load_json((folder/"precalibration_training.json").read_bytes())
    window = plan["fresh_calibration_calls"]
    def observed_sequence(rows):
        return [(r["evaluation"], r["candidate"], r["risk"]) for r in rows]

    if (report["fresh_calibration_calls"] != window
            or observed_sequence(shared[:-window]) != observed_sequence(training)):
        raise ValueError("Common training prefix and fresh held-out block differ")
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                            output/"unused", allow_mda=False)
    if problem.contract_id != plan["physical"]["contract_id"]:
        raise ValueError("Physical contract differs")
    learner = FullSystemLearning("ra_full", SystemSettings(**plan["settings"]), seed=0)
    models, metadata = learner.fit(
        shared, [asdict(v) for v in problem.space.variables], plan["physical"]["base_project"],
        sorted(set(problem.required_constraints)-set(problem.analytical_constraints)))
    output.mkdir(parents=True, exist_ok=False)
    save_new(output/"observations.json", shared)
    save_new(output/"model_metadata.json", metadata)
    save_new(output/"training_tables.json", training_rows(training))
    learner.features.save_checkpoint(output/"encoder.pt")
    sources = dict(plan["source_sha256"], **{
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    save_model_weights(output, models, metadata, plan["physical"]["base_project"], shared, sources)
    calibration = shared[-window:]
    raw = models[0].predict([r["candidate"] for r in calibration])
    raw_scores = {}
    for name, values in raw.items():
        known = [(p, observed_target(r["risk"], name)) for r, p in zip(calibration, values)
                 if observed_target(r["risk"], name) is not None]
        fitted = metadata["risk_diagnostics"][name]["status"] == "fitted_raw"
        raw_scores[name] = dict(
            base_fitted=fitted, known_labels=len(known), positive=sum(y for p, y in known),
            scores=(binary_metrics([p for p, y in known], [y for p, y in known]) if fitted else None))
    summary = dict(
        schema="ra_common_history_reference_bundle_v1", status="completed",
        created_utc=datetime.now(timezone.utc).isoformat(), additional_mda_calls=0,
        interpretation="Common-history seed-0 reference model; optimizer observations excluded",
        contract_id=problem.contract_id, label_contract_id=plan["label_contract_id"],
        historical_mda_cost=420,
        shared_acquisition_mda_cost=report["probe_calls_including_carried"]+window,
        unique_observations=len(shared), training_observations=len(training),
        calibration_observations=window, history_id=history_identity(shared),
        precalibration_observation_sequence_verified=True,
        precalibration_source_history_id=history_identity(training),
        training_support=mechanism_support(training),
        calibration_support=mechanism_support(shared[-window:]),
        active_mechanisms=list(models[0].active_mechanisms),
        raw_held_out_metrics=raw_scores,
        held_out_metric_scope="Raw models exclude all calibration labels; calibrated scores are not tested on their fit block",
        calibration_diagnostics=metadata["calibration"],
        full_model_bundle_roundtrip="passed",
        scope="Directed held-out acquisition; no calibration-fit-block generalization score",
        input_audit_sha256=hashlib.sha256((folder/"independent_audit.json").read_bytes()).hexdigest(),
        output_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in output.iterdir() if p.is_file()})
    save_new(output/"release.json", summary)
    (output/"release.source.py").write_bytes(Path(__file__).read_bytes())
    print("Shared reference released", len(training), "training", window, "calibration",
          summary["active_mechanisms"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    release(args.folder.resolve(), args.output.resolve())
