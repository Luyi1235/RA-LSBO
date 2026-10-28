"""Rebuild saved risk/GP forecasts from checkpoints and past labels only."""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.calibration import CalibratedRiskHeads
from ra_lsbo.mechanism_system import MechanismRiskHeads
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.structured_models import StructuredResponseHeads, StructuredRiskHeads
from ra_lsbo.system_experiment import save_new
from tools.encoder_framework_experiment import geometry
from tools.run_system_validation import read_lines


def assert_close(actual, expected, context):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("Forecast keys differ: "+context)
        for key in expected:
            assert_close(actual[key], expected[key], context+"/"+key)
    elif expected is None:
        if actual is not None:
            raise ValueError("Forecast availability differs: "+context)
    elif not np.allclose(actual, expected, rtol=1e-8, atol=1e-10):
        raise ValueError("Reconstructed forecast differs: "+context)


def rebuild(folder, fit, history, base, seed, method, steps, checkpoint):
    training = [r for r in history if r["evaluation"] in set(fit["training_indices"])]
    if training[-1]["evaluation"] >= fit.get("before_evaluation", 901):
        raise ValueError("Future observation in reconstructed training prefix")
    features = StructuredFeatures.load_checkpoint(folder/checkpoint, base, training)
    risks = StructuredRiskHeads(
        training, features, seed,
        view="hierarchical_onehot" if method == "ra_onehot_risk" else "latent")
    responses = StructuredResponseHeads(training, features, seed, steps)
    required = [name.removeprefix("constraint:") for name in
                fit["risk_diagnostics"]["final_feasibility"]["required_constraint_heads"]]
    combined = MechanismRiskHeads(risks, responses, required, method != "ra_no_mechanisms")
    if combined.model_id != fit["model_id"]:
        raise ValueError("Reconstructed raw-model identity differs")
    calibration = [r for r in history if r["evaluation"] in set(fit["calibration_indices"])]
    calibrated = CalibratedRiskHeads(combined, calibration) if calibration else None
    if (None if calibrated is None else calibrated.calibration_id) != fit["calibration_id"]:
        raise ValueError("Reconstructed calibration identity differs")
    return combined, responses, calibrated


def trial(run, base):
    folder = Path(run["directory"])
    manifest = load_json((folder/"manifest.json").read_bytes())
    initial = load_json(Path(manifest["history_path"]).read_bytes())
    history = initial+read_lines(folder/"observations.jsonl")
    proposals = read_lines(folder/"proposals.jsonl")
    fits = read_lines(folder/"model_fits.jsonl")
    count, structures = 0, []
    for index, fit in enumerate(fits):
        lower = fit["before_evaluation"]
        upper = fits[index+1]["before_evaluation"] if index+1 < len(fits) else len(history)+2
        relevant = [r for r in proposals if lower <= r["evaluation"] < upper
                    and r["selection"].get("forecast") is not None]
        models = rebuild(
            folder, fit, history[:lower-1], base, run["seed"], run["method"],
            manifest["settings"]["gp_steps"], fit["encoder_checkpoint"])
        structures.append({
            "before_evaluation": fit["before_evaluation"],
            "encoder_update": fit["encoder_update"],
            "geometry": geometry(models[0].features)})
        if not relevant:
            continue
        candidates = [r["proposal"]["candidate"] for r in relevant]
        raw = models[0].predict(candidates)
        responses = models[1].predict(candidates)
        cal = None if models[2] is None else models[2].predict(candidates)[
            "risk_probabilities_calibrated"]["final_feasibility"]
        for position, proposal in enumerate(relevant):
            actual = {
                "raw": {name: values[position] for name, values in raw.items()},
                "calibrated_final": None if cal is None else cal[position],
                "responses": {name: None if values is None else
                              {key: value[position] for key, value in values.items()}
                              for name, values in responses.items()},
                "active_mechanisms": list(models[0].active_mechanisms)}
            expected = proposal["selection"]["forecast"]
            if actual.pop("active_mechanisms") != expected["active_mechanisms"]:
                raise ValueError("Reconstructed active mechanisms differ")
            assert_close(actual, {k: v for k, v in expected.items() if k != "active_mechanisms"},
                         run["method"]+"/"+str(proposal["evaluation"]))
            count += 1
    return {"method": run["method"], "seed": run["seed"],
            "kind": run["kind"], "model_fits_rebuilt": len(fits), "selected_forecasts_checked": count,
            "checkpoint_structures": structures}


def static(folder, base):
    report = load_json((folder/"controlled_static_report.json").read_bytes())
    history = load_json((folder/"controlled_static_history.json").read_bytes())[:900]
    forecasts = read_lines(folder/"controlled_forecasts.jsonl")
    checked = 0
    for model in report["models"]:
        method, fit = model["method"], model["metadata"]
        combined, _, _ = rebuild(folder, fit, history, base, 0, method, 45,
                                  method+"_controlled_static.pt")
        selected = [p for p in forecasts if p["method"] == method]
        raw = combined.predict([p["candidate"] for p in selected])
        for position, item in enumerate(selected):
            assert_close({name: raw[name][position] for name in item["raw"]},
                         item["raw"], method+"/static/"+str(item["evaluation"]))
            checked += 1
    return checked


def reconstruct(folder, output, controlled_only=False):
    torch.set_num_threads(1)
    report = load_json((folder/"report.json").read_bytes())
    audit_path = folder/"independent_audit.json"
    audit = load_json(audit_path.read_bytes())
    if report["status"] != "terminal" or audit["status"] != "passed":
        raise ValueError("Require terminal independently audited study")
    verify_sources(audit["input_sha256"])
    plan = load_json((folder/"plan.json").read_bytes())
    verify_sources(plan["source_sha256"])
    base = plan["physical"]["base_project"]
    controlled = folder if controlled_only else Path(report["controlled_source"])
    selected = report["controlled"] + ([] if controlled_only else report["physical"])
    results = []
    for run in selected:
        result = trial(run, base)
        results.append(result)
        print("Rebuilt", {k: v for k, v in result.items() if k != "checkpoint_structures"}, flush=True)
    static_checked = static(controlled, base)
    result = {
        "status": "passed", "additional_mda_calls": 0, "runs": results,
        "static_raw_forecasts_rebuilt": static_checked,
        "relative_tolerance": 1e-8, "absolute_tolerance": 1e-10,
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "geometry_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("encoder_framework_experiment.py").read_bytes()).hexdigest(),
        "input_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest()}
    save_new(output, result)
    output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    print("Forecast reconstruction passed", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controlled-only", action="store_true")
    args = parser.parse_args()
    reconstruct(args.folder.resolve(), args.output.resolve(), args.controlled_only)
