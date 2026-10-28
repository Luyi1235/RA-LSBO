"""Complete the interrupted retrospective diagnostic with explicit simplex normalization."""

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.adaptive_risk import JointRiskEnsemble, JointRiskSettings, ReferenceTemperature
from ra_lsbo.audit import load_json
from ra_lsbo.margin_joint import MarginJointRisk
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.normalized_margin_joint import NormalizedMarginRisk
from ra_lsbo.system_experiment import save_new
from tools.release_system_training_data import load_model_bundle
from tools.run_margin_research import validate


def complete(folder):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    validate(plan)
    reference = Path(plan["retrospective"]["reference_bundle"])
    previous = Path(plan["predecessor"])/"physical_retrospective"
    output = folder/"physical_retrospective_normalized_v2"
    output.mkdir(exist_ok=False)
    source = Path(__file__).resolve()
    adapter = ROOT/"ra_lsbo/normalized_margin_joint.py"
    save_new(output/"repair_plan.json", {
        "additional_real_mda_calls": 0, "parent": str(folder),
        "reason": "Original retrospective binary_metrics rejected a float32 sum slightly greater than one",
        "rule": "Normalize each event member in float64 before calibration and scoring",
        "unchanged": "GP and neural training settings, data split, hidden failure labels, all frozen online trials",
        "scope": "Numerical repair of retrospective probability presentation, not a new optimizer efficacy trial",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, adapter)}})
    _, responses, _ = load_model_bundle(reference)
    rows = load_json((reference/"observations.json").read_bytes())
    spec = load_json((reference/"bundle_spec.json").read_bytes())
    training = [r for r in rows if r["evaluation"] in set(spec["training_indices"])]
    calibration = [r for r in rows if r["evaluation"] in set(spec["calibration_indices"])]
    test = load_json((previous/"joint_full_forecasts.json").read_bytes())["test"]
    candidates, labels = [r["candidate"] for r in test], [r["label"] for r in test]
    settings = JointRiskSettings()
    variants = {"joint_full": settings, "joint_aggregate": replace(settings, aggregate=True),
                "joint_onehot": replace(settings, latent=False),
                "joint_no_auxiliary": replace(settings, auxiliary=False)}
    metrics = []
    for name, risk_settings in variants.items():
        terminal = JointRiskEnsemble(training, responses.features, seed=0, settings=risk_settings)
        base = MarginJointRisk(terminal, responses)
        bridge = NormalizedMarginRisk(base)
        temperature = ReferenceTemperature(bridge, calibration)
        predicted = bridge.predict(candidates)
        transformed = temperature.transform(predicted["probabilities"], bridge.model_id)
        base.save_checkpoint(output/(name+".pt"))
        restored = NormalizedMarginRisk(MarginJointRisk.load_checkpoint(
            output/(name+".pt"), responses.features, training))
        np.testing.assert_allclose(restored.predict(candidates)["probabilities"], predicted["probabilities"],
                                   atol=1e-7, rtol=1e-6)
        if predicted["maximum_member_normalization_delta"] > 2e-6:
            raise ValueError("Repair exceeds declared numerical tolerance")
        metric = {"method": name, "raw": binary_metrics(predicted["final_feasibility"], labels),
                  "calibrated": binary_metrics(transformed[:, 0].tolist(), labels),
                  "calibration": temperature.to_dict(), "diagnostics": bridge.diagnostics,
                  "adapter_id": bridge.model_id, "base_bridge_id": base.model_id,
                  "maximum_normalization_delta": predicted["maximum_member_normalization_delta"]}
        metrics.append(metric)
        save_new(output/(name+"_forecasts.json"), {
            "model": metric, "raw": predicted, "transformed": transformed.tolist(), "test": test})
        print("Normalized bridge historical risk", name, metric["raw"]["brier"],
              metric["calibrated"]["brier"], metric["maximum_normalization_delta"], flush=True)
    save_new(output/"diagnostics.json", {
        "status": "completed", "additional_real_mda_calls": 0, "fixed_design_rows": len(test),
        "fixed_design_results": metrics, "training_count": len(training), "reference_count": len(calibration),
        "unchanged_hidden_transfer": str(previous/"diagnostics.json"),
        "limitations": "Numerically repaired retrospective development; no new physical or controlled optimizer trial"})


ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    complete(args.folder.resolve())
