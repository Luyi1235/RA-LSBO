"""Second, separately frozen development study: conditional margin/event bridge."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch
import uuid

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.adaptive_risk import JointRiskEnsemble, JointRiskSettings, ReferenceTemperature
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.margin_joint import MarginJointRisk, MarginLearning
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.system_experiment import append_json, sample_rows, save_new
from tools import audit_adaptive_research as independent
from tools import run_adaptive_research as core
from tools.release_system_training_data import load_model_bundle


ROOT, SEEDS = Path(__file__).resolve().parents[1], (71, 72, 73)


def validate(plan):
    if (plan["schema"] != "ra_margin_development_v1" or plan["additional_real_mda_calls"] != 0
            or plan["settings"] != core.SETTINGS or plan["methods"] != list(core.METHODS)
            or plan["cases"] != list(core.CASES) or plan["seeds"] != list(SEEDS)
            or plan["max_controlled_online_calls"] != 1584):
        raise ValueError("Margin development protocol differs")
    verify_sources(plan["source_sha256"])


def prepare(parent):
    previous = load_json((parent/"plan.json").read_bytes())
    core.validate_plan(previous)
    if load_json((parent/"independent_audit.json").read_bytes())["status"] != "passed":
        raise ValueError("First development study must be audited")
    folder = ROOT/"runs"/("margin_research_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                          +"_"+uuid.uuid4().hex[:8])
    folder.mkdir(exist_ok=False)
    paths = [*sorted((ROOT/"ra_lsbo").glob("*.py")), Path(__file__).resolve(),
             ROOT/"tools/run_adaptive_research.py", ROOT/"tools/audit_adaptive_research.py",
             ROOT/"docs/MARGIN_JOINT_PROTOCOL_20261021.md"]
    sources = dict(previous["source_sha256"])
    sources.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    plan = dict(previous, schema="ra_margin_development_v1", seeds=list(SEEDS),
                predecessor=str(parent), source_sha256=sources,
                created_utc=datetime.now(timezone.utc).isoformat(),
                method_family="conditional_margin_joint_v1",
                scope="Second development revision on known equations, fresh seeds; not confirmation")
    save_new(folder/"plan.json", plan)
    snapshot = folder/"source_snapshot"
    snapshot.mkdir()
    for i, path in enumerate(paths):
        (snapshot/(str(i)+"_"+path.name)).write_bytes(path.read_bytes())
    print("Prepared margin bridge", folder, flush=True)
    return folder


def run_trial(folder, case, method, seed):
    torch.set_num_threads(1)
    folder = Path(folder)
    plan = load_json((folder/"plan.json").read_bytes())
    validate(plan)
    problem = core.problem_for(case, plan["base_project"])
    if problem.contract_id != plan["contracts"][case]:
        raise ValueError("Controlled truth changed")
    output = folder/case/(method+"_"+str(seed))
    output.mkdir(parents=True, exist_ok=False)
    # Inject only the experimental learner; the frozen loop and four baselines are unchanged.
    with patch.object(core, "AdaptiveLearning", MarginLearning), (
            output/"execution.log").open("x", encoding="utf-8") as stream, redirect_stdout(stream):
        return core._trial(plan, folder, output, problem, case, method, seed)


class BridgeCheckpointLoader:
    @staticmethod
    def load_checkpoint(path, features):
        path = Path(path)
        manifest = load_json((path.parent/"manifest.json").read_bytes())
        initial = load_json((path.parent.parent/("initial_"+str(manifest["seed"])+".json")).read_bytes())
        observations = core.lines(path.parent/"observations.jsonl")
        eligible = [r for r in initial+observations if r["evaluation"] in set(features.training_indices)]
        return MarginJointRisk.load_checkpoint(path, features, eligible)


def analyze(folder):
    with patch.object(core, "validate_plan", validate), patch.object(core, "SEEDS", SEEDS), \
            patch.object(core, "JointRiskEnsemble", BridgeCheckpointLoader):
        core.analyze(folder)
    with patch.object(independent, "validate_plan", validate):
        independent.audit(folder)


def retrospective(folder):
    plan = load_json((folder/"plan.json").read_bytes())
    validate(plan)
    reference = Path(plan["retrospective"]["reference_bundle"])
    previous = Path(plan["predecessor"])/"physical_retrospective"
    old, responses, _ = load_model_bundle(reference)
    rows = load_json((reference/"observations.json").read_bytes())
    spec = load_json((reference/"bundle_spec.json").read_bytes())
    training = [r for r in rows if r["evaluation"] in set(spec["training_indices"])]
    calibration = [r for r in rows if r["evaluation"] in set(spec["calibration_indices"])]
    test = load_json((previous/"joint_full_forecasts.json").read_bytes())["test"]
    output = folder/"physical_retrospective"
    output.mkdir(exist_ok=False)
    candidates, labels = [r["candidate"] for r in test], [r["label"] for r in test]
    settings = JointRiskSettings()
    variants = {"joint_full": settings, "joint_aggregate": replace(settings, aggregate=True),
                "joint_onehot": replace(settings, latent=False),
                "joint_no_auxiliary": replace(settings, auxiliary=False)}
    metrics = []
    for name, risk_settings in variants.items():
        terminal = JointRiskEnsemble(training, responses.features, seed=0, settings=risk_settings)
        bridge = MarginJointRisk(terminal, responses)
        temperature = ReferenceTemperature(bridge, calibration)
        predicted = bridge.predict(candidates)
        transformed = temperature.transform(predicted["probabilities"], bridge.model_id)
        bridge.save_checkpoint(output/(name+".pt"))
        restored = MarginJointRisk.load_checkpoint(output/(name+".pt"), responses.features, training)
        np.testing.assert_allclose(restored.predict(candidates)["probabilities"], predicted["probabilities"],
                                   atol=1e-7, rtol=1e-6)
        metric = {"method": name, "raw": binary_metrics(predicted["final_feasibility"], labels),
                  "calibrated": binary_metrics(transformed[:, 0].tolist(), labels),
                  "calibration": temperature.to_dict(), "diagnostics": bridge.diagnostics}
        metrics.append(metric)
        save_new(output/(name+"_forecasts.json"), {
            "model": metric, "raw": predicted, "transformed": transformed.tolist(), "test": test})
        print("Bridge historical risk", name, metric["raw"]["brier"], metric["calibrated"]["brier"], flush=True)
    save_new(output/"diagnostics.json", {
        "status": "completed", "additional_real_mda_calls": 0, "fixed_design_rows": len(test),
        "fixed_design_results": metrics, "training_count": len(training), "reference_count": len(calibration),
        "unchanged_hidden_transfer": str(previous/"diagnostics.json"),
        "limitations": "Retrospective development; unchanged hidden mechanism mass; no physical optimizer trial"})


def execute(folder):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    validate(plan)
    save_new(folder/"execution_started.json", {"created_utc": datetime.now(timezone.utc).isoformat()})
    for case in plan["cases"]:
        (folder/case).mkdir()
        problem = core.problem_for(case, plan["base_project"])
        for seed in SEEDS:
            save_new(folder/case/("initial_"+str(seed)+".json"), sample_rows(problem, plan["initial_count"], seed))
            save_new(folder/case/("test_"+str(seed)+".json"), sample_rows(problem, plan["static_test_count"], seed+5000))
    with ProcessPoolExecutor(max_workers=plan["parallel_workers"]) as executor:
        jobs = [executor.submit(run_trial, str(folder), case, method, seed)
                for case in plan["cases"] for seed in SEEDS for method in plan["methods"]]
        for i, job in enumerate(as_completed(jobs)):
            result = job.result()
            append_json(folder/"completed.jsonl", result)
            print(i+1, "/", len(jobs), result["case"], result["method"], result["seed"],
                  result["status"], result["evaluations"], result["improvement_percent"], flush=True)
    analyze(folder)
    retrospective(folder)


if __name__ == "__main__":
    import os
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "analyze"))
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--parent", type=Path, default=ROOT/"runs/adaptive_research_20261021T071809Z_6fba5696")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.parent.resolve())
    else:
        if args.folder is None:
            parser.error("--folder required")
        (execute if args.action == "execute" else analyze)(args.folder.resolve())
