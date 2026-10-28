"""Retrospective masked-ICM diagnostic from existing, disjoint physical records."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import digest
from ra_lsbo.joint_constraints import JointConstraintPosterior, JointConstraintSettings
from ra_lsbo.structured_encoder import EncoderSettings, StructuredFeatures
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem

OLD = ROOT/"runs/system_validation_v5_20261020T192312Z_5f6531eb"
BASE = (ROOT/"runs/compact_mda_20261021T081006Z_20eb608d/"
        "qualification_recovered_20261025/frozen_base")
ENCODER = EncoderSettings(initial_epochs=600)
TRAIN_COUNT, STEPS, SAMPLES = 128, 60, 4096


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_data():
    shared_path = ROOT/"runs/system_shared_reference_v5_20261021/observations.json"
    shared = load_json(shared_path.read_bytes())
    rows = shared[:TRAIN_COUNT]
    problem = X57LiveProblem(BASE, OLD/"frozen_config.json", ROOT/"runs/unused_joint_diagnostic", False)
    old_plan = load_json((OLD/"plan.json").read_bytes())
    if problem.contract_id != old_plan["physical"]["contract_id"]:
        raise ValueError("Recovered original mission identity differs")
    if any(r["evaluator_contract_sha256"] != problem.contract_id for r in rows):
        raise ValueError("Mixed training physical identity")
    key = lambda raw: problem.space.key(problem.candidate_class(**raw))
    seen_train = {key(r["candidate"]) for r in rows}
    heldout, files, duplicates, overlaps = {}, [shared_path, OLD/"report.json", OLD/"frozen_config.json"], 0, 0
    report = load_json((OLD/"report.json").read_bytes())
    for run in report["physical"]:
        path = Path(run["directory"])/"observations.jsonl"
        files.append(path)
        for line in path.read_bytes().splitlines():
            row = load_json(line)
            identity = key(row["candidate"])
            if identity in seen_train:
                overlaps += 1
                continue
            if identity in heldout:
                prior = heldout[identity]["risk"]
                if (prior["feasible_label"] != row["risk"]["feasible_label"]
                        or prior["hidden_success"] != row["risk"]["hidden_success"]):
                    raise ValueError("Conflicting repeated physical outcome")
                duplicates += 1
            else:
                heldout[identity] = row
    return problem, rows, list(heldout.values()), files, duplicates, overlaps


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=False)
    problem, rows, heldout, files, duplicates, overlaps = input_data()
    paths = [Path(__file__).resolve(), *sorted((ROOT/"ra_lsbo").glob("*.py")), *files]
    paths += [p for p in BASE.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    save_new(folder/"plan.json", {
        "schema": "historical_masked_icm_diagnostic_v1", "date": "2026-10-25",
        "training_count": TRAIN_COUNT, "training_history_id": digest(rows),
        "heldout_unique": len(heldout), "consistent_repetitions_removed": duplicates,
        "training_overlap_records_excluded": overlaps, "seed": 0,
        "ranks": [0, 2], "gp_steps": STEPS, "mc_samples": SAMPLES,
        "encoder_settings": asdict(ENCODER), "physical_contract": problem.contract_id,
        "new_mda_calls": 0, "new_truth_calls": 0,
        "scope": "retrospective diagnostic on repeatedly-inspected data; no optimization efficacy claim",
        "control": "same inputs, normalization, train masks and shared kernel; only ICM rank differs",
        "primary_score": "mean predictive joint Gaussian negative log density on fully observed constraint vectors",
        "secondary_score": "raw latent joint feasibility Brier on fully observed vectors",
        "source_sha256": {str(p): sha(p) for p in paths}})
    print(folder/"plan.json", flush=True)


def execute(folder):
    plan = load_json((folder/"plan.json").read_bytes())
    verify_sources(plan["source_sha256"])
    if (plan["gp_steps"] != STEPS or plan["training_count"] != TRAIN_COUNT
            or plan["ranks"] != [0, 2] or plan["mc_samples"] != SAMPLES):
        raise ValueError("Joint GP diagnostic scope differs")
    problem, rows, heldout, _, _, _ = input_data()
    if digest(rows) != plan["training_history_id"] or len(heldout) != plan["heldout_unique"]:
        raise ValueError("Historical diagnostic data changed")
    save_new(folder/"execution_started.json", {"seed": 0})
    features = StructuredFeatures(BASE, [asdict(v) for v in problem.space.variables],
                                  0, ENCODER).fit(rows)
    features.save_checkpoint(folder/"encoder.pt")
    records = []
    for rank in plan["ranks"]:
        joint = JointConstraintPosterior(
            rows, features, problem.analytical_constraints, seed=0,
            settings=JointConstraintSettings(steps=STEPS, rank=rank))
        joint.require_supported()
        numeric = [r for r in heldout if all(r["risk"]["constraint_observed"][n] for n in joint.names)]
        if not numeric:
            raise ValueError("No fully observed held-out numeric vectors")
        nll, brier, predictions = [], [], []
        scales = torch.tensor(joint.scales, dtype=torch.double)
        for start in range(0, len(numeric), 16):
            block = numeric[start:start+16]
            candidates = [r["candidate"] for r in block]
            mean, covariance = joint.pointwise_moments(candidates)
            jitter = torch.diag(scales.square()*1e-8)
            latent = torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance+jitter)
            observed_covariance = covariance+torch.diag(
                scales.square()*float(joint.model.likelihood.noise.detach()))+jitter
            predictive = torch.distributions.MultivariateNormal(mean, covariance_matrix=observed_covariance)
            targets = torch.tensor([[r["risk"]["constraints"][n] for n in joint.names] for r in block],
                                   dtype=torch.double)
            losses = -predictive.log_prob(targets)
            with torch.random.fork_rng():
                torch.manual_seed(97000+start)
                samples = latent.sample(torch.Size([SAMPLES]))
            probability = (samples <= 0).all(-1).double().mean(0)
            outcomes = (targets <= 0).all(-1).double()
            errors = (probability-outcomes).square()
            nll.extend(losses.tolist())
            brier.extend(errors.tolist())
            for i, row in enumerate(block):
                predictions.append({
                    "candidate_key": digest(row["candidate"]),
                    "observed_constraints": targets[i].tolist(),
                    "mean": mean[i].tolist(), "latent_covariance": covariance[i].tolist(),
                    "raw_joint_feasibility": float(probability[i]),
                    "observed_joint_satisfaction": bool(outcomes[i]),
                    "predictive_joint_nll": float(losses[i]), "brier": float(errors[i]),
                })
        checkpoint = folder/("joint_rank_"+str(rank)+".pt")
        joint.save_checkpoint(checkpoint)
        restored = JointConstraintPosterior.load_checkpoint(checkpoint, features, rows)
        check_candidates = [r["candidate"] for r in numeric[:4]]
        torch.testing.assert_close(joint.sample(check_candidates, 98, 2),
                                   restored.sample(check_candidates, 98, 2))
        normalized = joint.model.task_covar_module.covar_matrix.to_dense().detach()
        sd = normalized.diag().sqrt()
        correlation = normalized/sd[:, None]/sd[None, :]
        record = {"rank": rank, "model_id": joint.model_id, "numeric_holdout_count": len(numeric),
                  "constraint_names": joint.names, "train_observations": joint.diagnostics,
                  "mean_predictive_joint_nll": float(np.mean(nll)),
                  "raw_conditional_feasibility_brier": float(np.mean(brier)),
                  "task_kernel_correlation": correlation.tolist(),
                  "initial_training_loss": joint.losses[0], "final_training_loss": joint.losses[-1],
                  "checkpoint_roundtrip": "passed", "predictions": predictions}
        save_new(folder/("rank_"+str(rank)+".json"), record)
        records.append(record)
        print(rank, len(numeric), record["mean_predictive_joint_nll"],
              record["raw_conditional_feasibility_brier"], flush=True)
    save_new(folder/"result.json", {
        "status": "complete", "new_mda_calls": 0, "new_truth_calls": 0,
        "training_count": len(rows), "unique_holdout_count": len(heldout),
        "encoder_training_excluded_all_holdout": True,
        "plan_sha256": sha(folder/"plan.json"),
        "models": [{k: v for k, v in r.items() if k != "predictions"} for r in records],
        "limits": [
            "One fixed historical training prefix and one model seed; no significance claim.",
            "Both variants share an input kernel; this diagonal control is not the old separate-ARD GP baseline.",
            "Predictive numeric score includes fitted surrogate noise; raw feasibility uses latent covariance.",
            "Fully observed vectors are a completion-conditional subset, not a missing-at-random sample.",
            "ICM correlation is a fitted model assumption, not verified physical dependence.",
            "No independent new task, no probability calibration, no optimization loop test.",
        ]})


if __name__ == "__main__":
    import os
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--folder", required=True, type=Path)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else execute)(args.folder.resolve())
