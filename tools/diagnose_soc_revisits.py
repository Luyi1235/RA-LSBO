"""Read frozen forecasts to diagnose SOC revisits without new evaluations."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.selector import feasibility_probability
from ra_lsbo.risk_core.mixed_gp import eicb_factor
from ra_lsbo.risk_core.optimizer import expected_improvement
from tools.run_partial_bo_development import lines, read, sha, write_new, verify_sources


def diagnose(source, output):
    plan = read(source/"plan.json")
    verify_sources(plan)
    rows, bindings = [], {str(source/"plan.json"): sha(source/"plan.json")}
    for folder in sorted((source/"optimization/aircraft_140km").iterdir()):
        if folder.name.startswith("original_"):
            continue
        specification = read(folder/"manifest.json")
        proposals, observations = lines(folder/"proposals.jsonl"), lines(folder/"observations.jsonl")
        history = list(specification["initial"])
        for name in ("manifest.json", "proposals.jsonl", "observations.jsonl"):
            bindings[str(folder/name)] = sha(folder/name)
        for proposal, observation in zip(proposals, observations):
            metadata = proposal["metadata"]
            if proposal["prefix_digest"] != digest(history):
                raise ValueError("Diagnosis prefix mismatch")
            index = next(index for index, candidate in enumerate(metadata["pool"])
                         if digest(candidate) == digest(proposal["candidate"]))
            forecast = metadata["forecasts"]
            soc = forecast["constraints"]["soc"]
            mean, variance = np.asarray(soc["mean"]), np.asarray(soc["variance"])
            probability = feasibility_probability(mean, variance)
            factor = eicb_factor(mean, variance)
            candidates = metadata["heads"]["soc"]["candidates"]
            evidence = np.array([candidate["log_evidence"] if candidate["status"] == "ok"
                                 else -np.inf for candidate in candidates])
            weights = softmax(evidence)
            objective = forecast["objective"]
            improvement = (expected_improvement(np.asarray(objective["mean"]),
                np.asarray(objective["variance"]), metadata["best_observed"])
                if objective is not None and metadata["best_observed"] is not None else None)
            raw_score = np.log(np.maximum(metadata["completion_probability"], 1e-300))
            for constraint in forecast["constraints"].values():
                raw_score += np.log(np.maximum(feasibility_probability(
                    constraint["mean"], constraint["variance"]), 1e-300))
            exploitation = metadata["mode"] == "balanced_feasible_improvement"
            if exploitation:
                raw_score += np.log(np.maximum(improvement, 1e-300))
            counterfactual = int(np.argmax(np.where(metadata["eligible"], raw_score, -np.inf)))
            key = digest(proposal["candidate"]["discrete"])
            previous_failed = [record for record in history
                if record["diagnostics"].get("termination_reason") == "physical_soc_reserve_abort"
                and digest(record["candidate"]["discrete"]) == key]
            rows.append({"method": specification["method"], "seed": specification["seed"],
                "iteration": proposal["iteration"], "candidate": proposal["candidate"],
                "termination": observation["diagnostics"]["termination_reason"],
                "soc_mean": mean[index], "soc_sd": np.sqrt(variance[index]),
                "soc_probability": probability[index], "soc_balanced_factor": factor[index],
                "objective_ei": None if improvement is None else improvement[index],
                "selected_soc_kernel": metadata["heads"]["soc"]["selected"],
                "mode": metadata["mode"], "same_branch_prior_soc_aborts": len(previous_failed),
                "kernel_evidence_gap": np.sort(evidence)[-1]-np.sort(evidence)[-2],
                "kernel_max_weight": weights.max(), "kernel_entropy": -np.sum(
                    weights*np.log(np.maximum(weights, 1e-300))),
                "probability_only_same_pool_changes_choice": counterfactual != index if exploitation else None,
                "counterfactual_prediction_only": True,
                "counterfactual_candidate": metadata["pool"][counterfactual] if exploitation else None})
            history.append(observation)
    failing = [row for row in rows if row["method"] == "ra_grouped_local" and row["seed"] == 8202]
    prior_like = [row for row in failing if row["termination"] == "physical_soc_reserve_abort"
                  and abs(row["soc_probability"]-.5) < .01]
    summary = {"target_seed_soc_aborts": sum(row["termination"] == "physical_soc_reserve_abort"
        for row in failing), "target_seed_prior_like_soc_aborts": len(prior_like),
        "prior_like_balanced_factors": [row["soc_balanced_factor"] for row in prior_like],
        "target_exploitation_same_pool_switches": sum(bool(row["probability_only_same_pool_changes_choice"])
            for row in failing),
        "interpretation": "Observed mechanism and prediction-only counterfactual; no causal outcome claim"}
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"diagnosis.json", {"summary": summary, "rows": rows, "sources": bindings,
        "tool_sha256": sha(__file__), "new_truth_calls": 0, "new_rcaide_mda_calls": 0,
        "cohort": "Historical development trajectories, not independent validation"})
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    diagnose(arguments.source, arguments.output)
