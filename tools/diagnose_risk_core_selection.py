"""Post-selection descriptive diagnostics from saved pre-outcome predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def diagnose(folder):
    audit = load(folder/"audit.json")
    if audit["status"] != "passed":
        raise ValueError("A complete audit is required")
    for name, expected in audit["evidence_sha256"].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Evidence changed after audit: "+name)
    runs = []
    for summary in lines(folder/"completed.jsonl"):
        if summary["method"] == "lsbo":
            continue
        directory = folder/summary["phase"]/summary["task"]/(
            summary["method"]+"_"+str(summary["seed"]))
        proposals, observations = lines(directory/"proposals.jsonl"), lines(directory/"observations.jsonl")
        feasibility, completion, log_losses = [], [], []
        numeric = {}
        fallback, censored = 0, 0
        for proposal, observation in zip(proposals, observations):
            metadata, truth = proposal["metadata"], observation["truth"]["risk"]
            p = metadata["raw_independent_feasibility"]
            y = float(truth["feasible_label"])
            feasibility.append((p-y)**2)
            bounded = np.clip(p, 1e-12, 1-1e-12)
            log_losses.append(-y*np.log(bounded)-(1-y)*np.log(1-bounded))
            completion.append((metadata["selected_completion_probability"]-truth["hidden_success"])**2)
            fallback += metadata["fallback"] is not None
            observed = observation["observed"]["risk"]
            censored += len(observed["evidence"]["verified_constraint_bounds"])
            for name, forecast in metadata["calibration_forecast"].items():
                record = numeric.setdefault(name, {
                    "exact_feedback": 0, "finite_upper": 0, "upper_miscoverage": 0,
                    "finite_upper_exact_feedback": 0, "finite_upper_miscoverage": 0})
                record["finite_upper"] += forecast["upper"] is not None
                if observed["constraint_observed"][name]:
                    record["exact_feedback"] += 1
                    # Null is an explicitly unsupported infinite upper, not a measured zero.
                    if forecast["upper"] is not None:
                        record["finite_upper_exact_feedback"] += 1
                        error = observed["constraints"][name] > forecast["upper"]
                        record["upper_miscoverage"] += error
                        record["finite_upper_miscoverage"] += error
        runs.append({
            "phase": summary["phase"], "task": summary["task"],
            "seed": summary["seed"], "method": summary["method"],
            "feasibility_brier": float(np.mean(feasibility)),
            "completion_brier": float(np.mean(completion)),
            "feasibility_log_loss": float(np.mean(log_losses)),
            "fallback_calls": fallback, "censored_output_records": censored,
            "numeric_calibration": numeric,
        })
    aggregate = []
    groups = sorted({(r["phase"], r["task"], r["method"]) for r in runs})
    for phase, task, method in groups:
        values = [r for r in runs if (r["phase"], r["task"], r["method"]) == (phase, task, method)]
        aggregate.append({
            "phase": phase, "task": task, "method": method, "n_runs": len(values),
            **{key: float(np.mean([r[key] for r in values]))
               for key in ("feasibility_brier", "completion_brier", "feasibility_log_loss")},
            "fallback_calls": sum(r["fallback_calls"] for r in values),
            "censored_output_records": sum(r["censored_output_records"] for r in values),
        })
    result = {
        "scope": "descriptive post-selection diagnostics; not used to select or tune methods",
        "evaluand": "saved chosen-point forecasts, before outcome; different policies sample different designs",
        "limitations": [
            "feasibility Brier uses the saved completion-times-numeric-GP proxy; it is not the ensemble boundary-band probability",
            "no predictive superiority inference across differently selected design distributions",
            "no common-probability output is extracted from the native LSBO baseline",
            "numeric residual coverage concerns exact observed subsets; not safety or censored-population coverage",
            "four run-level replicates per task; individual calls are not independent replicates"],
        "aggregate": aggregate, "runs": runs, "new_truth_labels": 0, "new_physical_mda_calls": 0,
    }
    with (folder/"prediction_diagnostics.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    diagnose(parser.parse_args().output.resolve())
