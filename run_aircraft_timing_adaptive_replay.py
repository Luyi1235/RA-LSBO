"""Exploratory four-query replay on the sealed PR125 screened candidate panels."""

import argparse
import json
from pathlib import Path
from statistics import mean

from ra_lsbo.budget import digest
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import ModelNumericalFailure
from run_aircraft_initial_budget import paired_interval, save, select
from run_aircraft_initial_feedback_diagnostic import passes_structural_screen
from run_aircraft_initial_timing_diagnostic import METHODS, completion_probabilities
from run_aircraft_initial_timing_validation import (
    TIMING_REASONS, load_validation, validated_seed_panels,
)
from run_aircraft_risk_cold_start import discovery_outcome, file_hash


INPUT_CODE_SHA = "e8c0158996e53e68bfaa1390a17c35f057dd76cf"
INITIAL = 8
QUERY_BUDGET = 4


def simulate(problem, history, panel, seed, method, initial_values):
    observed = list(history)
    remaining = list(panel)
    trace = []
    first = None
    for query in range(1, QUERY_BUDGET + 1):
        if not remaining:
            break
        designs = [row.candidate for row in remaining]
        if query == 1:
            probabilities = initial_values
        else:
            forecasts, _ = completion_probabilities(problem, observed, designs, seed,
                                                    methods=(method,))
            probabilities = forecasts[method]
        index = select(designs, probabilities)
        record = remaining.pop(index)
        trace.append({"query": query, "design_id": record.observation.design_id,
                      "completion_probability": float(probabilities[index]),
                      "outcome": discovery_outcome(record.to_dict()),
                      "history_digest": digest([row.to_dict() for row in observed])})
        if record.usable_feasible:
            first = query
            break
        observed.append(record)
    return {
        "first_usable_query": first, "capped_first": first or QUERY_BUDGET + 1,
        "discovered": first is not None, "queries": len(trace), "trace": trace,
    }


def summarize(rows):
    pairs = [row["methods"] for row in rows if "methods" in row]
    metrics = {"n": len(pairs), "excluded_n": len(rows) - len(pairs)}
    if not pairs:
        return metrics
    metrics["methods"] = {
        name: {
            "discovered": sum(row[name]["discovered"] for row in pairs),
            "mean_capped_first": mean(row[name]["capped_first"] for row in pairs),
            "replayed_queries": sum(row[name]["queries"] for row in pairs),
        } for name in METHODS
    }
    metrics["interval_minus_control"] = {}
    for control in ("timing_sign", "original_probit"):
        contrasts = {
            key: [int(row["timing_interval"][key]) - int(row[control][key]) for row in pairs]
            for key in ("discovered", "capped_first")
        }
        metrics["interval_minus_control"][control] = {
            key: {"mean": mean(deltas), "paired_95": paired_interval(deltas),
                  "positive": sum(delta > 0 for delta in deltas),
                  "negative": sum(delta < 0 for delta in deltas)}
            for key, deltas in contrasts.items()
        }
    return metrics


def analyze(folder):
    spec, problem, calls = load_validation(folder, INPUT_CODE_SHA)
    rows = []
    initially_solved = 0
    for seed, history, panel in validated_seed_panels(spec, problem, calls):
        history = history[:INITIAL]
        if any(record.usable_feasible for record in history):
            initially_solved += 1
            continue
        screened = [record for record in panel if passes_structural_screen(record)]
        row = {"seed": seed, "screened_candidates": len(screened),
               "screened_has_usable": any(record.usable_feasible for record in screened)}
        if any(record.diagnostics["termination_reason"] not in TIMING_REASONS
               for record in history + screened):
            rows.append({**row, "exclusion": "unsupported_termination"})
            continue
        if any(discovery_outcome(record.to_dict()) not in {"abort", "violation", "usable"}
               for record in screened):
            rows.append({**row, "exclusion": "unresolved_panel_outcome"})
            continue
        try:
            initial_values = {name: [] for name in METHODS}
            if screened:
                initial_values, _ = completion_probabilities(
                    problem, history, [record.candidate for record in screened], seed)
            methods = {name: simulate(problem, history, screened, seed, name, initial_values[name])
                       for name in METHODS}
        except ModelNumericalFailure as exc:
            rows.append({**row, "exclusion": "model_numerical_failure", "head": exc.head})
            continue
        rows.append({**row, "methods": methods})
        print(f"{spec['name']} {seed} replayed", flush=True)
    return {
        "status": "COMPLETE_TIMING_ADAPTIVE_REPLAY",
        "profile": spec["name"], "archive_source_sha": INPUT_CODE_SHA,
        "archive_seal_sha256": file_hash(folder / "seal.json"),
        "archive_calls": len(calls), "new_aircraft_calls": 0, "new_rcaide_mda_calls": 0,
        "initial": INITIAL, "query_budget": QUERY_BUDGET, "initially_solved": initially_solved,
        "cold": summarize(rows), "per_seed": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(folder.resolve() in args.output.resolve().parents for folder in args.input):
        parser.error("Write the output outside the sealed evidence directory")
    results = [analyze(folder) for folder in args.input]
    save(args.output, results)
    print(json.dumps({"profiles": [r["profile"] for r in results], "new_calls": 0}))
