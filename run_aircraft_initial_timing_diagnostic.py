"""Compare SOC timing intervals and matched binary feedback on sealed PR122 panels."""
import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.special import ndtr

from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import StableGroupedHead
from ra_lsbo.partial_bo.models import PrefixFeatures
from run_aircraft_initial_budget import brier_by_configuration, paired_interval, save
from run_aircraft_initial_feedback_diagnostic import (
    SOURCE_SHA, archived_seed_panels, first_ranking, load_pr122_profile, passes_structural_screen,
)
from run_aircraft_risk_cold_start import discovery_outcome, file_hash
from run_aircraft_risk_timing_replay import MARGIN_SCALE, timing_feedback

METHODS = ("timing_interval", "timing_sign", "original_probit")


def fit_completion_heads(problem, history, candidates, seed, methods=METHODS):
    features = PrefixFeatures(problem, BASE, history, seed, latent=False)
    x = features.transform([record.candidate for record in history])
    query = features.transform(candidates)
    history_records = [record.to_dict() for record in history]
    evidence = {
        use_interval: [timing_feedback(record, problem.settings, use_interval=use_interval,
                                       allow_hidden_reserve_stop=True) for record in history_records]
        for use_interval in (False, True)
    }
    heads = {}
    for name in methods:
        if name not in METHODS:
            raise ValueError(f"Unknown completion model {name}")
        probit = name == "original_probit"
        heads[name] = StableGroupedHead(x, evidence[name == "timing_interval"], seed=seed, name=name,
                                        continuous_count=len(problem.space.continuous),
                                        completion=probit, reference_scale=MARGIN_SCALE)
    return heads, query, sum(obs.lower is not None for obs in evidence[True])


def completion_probabilities(problem, history, candidates, seed, methods=METHODS):
    heads, query, stops = fit_completion_heads(problem, history, candidates, seed, methods)
    predictions = {}
    for name, head in heads.items():
        pred = head.predict(query)
        variance = pred["variance"] + (1. if name == "original_probit" else 0.)
        predictions[name] = ndtr(-pred["mean"] / np.sqrt(np.maximum(variance, 1e-14)))
    return predictions, stops


def analyze_prefix(problem, history, panel, seed):
    cold = not any(row.usable_feasible for row in history)
    labels = [discovery_outcome(row.to_dict()) for row in panel]
    if set(labels) - {"abort", "violation", "usable"}:
        raise ValueError("The sealed panel must have settled three-state observations")
    row = {"seed": seed, "initial": len(history), "cold": cold,
           "candidates": len(panel), "usable": labels.count("usable"),
           "completed_violations": labels.count("violation"), "scores": {}}
    if not panel:
        return row

    candidates = [record.candidate for record in panel]
    predictions, row["reserve_stops"] = completion_probabilities(problem, history, candidates, seed)
    for name, p in predictions.items():
        row["scores"][name] = {
            **first_ranking(candidates, p, labels),
            "completion_brier": brier_by_configuration(p, panel, [y != "abort" for y in labels]),
        }
    return row


def summarize(rows):
    ranked = [row for row in rows if row["scores"]]
    summary = {"n": len(ranked), "prefixes_without_candidate": len(rows) - len(ranked),
               "prefixes_with_usable_candidate": sum(row["usable"] > 0 for row in ranked)}
    if not ranked:
        return summary
    summary["methods"] = {
        name: {
            "top_one_hits": sum(row["scores"][name]["top_one_usable"] for row in ranked),
            "top_five_usable_fraction": mean(row["scores"][name]["top_five_usable_fraction"]
                                              for row in ranked),
            "completion_brier": mean(row["scores"][name]["completion_brier"] for row in ranked),
        }
        for name in METHODS
    }
    summary["interval_minus_control"] = {}
    for control in ("timing_sign", "original_probit"):
        pairs = [(row["scores"]["timing_interval"], row["scores"][control]) for row in ranked]
        effect = {"changed_first": sum(a["best_id"] != b["best_id"] for a, b in pairs)}
        for metric in ("completion_brier", "top_one_usable", "top_five_usable_fraction"):
            deltas = [float(a[metric]) - float(b[metric]) for a, b in pairs]
            effect[metric] = {"mean": mean(deltas), "paired_95": paired_interval(deltas)}
        summary["interval_minus_control"][control] = effect
    return summary


def replay_profile(folder):
    spec, problem, calls = load_pr122_profile(folder)
    sizes = sorted({arm["initial"] for arm in spec["arms"]})
    rows = []
    for seed, history, panel in archived_seed_panels(spec, calls, max(sizes)):
        screened = [record for record in panel if passes_structural_screen(record)]
        rows.extend(analyze_prefix(problem, history[:n], screened, seed) for n in sizes)
        print(f"{spec['name']} {seed} complete", flush=True)
    return {
        "profile": spec["name"], "input_source_sha": SOURCE_SHA,
        "input_seal_sha256": file_hash(folder / "seal.json"),
        "new_evaluator_calls": 0, "new_rcaide_mda_calls": 0,
        "by_initial": {
            str(n): {
                cohort: summarize([r for r in rows if r["initial"] == n
                                   and (cohort == "all" or r["cold"])])
                for cohort in ("all", "cold")
            } for n in sizes
        },
        "per_prefix": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(folder.resolve() in args.output.resolve().parents for folder in args.input):
        parser.error("Write the output outside the sealed inputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reports = [replay_profile(folder) for folder in args.input]
    save(args.output, reports)
    print(json.dumps({"profiles": [r["profile"] for r in reports], "new_calls": 0}))
