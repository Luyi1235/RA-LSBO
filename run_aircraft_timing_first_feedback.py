"""Compare the first real cold-start feedback in the sealed PR132 trajectories."""

import argparse
import json
from math import isclose
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.special import ndtr

from aircraft_timing_information import PROBABILITY_FLOOR, select_information
from ra_lsbo.budget import digest
from run_aircraft_initial_budget import (
    INFORMATION_METHODS, BenchmarkAircraft, brier_by_configuration, evidence_hashes, paired_interval,
    plan, read_ledger, save,
)
from run_aircraft_initial_timing_diagnostic import completion_probabilities, fit_completion_heads
from run_aircraft_risk_cold_start import discovery_outcome, file_hash, read, select, source_commit


INPUT_SOURCE = "2962da5729cf9c9c4a21c9ab9831809e63c89881"
INPUT_RESULT = "254dba9a935a477759d8335e9342be4963e6aa11"
INPUT_SEALS = {
    "timing-information-headwind": "0f7cdbbf8eeac56bdadc79aadd20a82ef43c2ef9605177cb8f18efa00ce5607b",
    "timing-information-payload": "a4afd0910f2749489e82c449f4cecf10537f7a53406672c054df5705613e65ee",
    "timing-information-smoke": "c5820a633c65a6d109295b61cd2cd4096c07a53ac74c63c62fcfa721af7333f5",
}
GREEDY, INFORMATION = INFORMATION_METHODS


def probability(head, query):
    moments = head.predict(query)
    return ndtr(-moments["mean"] / np.sqrt(np.maximum(moments["variance"], 1e-14)))


def analyze_pair(problem, seed, history, roster, panel, observed):
    n = len(history)
    used = {row.observation.design_id for row in history}
    remaining = [candidate for candidate in roster if digest(candidate) not in used]
    candidates = [row.candidate for row in panel]
    if {digest(candidate) for candidate in candidates} & (used | set(map(digest, remaining))):
        raise ValueError("Held-out labels must be from disjoint designs")
    labels = [discovery_outcome(row.to_dict()) for row in panel]
    labels = [None if value == "unknown" else value != "abort" for value in labels]
    heads, query, _ = fit_completion_heads(problem, history, remaining+candidates, seed,
                                          methods=(GREEDY,))
    head = heads[GREEDY]
    proposed, heldout = query[:len(remaining)], query[len(remaining):]
    p = probability(head, proposed)
    best = select(remaining, p)
    informative, informative_p, gain = select_information(head, proposed, remaining)
    before = brier_by_configuration(probability(head, heldout), panel, labels)
    history_digest = digest([row.to_dict() for row in history])
    choices = {}
    updated_losses = {}
    for method, index in ((GREEDY, best), (INFORMATION, informative)):
        receipt, record = observed[method]
        if (receipt["mode"] != method or receipt["call"] != n+1
                or receipt["history"] != history_digest
                or receipt["candidate"] != record.candidate or remaining[index] != record.candidate
                or not isclose(receipt["completion_probability"], p[index], rel_tol=1e-8, abs_tol=1e-10)):
            raise ValueError("First decision does not match the sealed experiment")
        key = record.observation.design_id
        if key not in updated_losses:
            forecast, _ = completion_probabilities(problem, history+[record], candidates, seed,
                                                   methods=(GREEDY,))
            updated_losses[key] = brier_by_configuration(forecast[GREEDY], panel, labels)
        after = updated_losses[key]
        choices[method] = {"design_id": key, "outcome": discovery_outcome(record.to_dict()),
                           "usable": record.usable_feasible, "probability": float(p[index]),
                           "brier_after": after, "brier_reduction": before-after}
    if not isclose(observed[INFORMATION][0]["expected_brier_reduction"], gain,
                   rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("First information score differs from the replay")
    return {"seed": seed, "initial": n, "heldout_brier_before": before,
            "heldout_count": sum(value is not None for value in labels),
            "screened_candidates": len(remaining),
            "in_probability_band": int(np.count_nonzero(p >= PROBABILITY_FLOOR*p[best])),
            "predicted_probability_drop": float(p[best]-informative_p),
            "expected_reduction_for_information_choice": gain,
            "changed": informative != best, "choices": choices}


def summarize(rows):
    result = {"initially_unsolved": len(rows), "changed_first_count": sum(row["changed"] for row in rows),
              "single_candidate_in_band": sum(row["in_probability_band"] == 1 for row in rows)}
    for cohort, matched in (("all_cold", rows), ("changed_first", [row for row in rows if row["changed"]])):
        if not matched:
            continue
        pairs = [(row["choices"][INFORMATION], row["choices"][GREEDY]) for row in matched]
        delta = [info["brier_after"]-greedy["brier_after"] for info, greedy in pairs]
        result[cohort] = {"n": len(delta),
                          "brier_after_information_minus_greedy": mean(delta),
                          "paired_95": paired_interval(delta),
                          "first_query_usable": {name: sum(row["choices"][name]["usable"] for row in matched)
                                                 for name in INFORMATION_METHODS},
                          "mean_brier_reduction": {
                              name: mean(row["choices"][name]["brier_reduction"] for row in matched)
                              for name in INFORMATION_METHODS}}
    return result


def read_profile(folder):
    header = read(folder/"plan.json")
    spec = header["experiment"]
    name = spec["name"]
    seal = folder/"seal.json"
    if (header["source_sha"] != INPUT_SOURCE or spec != plan(name)
            or file_hash(seal) != INPUT_SEALS[name] or read(seal) != evidence_hashes(folder)):
        raise ValueError("Input is not the fixed sealed PR132 run")
    problem = BenchmarkAircraft(spec)
    if header["contract"] != problem.contract_id:
        raise ValueError("Evaluator contract differs from the archive")
    calls = read_ledger(folder, header["contract"])
    if len(calls) != spec["max_calls"]:
        raise ValueError("Incorrect PR132 evaluation count")
    sizes = sorted({arm["initial"] for arm in spec["arms"]})
    rows = []
    for seed in spec["seeds"]:
        unit = folder/str(seed)
        roster = read(unit/"roster.json")
        initial = [calls[f"{seed}/initial/{j}"][1] for j in range(max(sizes))]
        panel = [calls[f"{seed}/heldout/{j}"][1] for j in range(spec["heldout"])]
        if [row.candidate for row in panel] != roster["heldout"]:
            raise ValueError("Held-out roster differs from the evidence")
        for size in sizes:
            history = initial[:size]
            if any(row.usable_feasible for row in history):
                continue
            observed = {}
            for method in INFORMATION_METHODS:
                arm = f"{method}-{size}"
                with (unit/arm/"proposals.jsonl").open(encoding="utf-8") as receipts:
                    receipt = json.loads(receipts.readline())
                observed[method] = (receipt, calls[f"{seed}/{arm}/{size}"][1])
            rows.append(analyze_pair(problem, seed, history, roster["pool"], panel, observed))
    return {"profile": name, "seal_sha256": INPUT_SEALS[name], "seed_count": len(spec["seeds"]),
            "by_initial": {str(size): summarize([row for row in rows if row["initial"] == size])
                           for size in sizes}, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    sha = source_commit(args.expected_sha, clean=True)
    output = args.output.resolve()
    if any(folder.resolve() in (output, *output.parents) for folder in args.input):
        parser.error("Write outside the sealed input directories")
    result = {"status": "VERIFIED_TIMING_FIRST_FEEDBACK", "source_sha": sha,
              "input_source_sha": INPUT_SOURCE, "input_result_sha": INPUT_RESULT,
              "new_evaluator_calls": 0, "new_rcaide_mda_calls": 0, "profiles": []}
    for folder in args.input:
        profile = read_profile(folder)
        result["profiles"].append(profile)
        print(result["status"], profile["profile"], len(profile["rows"]), flush=True)
    save(output, result)


if __name__ == "__main__":
    main()
