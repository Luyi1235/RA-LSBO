"""Replay PR135 risk rankings on its sealed, independently labeled panels."""

import argparse
from collections import Counter
from math import isclose
from pathlib import Path

from aircraft_timing_information import supports_timing
from ra_lsbo.budget import digest
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import ModelNumericalFailure
from run_aircraft_initial_timing_diagnostic import completion_probabilities
from run_aircraft_risk_cold_start import discovery_outcome, file_hash, read, source_commit
from run_aircraft_timing_interval_value import (
    DECISION_ARMS, HEAD, BenchmarkAircraft, brier_by_configuration, effect, evidence_hashes, plan,
    read_ledger, save, structural_margins,
)


INPUT_SOURCE = "9a8d841e8e3006070471ca9cad2dc82afb2328a8"
INPUT_RESULT = "3b6141131ec790a829f5267f285910a7b0a6f077"
INPUT_HASHES = {
    "timing-decision-smoke": (
        "9c42117e34c9e155493480cde40339c3ff953e75555adb9dcf05b7ef0f6b133f",
        "ccaf1f5a3aa173fc075c02ef83ce6bd83832a357ed8eff19d471cf4a6fcbfafb"),
    "timing-decision-headwind": (
        "067184364795fed9a0c5a9a468fd69309aefae72629bc0b185b445158f40ef4e",
        "2542d8a001a3feb696ab591931da7e1ee196083c9d20d0274397b41d3c9ea016"),
    "timing-decision-payload": (
        "7fbc821a63c153535717bc3d79360234d71107fe4c49367580c6077f0a2ec199",
        "fe69d110dcb5da1c708f2ca12f10564c38c49f8adf4ef207fd3ac63fe25c5dee"),
}
BASELINE = "before_feedback"


def rank_panel(panel, scores):
    order = sorted(range(len(panel)), key=lambda i: (scores[i], digest(panel[i].candidate)),
                   reverse=True)
    hit = [panel[i].usable_feasible for i in order]
    return {"top_one_usable": bool(hit and hit[0]), "top_three_usable": any(hit[:3]),
            "top_design_id": digest(panel[order[0]].candidate) if order else None,
            "first_usable_rank": next((rank for rank, y in enumerate(hit, 1) if y), None)}


def forecast(problem, history, panel, seed):
    predictions, _ = completion_probabilities(problem, history, [r.candidate for r in panel], seed,
                                               methods=(HEAD,))
    return predictions[HEAD]


def check_loss(panel, raw, forecast_values, archived):
    labels = [None if y == "unknown" else y != "abort" for y in raw]
    loss = brier_by_configuration(forecast_values, panel, labels)
    same = loss is archived if loss is None or archived is None else isclose(
        loss, archived, rel_tol=1e-8, abs_tol=1e-10)
    if not same:
        raise ValueError("Replayed risk forecast has different Brier loss")


def groups_for_panel(problem, history, panel, first_records, original):
    ids = [digest(record.candidate) for record in panel]
    if set(ids) & ({digest(r.candidate) for r in history} | set(first_records)):
        raise ValueError("Ranking panel intersects first-step training data")
    screened = [i for i, r in enumerate(panel) if all(
        margin <= 0. for margin in structural_margins(problem, r.candidate).values())]
    visible = [panel[i] for i in screened]
    raw = [discovery_outcome(r.to_dict()) for r in panel]
    counts = Counter(raw[i] for i in screened)
    seed = original["seed"]
    p0 = forecast(problem, history, panel, seed)
    check_loss(panel, raw, p0, original["heldout_brier_before"]["all"])
    ranks = {BASELINE: rank_panel(visible, p0[screened])}
    updated = {}
    for arm in DECISION_ARMS:
        row = original["arms"][arm]
        key = row["design_id"]
        record = first_records[key]
        if discovery_outcome(record.to_dict()) != row["outcome"] or record.usable_feasible != row["usable"]:
            raise ValueError("First feedback differs from the registered decision")
        if key not in updated:
            p = p0
            state = "unsupported_feedback"
            if supports_timing(record):
                try:
                    p = forecast(problem, history+[record], panel, seed)
                    state = "updated"
                except ModelNumericalFailure:
                    state = "update_model_failure"
            updated[key] = (state, p, rank_panel(visible, p[screened]))
        state, p, ranks[arm] = updated[key]
        if state != row["update"]:
            raise ValueError("First update state differs from PR135")
        check_loss(panel, raw, p, row["heldout_brier_after"]["all"])
    return counts, ranks


def audit_row(problem, calls, spec, next_records, original):
    seed, n = original["seed"], original["initial"]
    history = [calls[f"{seed}/initial/{j}"][1] for j in range(n)]
    panel = [calls[f"{seed}/heldout/{j}"][1] for j in range(spec["heldout"])]
    counts, ranks = groups_for_panel(problem, history, panel, next_records[seed, n], original)
    arms = original["arms"]
    actual = {arm: {"first": row["usable"], "second": row["second"]["usable"],
                    "either": row["any_usable_by_two"]} for arm, row in arms.items()}
    return {"seed": seed, "initial": n, "screened_outcomes": counts,
            "rankings": ranks, "actual": actual}


def summarize(rows):
    available = [row for row in rows if row["screened_outcomes"].get("usable", 0)>0]
    outcomes = Counter()
    for row in rows:
        outcomes.update(row["screened_outcomes"])
    rankings = {}
    for name in (BASELINE, *DECISION_ARMS):
        rankings[name] = {metric: sum(r["rankings"][name][metric] for r in available)
                          for metric in ("top_one_usable", "top_three_usable")}
    paired = {}
    for arm in ("decision_value", "gated_decision"):
        paired[arm] = {metric: effect([int(r["rankings"][arm][metric])-int(r["rankings"]["greedy"][metric])
                                      for r in available])
                       for metric in ("top_one_usable", "top_three_usable")}
    online = {name: {metric: sum(r["actual"][name][metric] for r in rows)
                     for metric in ("first", "second", "either")} for name in DECISION_ARMS}
    union = {
        metric: sum(not r["actual"]["greedy"][metric]
                    and any(values[metric] for values in r["actual"].values()) for r in rows)
        for metric in ("second", "either")
    }
    return {"n": len(rows), "panels_with_usable": len(available),
            "screened_outcomes": dict(outcomes), "rank_hits_given_panel_has_usable": rankings,
            "paired_rank_deltas_vs_greedy_given_panel_has_usable": paired,
            "online_hits": online, "online_greedy_misses_rescued_by_another_observed_arm": union}


def audit_profile(folder, analysis):
    header = read(folder/"plan.json")
    spec = header["spec"]
    name = spec["name"]
    seal = folder/"seal.json"
    required_seal, required_analysis = INPUT_HASHES[name]
    if (header["source_sha"] != INPUT_SOURCE or spec != plan(name)
            or file_hash(seal) != required_seal or file_hash(analysis) != required_analysis
            or read(seal) != evidence_hashes(folder)):
        raise ValueError("PR135 sealed input identity differs")
    problem = BenchmarkAircraft(spec)
    result = read(analysis)
    if header["contract"] != problem.contract_id or result["plan"] != spec:
        raise ValueError("PR135 evaluation contract differs")
    calls = read_ledger(folder, problem.contract_id)
    if len(calls) != result["actual_calls"]:
        raise ValueError("PR135 evidence has a different call count")
    next_records = {}
    for tag, (_, record, role) in calls.items():
        if role == "next":
            seed, initial, _, _ = tag.split("/")
            next_records.setdefault((int(seed), int(initial)), {})[digest(record.candidate)] = record
    rows = [audit_row(problem, calls, spec, next_records, row) for row in result["per_seed"]
            if row["state"] == "paired"]
    return {"name": name, "audit_kind": "retrospective_shadow_on_independent_panel",
            "summary": {str(n): summarize([r for r in rows if r["initial"] == n])
                        for n in spec["initial_sizes"]}, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--analyses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    sha = source_commit(args.expected_sha, clean=True)
    output = args.output.resolve()
    if any(folder.resolve() in (output, *output.parents) for folder in [*args.input, args.analyses]):
        parser.error("Write outside the sealed inputs")
    result = {"status": "VERIFIED_TIMING_DECISION_AUDIT", "source_sha": sha,
              "input_source_sha": INPUT_SOURCE, "input_result_sha": INPUT_RESULT,
              "new_evaluator_calls": 0, "new_rcaide_mda_calls": 0, "profiles": []}
    for folder in args.input:
        report = audit_profile(folder, args.analyses/(folder.name+".json"))
        result["profiles"].append(report)
        print(result["status"], report["name"], len(report["rows"]), flush=True)
    save(output, result)


if __name__ == "__main__":
    main()
