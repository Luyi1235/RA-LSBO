"""Replay risk predictions and two-query decisions on sealed PR122 panels."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

import numpy as np

from aircraft_risk_discovery import rollout_branch_probabilities, shortlist, stage_forecast
from ra_lsbo.budget import digest
from run_aircraft_initial_budget import (
    BenchmarkAircraft, brier_by_configuration, coverage_index, evidence_hashes, fit,
    paired_interval, plan, read, read_ledger, save, select,
)
from run_aircraft_risk_cold_start import discovery_outcome


SOURCE_SHA = "ae4b9be8ef8fa09b7b45e4be7d40d8956edb59ab"
STRUCTURAL_CONSTRAINTS = ("stall", "root_bending")


def first_ranking(query, probabilities, outcomes):
    ranked = sorted(range(len(query)), key=lambda i: (probabilities[i], digest(query[i])),
                    reverse=True)[:5]
    labels = [outcomes[i] for i in ranked]
    return {
        "best_id": digest(query[ranked[0]]),
        "top_one_usable": labels[0] == "usable",
        "top_five_usable_fraction": mean(label == "usable" for label in labels),
    }


def decision_replay(heads, x, query, forecast, outcomes):
    p, moments, z = forecast
    usable = p["usable_factored"]
    first = select(query, usable)
    choices = shortlist(p, list(map(digest, query)))
    abort, violation = rollout_branch_probabilities(heads, x, p, moments, z, choices)
    collapsed = abort + violation
    staged_scores = np.full(len(query), -1.)
    collapsed_scores = staged_scores.copy()
    staged_scores[choices] = np.minimum(1., usable[choices] + abort.max(1) + violation.max(1))
    collapsed_scores[choices] = np.minimum(1., usable[choices] + collapsed.max(1))

    def replay(action, use_stages):
        branch = choices.index(action)
        outcome = outcomes[action]
        result = {"first_id": digest(query[action]), "first_outcome": outcome,
                  "second_id": None, "second_outcome": None, "step_two_success": True}
        if outcome != "usable":
            scores = (abort if outcome == "abort" else violation) if use_stages else collapsed
            second_scores = scores[branch].copy()
            second_scores[action] = -1.
            second = select(query, second_scores)
            second_outcome = outcomes[second]
            result.update(second_id=digest(query[second]), second_outcome=second_outcome,
                          step_two_success=second_outcome == "usable")
        return result

    return {
        "greedy_first": replay(first, True),
        "collapsed": replay(select(query, collapsed_scores), False),
        "staged": replay(select(query, staged_scores), True),
    }


def passes_structural_screen(record):
    """Read archived margins computed from the design before mission integration."""
    constraints = dict(record.observation.constraints)
    margins = [constraints[name] for name in STRUCTURAL_CONSTRAINTS]
    if any(margin.kind != "exact" for margin in margins):
        raise ValueError("The PR122 structural replay requires exact archived margins")
    return all(margin.value <= 0. for margin in margins)


def screen_counts(records):
    counts = {"accepted": Counter(), "rejected": Counter()}
    for record in records:
        group = "accepted" if passes_structural_screen(record) else "rejected"
        counts[group][discovery_outcome(record.to_dict())] += 1
    return counts


def screened_ranking(problem, history, panel, outcomes, probabilities, eligible):
    """Keep the historical fit fixed and restrict every selector to the same points."""
    if not eligible:
        return {"candidates": 0, "usable": 0, "ranking": {}}
    kept = [panel[i] for i in eligible]
    query = [r.candidate for r in kept]
    observed = [outcomes[i] for i in eligible]
    coverage = coverage_index(problem, query, [r.candidate for r in history])
    return {
        "candidates": len(kept),
        "usable": sum(outcome == "usable" for outcome in observed),
        "coverage_first_usable": observed[coverage] == "usable",
        "ranking": {
            name: {
                **first_ranking(query, probabilities[head][eligible], observed),
                "usable_brier": brier_by_configuration(probabilities[head][eligible], kept,
                                                        [y == "usable" for y in observed]),
            } for name, head in (("factored", "usable_factored"),
                                 ("direct", "usable_direct"), ("completion", "mission"))
        },
    }


def diagnose_prefix(problem, history, panel, seed, eligible):
    features, heads = fit(problem, history, seed)
    query = [record.candidate for record in panel]
    outcomes = [discovery_outcome(record.to_dict()) for record in panel]
    if set(outcomes) - {"abort", "violation", "usable"}:
        raise ValueError("The fixed PR122 replay requires settled three-state panel outcomes")
    x = features.transform(query)
    forecast = stage_forecast(heads, x)
    p = forecast[0]
    observed = [discovery_outcome(record.to_dict()) for record in history]
    base = (sum(y == "usable" for y in observed) + 1) / (len(observed) + 2)
    labels = [y == "usable" for y in outcomes]
    ranking = {}
    for name, scores in (("factored", p["usable_factored"]), ("direct", p["usable_direct"])):
        ranking[name] = {
            "usable_brier": brier_by_configuration(scores, panel, labels),
            **first_ranking(query, scores, outcomes),
        }
    coverage = coverage_index(problem, query, [r.candidate for r in history])
    return {
        "seed": seed, "initial": len(history), "cold": "usable" not in observed,
        "panel_usable_count": sum(labels), "ranking": ranking,
        "prevalence_brier": brier_by_configuration([base] * len(query), panel, labels),
        "completion_brier": brier_by_configuration(p["mission"], panel,
                                                   [y != "abort" for y in outcomes]),
        "coverage_first_usable": labels[coverage],
        "two_query_replay": decision_replay(heads, x, query, forecast, outcomes),
        "common_structural_screen": screened_ranking(problem, history, panel, outcomes, p,
                                                     eligible),
    }


def summarize_screen(rows):
    screened = [r["common_structural_screen"] for r in rows]
    ranked = [r for r in screened if r["candidates"]]
    if not ranked:
        return {"n": 0, "prefixes_without_candidate": len(screened)}
    comparisons = {}
    for control in ("completion", "direct"):
        delta = {}
        for key in ("top_one_usable", "top_five_usable_fraction", "usable_brier"):
            values = [float(r["ranking"]["factored"][key]) - float(r["ranking"][control][key])
                      for r in ranked]
            delta[key] = {"mean": mean(values), "paired_95": paired_interval(values)}
        comparisons[control] = delta
    return {
        "n": len(ranked),
        "prefixes_without_candidate": len(screened) - len(ranked),
        "prefixes_with_usable_candidate": sum(r["usable"] > 0 for r in ranked),
        "first_usable": {
            **{name: sum(r["ranking"][name]["top_one_usable"] for r in ranked)
               for name in ("factored", "direct", "completion")},
            "coverage": sum(r["coverage_first_usable"] for r in ranked),
        },
        "factored_minus_control": comparisons,
    }


def summarize(rows):
    if not rows:
        return {"n": 0}
    ranking_deltas = {
        key: [r["ranking"]["factored"][key] - r["ranking"]["direct"][key] for r in rows]
        for key in ("usable_brier", "top_five_usable_fraction")
    }
    replay_deltas = {}
    for control in ("greedy_first", "collapsed"):
        replay_deltas[control] = [
            int(r["two_query_replay"]["staged"]["step_two_success"])
            - int(r["two_query_replay"][control]["step_two_success"]) for r in rows]
    return {
        "n": len(rows),
        "completion_brier": mean(r["completion_brier"] for r in rows),
        "usable_brier": {
            **{name: mean(r["ranking"][name]["usable_brier"] for r in rows)
               for name in ("factored", "direct")},
            "prevalence": mean(r["prevalence_brier"] for r in rows),
        },
        "factored_minus_direct": {k: {"mean": mean(v), "paired_95": paired_interval(v)}
                                  for k, v in ranking_deltas.items()},
        "first_usable": {
            **{name: sum(r["ranking"][name]["top_one_usable"] for r in rows)
               for name in ("factored", "direct")},
            "coverage": sum(r["coverage_first_usable"] for r in rows),
        },
        "staged_minus_control_in_two_queries": {
            k: {"mean": mean(v), "paired_95": paired_interval(v)} for k, v in replay_deltas.items()
        },
        "staged_changes_first": {
            control: sum(r["two_query_replay"]["staged"]["first_id"]
                         != r["two_query_replay"][control]["first_id"] for r in rows)
            for control in replay_deltas
        },
        "common_structural_screen": summarize_screen(rows),
    }


def replay_profile(folder):
    header = read(folder / "plan.json")
    spec = plan(header["experiment"]["name"])
    if (header["source_sha"] != SOURCE_SHA or header["experiment"] != spec
            or read(folder / "seal.json") != evidence_hashes(folder)):
        raise ValueError("Input does not match the sealed PR122 protocol")
    problem = BenchmarkAircraft(spec)
    if problem.contract_id != header["contract"]:
        raise ValueError("Evaluator contract mismatch")
    calls = read_ledger(folder, header["contract"])
    sizes = sorted({arm["initial"] for arm in spec["arms"]})
    rows = []
    for seed in spec["seeds"]:
        history = [calls[f"{seed}/initial/{j}"][1] for j in range(max(sizes))]
        panel = [calls[f"{seed}/heldout/{j}"][1] for j in range(spec["heldout"])]
        if {r.observation.design_id for r in history} & {r.observation.design_id for r in panel}:
            raise ValueError("Replay panel must be disjoint from the fit history")
        eligible = [i for i, record in enumerate(panel) if passes_structural_screen(record)]
        rows.extend(diagnose_prefix(problem, history[:n], panel, seed, eligible) for n in sizes)
    return {
        "profile": spec["name"], "input_source_sha": SOURCE_SHA, "new_evaluator_calls": 0,
        "structural_screen_counts": {
            "heldout": screen_counts(row for _, row, role in calls.values() if role == "heldout"),
            "optimization": screen_counts(row for _, row, role in calls.values() if role != "heldout"),
        },
        "by_initial": {
            str(n): {key: summarize([r for r in rows if r["initial"] == n
                                    and (key == "all" or r["cold"])]) for key in ("all", "cold")}
            for n in sizes
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
    results = [replay_profile(folder) for folder in args.input]
    save(args.output, results)
    print(json.dumps({"profiles": [r["profile"] for r in results], "new_calls": 0}))
