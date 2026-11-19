"""Measure screened pool expansion under the original four-query budget."""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from statistics import mean

from ra_lsbo.budget import digest
from run_aircraft_initial_budget import paired_interval, save
from run_aircraft_initial_timing_diagnostic import METHODS
from run_aircraft_risk_cold_start import discovery_outcome, file_hash, source_commit
from run_aircraft_timing_online import QUERY_BUDGET, execute_arm, roster
from run_aircraft_timing_pool_audit import load_archive, load_supplement, plan, reconcile, run
from run_aircraft_timing_two_step_replay import (
    SUPPLEMENT_RESULT_SHA, SUPPLEMENT_SEALS, SUPPLEMENT_SHA,
)


LARGE_PANEL = 64


def prepare(original, supplement):
    spec, problem, seeds = load_archive(original)
    name = spec["name"]
    if file_hash(supplement / "seal.json") != SUPPLEMENT_SEALS[name]:
        raise ValueError("Supplement must be the fixed PR128 result")
    base_plan = plan(spec, problem, seeds)
    calls = load_supplement(base_plan, problem, supplement, SUPPLEMENT_SHA)
    expanded = []
    used = 0
    for unit in seeds:
        panel, used = reconcile(unit, problem, calls, used)
        initial, enlarged = roster(problem, spec, unit.seed, panel_size=LARGE_PANEL)
        if (initial != [row.candidate for row in unit.history]
                or enlarged[:len(unit.screened)] != unit.screened
                or len(set(map(digest, enlarged))) != len(enlarged)):
            raise ValueError("The expanded roster must preserve the exact original prefix")
        expanded.append(replace(unit, screened=enlarged,
                                archived={row.observation.design_id: row for row in panel}))
    if used != len(calls):
        raise ValueError("Unregistered call in the PR128 supplement")
    protocol = {
        **plan(spec, problem, expanded),
        "base_pool_result_sha": SUPPLEMENT_RESULT_SHA,
        "base_pool_seal_sha256": SUPPLEMENT_SEALS[name],
        "panel_sizes_before_structural_screen": [spec["online_panel_before_screen"], LARGE_PANEL],
        "queries_per_policy": QUERY_BUDGET,
        "small_eligible": base_plan["eligible"],
    }
    return protocol, problem, expanded


def availability(panel):
    usable = sum(row.usable_feasible for row in panel)
    outcomes = Counter(discovery_outcome(row.to_dict()) for row in panel)
    unresolved = sum(count for name, count in outcomes.items()
                     if name not in {"abort", "violation", "usable"})
    status = "has_usable" if usable else "unresolved" if unresolved else "no_usable"
    return {"candidates": len(panel), "usable": usable, "status": status, "outcomes": outcomes}


def archived_outcome(records):
    first = next((i for i, row in enumerate(records, 1) if row.usable_feasible), None)
    return {"discovered": first is not None, "first_usable_query": first,
            "capped_first": first or QUERY_BUDGET+1, "queries": len(records),
            "design_ids": [row.observation.design_id for row in records]}


def audit_unit(problem, unit, panel):
    small = panel[:len(unit.archived)]
    records = {row.observation.design_id: row for row in panel}

    def reveal(candidate, _tag, _role):
        return records[digest(candidate)]

    methods = {}
    for name in METHODS:
        methods[name] = {
            "small": archived_outcome(unit.online[name]),
            "large": execute_arm(problem, unit.seed, name, unit.history, unit.screened, reveal),
        }
    return {"seed": unit.seed, "small_pool": availability(small), "large_pool": availability(panel),
            "methods": methods}


def effect(values):
    return {"mean": mean(values) if values else None, "paired_95": paired_interval(values),
            "positive": sum(x > 0 for x in values), "negative": sum(x < 0 for x in values)}


def summarize(rows):
    recovered = [row for row in rows
                 if row["small_pool"]["status"] == "no_usable" and row["large_pool"]["usable"]]
    if any(row["small_pool"]["usable"] > row["large_pool"]["usable"] for row in rows):
        raise ValueError("Nested expansion cannot remove a usable candidate")
    coverage = {
        size: dict(Counter(row[f"{size}_pool"]["status"] for row in rows))
        for size in ("small", "large")
    }
    coverage["recovered_seeds"] = [row["seed"] for row in recovered]
    methods = {}
    for name in METHODS:
        trajectories = [row["methods"][name] for row in rows]
        runs = {}
        for size in ("small", "large"):
            arms = [row[size] for row in trajectories]
            runs[size] = {
                "discovered": sum(arm["discovered"] for arm in arms),
                "mean_capped_first": mean(arm["capped_first"] for arm in arms) if arms else None,
                "replayed_queries": sum(arm["queries"] for arm in arms),
            }
        runs["large_minus_small"] = {
            key: effect([int(row["large"][key]) - int(row["small"][key]) for row in trajectories])
            for key in ("discovered", "capped_first")
        }
        runs["newly_covered_and_discovered"] = sum(
            row["methods"][name]["large"]["discovered"] for row in recovered)
        methods[name] = runs
    return coverage, methods


def verify(protocol, problem, units, folder, sha):
    source_commit(sha)
    calls = load_supplement(protocol, problem, folder, sha)
    rows, used = [], 0
    for unit in units:
        panel, used = reconcile(unit, problem, calls, used)
        rows.append(audit_unit(problem, unit, panel))
        print(f"{protocol['name']} {unit.seed} verified", flush=True)
    if used != len(calls):
        raise ValueError("Unregistered expansion evaluation")
    coverage, methods = summarize(rows)
    return {
        "status": "VERIFIED_TIMING_NESTED_POOL", "source_sha": sha, "protocol": protocol,
        "new_diagnostic_aircraft_calls": len(calls), "new_replay_calls": 0, "new_rcaide_mda_calls": 0,
        "coverage": coverage, "methods": methods, "per_seed": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Sealed PR127 profile")
    parser.add_argument("--supplement", type=Path, required=True, help="Sealed PR128 profile")
    parser.add_argument("--evidence", type=Path, help="PR130 expansion evaluation folder")
    parser.add_argument("--expected-sha")
    parser.add_argument("--output", type=Path, help="Analysis JSON for verify")
    args = parser.parse_args()
    inputs = [args.input.resolve(), args.supplement.resolve()]
    protocol, problem, units = prepare(*inputs)
    if args.task == "plan":
        print(json.dumps(protocol, indent=2))
    else:
        if not args.expected_sha or args.evidence is None:
            parser.error("--expected-sha and --evidence are required for run and verify")
        evidence = args.evidence.resolve()
        if any(evidence == folder or folder in evidence.parents for folder in inputs):
            parser.error("Write new evidence outside the original archives")
        if args.task == "run":
            run(protocol, problem, units, evidence, args.expected_sha)
        else:
            if args.output is None or any(
                    folder in args.output.resolve().parents for folder in [*inputs, evidence]):
                parser.error("Write --output outside all sealed folders")
            report = verify(protocol, problem, units, evidence, args.expected_sha)
            save(args.output, report)
            print(report["status"], protocol["name"], report["new_diagnostic_aircraft_calls"])
