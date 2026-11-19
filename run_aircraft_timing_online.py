"""Prospective equal-budget test of SOC timing feedback on fresh aircraft seeds."""

import argparse
from collections import Counter
import json
from math import isclose
from pathlib import Path

import numpy as np

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.runner import Ledger, save
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import ModelNumericalFailure
from ra_lsbo.partial_bo.problems import candidate_pool
from run_aircraft_initial_budget import (
    BenchmarkAircraft, coverage_index, evidence_hashes, nested_indices, read_ledger, select,
)
from run_aircraft_initial_timing_diagnostic import METHODS, completion_probabilities
from run_aircraft_initial_timing_validation import TIMING_REASONS
from run_aircraft_risk_cold_start import discovery_outcome, read, source_commit
from run_aircraft_timing_adaptive_replay import summarize


PROFILES = {
    # Include a known cold PR125 smoke seed to exercise the live selection path.
    "smoke": (28710000, 28610000),
    "headwind-online": tuple(range(28700000, 28704800, 100)),
}
INITIAL = 8
QUERY_BUDGET = 4
POOL_SIZE = 32


def plan(name):
    seeds = list(PROFILES[name])
    return {
        "name": name, "seeds": seeds, "headwind_m_s": 4., "payload_delta_kg": 0.,
        "initial_evaluations": INITIAL, "online_queries_per_method": QUERY_BUDGET,
        "geometry_pool_size": 256, "online_panel_before_screen": POOL_SIZE,
        "max_aircraft_calls": len(seeds) * (INITIAL + len(METHODS)*QUERY_BUDGET),
        "primary": {"cohort": "initially_unsolved", "method": "timing_interval",
                    "control": "timing_sign", "metric": "capped_first",
                    "undiscovered_score": QUERY_BUDGET + 1},
    }


def structural_margins(problem, candidate):
    """Compute the evaluator's two exact static margins before the mission call."""
    settings = problem.settings
    count, diameter, span, _ = problem.geometry(candidate)
    area = candidate["continuous"]["wing_area"]
    cells = 128*candidate["discrete"]["parallel_strings"]
    battery_mass = (cells-5120)*settings["cell_mass_kg"]/settings["pack_cell_mass_fraction"]
    fixed_mass = settings["fixed_mass_kg"] + battery_mass + problem.scenario.payload_delta_kg
    mass = (fixed_mass + settings["wing_areal_density_kg_m2"]*area
            + 2*count + 4*count*(diameter/1.5)**3)
    weight = mass*9.80665
    return {
        "stall": float(np.sqrt(2*weight/(settings["rho_kg_m3"]*area*settings["cl_max"]))
                       / settings["stall_limit_m_s"] - 1),
        "root_bending": float(settings["ultimate_load_factor"]*weight*span
                              / (8*settings["root_moment_limit_Nm"]) - 1),
    }


def validate_margins(problem, record):
    archived = dict(record.observation.constraints)
    for name, margin in structural_margins(problem, record.candidate).items():
        value = archived[name]
        if (value.kind != "exact" or not isclose(value.value, margin, rel_tol=0., abs_tol=1e-12)
                or (value.value <= 0.) != (margin <= 0.)):
            raise ValueError("Static screen differs from the evaluator's exact margins")


def roster(problem, spec, seed, *, panel_size=None):
    pool = candidate_pool(problem, seed + 1, spec["geometry_pool_size"])[0]
    size = spec["online_panel_before_screen"] if panel_size is None else panel_size
    panel = candidate_pool(problem, seed + 2, size, map(digest, pool))[0]
    order = nested_indices(problem, pool, INITIAL)
    screened = [candidate for candidate in panel
                if all(value <= 0. for value in structural_margins(problem, candidate).values())]
    return [pool[i] for i in order], screened


def choose(problem, observed, remaining, seed, method):
    if all(row.diagnostics.get("termination_reason") in TIMING_REASONS for row in observed):
        try:
            forecasts, _ = completion_probabilities(problem, observed, remaining, seed,
                                                    methods=(method,))
            index = select(remaining, forecasts[method])
            return index, "greedy", float(forecasts[method][index])
        except ModelNumericalFailure:
            reason = "model_numerical_failure"
    else:
        reason = "unsupported_termination"
    index = coverage_index(problem, remaining, [row.candidate for row in observed])
    return index, reason, None


def execute_arm(problem, seed, method, history, screened, obtain):
    observed = list(history)
    remaining = list(screened)
    trace = []
    first = None
    for query in range(1, QUERY_BUDGET+1):
        if not remaining:
            break
        index, reason, probability = choose(problem, observed, remaining, seed, method)
        candidate = remaining.pop(index)
        record = obtain(candidate, f"{seed}/{method}/{query}", "online")
        validate_margins(problem, record)
        trace.append({"query": query, "design_id": record.observation.design_id,
                      "reason": reason, "completion_probability": probability,
                      "outcome": discovery_outcome(record.to_dict()),
                      "history_digest": digest([row.to_dict() for row in observed])})
        if record.usable_feasible:
            first = query
            break
        observed.append(record)
    return {"first_usable_query": first, "capped_first": first or QUERY_BUDGET+1,
            "discovered": first is not None, "queries": len(trace), "trace": trace}


def campaign(spec, problem, obtain, phase):
    rows = []
    for seed in spec["seeds"]:
        initial, screened = roster(problem, spec, seed)
        history = [obtain(candidate, f"{seed}/initial/{i}", "initial")
                   for i, candidate in enumerate(initial)]
        for record in history:
            validate_margins(problem, record)
        solved = any(record.usable_feasible for record in history)
        row = {"seed": seed, "initially_solved": solved, "screened_candidates": len(screened)}
        if not solved:
            row["methods"] = {method: execute_arm(problem, seed, method, history, screened, obtain)
                              for method in METHODS}
        rows.append(row)
        print(f"{spec['name']} {seed} {phase}", flush=True)
    return rows


def run(spec, sha, output):
    source_commit(sha, clean=True)
    root = Path(__file__).resolve().parent
    if root == output or root in output.parents:
        raise ValueError("Use an evidence directory outside the source worktree")
    output.mkdir(parents=True)
    problem = BenchmarkAircraft(spec)
    save(output / "plan.json", {"source_sha": sha, "experiment": spec,
                                "contract": problem.contract_id})
    ledger = Ledger(output, spec["max_aircraft_calls"])
    campaign(spec, problem, lambda candidate, tag, role: ledger.evaluate(problem, candidate, tag, role),
             "evaluated")
    save(output / "seal.json", evidence_hashes(output))


def verify(folder, sha):
    source_commit(sha)
    if read(folder / "seal.json") != evidence_hashes(folder):
        raise ValueError("Evidence seal mismatch")
    header = read(folder / "plan.json")
    spec = plan(header["experiment"]["name"])
    problem = BenchmarkAircraft(spec)
    if (header["source_sha"] != sha or header["experiment"] != spec
            or header["contract"] != problem.contract_id):
        raise ValueError("Input plan, source or evaluator differs from the registered protocol")
    calls = read_ledger(folder, problem.contract_id)
    if len(calls) > spec["max_aircraft_calls"]:
        raise ValueError("Evaluator call budget exceeded")
    used = 0
    roles = Counter()

    def consume(candidate, tag, role):
        nonlocal used
        number, record, actual_role = calls[tag]
        used += 1
        if number != used or record.candidate != candidate or actual_role != role:
            raise ValueError("Recorded evaluation differs from the frozen sequential policy")
        roles[role] += 1
        return record

    rows = campaign(spec, problem, consume, "verified")
    if used != len(calls):
        raise ValueError("Unregistered evaluator call")
    cold_rows = [row for row in rows if not row["initially_solved"]]
    stats = summarize(cold_rows)
    fallbacks = {
        method: dict(Counter(step["reason"] for row in cold_rows
                             for step in row["methods"][method]["trace"]
                             if step["reason"] != "greedy"))
        for method in METHODS
    }
    return {
        "status": "VERIFIED_TIMING_ONLINE", "profile": spec["name"], "source_sha": sha,
        "new_aircraft_calls": len(calls), "new_rcaide_mda_calls": 0, "roles": roles,
        "cap": spec["max_aircraft_calls"], "initially_solved": len(rows)-len(cold_rows),
        "primary": stats.get("interval_minus_control", {}).get("timing_sign", {}).get("capped_first"),
        "cold": stats, "fallbacks": fallbacks, "per_seed": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify"), required=True)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--expected-sha")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.task != "verify" and not args.profile:
        parser.error("--profile is required for plan and run")
    if args.task == "plan":
        print(json.dumps(plan(args.profile), indent=2))
    else:
        if not args.expected_sha or args.output is None:
            parser.error("--expected-sha and --output are required for run and verify")
        if args.task == "run":
            run(plan(args.profile), args.expected_sha, args.output.resolve())
        else:
            if args.input is None or args.input.resolve() in args.output.resolve().parents:
                parser.error("Specify --input and put --output outside the sealed evidence")
            analysis = verify(args.input, args.expected_sha)
            save(args.output, analysis)
            print(analysis["status"], analysis["profile"], analysis["new_aircraft_calls"])
