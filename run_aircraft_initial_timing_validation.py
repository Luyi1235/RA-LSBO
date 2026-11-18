"""Prospective seed replication of the PR124 screened ranking comparison."""

import argparse
from collections import Counter
import json
from pathlib import Path

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.runner import Ledger, save
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import ModelNumericalFailure
from ra_lsbo.partial_bo.problems import candidate_pool
from run_aircraft_initial_budget import BenchmarkAircraft, evidence_hashes, nested_indices, read_ledger
from run_aircraft_initial_feedback_diagnostic import passes_structural_screen
from run_aircraft_initial_timing_diagnostic import analyze_prefix, summarize
from run_aircraft_risk_cold_start import discovery_outcome, read, source_commit


PROFILES = {
    "smoke": (28610000, 28610100),
    "headwind-confirm": tuple(range(28600000, 28604800, 100)),
}
INITIAL = (4, 8)
TIMING_REASONS = {"complete", "physical_soc_reserve_abort"}


def plan(name):
    seeds = list(PROFILES[name])
    return {
        "name": name, "seeds": seeds, "headwind_m_s": 4., "payload_delta_kg": 0.,
        "pool": 256, "heldout": 32, "initial": list(INITIAL),
        "budget": len(seeds) * (max(INITIAL) + 32),
        "training": "PR122 geometry-screened balanced coverage prefixes",
        "panel": "PR124 common structural screen before scoring",
        "primary": {"initial": 4, "cohort": "cold", "control": "timing_sign",
                    "metric": "top_five_usable_fraction"},
    }


def panels(problem, spec, seed):
    pool = candidate_pool(problem, seed + 1, spec["pool"])[0]
    heldout = candidate_pool(problem, seed + 2, spec["heldout"], map(digest, pool))[0]
    order = nested_indices(problem, pool, max(spec["initial"]))
    return {"initial": [pool[i] for i in order], "heldout": heldout}


def run(spec, sha, output):
    source_commit(sha, clean=True)
    root = Path(__file__).resolve().parent
    if root == output or root in output.parents:
        raise ValueError("Use an evidence directory outside the source worktree")
    output.mkdir(parents=True)
    problem = BenchmarkAircraft(spec)
    save(output / "plan.json", {"source_sha": sha, "experiment": spec,
                                "contract": problem.contract_id})
    ledger = Ledger(output, spec["budget"])
    for seed in spec["seeds"]:
        roster = panels(problem, spec, seed)
        for role, candidates in roster.items():
            for i, candidate in enumerate(candidates):
                ledger.evaluate(problem, candidate, f"{seed}/{role}/{i}", role)
        print(f"{spec['name']} {seed} evaluated", flush=True)
    save(output / "seal.json", evidence_hashes(output))


def score_prefix(problem, seed, history, screened):
    # Exclude a matched prefix if the registered timing model cannot represent a
    # training outcome, or a retained panel member has an unresolved label.
    row = {"seed": seed, "initial": len(history), "scores": {},
           "cold": not any(record.usable_feasible for record in history),
           "candidates": len(screened), "usable": sum(record.usable_feasible for record in screened)}
    unusual = sorted({record.diagnostics["termination_reason"] for record in history
                      if record.diagnostics["termination_reason"] not in TIMING_REASONS})
    if unusual:
        return {**row, "exclusion": "unsupported_training_termination", "reasons": unusual}
    if any(discovery_outcome(record.to_dict()) not in {"abort", "violation", "usable"}
           for record in screened):
        return {**row, "exclusion": "unresolved_panel_label"}
    try:
        return analyze_prefix(problem, history, screened, seed)
    except ModelNumericalFailure as exc:
        return {**row, "exclusion": "model_numerical_failure", "head": exc.head}


def summarize_cohort(rows):
    exclusions = Counter(row["exclusion"] for row in rows if "exclusion" in row)
    analyzed = [row for row in rows if "exclusion" not in row]
    return {**summarize(analyzed), "registered_prefixes": len(rows),
            "exclusions": exclusions}


def load_validation(folder, sha):
    if read(folder / "seal.json") != evidence_hashes(folder):
        raise ValueError("Evidence seal mismatch")
    header = read(folder / "plan.json")
    spec = plan(header["experiment"]["name"])
    problem = BenchmarkAircraft(spec)
    if (header["source_sha"] != sha or header["experiment"] != spec
            or header["contract"] != problem.contract_id):
        raise ValueError("Input plan, source or evaluator differs from the registered protocol")
    calls = read_ledger(folder, problem.contract_id)
    if len(calls) != spec["budget"]:
        raise ValueError("Evaluator call count differs from registered budget")
    return spec, problem, calls


def validated_seed_panels(spec, problem, calls):
    used = 0
    for seed in spec["seeds"]:
        records = {"initial": [], "heldout": []}
        for role, candidates in panels(problem, spec, seed).items():
            for i, candidate in enumerate(candidates):
                number, record, actual_role = calls[f"{seed}/{role}/{i}"]
                used += 1
                if number != used or record.candidate != candidate or actual_role != role:
                    raise ValueError("Recorded evaluation differs from the frozen seed design")
                records[role].append(record)
        yield seed, records["initial"], records["heldout"]
    if used != len(calls):
        raise ValueError("Unregistered evaluator call")


def verify(folder, sha):
    source_commit(sha)
    spec, problem, calls = load_validation(folder, sha)
    rows = []
    panel_counts = Counter()
    for seed, history, panel in validated_seed_panels(spec, problem, calls):
        screened = [record for record in panel if passes_structural_screen(record)]
        panel_counts.update({
            "screened_out": len(panel) - len(screened),
            "retained": len(screened),
            "retained_usable": sum(row.usable_feasible for row in screened),
            "retained_completed_violation": sum(discovery_outcome(row.to_dict()) == "violation"
                                                for row in screened),
        })
        rows.extend(score_prefix(problem, seed, history[:n], screened) for n in spec["initial"])
        print(f"{spec['name']} {seed} verified", flush=True)
    by_initial = {
        str(n): {
            cohort: summarize_cohort([r for r in rows if r["initial"] == n
                                      and (cohort == "all" or r["cold"])])
            for cohort in ("all", "cold")
        } for n in spec["initial"]
    }
    return {
        "status": "VERIFIED_TIMING_VALIDATION", "profile": spec["name"], "source_sha": sha,
        "new_aircraft_calls": len(calls), "new_rcaide_mda_calls": 0, "panel_counts": panel_counts,
        "primary": by_initial["4"]["cold"], "by_initial": by_initial, "per_prefix": rows,
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
