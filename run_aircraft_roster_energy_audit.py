"""Complete the finite PR149 candidate rosters for an offline energy audit."""

import argparse
import json
from math import isfinite
from pathlib import Path

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.runner import Ledger, save
from run_aircraft_external_full_budget import (
    POOL_ARMS, SCENARIOS, analyze, passes_exact_screen, plan, read, read_ledger,
)
from run_aircraft_risk_cold_start import discovery_outcome, file_hash, source_commit
from run_aircraft_timing_interval_value import effect, evidence_hashes
from run_aircraft_timing_usable_value import UsableAircraft


PR149_CODE = "2393ff261e39e461ac1b68e15b2a9458791a2f6d"
PR149_RESULT = "f3dd32eba5d97f8523b72fc850ae283171790f96"
PR149_MANIFEST = "72951311edc3fedd39242932de50dae875d6db4e7b846141d25b03370c377c0f"
EXTRA_CALLS = {"payload": 145, "headwind": 118}


def usable_energy(record):
    result = discovery_outcome(record.to_dict())
    if result == "usable" and record.usable_feasible and isfinite(record.observation.objective.value):
        return record.observation.objective.value
    if result in ("abort", "violation") and not record.usable_feasible:
        return None
    raise ValueError("The finite-roster energy audit requires a resolved usable-design label")


def original_data(root, scenario):
    if file_hash(root/"SHA256SUMS.txt") != PR149_MANIFEST:
        raise ValueError("The input must be the archived PR149 confirmation commit")
    spec = plan(scenario, "confirmation")
    folders = [root/scenario/str(seed) for seed in spec["seeds"]]
    source = analyze(spec, folders, PR149_CODE)
    if source != read(root/(scenario+"-analysis.json")):
        raise ValueError("Source analysis differs from its sealed trajectories")
    cases = []
    for folder, row in zip(folders, source["rows"]):
        header = read(folder/"plan.json")
        ledger = read_ledger(folder, header["contract"])
        pool = {digest(point): point for point in read(folder/"roster.json")["covered"]}
        measured = {}
        for tag, (_, record, role) in ledger.items():
            if role != "online" or tag.split("/")[1] not in POOL_ARMS:
                continue
            key = record.observation.design_id
            energy = usable_energy(record)
            if key not in pool or (key in measured and energy != measured[key]):
                raise ValueError("Inconsistent observation within the registered roster")
            measured[key] = energy
        prefix = [usable_energy(ledger[f"initial/{i}"][1]) for i in range(1, max(spec["initial_sizes"])+1)]
        cases.append({"seed": row["seed"], "by_initial": row["by_initial"], "prefix": prefix,
                      "pool": pool, "measured": measured})
    return spec, cases


def pending(cases):
    return [(case["seed"], key, point) for case in cases for key, point in case["pool"].items()
            if key not in case["measured"]]


def audit_plan(spec, cases, code_sha):
    missing = pending(cases)
    if len(missing) != EXTRA_CALLS[spec["scenario"]]:
        raise ValueError("The set of missing PR149 labels differs from the frozen count")
    return {"source_sha": code_sha, "input_code_sha": PR149_CODE, "input_result_sha": PR149_RESULT,
            "scenario": spec["scenario"], "contract": UsableAircraft(spec).contract_id,
            "seed_counts": [{"seed": case["seed"], "covered": len(case["pool"]),
                             "prior_labels": len(case["measured"])} for case in cases],
            "extra_calls": len(missing)}


def run(spec, cases, registration, output):
    output.mkdir(parents=True)
    save(output/"plan.json", registration)
    problem = UsableAircraft(spec)
    ledger = Ledger(output, registration["extra_calls"])
    for seed, key, point in pending(cases):
        if not passes_exact_screen(problem, point):
            raise ValueError("Roster point fails the common engineering screen")
        ledger.evaluate(problem, point, f"{seed}/{key}", "roster_supplement")
    save(output/"seal.json", evidence_hashes(output))


def summarize(rows):
    gap = [row["pooled_minus_finite_optimum_Wh"] for row in rows]
    native = [row["archsbo_minus_finite_optimum_Wh"] for row in rows]
    return {"n": len(rows), "pooled_at_finite_optimum": sum(value == 0 for value in gap),
            "archsbo_below_finite_optimum": sum(value < 0 for value in native),
            "archsbo_advantage_eliminated": sum(row["archsbo_beats_pooled"] and value >= 0
                                                for row, value in zip(rows, native)),
            "pooled_minus_finite_optimum_Wh": effect(gap),
            "archsbo_minus_finite_optimum_Wh": effect(native)}


def verify(spec, cases, registration, folder):
    if read(folder/"plan.json") != registration or read(folder/"seal.json") != evidence_hashes(folder):
        raise ValueError("Supplemental ledger is not the registered, sealed run")
    ledger = read_ledger(folder, registration["contract"])
    additions = {case["seed"]: {} for case in cases}
    missing = pending(cases)
    if len(ledger) != len(missing):
        raise ValueError("Supplemental evaluation count differs from the plan")
    for i, (seed, key, point) in enumerate(missing, 1):
        number, record, role = ledger[f"{seed}/{key}"]
        if (number, record.candidate, role) != (i, point, "roster_supplement"):
            raise ValueError("Unexpected supplemental evaluation")
        additions[seed][key] = usable_energy(record)
    groups = {str(size): [] for size in spec["initial_sizes"]}
    for case in cases:
        extra = additions[case["seed"]]
        measured = {**case["measured"], **extra}
        if measured.keys() != case["pool"].keys():
            raise ValueError("The roster does not have complete labels")
        for n, rows in groups.items():
            original = case["by_initial"][n]
            arms = original["arms"]
            values = [*case["prefix"][:int(n)], *measured.values()]
            optimum = min(value for value in values if value is not None)
            pooled = arms["interval_greedy"]["best_usable_Wh"]
            native = arms["archsbo_screened"]["best_usable_Wh"]
            if pooled < optimum:
                raise ValueError("Original pooled energy is below the complete roster optimum")
            rows.append({"seed": case["seed"], "cold": original["cold"],
                         "finite_roster_optimum_Wh": optimum, "pooled_Wh": pooled, "archsbo_Wh": native,
                         "pooled_minus_finite_optimum_Wh": pooled-optimum,
                         "archsbo_minus_finite_optimum_Wh": native-optimum,
                         "archsbo_beats_pooled": native < pooled, "supplemental_labels": len(extra)})
    return {"status": "VERIFIED_FINITE_ROSTER_ENERGY_AUDIT", "plan": registration,
            "rcaide_mda_calls": 0, "summary": {n: summarize(rows) for n, rows in groups.items()},
            "rows": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify"), required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--input-archive", type=Path, required=True)
    parser.add_argument("--input-run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sha = source_commit(args.expected_sha, clean=True)
    archive = args.input_archive.resolve()
    spec, cases = original_data(archive, args.scenario)
    registration = audit_plan(spec, cases, sha)
    if args.task == "plan":
        print(json.dumps(registration, indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    forbidden = (archive, Path(__file__).resolve().parent)
    if output.exists() or any(parent in forbidden for parent in (output, *output.parents)):
        parser.error("Use a new output outside the source worktree and input archive")
    if args.task == "run":
        run(spec, cases, registration, output)
    else:
        if args.input_run is None or args.input_run.resolve() in (output, *output.parents):
            parser.error("Provide the sealed --input-run and write outside it")
        report = verify(spec, cases, registration, args.input_run)
        save(output, report)
        print(report["status"], args.scenario, registration["extra_calls"], flush=True)


if __name__ == "__main__":
    main()
