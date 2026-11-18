"""Complete the frozen PR127 candidate panels and audit missed usable designs."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.runner import Ledger, save
from run_aircraft_initial_timing_diagnostic import METHODS
from run_aircraft_initial_timing_validation import score_prefix, summarize_cohort
from run_aircraft_risk_cold_start import discovery_outcome, file_hash, read, source_commit
from run_aircraft_timing_online import (
    BenchmarkAircraft, evidence_hashes, plan as original_plan, read_ledger, roster, validate_margins,
)

SOURCE_SHA = "863db9e2a6da2a53c5dda58a19b945c4b79b142a"
RESULT_SHA = "e23b1510e4638582b1766e51f57fbdd0b48137d5"
SEALS = {
    "smoke": "0e456a8197482edf072b7b4c775b2d77b9ad7614349aec36a4e4ecac23f3f357",
    "headwind-online": "cf8ad4ea2521cc18999deac22631929ba44ab3a676e3dc3eb291bc818f6818e1",
}


@dataclass
class SeedPanel:
    seed: int
    history: list
    screened: list
    archived: dict
    discovered: dict

    def missing(self):
        return [candidate for candidate in self.screened if digest(candidate) not in self.archived]


def load_archive(folder):
    header = read(folder / "plan.json")
    profile = header["experiment"]["name"]
    spec = original_plan(profile)
    problem = BenchmarkAircraft(spec)
    if (header["source_sha"] != SOURCE_SHA or header["experiment"] != spec
            or header["contract"] != problem.contract_id
            or file_hash(folder / "seal.json") != SEALS[profile]
            or read(folder / "seal.json") != evidence_hashes(folder)):
        raise ValueError("Input is not the frozen PR127 evaluation archive")
    calls = read_ledger(folder, problem.contract_id)
    online = defaultdict(lambda: defaultdict(list))
    for tag, (_, record, role) in calls.items():
        if role == "online":
            seed, method, _ = tag.split("/")
            online[int(seed)][method].append(record)

    seeds = []
    for seed in spec["seeds"]:
        initial, screened = roster(problem, spec, seed)
        history = [calls[f"{seed}/initial/{i}"][1] for i in range(len(initial))]
        if initial != [record.candidate for record in history]:
            raise ValueError("Archived initial design differs from seed roster")
        arms = online[seed]
        if any(record.usable_feasible for record in history):
            if arms:
                raise ValueError("Initially solved seed has unexpected online calls")
            continue

        ids = set(map(digest, screened))
        archived = {}
        for records in arms.values():
            for record in records:
                key = record.observation.design_id
                if key not in ids:
                    raise ValueError("Online record is outside the screened panel")
                previous = archived.setdefault(key, record)
                if previous.to_dict() != record.to_dict():
                    raise ValueError("Identical designs have conflicting archived outcomes")
        discovered = {method: any(row.usable_feasible for row in arms[method]) for method in METHODS}
        seeds.append(SeedPanel(seed, history, screened, archived, discovered))
    return spec, problem, seeds


def plan(spec, problem, seeds):
    return {
        "name": spec["name"], "input_code_sha": SOURCE_SHA, "input_result_sha": RESULT_SHA,
        "input_seal_sha256": SEALS[spec["name"]], "contract": problem.contract_id,
        "cold_seeds": [unit.seed for unit in seeds],
        "eligible": sum(len(unit.screened) for unit in seeds),
        "archived_unique": sum(len(unit.archived) for unit in seeds),
        "extra_aircraft_calls": sum(len(unit.missing()) for unit in seeds),
        "new_rcaide_mda_calls": 0,
    }


def run(protocol, problem, seeds, folder, sha):
    source_commit(sha, clean=True)
    root = Path(__file__).resolve().parent
    if folder == root or root in folder.parents:
        raise ValueError("Use an evidence directory outside the source worktree")
    folder.mkdir(parents=True)
    save(folder / "plan.json", {"source_sha": sha, "experiment": protocol})
    maximum = protocol["extra_aircraft_calls"]
    ledger = Ledger(folder, maximum) if maximum else None
    for unit in seeds:
        for i, candidate in enumerate(unit.missing()):
            record = ledger.evaluate(problem, candidate, f"{unit.seed}/pool/{i}", "pool_audit")
            validate_margins(problem, record)
        print(f"{protocol['name']} {unit.seed} panel completed", flush=True)
    save(folder / "seal.json", evidence_hashes(folder))


def reconcile(unit, problem, calls, offset):
    records = dict(unit.archived)
    for i, candidate in enumerate(unit.missing()):
        number, record, role = calls[f"{unit.seed}/pool/{i}"]
        offset += 1
        if number != offset or role != "pool_audit" or record.candidate != candidate:
            raise ValueError("New evaluation differs from the frozen missing-design list")
        validate_margins(problem, record)
        records[record.observation.design_id] = record
    return [records[digest(candidate)] for candidate in unit.screened], offset


def audit_panel(unit, problem, panel):
    outcomes = Counter(discovery_outcome(row.to_dict()) for row in panel)
    usable = sum(row.usable_feasible for row in panel)
    unresolved = sum(count for key, count in outcomes.items()
                     if key not in {"abort", "violation", "usable"})
    status = "has_usable" if usable else "unresolved" if unresolved else "no_usable"
    return {
        "seed": unit.seed, "pool": status, "outcomes": outcomes, "usable": usable,
        "usable_already_observed": sum(row.usable_feasible for row in unit.archived.values()),
        "online_discovered": unit.discovered,
        "initial_ranking": score_prefix(problem, unit.seed, unit.history, panel),
    }


def verify(protocol, problem, seeds, folder, sha):
    source_commit(sha)
    if (read(folder / "plan.json") != {"source_sha": sha, "experiment": protocol}
            or read(folder / "seal.json") != evidence_hashes(folder)):
        raise ValueError("Output plan, source or evidence seal differs")
    calls = read_ledger(folder, problem.contract_id) if protocol["extra_aircraft_calls"] else {}
    if len(calls) != protocol["extra_aircraft_calls"]:
        raise ValueError("Pool-completion call count differs from the fixed budget")
    rows, used = [], 0
    for unit in seeds:
        panel, used = reconcile(unit, problem, calls, used)
        rows.append(audit_panel(unit, problem, panel))
        print(f"{protocol['name']} {unit.seed} verified", flush=True)
    if used != len(calls):
        raise ValueError("Unregistered pool-completion call")
    decomposition = {}
    for method in METHODS:
        counts = Counter()
        for row in rows:
            if row["online_discovered"][method]:
                counts["discovered"] += 1
            elif row["pool"] == "has_usable":
                counts["missed_usable_pool"] += 1
            else:
                counts[row["pool"]] += 1
        decomposition[method] = counts
    retrospective_ranks = summarize_cohort([row["initial_ranking"] for row in rows])
    return {
        "status": "VERIFIED_TIMING_POOL_AUDIT", "source_sha": sha, "protocol": protocol,
        "new_aircraft_calls": len(calls), "new_rcaide_mda_calls": 0,
        "pools": Counter(row["pool"] for row in rows),
        "previous_online_outcomes": decomposition, "initial_ranking": retrospective_ranks,
        "per_seed": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Sealed PR127 profile folder")
    parser.add_argument("--evidence", type=Path, help="PR128 supplemental evaluation folder")
    parser.add_argument("--expected-sha")
    parser.add_argument("--output", type=Path, help="Output analysis JSON for verify")
    args = parser.parse_args()
    archive = args.input.resolve()
    spec, problem, seeds = load_archive(archive)
    protocol = plan(spec, problem, seeds)
    if args.task == "plan":
        print(json.dumps(protocol, indent=2))
    else:
        if not args.expected_sha or args.evidence is None:
            parser.error("--expected-sha and --evidence are required for run and verify")
        evidence = args.evidence.resolve()
        if evidence == archive or archive in evidence.parents:
            parser.error("Write new evidence outside the original archive")
        if args.task == "run":
            run(protocol, problem, seeds, evidence, args.expected_sha)
        else:
            if (args.output is None or evidence in args.output.resolve().parents
                    or archive in args.output.resolve().parents):
                parser.error("Write --output outside both sealed evidence folders")
            result = verify(protocol, problem, seeds, evidence, args.expected_sha)
            save(args.output, result)
            print(result["status"], protocol["name"], result["new_aircraft_calls"])
