"""Complete the previously registered PR98 pools and audit search opportunity."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import platform
import sys
import traceback

from aircraft_risk_protocol import OPT_ARMS, plan
from ra_lsbo.budget import digest
from run_aircraft_risk_cold_start import (
    ROOT, checked_label, file_hash, read, source_commit, verify_evidence,
)

SOURCE_SHA = "971bd052afb546dba501a4864ae3954dc3babeaa"
RESULT_SHA = "5e521d234d8ce264156f72b67155949e9423de23"
RESULT_PATH = "results/aircraft-risk-energy-20261113-971bd05"
SOURCE_MANIFEST = "e96a11e9416d1bdc06870f82be08a6a6f3ba7e4304d621c94617d6a68ee20dfc"
CAPS = {"smoke": 20, "pilot": 836}
STATUS = "COMPLETE_AIRCRAFT_RISK_POOL_AUDIT"


def json_lines(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def source_archive(folder):
    manifest = folder/"SHA256SUMS.txt"
    if file_hash(manifest) != SOURCE_MANIFEST:
        raise ValueError("Archive SHA256SUMS.txt differs from the frozen PR98 result")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        checksum, path = line.split(" *", 1)
        if file_hash(folder/path) != checksum:
            raise ValueError(f"Source archive checksum mismatch: {path}")


def load_roster(folder, profile):
    source_archive(folder)
    run_folder = folder/profile
    spec = plan("energy-"+profile)
    report, _, _ = verify_evidence(run_folder, SOURCE_SHA)
    if read(run_folder/"plan.json")["profile"] != spec["profile"]:
        raise ValueError("Wrong sealed source profile")
    contract = read(run_folder/"contract.json")["id"]
    observed = {}
    for event in json_lines(run_folder/"calls.jsonl"):
        if event["event"] == "settled":
            observed[event["tag"]] = event["record"]

    units, pending = {}, []
    for seed in spec["seeds"]:
        roster = read(run_folder/str(seed)/"roster.json")
        pool = {digest(candidate): candidate for candidate in roster["pool"]["designs"]}
        paid = {}
        for arm in OPT_ARMS:
            for n in range(spec["initial"], spec["total"]):
                tag = f"{seed}/{arm}/{n}"
                record = observed[tag]
                identity = record["observation"]["design_id"]
                if identity not in pool or record["candidate"] != pool[identity]:
                    raise ValueError(f"Historical call outside the candidate pool: {tag}")
                previous = paid.setdefault(identity, (tag, record))
                if digest(previous[1]) != digest(record):
                    raise ValueError(f"Conflicting repeat evaluation: {tag}")
        initial = [observed[f"{seed}/initial/{i}"] for i in range(spec["initial"])]
        units[seed] = (pool, paid, initial)
        pending.extend({"tag": f"{seed}/pool/{identity}", "seed": seed, "id": identity,
                        "candidate": candidate}
                       for identity, candidate in pool.items() if identity not in paid)
    if len(pending) != CAPS[profile]:
        raise ValueError("Unseen pool count does not match the frozen call budget")
    audit_plan = {"source_sha": SOURCE_SHA, "result_sha": RESULT_SHA, "profile": profile,
                  "contract": contract, "source_calls": report["calls"],
                  "new_calls": len(pending), "rcaide_mda_calls": 0, "pending": pending}
    return audit_plan, units, run_folder


def execute(audit_plan, output, expected):
    from ra_lsbo.experiments.core.problems import AircraftCase
    from ra_lsbo.experiments.core.protocol import BASE
    from ra_lsbo.experiments.core.runner import Ledger, save

    problem = AircraftCase(BASE, "sparse", integration_step=20.)
    if problem.contract_id != audit_plan["contract"]:
        raise ValueError("Local aircraft evaluator differs from the archived evaluator")
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Use a fresh evidence directory outside the code worktree")
    output.mkdir(parents=True)
    save(output/"audit_plan.json", audit_plan)
    ledger = Ledger(output, audit_plan["new_calls"])
    try:
        for row in audit_plan["pending"]:
            ledger.evaluate(problem, row["candidate"], row["tag"], "pool_audit")
        save(output/"report.json", {"status": STATUS, "source_sha": expected,
             "calls": ledger.count, "settled": sum(ledger.roles.values()),
             "rcaide_mda_calls": 0, "python": sys.version, "platform": platform.platform()})
        save(output/"manifest.json", {path.name: file_hash(path) for path in output.iterdir()})
    except Exception:
        save(output/"failure.json", {"started": ledger.count, "settled": sum(ledger.roles.values()),
             "pending": ledger.pending, "traceback": traceback.format_exc()})
        raise


def completed_records(audit_plan, output, expected):
    files = {p.name: file_hash(p) for p in output.iterdir() if p.name != "manifest.json"}
    if set(files) != {"audit_plan.json", "calls.jsonl", "report.json"} or files != read(output/"manifest.json"):
        raise ValueError("Pool audit evidence manifest mismatch")
    if audit_plan != read(output/"audit_plan.json"):
        raise ValueError("Pool audit plan differs from the archived roster")
    report = read(output/"report.json")
    if (report["status"] != STATUS or report["source_sha"] != expected
            or report["calls"] != audit_plan["new_calls"] or report["settled"] != report["calls"]
            or report["rcaide_mda_calls"] != 0):
        raise ValueError("Pool audit report does not match the budget")
    records = {}
    events = iter(json_lines(output/"calls.jsonl"))
    for number, row in enumerate(audit_plan["pending"], 1):
        call = {"number": number, "tag": row["tag"], "role": "pool_audit",
                "candidate": row["candidate"], "contract": audit_plan["contract"],
                "new_rcaide_mda": False}
        identity = digest(call)
        start, end = next(events), next(events)
        record = end["record"]
        if (start != {"event": "started", "call_id": identity, **call}
                or end["event"] != "settled" or end["tag"] != row["tag"] or end["call_id"] != identity
                or record["candidate"] != row["candidate"]
                or record["observation"]["design_id"] != row["id"]
                or record["observation"]["contract_id"] != audit_plan["contract"]
                or end["record_digest"] != digest(record)):
            raise ValueError(f"Invalid pool audit record: {row['tag']}")
        records[row["seed"], row["id"]] = (row["tag"], record)
    if next(events, None) is not None:
        raise ValueError("Unexpected extra evaluation records")
    return records


def usable_energy(record):
    observation = record["observation"]
    if (checked_label(record) is True and observation["status"] == "completed"
            and observation["objective"]["kind"] == "exact"):
        return observation["objective"]["value"]
    return None


def audit_rows(units, new_records, run_folder):
    spec = plan("energy-"+run_folder.name)
    design_rows, endpoints, steps = [], [], []
    for seed, (pool, paid, initial) in units.items():
        energy, labels = {}, {}
        for key in pool:
            tag, record = paid[key] if key in paid else new_records[seed, key]
            energy[key], labels[key] = usable_energy(record), checked_label(record)
            design_rows.append({"seed": seed, "candidate": key, "call_tag": tag,
                                "new_call": key not in paid, "physical_feasible": labels[key],
                                "usable_energy_Wh": energy[key]})
        start = min((value for record in initial if (value := usable_energy(record)) is not None),
                    default=None)
        optimum = min((value for value in [start, *energy.values()] if value is not None),
                      default=None)
        for arm in OPT_ARMS:
            remaining, incumbent = list(pool), start
            snapshots = []
            for row in json_lines(run_folder/str(seed)/(arm+".jsonl")):
                if row["event"] != "proposal":
                    continue
                ranked = sorted(zip(remaining, row["scores"]), key=lambda x: (x[1], x[0]), reverse=True)
                chosen = row["candidate"]
                if ranked[0][0] != chosen:
                    raise ValueError("Frozen proposal ranking differs from its remaining pool")
                improvements = [k for k, _ in ranked if energy[k] is not None
                                and (incumbent is None or energy[k] < incumbent)]
                first_rank = next((i for i, (k, _) in enumerate(ranked, 1)
                                   if k in improvements), None)
                best_in_pool = min((energy[k] for k in improvements), default=None)
                receipt = row["energy"]
                ei_choice = cei_choice = feasible_ei_choice = None
                if receipt["improvement"] is not None:
                    forecasts = dict(zip(remaining, zip(receipt["improvement"], receipt["probability"])))
                    ei_choice = max(remaining, key=lambda k: (forecasts[k][0], k))
                    cei_choice = max(remaining, key=lambda k: (math.prod(forecasts[k]), k))
                    feasible_ei_choice = max((k for k in remaining if energy[k] is not None),
                                             key=lambda k: (forecasts[k][0], k), default=None)
                snapshots.append({"seed": seed, "arm": arm, "n": row["n"],
                                  "mode": "coverage" if row["energy"]["improvement"] is None else arm.split("_")[0],
                                  "incumbent_Wh": incumbent, "unseen_better_count": len(improvements),
                                  "best_better_Wh": best_in_pool,
                                  "best_improver_rank": next((i for i, (k, _) in enumerate(ranked, 1)
                                                               if energy[k] == best_in_pool), None)
                                  if best_in_pool is not None else None,
                                  "first_improver_rank": first_rank, "selected_improves": chosen in improvements,
                                  "unknown_count": sum(labels[k] is None for k in remaining),
                                  "candidate": chosen, "selected_energy_Wh": energy[chosen],
                                  "same_state_ei_choice": ei_choice,
                                  "same_state_cei_choice": cei_choice,
                                  "same_state_known_usable_ei_choice": feasible_ei_choice,
                                  "same_state_ei_improves": None if ei_choice is None else ei_choice in improvements,
                                  "same_state_cei_improves": None if cei_choice is None else cei_choice in improvements,
                                  "same_state_known_usable_ei_improves": None if feasible_ei_choice is None
                                  else feasible_ei_choice in improvements})
                if chosen in improvements:
                    incumbent = energy[chosen]
                remaining.remove(chosen)
            if len(snapshots) != spec["total"]-spec["initial"]:
                raise ValueError("Missing frozen proposals")
            steps.extend(snapshots)
            endpoints.append({"seed": seed, "arm": arm, "initial_best_Wh": start,
                              "final_best_Wh": incumbent, "known_finite_pool_best_Wh": optimum,
                              "known_pool_gap_Wh": incumbent-optimum
                              if incumbent is not None and optimum is not None else None,
                              "unseen_better_count_at_end": sum(energy[k] is not None
                                  and (incumbent is None or energy[k] < incumbent) for k in remaining),
                              "pool_unknown_count": sum(v is None for v in labels.values()),
                              "opportunity_steps": sum(r["unseen_better_count"] > 0 for r in snapshots),
                              "conversion_steps": sum(r["selected_improves"] for r in snapshots)})
    return design_rows, endpoints, steps


def write_analysis(folder, rows, audit_plan):
    folder.mkdir(parents=True)
    for name, table in zip(("pool_truth", "endpoints", "steps"), rows):
        with (folder/(name+".csv")).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    design, endpoints, steps = rows
    summary = {"status": "COMPLETE_AIRCRAFT_RISK_POOL_ANALYSIS", "source_sha": SOURCE_SHA,
               "result_sha": RESULT_SHA, "profile": audit_plan["profile"],
               "new_calls": audit_plan["new_calls"], "pool_designs": len(design),
               "known_usable_pool_designs": sum(r["usable_energy_Wh"] is not None for r in design),
               "unknown_pool_designs": sum(r["physical_feasible"] is None for r in design),
               "endpoint_rows": len(endpoints), "step_rows": len(steps)}
    (folder/"analysis.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify", "analyze"), required=True)
    parser.add_argument("--profile", choices=CAPS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.task in {"run", "analyze"} and args.output is None:
        parser.error("--output is required")
    if args.task in {"verify", "analyze"} and args.input is None:
        parser.error("--input is required")
    source_commit(args.expected_sha, clean=args.task == "run")
    source = args.source.resolve()
    output = args.output.resolve() if args.output else None
    if output is not None and (output == source or source in output.parents):
        parser.error("Output must be outside the frozen source archive")
    audit_plan, units, run_folder = load_roster(source, args.profile)
    if args.task == "plan":
        print(json.dumps({k: v for k, v in audit_plan.items() if k != "pending"}, indent=2))
    elif args.task == "run":
        execute(audit_plan, output, args.expected_sha)
    else:
        records = completed_records(audit_plan, args.input.resolve(), args.expected_sha)
        if args.task == "verify":
            print(json.dumps({"status": "VERIFIED_AIRCRAFT_RISK_POOL_AUDIT", "calls": len(records)}))
        else:
            rows = audit_rows(units, records, run_folder)
            print(json.dumps(write_analysis(output, rows, audit_plan)))


if __name__ == "__main__":
    main()
