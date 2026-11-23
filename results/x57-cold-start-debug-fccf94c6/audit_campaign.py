"""Check a settled X-57 diagnostic against raw artifacts and produce a flat trajectory."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from ra_lsbo.x57_energy_stop import STATUS, verified_stop


def read(path):
    return json.loads(path.read_bytes())


def read_ledger(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def key(row):
    return row["journal"], row["call_id"]


def peak(rows):
    edges = [(stamp, delta) for started, ended in rows for stamp, delta in ((started, 1), (ended, -1))]
    live = highest = 0
    for _, change in sorted(edges):
        live += change
        highest = max(live, highest)
        if live < 0:
            raise ValueError("Worker settlement precedes launch")
    if live:
        raise ValueError("Worker ledger was not fully settled")
    return highest


def audit(output, data):
    plan_file = data / "campaign_plan.json"
    plan = read(plan_file)
    report = read(data / "campaign_report.json")
    if hashlib.sha256(plan_file.read_bytes()).hexdigest() != report["plan_sha256"]:
        raise ValueError("The frozen plan digest changed")
    ledger = read_ledger(data / "calls.jsonl")
    starts = [r for r in ledger if r["event"] == "worker_started"]
    end_events = [r for r in ledger if r["event"] in ("settled", "interrupted") and r["charged"]]
    if Counter(map(key, starts)) != Counter(map(key, end_events)) or len(set(map(key, starts))) != len(starts):
        raise ValueError("Worker starts and charges do not reconcile")
    if len(starts) != report["rcaide_calls"] or len(starts) > 112:
        raise ValueError("Real call limit exceeded or report does not match charges")
    end = {key(row): row for row in end_events}
    measurements = {}
    for name, subset in {
        "global": starts,
        "calibration": [r for r in starts if r["batch"] == "calibration"],
        "pilot_initial": [r for r in starts if r["batch"] == "pilot" and not r["online"]],
        "pilot_online": [r for r in starts if r["batch"] == "pilot" and r["online"]],
    }.items():
        measurements[name] = {"count": len(subset), "peak": peak(
            [(r["started_at_utc"], end[key(r)]["finished_at_utc"]) for r in subset])}
    if measurements["global"]["peak"] > 12 or measurements["pilot_online"]["peak"] > 6:
        raise ValueError("Recorded worker overlap exceeds the concurrency limits")
    stops = []
    for row in end_events:
        worker_dir = data / row["journal"] / "workers" / row["call_id"]
        raw = row["worker_result"]
        response = worker_dir / "response.json"
        if response.is_file() and read(response) != raw and row["event"] == "settled":
            raise ValueError(f"Response and ledger differ for {key(row)}")
        if row["feedback"]["outcome"] != STATUS:
            continue
        certificate = verified_stop(raw)
        feedback = row["feedback"]
        if (certificate is None or certificate != read(worker_dir / "energy_stop_audit.json")
                or feedback["objective_kind"] != "missing" or feedback["completion"] is not None
                or feedback["known_constraint_violations"] != ["usable_energy"]):
            raise ValueError(f"Engineering stop evidence disagrees for {key(row)}")
        log = (worker_dir / "solver.log").read_text(encoding="utf-8", errors="replace")
        marker = f"Solving {certificate['next_segment_not_started']} segment."
        if marker in log:
            raise ValueError(f"Solver started a later segment for {key(row)}")
        stops.append({"journal": row["journal"], "call_id": row["call_id"], "batch": row["batch"],
                      "seed": row["seed"], "arm": row["arm"], "evaluation": row["evaluation"],
                      "layout": row["candidate"]["discrete"]["prop_layout"],
                      "segment": certificate["stop_after_segment"],
                      "terminal_energy_lower_Wh": certificate["terminal_energy_lower_Wh"]})
    with (output / "trajectory.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ("seed", "arm", "cumulative_calls", "online_calls", "candidate_id",
                                         "outcome", "usable", "objective_Wh", "best_usable_Wh", "target_reached"))
        writer.writeheader()
        for seed in plan["pilot_seeds"]:
            for arm in plan["arms"]:
                path = data / f"seed-{seed}" / arm / "arm_report.json"
                if not path.is_file():
                    continue
                arm_report = read(path)
                best = None
                for row in arm_report["history"]:
                    feedback = row["feedback"]
                    if feedback["usable"]:
                        energy = feedback["objective_Wh"]
                        best = energy if best is None else min(best, energy)
                    if best != arm_report["best_usable_trajectory_Wh"][row["evaluation"] - 1]:
                        raise ValueError("Recorded usable energy trajectory is inconsistent")
                    writer.writerow({
                        "seed": seed, "arm": arm, "cumulative_calls": row["evaluation"],
                        "online_calls": max(0, row["evaluation"] - plan["shared_initial_per_seed"]),
                        "candidate_id": row["candidate_id"], "outcome": feedback["outcome"], "usable": feedback["usable"],
                        "objective_Wh": feedback["objective_Wh"], "best_usable_Wh": best,
                        "target_reached": best is not None and best <= plan["target_objective_Wh"]})
    return {"status": report["status"], "budget_and_concurrency": measurements,
            "engineering_stops_checked": len(stops), "engineering_stops": stops}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.output, args.input)
    with (args.output / "physical_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
