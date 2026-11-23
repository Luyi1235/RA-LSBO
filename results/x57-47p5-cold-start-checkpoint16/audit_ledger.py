"""Independently check the X-57 start charges, shared rows and settled order."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def audit(directory, checkpoint):
    plan_path = directory / "run_plan.json"
    plan = read(plan_path)
    report = read(directory / "report.json")
    require(report["plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "Report and plan hashes differ")
    require(plan["total_budget_per_arm"] == report["total_budget_per_arm"] == 320, "Invalid horizon")
    require(report["status"] == "complete" and report["checkpoint"] == checkpoint, "Incomplete report")
    initial_size = plan["initial_count"]
    all_rows = defaultdict(list)
    for line in (directory / "calls.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        all_rows[entry["task_id"]].append(entry)

    windows = []
    roster = {}
    tracked_online = {}
    confirmation = set(plan["confirmation_exclusions"])
    excluded = set(plan["initial_exclusions"])
    seen = set()
    for seed in plan["seeds"]:
        rows = plan["rosters"][str(seed)]["records"]
        require(len(rows) == initial_size, f"Invalid frozen prefix length for {seed}")
        require(Counter(r["layout"] for r in rows) == {
            "wingtip": initial_size // 2, "leading_edge_distributed": initial_size // 2,
        }, f"Unbalanced frozen prefix for {seed}")
        initial = []
        for evaluation, expected in enumerate(rows, 1):
            job = f"seed-{seed}/shared/{evaluation:03d}"
            intent, started, settled = checked_call(all_rows.pop(job), job, plan["backend"], windows)
            require(settled["candidate_id"] == expected["candidate_id"] and
                    settled["candidate"] == expected["candidate"] and settled["precheck"] == expected["precheck"]
                    and settled["candidate_id"] not in seen | excluded, f"Changed frozen candidate: {job}")
            seen.add(settled["candidate_id"])
            initial.append(settled)
        roster[seed] = initial
        for arm in plan["arms"]:
            online = []
            previously_seen = {r["candidate_id"] for r in initial}
            previous_finish = max(r["finished_at_utc"] for r in initial)
            for position in range(initial_size + 1, checkpoint + 1):
                job = f"seed-{seed}/{arm}/{position:03d}"
                intent, started, settled = checked_call(all_rows.pop(job), job, plan["backend"], windows)
                require(settled["candidate_id"] not in previously_seen | confirmation,
                        f"Repeated or excluded design: {job}")
                require(previous_finish <= started["started_at_utc"], f"Out-of-order trajectory: {job}")
                previous_finish = settled["finished_at_utc"]
                previously_seen.add(settled["candidate_id"])
                online.append(settled)
            tracked_online[seed, arm] = online
            checkpoint_path = directory / f"seed-{seed}" / arm / "checkpoint.json"
            checked_checkpoint = read(checkpoint_path)
            require(checked_checkpoint["plan_sha256"] == report["plan_sha256"] and
                    checked_checkpoint["evaluation"] == checkpoint, f"Invalid checkpoint: {checkpoint_path}")
            for key, expected in (("initial", initial), ("online", online)):
                projected = [{k: v for k, v in row.items() if k not in ("event", "task_id")}
                             | {"source_id": row["task_id"]} for row in expected]
                require(checked_checkpoint["optimizer"][key] == projected,
                        f"Checkpoint observation differs in {key}: {checkpoint_path}")
    require(not all_rows, f"Unexpected dispatches: {list(all_rows)}")
    active = peak = 0
    online_active = online_peak = 0
    for _, delta, online in sorted(windows):
        active += delta
        peak = max(active, peak)
        if online:
            online_active += delta
            online_peak = max(online_active, online_peak)
    starts = len(windows) // 2
    require(starts == report["worker_starts"] and peak == report["peak_workers"] <= plan["max_workers"],
            "Report and ledger disagree on charges or concurrency")
    require(report["actual_rcaide_calls"] == (starts if plan["backend"] == "rcaide" else 0),
            "Physical charge differs from backend starts")
    return {"status": "verified", "backend": plan["backend"], "checkpoint": checkpoint,
            "shared_starts": sum(map(len, roster.values())), "online_starts": sum(map(len, tracked_online.values())),
            "peak_workers": peak, "peak_online_workers": online_peak, "trajectories": len(tracked_online)}


def checked_call(events, job, backend, windows):
    require([e["event"] for e in events] == ["dispatch_intent", "worker_started", "settled"],
            f"Expected exactly one start and settlement: {job}")
    intent, started, settled = events
    seed, arm, position = job.split("/")
    require(all((str(e["seed"]), e["arm"], e["evaluation"], e["candidate_id"])
                == (seed.split("-", 1)[1], arm, int(position), settled["candidate_id"]) for e in events),
            f"Ledger identity changed: {job}")
    require(started["charged"] is True and settled["charged"] is True and started["backend"] == backend,
            f"Unexpected charge or backend: {job}")
    feedback = settled["feedback"]
    require(feedback["outcome"] != "infrastructure_error", f"Infrastructure failure: {job}")
    exact = feedback["objective_kind"] == "exact"
    require(exact == (feedback["completion"] is True) and exact == (feedback["objective_Wh"] is not None),
            f"Completion and objective mismatch: {job}")
    start = datetime.fromisoformat(started["started_at_utc"])
    finish = datetime.fromisoformat(settled["finished_at_utc"])
    require(start < finish, f"Invalid worker times: {job}")
    online = arm != "shared"
    windows.extend(((start, 1, online), (finish, -1, online)))
    return intent, started, settled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--checkpoint", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.checkpoint), indent=2))
