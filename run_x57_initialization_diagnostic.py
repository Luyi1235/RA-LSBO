"""Evaluate one frozen 32-point X-57 initialization at the archived 40 nmi setting."""

import argparse
from collections import Counter
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_x57_distance_diagnostic import (
    QUALIFICATION_COMMIT, QUALIFICATION_LEDGER_SHA256, QUALIFICATION_PLAN_SHA256,
    ROOT, THREAD_VARS, TIMEOUT, elapsed_concurrency, observation, read_events, run_wave,
)
from ra_lsbo.budget import digest
from ra_lsbo.x57_cold_sampling import LAYOUT, LAYOUTS, PER_LAYOUT, active_names, freeze_roster, violations
from ra_lsbo.x57_cold_worker import Dispatcher, dump, sha256
from ra_lsbo.x57_live import X57LiveProblem, evaluator_files

CODE_BASE = "17bf9d327a2eff5dff170a972bbdfb4e3779a421"
DIAGNOSTIC_COMMIT = "214e11fa9969b3844d922a8872f2d647ec7077d3"
DIAGNOSTIC_LEDGER_SHA256 = "2059e74eaf53c4337d1b70851d163b6cd7dab1f98e2bdd895ea877c425a27293"
CONFIG_SHA256 = "2b884db6abf4ae22bbd482c296e58453cc420441b3f482323507e0b7fb506709"
WORKERS = 12
MAX_CALLS = PER_LAYOUT * len(LAYOUTS)


def verified_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Input differs from its archived bytes: {path}")
    return actual


def prepare(args):
    paths = (
        (args.qualification_ledger, QUALIFICATION_LEDGER_SHA256),
        (args.diagnostic_ledger, DIAGNOSTIC_LEDGER_SHA256),
        (args.qualification_plan, QUALIFICATION_PLAN_SHA256),
        (args.config, CONFIG_SHA256),
    )
    for path, expected in paths:
        verified_hash(path, expected)
    qualification = json.loads(args.qualification_plan.read_bytes())
    worker_root = args.qualification_plan.parent / "physical-base"
    worker_hashes = qualification["worker"]["worker_sha256"]
    if evaluator_files(worker_root) != worker_hashes:
        raise ValueError("Physical worker differs from the archived manifest")

    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    threads = {var: os.environ[var] for var in THREAD_VARS}
    dependencies = {name: importlib.metadata.version(name) for name in qualification["identity"]["dependencies"]}
    rcaide_hash = sha256(Path(importlib.util.find_spec("RCAIDE").origin))
    if args.execute_rcaide:
        identity = qualification["identity"]
        if (sys.version != identity["python_version"] or dependencies != identity["dependencies"]
                or rcaide_hash != identity["rcaide_module"]["sha256"] or set(threads.values()) != {"1"}):
            raise ValueError("Interpreter, libraries, or thread settings differ from the frozen run")
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                   cwd=ROOT, text=True).strip():
            raise ValueError("Real initialization requires committed code")
        subprocess.run(["git", "merge-base", "--is-ancestor", CODE_BASE, code_sha], cwd=ROOT, check=True)

    exclusions = {}
    for name, path in (("qualification", args.qualification_ledger), ("distance", args.diagnostic_ledger)):
        events = [row for row in read_events(path) if row["event"] == "worker_started"]
        exclusions[name] = sorted({row["candidate_id"] for row in events})
    excluded = set().union(*exclusions.values())
    args.output.mkdir(parents=True, exist_ok=False)
    config = args.output / "x57_qualification_40_nmi.json"
    shutil.copyfile(args.config, config)
    verified_hash(config, CONFIG_SHA256)
    problem = X57LiveProblem(worker_root, config, args.output / "tasks")
    if problem.config["mission"]["cruise_distance_nmi"] != 40 or problem.config["runtime"]["timeout_seconds"] != TIMEOUT:
        raise ValueError("Unexpected distance or worker timeout")

    candidates, coverage = freeze_roster(problem.problem, args.seed, exclude_keys=excluded)
    tasks = []
    for position, point in enumerate(candidates, 1):
        candidate = point.to_dict()
        tasks.append({"task_id": f"{position:02d}", "position": position, "layout": point.discrete[LAYOUT],
                      "candidate": candidate, "candidate_id": problem.space.key(point),
                      "precheck": problem.problem.precheck(point),
                      "active_variables": active_names(problem.space, point)})
    ids = {task["candidate_id"] for task in tasks}
    if (len(tasks) != MAX_CALLS or len(ids) != MAX_CALLS or ids & excluded
            or [task["layout"] for task in tasks] != list(LAYOUTS) * PER_LAYOUT
            or any(violations(task["precheck"]) for task in tasks)):
        raise ValueError("Roster has an invalid layout quota, duplicate, or precheck")

    backend = "rcaide" if args.execute_rcaide else "mock"
    plan = {"schema": "x57_40nmi_initialization_v1", "backend": backend, "seed": args.seed,
            "code_sha": code_sha, "code_base_sha": CODE_BASE, "distance_nmi": 40,
            "config_sha256": CONFIG_SHA256, "qualification_plan_sha256": QUALIFICATION_PLAN_SHA256,
            "sources": {"qualification": {"commit": QUALIFICATION_COMMIT, "sha256": QUALIFICATION_LEDGER_SHA256},
                        "distance": {"commit": DIAGNOSTIC_COMMIT, "sha256": DIAGNOSTIC_LEDGER_SHA256}},
            "excluded_candidate_ids": {**exclusions, "union": sorted(excluded)},
            "coverage": coverage, "workers": WORKERS, "max_calls": MAX_CALLS, "timeout_seconds": TIMEOUT,
            "physical_worker_manifest_sha256": digest(worker_hashes),
            "environment": {"python_executable": sys.executable, "python_version": sys.version,
                            "dependencies": dependencies, "rcaide_module_sha256": rcaide_hash, "threads": threads},
            "roster_sha256": digest(tasks), "tasks": tasks}
    plan_path = args.output / "initialization_plan.json"
    dump(plan_path, plan)
    print(f"Frozen roster SHA256: {plan['roster_sha256']}", flush=True)
    return plan, sha256(plan_path), problem, worker_hashes


def summarize(rows):
    ordered_usable = [row for row in rows if row.get("usable")]

    def metrics(selected):
        outcomes = Counter(row["outcome"] for row in selected)
        usable = [row for row in selected if row.get("usable")]
        return {"outcomes": dict(outcomes), "usable": len(usable),
                "first_three_usable_positions": [row["position"] for row in usable[:3]],
                "best_usable_objective_Wh": min((row["objective_Wh"] for row in usable), default=None),
                "verified_terminal_energy_lowers": sum(row.get("terminal_energy_lower_Wh") is not None
                                                       for row in selected),
                "verified_risk_intervals": sum(row.get("risk_interval_W") is not None for row in selected)}

    return {"layouts": {layout: metrics([row for row in rows if row["layout"] == layout]) for layout in LAYOUTS},
            "overall": {**metrics(rows), "initialization_group": (
                "0" if not ordered_usable else "1-2" if len(ordered_usable) < 3 else "3+")}}


def execute(args):
    started = time.perf_counter()
    plan, plan_hash, problem, worker_hashes = prepare(args)
    output = args.output
    tasks_dir = output / "tasks"
    tasks_dir.mkdir()

    def launch(task):
        folder = tasks_dir / task["task_id"]
        folder.mkdir()
        dispatcher = Dispatcher(problem, folder, backend=plan["backend"], timeout=TIMEOUT, max_calls=1,
                                worker_root=problem.base if args.execute_rcaide else ROOT,
                                expected_worker=worker_hashes, mock_profile="distance-diagnostic")
        point = problem.candidate_class(**task["candidate"])
        row = dispatcher.evaluate(point, seed=plan["seed"], arm="shared_initial", evaluation=task["position"],
                                  online=0, phase="initial", proposal={"source": "frozen_diagnostic_roster"})
        result = observation(row)
        return {**result, "status": result["outcome"]}

    results = run_wave(plan["tasks"], WORKERS, launch)
    rows, counts = [], Counter()
    with (output / "calls.jsonl").open("x", encoding="utf-8") as merged:
        for task in plan["tasks"]:
            task_id = task["task_id"]
            journal = tasks_dir / task_id / "calls.jsonl"
            events = read_events(journal) if journal.is_file() else []
            counts.update(event["event"] for event in events)
            for event in events:
                merged.write(json.dumps({"task_id": task_id, **event}, allow_nan=False) + "\n")
            worker = next((event for event in events if event["event"] == "worker_started"), {})
            end = next((event for event in events if event["event"] in ("settled", "interrupted")), {})
            result = results.get(task_id, {"status": "skipped_infrastructure_stop"})
            record = {key: task[key] for key in ("task_id", "position", "layout", "candidate_id")}
            rows.append({**record, **result, "outcome": result["status"], "charged": bool(worker),
                         "pid": worker.get("pid"), "started_at_utc": worker.get("started_at_utc"),
                         "finished_at_utc": end.get("finished_at_utc")})
    status = "complete" if all(row["outcome"] not in ("infrastructure_error", "skipped_infrastructure_stop")
                               for row in rows) else "infrastructure_stop"
    report = {"status": status, "backend": plan["backend"], "seed": plan["seed"],
              "plan_sha256": plan_hash, "roster_sha256": plan["roster_sha256"],
              "worker_charges": counts["worker_started"], "rcaide_calls": counts["worker_started"] if args.execute_rcaide else 0,
              "settlements": counts["settled"], "interruptions": counts["interrupted"],
              "peak_concurrent_workers": elapsed_concurrency(rows), "wall_seconds": time.perf_counter() - started,
              **summarize(rows), "tasks": rows}
    dump(output / "initialization_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qualification-ledger", "qualification-plan", "diagnostic-ledger", "config", "output"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute-rcaide", action="store_true")
    args = parser.parse_args()
    report = execute(args)
    print(args.output / "initialization_report.json")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
