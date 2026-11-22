"""Pair 12 frozen X-57 designs across 40/45/50 nmi and replay energy feedback."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict
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
from run_x57_initialization_diagnostic import CONFIG_SHA256, DIAGNOSTIC_COMMIT, summarize, verified_hash
from ra_lsbo.budget import digest
from ra_lsbo.x57_cold_feedback import NUMERICAL, usable_energy_feedback
from ra_lsbo.x57_cold_sampling import LAYOUTS, active_names, violations
from ra_lsbo.x57_cold_worker import Dispatcher, dump, sha256
from ra_lsbo.x57_live import X57LiveProblem, evaluator_files

CODE_BASE = "c7631678eee403826ebed6bab427f844d91b21b2"
INITIALIZATION_COMMIT = "9cd0021b7c25d4813210c7164228031d7e857863"
INITIALIZATION_PLAN_SHA256 = "968e7056fd6e71c41d863a517533e5142e2f5376a2ca74e49660ec0ab081da05"
INITIALIZATION_LEDGER_SHA256 = "679440879f0e8ac6a1c4509b6baaa99f9e6f4db6804b7bef0a4e4f3f654b8d30"
DISTANCE_CONFIGS = {
    45: "3d2cdb22bad14ae7da3988c210b29bfce4ec87fbc0002d0f8145c1a9ce1f6118",
    50: "8ed89ff623a966f8bb033a04eb0de7b58448fb9dfa8f190162a68a44d50d0495",
}
MAX_CALLS = 24
WORKERS = 12


def replay_feedback(qualification_ledger):
    verified_hash(qualification_ledger, QUALIFICATION_LEDGER_SHA256)
    records = []
    for row in read_events(qualification_ledger):
        if row["event"] != "settled" or row["worker_result"]["status"] not in NUMERICAL:
            continue
        result = observation(row)
        metrics = row["worker_result"]["metrics"]
        lower, cap = result["terminal_energy_lower_Wh"], metrics["usable_terminal_energy_Wh"]
        if (result["outcome"] != "numerical_nonconvergence" or result["completion"] is not None
                or result["objective_kind"] != "missing" or result["objective_Wh"] is not None
                or metrics.get("mda_evaluated") is not True or cap != 47000
                or metrics.get("mission_profile", {}).get("cruise_distance_nmi") != 55
                or row["feedback"]["constraints"]["usable_energy"] is not None):
            raise ValueError(f"Unexpected 55 nmi source evidence for call {row['call_id']}")
        evidence = f"{QUALIFICATION_COMMIT}:{row['call_id']}:converged_terminal_prefix"
        shared = {key: result[key] for key in
                  ("outcome", "completion", "objective_kind", "objective_Wh", "risk_interval_W")}
        arms = {}
        for arm, use_interval in (("sign", False), ("interval", True)):
            observed = usable_energy_feedback(evidence, lower, cap, use_interval=use_interval)
            arms[arm] = {**shared, "usable_energy": asdict(observed),
                         "constraint_satisfied": observed.constraint_satisfaction()}
        if (arms["sign"]["constraint_satisfied"] != arms["interval"]["constraint_satisfied"]
                or any(arms["sign"][key] != arms["interval"][key] for key in shared)):
            raise ValueError("Projected engineering feedback altered shared evidence")
        records.append({"call_id": row["call_id"], "candidate_id": row["candidate_id"],
                        "failed_segment": result["failed_segment"], "terminal_energy_lower_Wh": lower,
                        "converged_energy_prefix": result["converged_energy_prefix"], "arms": arms})
    violators = [r for r in records if r["arms"]["sign"]["constraint_satisfied"] is False]
    return {"source_commit": QUALIFICATION_COMMIT, "source_ledger_sha256": QUALIFICATION_LEDGER_SHA256,
            "constraint": "usable_energy", "margin_definition": "terminal_energy_Wh / 47000 - 1",
            "terminal_cap_Wh": 47000, "failed_calls": len(records),
            "unique_failed_designs": len({r["candidate_id"] for r in records}),
            "calls_with_verified_terminal_lower": sum(r["terminal_energy_lower_Wh"] is not None for r in records),
            "calls_with_proven_violation": len(violators),
            "unique_designs_with_proven_violation": len({r["candidate_id"] for r in violators}),
            "identical_completion_and_objective_records": True, "records": records}


def prepare(args):
    init_folder = args.initialization_archive
    init_plan_path, init_ledger = init_folder / "initialization_plan.json", init_folder / "calls.jsonl"
    configs = {distance: args.distance_archive / "configs" / f"x57_qualification_{distance}_nmi.json"
               for distance in DISTANCE_CONFIGS}
    config_40 = init_folder / "x57_qualification_40_nmi.json"
    for path, expected in ((args.qualification_plan, QUALIFICATION_PLAN_SHA256),
                           (init_plan_path, INITIALIZATION_PLAN_SHA256),
                           (init_ledger, INITIALIZATION_LEDGER_SHA256), (config_40, CONFIG_SHA256),
                           *((path, DISTANCE_CONFIGS[distance]) for distance, path in configs.items())):
        verified_hash(path, expected)
    replay = replay_feedback(args.qualification_ledger)
    init_plan = json.loads(init_plan_path.read_bytes())
    if digest(init_plan["tasks"]) != init_plan["roster_sha256"]:
        raise ValueError("Frozen initialization roster digest mismatch")
    prefix = init_plan["tasks"][:12]
    if ([row["position"] for row in prefix] != list(range(1, 13))
            or Counter(row["layout"] for row in prefix) != dict.fromkeys(LAYOUTS, 6)):
        raise ValueError("Unexpected frozen prefix position or layout count")
    baselines = {row["evaluation"]: row for row in read_events(init_ledger) if row["event"] == "settled"}
    for row in prefix:
        baseline = baselines[row["position"]]
        if (not baseline["charged"] or any(baseline[key] != row[key] for key in
                                         ("candidate", "candidate_id", "precheck", "active_variables"))):
            raise ValueError(f"40 nmi settlement differs from plan at position {row['position']}")

    qualification = json.loads(args.qualification_plan.read_bytes())
    worker_root = args.qualification_plan.parent / "physical-base"
    worker_hashes = qualification["worker"]["worker_sha256"]
    if evaluator_files(worker_root) != worker_hashes:
        raise ValueError("Physical worker differs from the archived manifest")
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    threads = {name: os.environ[name] for name in THREAD_VARS}
    if args.execute_rcaide:
        identity = qualification["identity"]
        deps = {name: importlib.metadata.version(name) for name in identity["dependencies"]}
        rcaide_hash = sha256(Path(importlib.util.find_spec("RCAIDE").origin))
        if (sys.version != identity["python_version"] or deps != identity["dependencies"]
                or rcaide_hash != identity["rcaide_module"]["sha256"] or set(threads.values()) != {"1"}):
            raise ValueError("Interpreter, packages, or thread settings differ from the archived run")
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                   cwd=ROOT, text=True).strip():
            raise ValueError("Real evaluations require a committed checkout")
        subprocess.run(["git", "merge-base", "--is-ancestor", CODE_BASE, code_sha], cwd=ROOT, check=True)

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    config_dir = output / "configs"
    config_dir.mkdir()
    base = json.loads(config_40.read_bytes())
    shutil.copyfile(config_40, config_dir / config_40.name)
    problems = {}
    for distance, source in configs.items():
        expected = deepcopy(base)
        expected["mission"]["cruise_distance_nmi"] = distance
        if expected != json.loads(source.read_bytes()):
            raise ValueError(f"Config changes fields besides cruise distance for {distance} nmi")
        copied = config_dir / source.name
        shutil.copyfile(source, copied)
        verified_hash(copied, DISTANCE_CONFIGS[distance])
        problems[distance] = X57LiveProblem(worker_root, copied, output / "tasks")

    tasks = []
    for row in prefix:
        for distance, problem in problems.items():
            canonical = problem.canonical(row["candidate"])
            candidate = problem.candidate_class(**canonical)
            checks = problem.problem.precheck(candidate)
            if (canonical != row["candidate"] or problem.space.key(candidate) != row["candidate_id"]
                    or checks != row["precheck"] or violations(checks)
                    or active_names(problem.space, candidate) != row["active_variables"]):
                raise ValueError(f"Candidate/precheck changed at position {row['position']} and {distance} nmi")
            tasks.append({**row, "task_id": f"{row['position']:02d}-{distance}",
                          "distance_nmi": distance, "config_sha256": DISTANCE_CONFIGS[distance]})
    if len(tasks) != MAX_CALLS or len({t["task_id"] for t in tasks}) != MAX_CALLS:
        raise ValueError("Unexpected task roster")

    plan = {"schema": "x57_paired_distance_feedback_v1", "code_sha": code_sha, "code_base_sha": CODE_BASE,
            "initialization_commit": INITIALIZATION_COMMIT, "initialization_plan_sha256": INITIALIZATION_PLAN_SHA256,
            "initialization_ledger_sha256": INITIALIZATION_LEDGER_SHA256,
            "initialization_roster_sha256": init_plan["roster_sha256"],
            "distance_config_commit": DIAGNOSTIC_COMMIT,
            "config_sha256": {40: CONFIG_SHA256, **DISTANCE_CONFIGS},
            "qualification_plan_sha256": QUALIFICATION_PLAN_SHA256,
            "physical_worker_manifest_sha256": digest(worker_hashes),
            "backend": "rcaide" if args.execute_rcaide else "mock", "seed": init_plan["seed"],
            "threads": threads, "workers": WORKERS, "timeout_seconds": TIMEOUT,
            "max_calls": MAX_CALLS, "task_roster_sha256": digest(tasks), "tasks": tasks}
    dump(output / "run_plan.json", plan)
    dump(output / "feedback_replay.json", replay)
    (output / "tasks").mkdir()
    return plan, problems, worker_hashes, baselines


def execute(args):
    started = time.perf_counter()
    plan, problems, worker_hashes, baselines = prepare(args)
    output = args.output

    def launch(task):
        problem = problems[task["distance_nmi"]]
        folder = output / "tasks" / task["task_id"]
        folder.mkdir()
        dispatcher = Dispatcher(problem, folder, backend=plan["backend"], timeout=TIMEOUT, max_calls=1,
                                worker_root=problem.base if args.execute_rcaide else ROOT,
                                expected_worker=worker_hashes, mock_profile="distance-diagnostic")
        point = problem.candidate_class(**task["candidate"])
        row = dispatcher.evaluate(point, seed=plan["seed"], arm="paired_distance",
                                  evaluation=task["position"], online=0, phase="diagnostic",
                                  proposal={"distance_nmi": task["distance_nmi"], "task_id": task["task_id"]})
        result = observation(row)
        return {**result, "status": result["outcome"]}

    results = run_wave(plan["tasks"], WORKERS, launch)
    rows, counts = [], Counter()
    with (output / "calls.jsonl").open("x", encoding="utf-8") as merged:
        for task in plan["tasks"]:
            task_id = task["task_id"]
            journal = output / "tasks" / task_id / "calls.jsonl"
            events = read_events(journal) if journal.is_file() else []
            counts.update(event["event"] for event in events)
            for event in events:
                merged.write(json.dumps({"task_id": task_id, **event}, allow_nan=False) + "\n")
            worker = next((e for e in events if e["event"] == "worker_started"), {})
            end = next((e for e in events if e["event"] in ("settled", "interrupted")), {})
            result = results.get(task_id, {"status": "skipped_infrastructure_stop"})
            reference = observation(baselines[task["position"]])
            objective, baseline = result.get("objective_Wh"), reference["objective_Wh"]
            rows.append({**{k: task[k] for k in ("task_id", "position", "distance_nmi", "candidate_id", "layout")},
                         **result, "outcome": result["status"], "baseline_40_nmi": reference,
                         "objective_delta_vs_40_Wh": (objective - baseline if objective is not None
                                                      and baseline is not None else None),
                         "charged": bool(worker), "pid": worker.get("pid"),
                         "started_at_utc": worker.get("started_at_utc"), "finished_at_utc": end.get("finished_at_utc")})
    failed = {"infrastructure_error", "skipped_infrastructure_stop"}
    report = {"status": "infrastructure_stop" if any(row["outcome"] in failed for row in rows) else "complete",
              "backend": plan["backend"], "plan_sha256": sha256(output / "run_plan.json"),
              "worker_charges": counts["worker_started"],
              "rcaide_calls": counts["worker_started"] if args.execute_rcaide else 0,
              "settlements": counts["settled"], "interruptions": counts["interrupted"],
              "peak_concurrent_workers": elapsed_concurrency(rows), "wall_seconds": time.perf_counter() - started,
              "distances": {distance: summarize([row for row in rows if row["distance_nmi"] == distance])["layouts"]
                            for distance in DISTANCE_CONFIGS},
              "tasks": rows}
    dump(output / "diagnostic_report.json", report)
    fields = ("position", "layout", "distance_nmi", "task_id", "outcome", "baseline_40_outcome", "failed_segment",
              "constraint_violations", "terminal_energy_lower_Wh", "terminal_energy_Wh", "final_soc",
              "objective_Wh", "baseline_40_objective_Wh", "objective_delta_vs_40_Wh", "objective_kind",
              "elapsed_seconds", "pid", "started_at_utc", "finished_at_utc")
    with (output / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {key: row.get(key) for key in fields}
            values.update(constraint_violations=json.dumps(row.get("constraint_violations", {})),
                          baseline_40_outcome=row["baseline_40_nmi"]["outcome"],
                          baseline_40_objective_Wh=row["baseline_40_nmi"]["objective_Wh"])
            writer.writerow(values)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("initialization-archive", "distance-archive", "qualification-plan", "qualification-ledger", "output"):
        parser.add_argument("--" + option, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--execute-rcaide", action="store_true")
    args = parser.parse_args()
    report = execute(args)
    print(args.output / "diagnostic_report.json")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
