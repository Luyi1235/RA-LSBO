"""Run the paired X-57 cruise-distance diagnostic with isolated workers."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import csv
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
import time

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for variable in THREAD_VARS:
    os.environ.setdefault(variable, "1")

from ra_lsbo.budget import digest
from ra_lsbo.risk_core.observations import finite
from ra_lsbo.x57_cold_sampling import LAYOUT, LAYOUTS, active_names, violations
from ra_lsbo.x57_cold_worker import Dispatcher, dump, sha256
from ra_lsbo.x57_energy_stop import prefix_energy, verified_stop
from ra_lsbo.x57_live import X57LiveProblem, evaluator_files

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs/x57_cold_start_qualification_55.json"
CODE_BASE = "94abf5a455e7913f81eb9ec39da01006e77bb162"
QUALIFICATION_COMMIT = "838c85fe8c9bd4662278d9ff05386fe63a690336"
QUALIFICATION_LEDGER_SHA256 = "3dd0c35fc715b22bae276684f868f282f88d600733467636e464d6aae6263780"
QUALIFICATION_PLAN_SHA256 = "22ee810579fb83d35bf32869f388911d21cea51b2e53a73321c4d29df8bb34c8"
DISTANCES = (40, 45, 50)
BASELINE_CALLS = (("000001", LAYOUTS[0]), ("000015", LAYOUTS[0]),
                  ("000002", LAYOUTS[1]), ("000012", LAYOUTS[1]))
TIMEOUT = 1800
MAX_CALLS = 14


def read_events(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def qualification_inputs(ledger_path, plan_path):
    ledger_hash, plan_hash = sha256(ledger_path), sha256(plan_path)
    if ledger_hash != QUALIFICATION_LEDGER_SHA256 or plan_hash != QUALIFICATION_PLAN_SHA256:
        raise ValueError("Input differs from the archived 55 nmi ledger or plan")
    plan = json.loads(plan_path.read_bytes())
    points = plan["points"]
    if (plan["schema"] != "x57_cold_qualification_v1" or len(points) != 20
            or plan["roster_sha256"] != digest(points)):
        raise ValueError("Invalid frozen qualification roster")
    settled = {r["call_id"]: r for r in read_events(ledger_path) if r["event"] == "settled"}
    anchors = {}
    for call_id, layout in BASELINE_CALLS:
        row, frozen = settled[call_id], points[int(call_id) - 1]
        fields = ("candidate", "candidate_id", "precheck", "active_variables")
        if (row["evaluation"] != int(call_id) or not row["charged"] or frozen["layout"] != layout
                or row["candidate"]["discrete"][LAYOUT] != layout
                or any(row[k] != frozen[k] for k in fields) or violations(row["precheck"])):
            raise ValueError(f"Archived candidate or precheck mismatch for {call_id}")
        anchors[call_id] = row
    return plan, anchors


def prepare(output, ledger_path, plan_path, workers, execute):
    plan, anchors = qualification_inputs(ledger_path, plan_path)
    worker_root = plan_path.parent / "physical-base"
    worker_hashes = plan["worker"]["worker_sha256"]
    if evaluator_files(worker_root) != worker_hashes:
        raise ValueError("physical-base differs from the archived worker manifest")
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    rcaide = importlib.util.find_spec("RCAIDE")
    environment = {"python_executable": sys.executable, "python_version": sys.version,
                   "dependencies": {name: importlib.metadata.version(name)
                                    for name in plan["identity"]["dependencies"]},
                   "rcaide_module_sha256": sha256(Path(rcaide.origin)),
                   "threads": {name: os.environ[name] for name in THREAD_VARS}}
    if execute:
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                   cwd=ROOT, text=True).strip():
            raise ValueError("Real diagnostics require committed code")
        subprocess.run(["git", "merge-base", "--is-ancestor", CODE_BASE, code_sha], cwd=ROOT, check=True)
        baseline_identity = plan["identity"]
        if (environment["python_version"] != baseline_identity["python_version"]
                or environment["dependencies"] != baseline_identity["dependencies"]
                or environment["rcaide_module_sha256"] != baseline_identity["rcaide_module"]["sha256"]
                or any(value != "1" for value in environment["threads"].values())):
            raise ValueError("Real diagnostics require the frozen interpreter, libraries and one compute thread")
    base_config = json.loads(CONFIG.read_bytes())
    if (sha256(CONFIG) != plan["identity"]["config_sha256"] or base_config != plan["identity"]["config"]
            or base_config["mission"]["cruise_distance_nmi"] != 55
            or base_config["runtime"]["timeout_seconds"] != TIMEOUT):
        raise ValueError("The 55 nmi qualification config changed")
    config_dir = output / "configs"
    config_dir.mkdir()
    configs, problems = {}, {}
    for distance in DISTANCES:
        path = config_dir / f"x57_qualification_{distance}_nmi.json"
        config = deepcopy(base_config)
        config["mission"]["cruise_distance_nmi"] = float(distance)
        dump(path, config)
        configs[distance] = {"path": path.relative_to(output).as_posix(), "sha256": sha256(path)}
        problems[distance] = X57LiveProblem(worker_root, path, output / "tasks")
    tasks, candidates = [], {}
    for distance in DISTANCES:
        problem = problems[distance]
        for call_id, layout in BASELINE_CALLS:
            anchor = anchors[call_id]
            canonical = problem.canonical(anchor["candidate"])
            candidate = problem.candidate_class(**canonical)
            if (canonical != anchor["candidate"] or problem.space.key(candidate) != anchor["candidate_id"]
                    or active_names(problem.space, candidate) != anchor["active_variables"]
                    or problem.problem.precheck(candidate) != anchor["precheck"]):
                raise ValueError(f"Candidate/precheck changed at {distance} nmi for {call_id}")
            candidates[distance, call_id] = candidate
            tasks.append({"task_id": f"{len(tasks) + 1:02d}", "phase": "diagnostic", "distance_nmi": distance,
                          "baseline_call_id": call_id, "layout": layout, "candidate_id": anchor["candidate_id"],
                          "precheck": anchor["precheck"], "config_sha256": configs[distance]["sha256"]})
    run_plan = {"schema": "x57_distance_diagnostic_v1", "backend": "rcaide" if execute else "mock",
                "code_sha": code_sha, "code_base_sha": CODE_BASE, "qualification_commit": QUALIFICATION_COMMIT,
                "qualification_ledger_sha256": QUALIFICATION_LEDGER_SHA256,
                "qualification_plan_sha256": QUALIFICATION_PLAN_SHA256,
                "qualification_config_sha256": sha256(CONFIG), "configs": configs,
                "physical_worker": {"manifest_sha256": digest(worker_hashes), "file_count": len(worker_hashes)},
                "environment": environment, "workers": workers, "timeout_seconds": TIMEOUT,
                "max_worker_calls": MAX_CALLS, "tasks": tasks,
                "repeat_policy": "Longest common usable distance, lowest objective per layout, then task ID"}
    dump(output / "run_plan.json", run_plan)
    return tasks, problems, candidates, anchors, worker_hashes


def run_wave(tasks, workers, launch):
    """Drain admitted calls after a fault and stop admitting new tasks."""
    stop = Event()
    pending = {}
    results = {}
    remaining = iter(tasks)

    def guarded(task):
        if stop.is_set():
            return None
        try:
            result = launch(task)
        except Exception as error:
            result = {"status": "infrastructure_error", "error": f"{type(error).__name__}: {error}"}
        if result["status"] == "infrastructure_error":
            stop.set()
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        def fill():
            while not stop.is_set() and len(pending) < workers:
                task = next(remaining, None)
                if task is None:
                    break
                pending[pool.submit(guarded, task)] = task["task_id"]

        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = pending.pop(future)
                if (result := future.result()) is not None:
                    results[task_id] = result
            fill()
    return results


def observation(row):
    feedback, raw = row["feedback"], row["worker_result"]
    metrics = raw.get("metrics", {})
    segments = metrics.get("mission_solver_segments", [])
    completed = feedback["completion"] is True
    audited = completed and metrics.get("energy_audit_status") == "passed"
    lower, prefix = (None, []) if completed else prefix_energy(segments)
    result = {"outcome": feedback["outcome"], "completion": feedback["completion"], "usable": feedback["usable"],
            "worker_status": raw.get("status"), "failure_stage": metrics.get("failure_stage"),
            "failure_message": metrics.get("failure_message") or metrics.get("message"),
            "failed_segment": next((s["segment"] for s in segments if s.get("converged") is False), None),
            "constraint_violations": {k: v for k, v in feedback["constraints"].items() if finite(v) and v > 0},
            "unobserved_constraints": [k for k, v in feedback["constraints"].items() if v is None],
            "final_soc": metrics.get("final_soc") if audited else None,
            "terminal_energy_Wh": metrics.get("terminal_energy_Wh") if audited else None,
            "minimum_loaded_pack_voltage_V": metrics.get("minimum_loaded_pack_voltage_V") if audited else None,
            "stall_speed_mps": metrics.get("stall_speed_mps") if completed else None,
            "objective_Wh": feedback["objective_Wh"], "objective_kind": feedback["objective_kind"],
            "risk_interval_W": feedback["risk_interval_W"],
            "terminal_energy_lower_Wh": lower, "converged_energy_prefix": prefix,
            "prefix_end_soc": prefix[-1]["final_soc"] if prefix else None,
            "elapsed_seconds": row["elapsed_seconds"]}
    if stop := verified_stop(raw):
        result.update(stop_after_segment=stop["stop_after_segment"],
                      known_constraint_violations=feedback["known_constraint_violations"])
    return result


def choose_repeats(tasks, results):
    for distance in reversed(DISTANCES):
        selected = []
        for layout in LAYOUTS:
            available = [t for t in tasks if t["distance_nmi"] == distance and t["layout"] == layout
                         and results[t["task_id"]].get("usable")]
            if not available:
                break
            selected.append(min(available, key=lambda t: (results[t["task_id"]]["objective_Wh"], t["task_id"])))
        if len(selected) == len(LAYOUTS):
            return [{**task, "task_id": str(i), "phase": "repeat", "repeat_of": task["task_id"]}
                    for i, task in enumerate(selected, start=13)]
    return []


def elapsed_concurrency(rows):
    edges = [(row[key], step) for row in rows
             if row.get("started_at_utc") and row.get("finished_at_utc")
             for key, step in (("started_at_utc", 1), ("finished_at_utc", -1))]
    peak = current = 0
    for _, step in sorted(edges):
        current += step
        peak = max(peak, current)
    return peak


def execute_diagnostic(output, ledger, plan, workers, execute):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    tasks, problems, candidates, anchors, worker_hashes = prepare(output, ledger, plan, workers, execute)
    baseline = {call_id: observation(row) for call_id, row in anchors.items()}
    (output / "tasks").mkdir()
    backend = "rcaide" if execute else "mock"

    def launch(task):
        task_id, distance, anchor = (task[k] for k in ("task_id", "distance_nmi", "baseline_call_id"))
        folder = output / "tasks" / task_id
        folder.mkdir()
        dispatcher = Dispatcher(problems[distance], folder, backend=backend, timeout=TIMEOUT, max_calls=1,
                                worker_root=plan.parent / "physical-base" if execute else ROOT,
                                expected_worker=worker_hashes if execute else None, mock_profile="distance-diagnostic")
        row = dispatcher.evaluate(candidates[distance, anchor], seed=None, arm="distance_diagnostic",
                                  evaluation=int(task_id), online=0, phase=task["phase"], proposal=task)
        return {**observation(row), "status": row["feedback"]["outcome"]}

    results = run_wave(tasks, workers, launch)
    fault = any(r["status"] == "infrastructure_error" for r in results.values())
    repeats = [] if fault else choose_repeats(tasks, results)
    if repeats:
        dump(output / "repeat_plan.json", {"distance_nmi": repeats[0]["distance_nmi"], "tasks": repeats})
        results.update(run_wave(repeats, 2, launch))
        fault = any(r["status"] == "infrastructure_error" for r in results.values())

    rows = []
    charges = settled = interrupted = 0
    with (output / "calls.jsonl").open("x", encoding="utf-8") as merged:
        for task in tasks + repeats:
            task_id = task["task_id"]
            journal = output / "tasks" / task_id / "calls.jsonl"
            events = read_events(journal) if journal.is_file() else []
            for event in events:
                merged.write(json.dumps({"task_id": task_id, **event}, allow_nan=False) + "\n")
            starts = [event for event in events if event["event"] == "worker_started"]
            charges += len(starts)
            settled += sum(event["event"] == "settled" for event in events)
            interrupted += sum(event["event"] == "interrupted" for event in events)
            end = next((e for e in reversed(events) if e["event"] in ("settled", "interrupted")), {})
            result = results.get(task_id, {"status": "skipped_infrastructure_stop"})
            row = {**task, **result, "charged": bool(starts), "pid": starts[0]["pid"] if starts else None,
                   "started_at_utc": starts[0]["started_at_utc"] if starts else None,
                   "finished_at_utc": end.get("finished_at_utc")}
            reference = baseline[task["baseline_call_id"]]["objective_Wh"]
            objective = result.get("objective_Wh")
            row["baseline_objective_Wh"] = reference
            row["objective_delta_vs_55_Wh"] = objective - reference if finite(objective) and finite(reference) else None
            rows.append(row)
    comparisons = []
    for task in repeats:
        first, repeat = (results.get(task[key], {}) for key in ("repeat_of", "task_id"))
        before, after = first.get("objective_Wh"), repeat.get("objective_Wh")
        comparisons.append({"first_task": task["repeat_of"], "repeat_task": task["task_id"],
                            "same_outcome": bool(repeat) and first.get("outcome") == repeat.get("outcome"),
                            "both_usable": first.get("usable") is True and repeat.get("usable") is True,
                            "energy_delta_Wh": after - before if finite(before) and finite(after) else None})
    report = {"status": "infrastructure_stop" if fault else "complete", "backend": backend,
              "worker_charges": charges, "rcaide_calls": charges if execute else 0,
              "settled_results": settled, "interrupted_results": interrupted,
              "peak_concurrent_workers": elapsed_concurrency(rows),
              "common_usable_distance_nmi": repeats[0]["distance_nmi"] if repeats else None,
              "repeat_comparisons": comparisons, "wall_seconds": time.perf_counter() - started,
              "baseline_55_nmi": baseline, "tasks": rows}
    dump(output / "diagnostic_report.json", report)
    fields = ("task_id", "phase", "repeat_of", "layout", "distance_nmi", "baseline_call_id", "status",
              "completion", "usable", "failed_segment", "failure_stage", "failure_message", "constraint_violations",
              "final_soc", "terminal_energy_Wh", "terminal_energy_lower_Wh", "prefix_end_soc",
              "minimum_loaded_pack_voltage_V", "stall_speed_mps", "objective_Wh", "objective_kind",
              "baseline_objective_Wh", "objective_delta_vs_55_Wh", "elapsed_seconds",
              "charged", "pid", "started_at_utc", "finished_at_utc")
    with (output / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: (json.dumps(row[field]) if field == "constraint_violations" and field in row
                                     else row.get(field)) for field in fields})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-ledger", type=Path, required=True)
    parser.add_argument("--qualification-plan", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12, choices=range(1, 13), metavar="1..12")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-rcaide", action="store_true")
    args = parser.parse_args()
    report = execute_diagnostic(args.output.resolve(), args.qualification_ledger.resolve(),
                                args.qualification_plan.resolve(), args.workers, args.execute_rcaide)
    print(args.output.resolve() / "diagnostic_report.json")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
