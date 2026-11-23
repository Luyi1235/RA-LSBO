"""Verify, calibrate and run the bounded two-seed X-57 cold-start diagnostic."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from threading import BoundedSemaphore, Event, Lock
from types import SimpleNamespace

from run_x57_distance_diagnostic import (
    ROOT, THREAD_VARS, TIMEOUT, elapsed_concurrency, observation, prefix_energy, read_events, run_wave,
)
from run_x57_engineering_energy_pilot import (
    ARMS, INITIAL, ONLINE, code_sha, offline, pilot, points, verify_inputs,
)
from run_x57_cold_debug_replay import ARCHIVE, LEDGER_HASH, archive_rows, proposals, stop_replay
from run_x57_initialization_diagnostic import verified_hash
from ra_lsbo.budget import digest
from ra_lsbo.x57_cold_campaign import identity
from ra_lsbo.x57_cold_sampling import LAYOUTS
from ra_lsbo.x57_cold_worker import Dispatcher, dump, sha256
from ra_lsbo.x57_energy_stop import CAP_WH, POLICY
from ra_lsbo.x57_energy_surrogate import energy_training_row
from ra_lsbo.x57_live import X57LiveProblem

import torch

CALIBRATION_SEED = 20261126
SEEDS = (20261127, 20261129)
DISTANCES = (45, 47.5)
TARGET_WH = 46000.
N_CALIBRATION = 12
WORKERS = 12
FIXED_CALL_CAP = 112
CHECKPOINTS = (8, 9, 10, 12, 16, 20)
CHOICE_RULE = ("Prefer a distance with verified partial violations and usable designs, then the number of "
               "layouts with a usable design, then verified partial violations, then the longer distance. "
               "With zero usable designs at both distances, keep the previously solvable 45 nmi reference.")
EXCLUSION_LEDGERS = {
    "distance_diagnostic": ("214e11fa9969b3844d922a8872f2d647ec7077d3",
                            "2059e74eaf53c4337d1b70851d163b6cd7dab1f98e2bdd895ea877c425a27293"),
    "engineering_pilot": ("613983af769de05ce7ca8a15a7c0be7bd38963aa",
                          "f413027de9c35a2b133459a3e466d701f5eb6c8b8362cef200e6cf032add35d3"),
}


def verify_environment(problem, qualification):
    environment = identity(problem)
    reference = qualification["identity"]
    if (any(environment[k] != reference[k] for k in ("python_version", "dependencies", "lsbo_sha256"))
            or environment["rcaide_module"]["sha256"] != reference["rcaide_module"]["sha256"]
            or any(os.environ[v] != "1" for v in THREAD_VARS)):
        raise ValueError("Run with the qualified environment and four single-thread settings")
    return environment


def frozen_setup(output, base, configs, excluded):
    directory = output / "configs"
    directory.mkdir()
    original = json.loads(configs[45].read_bytes())
    config_paths = {}
    problems = {}
    for distance in DISTANCES:
        path = directory / f"distance_{distance}.json"
        if distance == 45:
            path.write_bytes(configs[45].read_bytes())
        else:
            config = deepcopy(original)
            config["mission"]["cruise_distance_nmi"] = distance
            dump(path, config)
        config_paths[distance] = path
        problems[distance] = X57LiveProblem(base, path, output / "workers")
    problem = problems[45]
    roster, calibration, sampling = points(problem, CALIBRATION_SEED, excluded, N_CALIBRATION)
    used = set(excluded) | {p["candidate_id"] for p in calibration}
    frozen = {}
    for seed in SEEDS:
        _, records, audit = points(problem, seed, used, INITIAL)
        frozen[seed] = {"roster": records, "sampling": audit, "exclude": sorted(used)}
        used.update(p["candidate_id"] for p in records)
    for distance in DISTANCES:
        for point in roster:
            if problems[distance].space.key(point) != problem.space.key(point) or (
                    problems[distance].problem.precheck(point) != problem.problem.precheck(point)):
                raise ValueError("Distance change altered candidate identity or precheck")
    tasks = [{**row, "task_id": f"{row['position']:02d}-{distance}", "distance_nmi": distance}
             for row in calibration for distance in DISTANCES]
    return problems, config_paths, list(zip(roster, calibration)), tasks, sampling, frozen


def merge_journals(output, journals):
    events, windows = [], []
    for name, metadata in journals:
        path = output / name / "calls.jsonl"
        if not path.is_file():
            continue
        rows = read_events(path)
        for event in rows:
            events.append({**event, **metadata, "journal": f"{name}/{event['journal']}"
                           if "journal" in event else name})
        ends = {(r.get("journal"), r["call_id"]): r for r in rows if r["event"] in ("settled", "interrupted")}
        for event in rows:
            if event["event"] == "worker_started":
                key = (event.get("journal"), event["call_id"])
                windows.append({**metadata, "seed": event["seed"], "arm": event["arm"],
                                "pid": event["pid"], "started_at_utc": event["started_at_utc"],
                                "finished_at_utc": ends.get(key, {}).get("finished_at_utc")})
    with (output / "calls.jsonl").open("x", encoding="utf-8") as stream:
        for row in events:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return windows


def calibration(output, problems, worker, roster, tasks, args):
    def launch(task):
        distance = task["distance_nmi"]
        point = roster[task["position"] - 1][0]
        folder = output / task["task_id"]
        folder.mkdir()
        dispatcher = Dispatcher(
            problems[distance], folder, backend="rcaide" if args.execute_rcaide else "mock",
            worker_root=problems[distance].base if args.execute_rcaide else ROOT, timeout=TIMEOUT,
            expected_worker=worker, max_calls=1, energy_stop=True, mock_profile="cold-start-debug")
        row = dispatcher.evaluate(point, seed=CALIBRATION_SEED, arm="calibration", evaluation=task["position"],
                                  online=0, phase="calibration",
                                  proposal={"calibration_task_id": task["task_id"], "distance_nmi": distance})
        evidence, _ = energy_training_row(row, task["task_id"], distance, prefix_energy, True)
        return {"status": row["feedback"]["outcome"], **observation(row),
                "verified_energy_violation": evidence.kind == "interval", "energy_bound": evidence.lower}

    completed = run_wave(tasks, args.workers, launch)
    rows = [{**task, **completed.get(task["task_id"], {"status": "not_admitted"})} for task in tasks]
    windows = merge_journals(output, [(r["task_id"], {"calibration_task": r["task_id"]}) for r in tasks])
    groups = {str(distance): {layout: {
        "outcomes": dict(Counter(r["status"] for r in rows
                                 if r["distance_nmi"] == distance and r["layout"] == layout)),
        "usable": sum(r.get("usable", False) for r in rows
                      if r["distance_nmi"] == distance and r["layout"] == layout),
        "verified_energy_violations": sum(r.get("verified_energy_violation", False) for r in rows
                                          if r["distance_nmi"] == distance and r["layout"] == layout),
        "valid_discharge_prefixes": sum(r.get("terminal_energy_lower_Wh") is not None for r in rows
                                        if r["distance_nmi"] == distance and r["layout"] == layout),
    } for layout in LAYOUTS} for distance in DISTANCES}
    rank = {}
    for distance in DISTANCES:
        layouts = list(groups[str(distance)].values())
        usable_layouts = sum(group["usable"] > 0 for group in layouts)
        partials = sum(group["verified_energy_violations"] for group in layouts)
        rank[distance] = (bool(usable_layouts and partials), usable_layouts, partials, distance)
    picked = max(rank, key=rank.get) if any(value[1] for value in rank.values()) else 45
    finished = (len(windows) == len(tasks)
                and all(row["status"] not in ("infrastructure_error", "not_admitted") for row in rows))
    result = {"status": "complete" if finished else "incomplete", "worker_starts": len(windows),
              "peak_workers": elapsed_concurrency(windows), "worker_times": windows,
              "groups": groups, "rows": rows, "distance_choice_rule": CHOICE_RULE,
              "selected_distance_nmi": picked if finished else None}
    dump(output / "calibration_report.json", result)
    return result


def pilot_indicators(reports):
    groups = {"0": [], "1-2": [], "3+": []}
    seeds = {}
    for seed, report in reports.items():
        initial = report["shared_initial_usable"]
        group = "0" if not initial else "1-2" if initial < 3 else "3+"
        groups[group].append(seed)
        metrics = {}
        for arm, result in report["arms"].items():
            if "best_usable_trajectory_Wh" not in result:
                continue
            metrics[arm] = {key: result[key] for key in (
                "first_usable_call", "first_usable_online", "first_target_call", "target_reached", "best_usable_Wh",
                "best_usable_trajectory_Wh", "partial_feedback_fitted", "fitted_partial_ids", "fallbacks")}
        endpoints = {}
        for node in CHECKPOINTS:
            best = {arm: (metric["best_usable_trajectory_Wh"][node - 1]
                          if len(metric["best_usable_trajectory_Wh"]) >= node else None)
                    for arm, metric in metrics.items()}
            endpoints[str(node)] = {
                "best_usable_Wh": best,
                "paired_delta_Wh": {f"{left}_minus_{right}": best[left] - best[right]
                                    for left, right in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[2], ARMS[1]))
                                    if best.get(left) is not None and best.get(right) is not None}}
        seeds[seed] = {"initial_usable": initial, "group": group, "arms": metrics, "checkpoints": endpoints}
    rates = {}
    for name, members in groups.items():
        rates[name] = {arm: {
            "seeds": len(members),
            "first_feasible_attainment": sum(seeds[s]["arms"].get(arm, {}).get("first_usable_call") is not None
                                            for s in members) / len(members) if members else None,
            "target_attainment": sum(bool(seeds[s]["arms"].get(arm, {}).get("target_reached"))
                                     for s in members) / len(members) if members else None,
            "fraction_without_usable_at_budget": {
                str(node): sum(seeds[s]["checkpoints"][str(node)]["best_usable_Wh"].get(arm) is None
                               for s in members) / len(members) if members else None for node in CHECKPOINTS},
        } for arm in ARMS}
    return {"target_objective_Wh": TARGET_WH, "initial_feasibility_groups": groups, "group_rates": rates,
            "seeds": seeds}


def campaign(args, base, worker, qualification, configs, sources, excluded):
    output = args.output
    problems, paths, roster, tasks, sampling, frozen = frozen_setup(output, base, configs, excluded)
    observed_identity = verify_environment(problems[45], qualification)
    plan = {"code_sha": code_sha(), "sources": sources, "identity": observed_identity,
            "worker_manifest_sha256": digest(worker),
            "worker_overlay_sha256": sha256(ROOT / "run_x57_energy_stop_worker.py"),
            "config_sha256": {str(distance): sha256(path) for distance, path in paths.items()},
            "verification_sha256": args.verification_sha256, "stop_policy": POLICY, "terminal_cap_Wh": CAP_WH,
            "workers": args.workers, "timeout_seconds": TIMEOUT, "calibration_seed": CALIBRATION_SEED,
            "calibration_tasks": tasks, "calibration_sampling": sampling, "pilot_seeds": SEEDS,
            "pilot_rosters": {seed: {k: v for k, v in data.items() if k != "exclude"} for seed, data in frozen.items()},
            "pilot_roster_sha256": {seed: digest(data["roster"]) for seed, data in frozen.items()},
            "historical_exclusion_sha256": digest(sorted(excluded)), "target_objective_Wh": TARGET_WH,
            "distance_choice_rule": CHOICE_RULE, "shared_initial_per_seed": INITIAL, "online_per_arm": args.online,
            "calibration_call_cap": len(tasks), "pilot_call_cap": len(SEEDS) * (INITIAL + len(ARMS) * args.online),
            "authorized_rcaide_call_cap": FIXED_CALL_CAP, "arms": ARMS,
            "reserve_non_elite_local_slot": True,
            "performance_search": "joint score for both local optimization and final rescore"}
    dump(output / "campaign_plan.json", plan)
    calibration_dir = output / "calibration"
    calibration_dir.mkdir()
    print(f"Frozen calibration {digest(tasks)}. Starting {len(tasks)} tasks with {args.workers} slots.",
          flush=True)
    cal = calibration(calibration_dir, problems, worker, roster, tasks, args)
    journals = [("calibration", {"batch": "calibration"})]
    reports = {}
    if cal["status"] == "complete":
        selected = cal["selected_distance_nmi"]
        stop = Event()
        slots = BoundedSemaphore(args.workers)
        optimizer_lock = Lock()
        print(f"Calibration complete. Selected {selected} nmi. Starting both frozen seeds.", flush=True)

        def run_seed(seed):
            name = f"seed-{seed}"
            folder = output / name
            folder.mkdir()
            flags = SimpleNamespace(
                output=folder, seed=seed, workers=args.workers, online=args.online,
                execute_rcaide=args.execute_rcaide, optimizer_lock=optimizer_lock,
                offline_report=args.offline_report, offline_report_sha256=args.offline_report_sha256)
            try:
                return pilot(flags, base, worker, qualification, paths[selected], sources,
                             set(frozen[seed]["exclude"]), frozen=frozen[seed]["roster"],
                             trial_stop=stop, worker_slots=slots, energy_stop=True, reserve_slot=True,
                             target=TARGET_WH)
            except Exception as error:
                stop.set()
                return {"status": "error", "error": f"{type(error).__name__}: {error}"}

        with ThreadPoolExecutor(max_workers=len(SEEDS)) as both:
            jobs = {seed: both.submit(run_seed, seed) for seed in SEEDS}
            for seed, job in jobs.items():
                reports[seed] = job.result()
                journals.append((f"seed-{seed}", {"batch": "pilot"}))
    windows = merge_journals(output, journals)
    peak = elapsed_concurrency(windows)
    online_peak = elapsed_concurrency([row for row in windows if row["batch"] == "pilot" and row["arm"] in ARMS])
    if (len(windows) > plan["calibration_call_cap"] + plan["pilot_call_cap"] or len(windows) > FIXED_CALL_CAP
            or peak > args.workers or online_peak > 6):
        raise RuntimeError("Campaign exceeded a fixed call or concurrency limit")
    status = ("complete" if len(reports) == len(SEEDS) and all(r["status"] == "complete" for r in reports.values())
              and len(windows) == plan["calibration_call_cap"] + plan["pilot_call_cap"] else "incomplete")
    result = {"status": status, "code_sha": plan["code_sha"], "plan_sha256": sha256(output / "campaign_plan.json"),
              "worker_starts": len(windows), "rcaide_calls": len(windows) if args.execute_rcaide else 0,
              "calibration_calls": cal["worker_starts"], "calibration_peak_workers": cal["peak_workers"],
              "peak_workers": peak, "online_peak_workers": online_peak, "worker_times": windows,
              "selected_distance_nmi": cal["selected_distance_nmi"],
              "pilot_statuses": {seed: {k: r.get(k) for k in ("status", "error", "worker_starts",
                                                             "shared_initial_starts", "online_starts")}
                                 for seed, r in reports.items()},
              "indicators": pilot_indicators({seed: r for seed, r in reports.items() if "arms" in r})}
    dump(output / "campaign_report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "campaign"))
    for flag in ("qualification-plan", "qualification-ledger", "paired-archive", "initialization-ledger",
                 "old-pilot-archive", "distance-diagnostic", "engineering-pilot", "output"):
        parser.add_argument("--" + flag, required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--online", type=int, default=ONLINE)
    parser.add_argument("--execute-rcaide", action="store_true")
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--verification-sha256")
    parser.add_argument("--mock-report", type=Path)
    parser.add_argument("--mock-report-sha256")
    args = parser.parse_args()
    if not 1 <= args.workers <= WORKERS or not 1 <= args.online <= ONLINE:
        parser.error("Require 1–12 workers and 1–12 online calls per arm")
    args.offline_report = args.offline_report_sha256 = None
    if args.execute_rcaide:
        if (args.mode != "campaign" or args.workers != WORKERS or args.online != ONLINE
                or not all((args.verification, args.verification_sha256, args.mock_report, args.mock_report_sha256))):
            parser.error("Real campaign requires full frozen settings and verified offline and mock reports")
        sha = code_sha()
        verified_hash(args.verification, args.verification_sha256)
        verified_hash(args.mock_report, args.mock_report_sha256)
        proof, mock = [json.loads(path.read_bytes()) for path in (args.verification, args.mock_report)]
        if (proof.get("code_sha") != sha or proof.get("new_worker_calls") != 0
                or proof.get("pre_final_energy_stops") != 5 or mock.get("code_sha") != sha
                or mock.get("rcaide_calls") != 0 or mock.get("status") != "complete"
                or mock.get("worker_starts") != FIXED_CALL_CAP
                or any(seed_data["arms"][arm]["partial_feedback_fitted"] < 1
                       for seed_data in mock["indicators"]["seeds"].values() for arm in ARMS[1:])
                or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                           cwd=ROOT, text=True).strip()):
            raise ValueError("Real run requires the matching committed verification and full mock run")
        args.offline_report = args.verification.parent / "offline_report.json"
        args.offline_report_sha256 = proof["offline_report_sha256"]
        verified_hash(args.offline_report, args.offline_report_sha256)
    torch.set_num_threads(1)
    base, worker, qualification, configs, sources, ledgers, excluded = verify_inputs(args)
    offline_excluded = set(excluded)
    initial, settled = archive_rows(args.old_pilot_archive)
    sources["old_pilot"] = {"commit": ARCHIVE, "sha256": LEDGER_HASH}
    excluded.update(row["candidate_id"] for row in settled)
    for name, (commit, checksum) in EXCLUSION_LEDGERS.items():
        path = getattr(args, name)
        verified_hash(path, checksum)
        sources[name] = {"commit": commit, "sha256": checksum}
        excluded.update(row["candidate_id"] for row in read_events(path) if row["event"] == "worker_started")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "verify":
        verify_environment(X57LiveProblem(base, configs[45], args.output / "workers"), qualification)
        stops = stop_replay(settled)
        dump(args.output / "energy_stop_replay.json", stops)
        proposal_report = proposals(base, configs[45], args.output, initial, settled)
        dump(args.output / "proposal_replay.json", proposal_report)
        offline(args, base, configs, sources, ledgers, offline_excluded)
        dump(args.output / "verification.json", {
            "code_sha": code_sha(), "new_worker_calls": 0, "pre_final_energy_stops": stops["triggered"],
            "proposal_replay_sha256": sha256(args.output / "proposal_replay.json"),
            "energy_stop_replay_sha256": sha256(args.output / "energy_stop_replay.json"),
            "offline_report_sha256": sha256(args.output / "offline_report.json")})
        return 0
    result = campaign(args, base, worker, qualification, configs, sources, excluded)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
