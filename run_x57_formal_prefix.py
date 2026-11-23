"""Evaluate the six sealed eight-point prefixes and up to four distance confirmations."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path

from ra_lsbo.budget import digest
from ra_lsbo.x57_cold_sampling import LAYOUTS, violations
from ra_lsbo.x57_cold_worker import Dispatcher, dump, sha256
from ra_lsbo.x57_energy_surrogate import energy_training_row
from ra_lsbo.x57_live import X57LiveProblem, evaluator_files
from run_x57_cold_start_debug import merge_journals, verify_environment
from run_x57_distance_diagnostic import (
    QUALIFICATION_PLAN_SHA256, ROOT, THREAD_VARS, TIMEOUT, elapsed_concurrency,
    observation, prefix_energy, read_events, run_wave,
)
from run_x57_engineering_energy_pilot import code_sha
from run_x57_initialization_diagnostic import verified_hash

REFERENCE = "fccf94c61d87b170824476046c8f3d2a4d503229"
ARCHIVE = "de1d060bcb17228c7076d4eb88a19a8b1d0ab630"
INPUT_HASHES = {
    "formal_protocol.json": "aaa810d3385b73aa96940d3270c05e4a4c591f80a63887271321d9efbbe6d268",
    "formal_rosters.json": "93ca8c54454c5daa1a673567e995e0bed910535d1c50ac4e2bb9815c29f65c74",
    "configs/distance_45.json": "3d2cdb22bad14ae7da3988c210b29bfce4ec87fbc0002d0f8145c1a9ce1f6118",
    "configs/distance_47.5.json": "9902197262e862b13a3b3ac4a75c6f2c7892c91d8a69d6ce1af8aa884f6e6c3d",
}
CAP = 52


def prepare(args):
    inputs = {name: verified_hash(args.archive / name, expected) for name, expected in INPUT_HASHES.items()}
    protocol = json.loads((args.archive / "formal_protocol.json").read_bytes())
    rosters = json.loads((args.archive / "formal_rosters.json").read_bytes())
    verified_hash(args.qualification_plan, QUALIFICATION_PLAN_SHA256)
    qualification = json.loads(args.qualification_plan.read_bytes())
    base = args.qualification_plan.parent / "physical-base"
    worker = qualification["worker"]["worker_sha256"]
    mission, sizes, budget = (protocol[name] for name in ("mission", "shared_designs", "budgets"))
    overlay = sha256(ROOT / "run_x57_energy_stop_worker.py")
    if (worker != evaluator_files(base) or digest(worker) != mission["worker_manifest_sha256"]
            or overlay != mission["worker_overlay_sha256"] or protocol["reference_code_sha"] != REFERENCE
            or inputs["formal_protocol.json"] != rosters["protocol_sha256"]
            or sizes["sensitivity_count"] != 8 or sizes["main_count"] != 32 or budget["per_arm_total"] != 320
            or budget["combined_max_starts"] != 18192 or TIMEOUT != 1800):
        raise ValueError("Qualification or formal protocol identity changed")
    problems = {}
    for distance in (45, 47.5):
        problems[distance] = X57LiveProblem(base, args.archive / f"configs/distance_{distance}.json",
                                            args.output / "unused")
    derived = deepcopy(problems[45].config)
    derived["mission"]["cruise_distance_nmi"] = 47.5
    if (derived != problems[47.5].config or problems[45].config["mission"]["cruise_distance_nmi"] != 45):
        raise ValueError("Only the cruise distance may change")
    environment = verify_environment(problems[45], qualification) if args.execute_rcaide else {}
    seen, tasks = set(), []
    for seed in protocol["seeds"]:
        roster = rosters["rosters"][str(seed)]["records"]
        if (len(roster) != 32 or [item["layout"] for item in roster] != list(LAYOUTS) * 16
                or [item["position"] for item in roster] != list(range(1, 33))):
            raise ValueError(f"Invalid frozen roster for {seed}")
        for row in roster:
            for problem in problems.values():
                candidate = problem.candidate_class(**row["candidate"])
                if (problem.canonical(row["candidate"]) != row["candidate"]
                        or problem.space.key(candidate) != row["candidate_id"]
                        or problem.problem.precheck(candidate) != row["precheck"] or violations(row["precheck"])):
                    raise ValueError(f"Frozen candidate identity or precheck changed for {seed}")
            if row["candidate_id"] in seen:
                raise ValueError("Duplicate candidate in sealed formal rosters")
            seen.add(row["candidate_id"])
        tasks.extend({**row, "task_id": f"{seed}-{row['position']:02d}", "seed": seed,
                      "distance_nmi": 45, "phase": "formal_shared_initial"} for row in roster[:8])
    if len(tasks) != 48:
        raise ValueError("Expected six complete shared prefixes")
    plan = {"code_sha": code_sha(), "reference_sha": REFERENCE, "archive_sha": ARCHIVE,
            "input_sha256": inputs, "qualification_plan_sha256": QUALIFICATION_PLAN_SHA256,
            "worker_manifest_sha256": digest(worker), "worker_overlay_sha256": overlay,
            "environment": environment, "workers": args.workers, "timeout_seconds": TIMEOUT,
            "thread_settings": {name: os.environ[name] for name in THREAD_VARS}, "call_cap": CAP,
            "backend": "rcaide" if args.execute_rcaide else "mock", "formal_budgets": budget,
            "seeds": protocol["seeds"], "prefix_tasks": tasks}
    return plan, problems, worker


def confirmation_tasks(rows):
    selected = []
    for layout in LAYOUTS:
        eligible = sorted((r for r in rows if r["layout"] == layout and r.get("usable")),
                          key=lambda r: (r["objective_Wh"], r["candidate_id"]))
        unique = {}
        for row in eligible:
            unique.setdefault(row["candidate_id"], row)
        for row in list(unique.values())[:2]:
            selected.append({key: row[key] for key in ("seed", "position", "layout", "candidate",
                                                       "candidate_id", "precheck")})
            selected[-1].update(task_id=f"confirm-{len(selected):02d}", phase="distance_confirmation",
                                distance_nmi=47.5, paired_task_id=row["task_id"],
                                paired_45nmi_objective_Wh=row["objective_Wh"], training_eligible=False)
    return selected


def execute_wave(args, tasks, problems, worker):
    def launch(task):
        distance = task["distance_nmi"]
        problem = problems[distance]
        directory = args.output / task["task_id"]
        directory.mkdir()
        dispatcher = Dispatcher(problem, directory, backend="rcaide" if args.execute_rcaide else "mock",
                                worker_root=problem.base if args.execute_rcaide else ROOT, timeout=TIMEOUT,
                                expected_worker=worker, max_calls=1, energy_stop=True, mock_profile="formal-prefix")
        row = dispatcher.evaluate(problem.candidate_class(**task["candidate"]), seed=task["seed"],
                                  arm="shared" if distance == 45 else "confirmation",
                                  evaluation=task["position"], online=0, phase=task["phase"],
                                  proposal={"frozen_task": task["task_id"], "distance_nmi": distance})
        evidence, _ = energy_training_row(row, task["task_id"], distance, prefix_energy, True)
        return {**observation(row), "status": row["feedback"]["outcome"],
                "verified_partial_feedback": evidence.kind == "interval",
                "normalized_energy_violation_lower": evidence.lower}

    outcomes = run_wave(tasks, args.workers, launch)
    return [{**task, **outcomes.get(task["task_id"], {"status": "not_admitted"})} for task in tasks]


def summarize(rows):
    return {"exact_objectives": sum(row.get("objective_kind") == "exact" for row in rows),
            "usable": sum(row.get("usable", False) for row in rows),
            "verified_partials": sum(row.get("verified_partial_feedback", False) for row in rows),
            "outcomes": dict(Counter(row["status"] for row in rows)),
            "first_usable_position": next((row["position"] for row in rows if row.get("usable")), None),
            "best_usable_Wh": min((row["objective_Wh"] for row in rows if row.get("usable")), default=None)}


def settled_wave(rows, grouped):
    return all(row["status"] not in ("infrastructure_error", "not_admitted")
               and [e["event"] for e in grouped.get(row["task_id"], [])]
               == ["dispatch_intent", "worker_started", "settled"]
               and all(e["candidate_id"] == row["candidate_id"] for e in grouped[row["task_id"]])
               and grouped[row["task_id"]][-1]["charged"] is True for row in rows)


def campaign(args):
    for name in THREAD_VARS:
        os.environ[name] = "1"
    plan, problems, worker = prepare(args)
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "run_plan.json", plan)
    prefix = execute_wave(args, plan["prefix_tasks"], problems, worker)
    dump(args.output / "prefix_outcomes.json", prefix)
    prefix_journals = {r["task_id"]: read_events(args.output / r["task_id"] / "calls.jsonl")
                       for r in prefix if (args.output / r["task_id"] / "calls.jsonl").is_file()}
    prefix_complete = settled_wave(prefix, prefix_journals)
    confirm = confirmation_tasks(prefix) if prefix_complete else []
    dump(args.output / "confirmation_plan.json", confirm)
    confirmation = execute_wave(args, confirm, problems, worker)
    dump(args.output / "confirmation_outcomes.json", confirmation)
    windows = merge_journals(args.output, [(r["task_id"], {"task_id": r["task_id"],
                             "distance_nmi": r["distance_nmi"]}) for r in prefix + confirmation])
    journals = {}
    for event in read_events(args.output / "calls.jsonl"):
        journals.setdefault(event["task_id"], []).append(event)
    starts = len(windows)
    peak = elapsed_concurrency(windows)
    complete = prefix_complete and settled_wave(confirmation, journals)
    if starts > CAP or peak > args.workers or (complete and starts != 48 + len(confirm)):
        raise RuntimeError("The durable ledger violates the call or concurrency budget")
    summaries = {str(seed): summarize([r for r in prefix if r["seed"] == seed]) for seed in plan["seeds"]}
    no_usable = [int(seed) for seed, summary in summaries.items() if summary["usable"] == 0]
    report = {"status": "complete" if complete else "incomplete", "backend": plan["backend"],
              "code_sha": plan["code_sha"], "actual_rcaide_calls": starts if args.execute_rcaide else 0,
              "worker_starts": starts, "peak_workers": peak, "worker_times": windows, "prefix": summaries,
              "prefix_zero_usable_seeds": no_usable if prefix_complete else None,
              "confirmation": confirmation,
              "confirmation_counts_by_layout": {layout: summarize([r for r in confirmation if
                                                 r["layout"] == layout]) for layout in LAYOUTS},
              "45nmi_prefix_fully_settled": prefix_complete,
              "47_5nmi_usable_confirmed": any(r.get("usable") for r in confirmation),
              "eligible_to_preregister_47_5nmi": bool(complete and not no_usable
                                                    and any(r.get("usable") for r in confirmation))}
    if prefix_complete and args.execute_rcaide:
        directory = args.output / "formal_initial"
        directory.mkdir()
        for seed in plan["seeds"]:
            dump(directory / f"{seed}.json", [prefix_journals[r["task_id"]][-1]
                                              for r in prefix if r["seed"] == seed])
        maximum = plan["formal_budgets"]["combined_max_starts"]
        dump(directory / "credit.json", {"total_budget_per_arm": 320, "formal_call_cap": maximum,
                                        "credited_starts": 48, "remaining_starts": maximum - 48,
                                        "run_plan_sha256": sha256(args.output / "run_plan.json"),
                                        "initial_sha256": {str(seed): sha256(directory / f"{seed}.json")
                                                           for seed in plan["seeds"]}})
    dump(args.output / "report.json", report)
    print(f"{report['status']} | starts={starts} | peak={peak} | output={args.output}", flush=True)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--qualification-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--execute-rcaide", action="store_true")
    args = parser.parse_args()
    args.archive = args.archive.resolve()
    args.qualification_plan = args.qualification_plan.resolve()
    args.output = args.output.resolve()
    raise SystemExit(campaign(args))


if __name__ == "__main__":
    main()
