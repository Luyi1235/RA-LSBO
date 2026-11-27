"""Continue the fixed X-57 five-arm experiment from evaluation 16 to 32."""

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from statistics import mean
import sys
from tempfile import TemporaryDirectory

RESULT = "db33f2fe8da4323480064b06fb1e5ebb7031ab8b"
ARCHIVE_DIR = "results/x57-47p5-five-arm-checkpoint16"
PLAN_HASH = "27656fb9e1043ffdcb2eb215564731cd67bfaf988220320aee7c4e1fce729a96"
LEDGER_HASH = "19070ae1b4bb1d908724b607c485e0e2b8105cf1b785ebe35021ec3b0d5435e0"
END = 32
PREFIX = 16


def parse_prefix(data, plan):
    if hashlib.sha256(data).hexdigest() != LEDGER_HASH:
        raise ValueError("Checkpoint-16 ledger hash mismatch")
    expected = {f"seed-{s}/shared/{i:03}" for s in plan["seeds"] for i in range(1, 9)}
    expected.update(f"seed-{s}/{a}/{i:03}" for s in plan["seeds"] for a in plan["arms"]
                    for i in range(9, PREFIX + 1))
    tasks = defaultdict(list)
    for line in data.splitlines():
        row = json.loads(line)
        task = row.pop("task_id")
        if task not in expected:
            raise ValueError(f"Unexpected archived task {task}")
        tasks[task].append(row)
    if tasks.keys() != expected or len(expected) != 288:
        raise ValueError("Incomplete historical prefix")
    for task, rows in tasks.items():
        if ([r["event"] for r in rows] != ["dispatch_intent", "worker_started", "settled"]
                or not rows[-1]["charged"] or rows[-1]["feedback"]["outcome"] == "infrastructure_error"):
            raise ValueError(f"Invalid historical settlement {task}")
    return tasks


def stage(inputs, output):
    plan = (inputs / "run_plan.json").read_bytes()
    if hashlib.sha256(plan).hexdigest() != PLAN_HASH:
        raise ValueError("Checkpoint-16 plan hash mismatch")
    tasks = parse_prefix((inputs / "calls.jsonl").read_bytes(), json.loads(plan))
    if output.exists():
        if (output / "run_plan.json").read_bytes() != plan:
            raise ValueError(f"Unexpected plan in {output}")
        for task, expected in tasks.items():
            entries = (output / task / "calls.jsonl").read_bytes().splitlines()
            if [json.loads(line) for line in entries] != expected:
                raise ValueError(f"Historical prefix differs at {task}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        new = Path(temp) / "campaign"
        new.mkdir()
        (new / "run_plan.json").write_bytes(plan)
        for task, rows in tasks.items():
            job = new / task
            job.mkdir(parents=True)
            with (job / "calls.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    print(json.dumps(row, allow_nan=False), file=stream)
        new.rename(output)


def replay_gate(code):
    """Run the physical optimizer environments, intercept every physical dispatch."""
    sys.path.insert(0, str(code))
    import run_x57_five_arm as engine

    proposals = {}
    evaluate = engine.Campaign.evaluate

    def replay(campaign, job, point, **kwargs):
        if kwargs["evaluation"] == PREFIX + 1:
            proposals[(kwargs["seed"], kwargs["arm"])] = {
                "evaluation": PREFIX + 1, "candidate_id": campaign.problem.space.key(point),
                "phase": kwargs["phase"],
            }
            return None
        if not (campaign.output / job / "calls.jsonl").is_file():
            raise ValueError(f"Missing historical settlement {job}")
        return evaluate(campaign, job, point, **kwargs)

    def no_dispatch(*args, **kwargs):
        raise RuntimeError("A physical evaluation was requested during offline replay")

    def finish(campaign, errors, trajectories, services, execution_code_sha):
        expected = {(s, a) for s in campaign.plan["seeds"] for a in campaign.arms}
        if (errors or set(proposals) != expected or set(trajectories) != expected or campaign.admitted
                or any(len(t["rows"]) != 8 for t in trajectories.values())):
            raise RuntimeError(f"Checkpoint-16 selection replay failed: {errors}")
        for (seed, arm), proposal in proposals.items():
            engine.save(campaign.output / f"seed-{seed}" / arm / "next_proposal.json", proposal)
        engine.save(campaign.output / "prefix_replay.json", {
            "result_commit": RESULT, "prefix_settlements": 288, "new_physical_calls": 0,
            "next_proposals": {f"{s}/{a}": p for (s, a), p in sorted(proposals.items())},
        })
        print(f"Verified 288 settlements; saved {len(proposals)} next proposals.", flush=True)
        return 0

    engine.Campaign.evaluate = replay
    engine.Dispatcher.evaluate = no_dispatch
    engine.report = finish
    engine.main()


def summarize(report, path):
    arms = [arm for arm in next(iter(report["per_seed"].values())) if arm != "shared"]
    records = []
    for arm in arms:
        rows = [seed[arm] for seed in report["per_seed"].values()]
        online = [r for row in rows for r in row["trajectory"][8:]]
        records.append({
            "checkpoint": report["checkpoint"], "arm": arm,
            "seeds": len(rows), "quality_hits": sum(r["first_quality_call"] is not None for r in rows),
            "online_usable": sum(r["usable"] for r in online), "online_evaluations": len(online),
            "mean_best_feasible_Wh": mean(r["best_usable_Wh"] for r in rows),
            "mean_running_best_online_Wh": mean(r["best_usable_Wh"] for r in online),
        })
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def load_launcher():
    path = Path(__file__).resolve().parents[2] / ARCHIVE_DIR / "launcher/run.py"
    spec = importlib.util.spec_from_file_location("checkpoint16_launcher", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("D:/RA-LSBO"))
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Use the Windows machine with the qualified X-57/RCAIDE environment")
    launch = load_launcher()
    root = args.workspace.resolve()
    repo = root / "x57-cold-start"
    code = launch.checkout(repo, root)
    qualification = root / "x57-qualification-v1-final/qualification_plan.json"
    external = root / "x57-baseline-py311/Scripts/python.exe"
    physical = Path.home() / ".conda/envs/Rcaide_vsp/python.exe"
    env = launch.launch_environment(external, code)
    archive = launch.prepare_inputs(repo, root / "x57-cold-debug-publish/results",
                                    root / "x57-native-doe64-inputs" / launch.QUALIFICATION_SHA,
                                    launch.QUALIFICATION_SHA, launch.QUALIFICATION_INPUTS)
    # Read the immutable archive blobs rather than a working copy with autocrlf filters.
    files = {"x57-47p5-five-arm-checkpoint16/" + name: digest for name, digest in
             [("run_plan.json", PLAN_HASH), ("calls.jsonl", LEDGER_HASH)]}
    cached = launch.prepare_inputs(repo, root / "x57-frozen-prefix" / RESULT,
                                   root / "x57-frozen-prefix" / RESULT, RESULT, files)
    output = root / "x57-47p5-five-arm-n8-checkpoint32"
    stage(cached / "x57-47p5-five-arm-checkpoint16", output)
    tail = ["--archive", str(archive), "--cold-baselines", "--execute-rcaide",
            "--qualification-plan", str(qualification), "--external-python", str(external),
            "--output", str(output), "--workers", "12"]
    gate = output / "prefix_replay.json"
    if not gate.exists():
        wrapper = [str(physical), "-B", str(Path(__file__).resolve()), "--gate", str(code)]
        status = launch.run_stage(wrapper + tail, code, output, PREFIX + 1, True, 12, env)
        if status:
            return status
    proof = launch.read_json(gate)
    plan = launch.read_json(output / "run_plan.json")
    expected = {f"{s}/{a}" for s in plan["seeds"] for a in plan["arms"]}
    if proof.get("result_commit") != RESULT or proof.get("next_proposals", {}).keys() != expected:
        raise ValueError("Incomplete replay evidence")
    for key, preview in proof["next_proposals"].items():
        if launch.read_json(output / ("seed-" + key) / "next_proposal.json") != preview:
            raise ValueError(f"Stored proposal changed: {key}")
    if args.replay_only:
        return 0
    report = launch.read_json(output / "report.json")
    if report.get("checkpoint") != END or report.get("status") != "complete":
        command = [str(physical), "-B", str(code / "run_x57_five_arm.py")] + tail
        status = launch.run_stage(command, code, output, END, True, 12, env)
        if status:
            return status
        report = launch.read_json(output / "report.json")
    replay = report.get("next_proposal_replay_verified", {})
    if (report.get("status") != "complete" or report.get("checkpoint") != END
            or report.get("plan_sha256") != PLAN_HASH or report["actual_rcaide_calls"] != 768
            or replay.keys() != expected or not all(replay.values())):
        raise ValueError("Missing checkpoint-32 settlements or next-proposal verification")
    summarize(report, output / "arm_summary.csv")
    print(f"Checkpoint 32 complete. 288 archived + 480 additional settlements. {output}", flush=True)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--gate"]:
        del sys.argv[1]
        replay_gate(Path(sys.argv.pop(1)))
    else:
        raise SystemExit(main())
