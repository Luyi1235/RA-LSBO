"""Test a posterior feasibility screen on six paired X-57 performance-search continuations."""

import argparse
import csv
from hashlib import sha256
import importlib.util
import os
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/x57-recovery-pool-pilot-1ecea8d6"
SOURCE = "e5fcec2710f1a8f3b5326974a76fff23f67ed917"
REPORT_HASH = "0110367f1936760863954d36b475ef6ac827db687d313b292acd56d6b2ee7852"
SEEDS = [str(i) for i in range(2026112801, 2026112807)]
PRIMARY = ("lsbo_full_budget", "ra_lsbo_energy_interval", "ra_lsbo_energy_interval_recovery_pool")
EXTERNAL = ("logcei_screened_online", "scbo_screened_online")


def read(path):
    return json.loads(path.read_bytes())


def paired(old, current, path):
    arm = "ra_lsbo_energy_interval"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "common_best_16_Wh", "original_best_32_Wh", "screen_best_32_Wh",
                         "screen_minus_original_Wh", "original_usable_17_32", "screen_usable_17_32"])
        for seed in SEEDS:
            a, b = (report["per_seed"][seed][arm]["trajectory"] for report in (old, current))
            if len(b) != 32 or a[:16] != b[:16]:
                raise ValueError(f"Prefix or trajectory mismatch for {seed}")
            x, y = a[-1]["best_usable_Wh"], b[-1]["best_usable_Wh"]
            writer.writerow([seed, a[15]["best_usable_Wh"], x, y, y-x,
                             sum(t["usable"] for t in a[16:]), sum(t["usable"] for t in b[16:])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("D:/RA-LSBO"))
    parser.add_argument("--prepare-only", action="store_true", help="Replay all six common prefixes")
    args = parser.parse_args()
    module_file = ROOT / "results/x57-47p5-five-arm-checkpoint16/launcher/run.py"
    spec = importlib.util.spec_from_file_location("workstation_launcher", module_file)
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    root = args.workspace.resolve()
    external = sys.executable
    physical_plan = root / "x57-qualification-v1-final/qualification_plan.json"
    if not launch.matches(physical_plan, launch.PHYSICAL_PLAN_HASH):
        raise ValueError("Unexpected physical qualification plan")
    source_bytes = (DATA / "report.json").read_bytes()
    if sha256(source_bytes).hexdigest() != REPORT_HASH:
        raise ValueError("Recovery-pilot source report hash mismatch")
    old = json.loads(source_bytes)
    env = {**os.environ, **dict.fromkeys(launch.THREAD_VARS, "1"), "PYTHONUNBUFFERED": "1",
           "PYTHONIOENCODING": "utf-8", "PATH": str(Path(launch.git_exe()).parent) + os.pathsep + os.environ.get("PATH", "")}
    code = launch.git(ROOT, "rev-parse", "HEAD").decode().strip()
    archive = launch.prepare_inputs(root / "x57-cold-start", root / "x57-cold-debug-publish/results",
                                    root / "x57-native-doe64-inputs" / launch.QUALIFICATION_SHA,
                                    launch.QUALIFICATION_SHA, launch.QUALIFICATION_INPUTS)
    output = root / ("x57-posterior-screen-32-" + code[:8])
    command = [sys.executable, "-B", str(ROOT / "run_x57_five_arm.py"), "--risk-screen-fork",
               "--archive", str(archive), "--initial-replay", str(DATA), "--qualification-plan", str(physical_plan),
               "--external-python", str(external), "--output", str(output), "--execute-rcaide", "--workers", "12"]
    plan_path = output / "run_plan.json"
    current = launch.read_json(output / "report.json")
    if plan_path.exists():
        plan = read(plan_path)
        if (plan.get("schema") != "x57_posterior_screen_fork_v1" or plan["code_sha"] != code
                or plan["reused_shared_prefix"]["commit"] != SOURCE):
            raise ValueError("Output belongs to another comparison")
    if not current or (current.get("checkpoint") == 16 and current.get("status") != "complete"):
        status = launch.run_stage(command, ROOT, output, 16, plan_path.exists(), 12, env)
        if status:
            return status
        current = read(output / "report.json")
    if current["reused_shared_calls"] != 48 or current["reused_online_prefix_calls"] != 48:
        raise ValueError("Incomplete shared prefix")
    if args.prepare_only:
        print(f"Verified 96 archived prefix settlements. No new evaluation requested. {output}", flush=True)
        return 0
    if current.get("status") != "complete" or current.get("checkpoint") != 32:
        status = launch.run_stage(command, ROOT, output, 32, True, 12, env)
        if status:
            return status
        current = read(output / "report.json")
    if (current.get("status") != "complete" or current.get("checkpoint") != 32
            or current["actual_rcaide_calls"] + len(current["reused_matching_late_feedback"]) != 96 or current["reused_shared_calls"] != 48):
        raise ValueError("Incomplete external trajectories")
    paired(old, current, output / "screen_paired_metrics.csv")
    print(f"Finished all six paired forks. New physical calls: {current['actual_rcaide_calls']}. {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
