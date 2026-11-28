"""Continue the frozen RA-LSBO and original LSBO policies on matched X-57 seeds."""

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
ARMS = ("lsbo_full_budget", "ra_lsbo_energy_interval")


def read(path):
    return json.loads(path.read_bytes())


def validate_and_export(old, current, output):
    end = current["checkpoint"]
    if (current["status"] != "complete" or current["reused_shared_calls"] != 48
            or current["reused_online_prefix_calls"] != 288 or current["actual_rcaide_calls"] != 12 * (end - 32)):
        raise ValueError(f"Invalid budget at checkpoint {end}")
    rows = []
    for seed in SEEDS:
        group = current["per_seed"][seed]
        values = []
        for arm in ARMS:
            trajectory = group[arm]["trajectory"]
            if len(trajectory) != end or trajectory[:32] != old["per_seed"][seed][arm]["trajectory"]:
                raise ValueError(f"Changed trajectory prefix: {seed}/{arm}")
            values.append(trajectory[-1]["best_usable_Wh"])
        rows.append((seed, end, *values, values[1] - values[0]))
    with (output / f"paired_{end}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "checkpoint", "lsbo_best_Wh", "ra_best_Wh", "ra_minus_lsbo_Wh"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("D:/RA-LSBO"))
    parser.add_argument("--prepare-only", action="store_true", help="Replay all six common prefixes")
    parser.add_argument("--through", type=int, choices=(64, 128, 320), default=64)
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
    output = root / ("x57-frozen-long-pair-" + code[:8])
    command = [sys.executable, "-B", str(ROOT / "run_x57_five_arm.py"), "--long-pair",
               "--archive", str(archive), "--initial-replay", str(DATA), "--qualification-plan", str(physical_plan),
               "--external-python", str(external), "--output", str(output), "--execute-rcaide", "--workers", "12"]
    plan_path = output / "run_plan.json"
    current = launch.read_json(output / "report.json")
    if plan_path.exists():
        plan = read(plan_path)
        if (plan.get("schema") != "x57_long_pair_v1" or plan["code_sha"] != code
                or plan["reused_shared_prefix"]["commit"] != SOURCE):
            raise ValueError("Output belongs to another comparison")
    target = 32 if args.prepare_only else args.through
    for end in (32, 64, 128, 320):
        if end > target:
            break
        latest = current.get("checkpoint", 0)
        if latest > end:
            continue
        if latest < end or current.get("status") != "complete":
            rc = launch.run_stage(command, ROOT, output, end, plan_path.exists(), 12, env)
            if rc:
                return rc
            current = read(output / "report.json")
        validate_and_export(old, current, output)
        print(f"Verified checkpoint {end}, paired results: {output / f'paired_{end}.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
