"""Compare frozen LogCEI and SCBO policies on the six recovery-pilot X-57 prefixes."""

import argparse
import csv
from hashlib import sha256
import importlib.util
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


def merge_metrics(source, external, output):
    """Blank event times denote no hit by the checkpoint. Preserve zero-feasible seeds."""
    if (set(source["per_seed"]) != set(SEEDS) or set(external["per_seed"]) != set(SEEDS)
            or source["quality_target_Wh"] != external["quality_target_Wh"]
            or source["quality_target_Wh"] != 47500):
        raise ValueError("Seed or quality target mismatch")
    rows = []
    for seed in SEEDS:
        native, comparison = source["per_seed"][seed], external["per_seed"][seed]
        initial = native["shared"]["trajectory"]
        if (len(initial) != 8 or comparison["shared"]["trajectory"] != initial
                or set(native) != {"shared", *PRIMARY} or set(comparison) != {"shared", *EXTERNAL}):
            raise ValueError(f"Paired initialization or arms differ for {seed}")
        for arm in PRIMARY + EXTERNAL:
            trajectory = (native if arm in PRIMARY else comparison)[arm]["trajectory"]
            if (trajectory[:8] != initial or [t["evaluation"] for t in trajectory] != list(range(1, 33))):
                raise ValueError(f"Invalid ordered prefix: {seed}/{arm}")
            for end in (16, 32):
                prefix = trajectory[:end]
                usable = [t for t in prefix if t["usable"]]
                hit = next((t["evaluation"] for t in usable if t["objective_Wh"] <= 47500), None)
                rows.append((seed, arm, sum(t["usable"] for t in initial), end,
                             usable[0]["evaluation"] if usable else None, hit,
                             len([t for t in prefix[8:] if t["usable"]]), prefix[-1]["best_usable_Wh"]))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed", "arm", "initial_usable", "checkpoint", "first_feasible", "first_47500Wh",
                         "online_feasible", "best_feasible_Wh"))
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("D:/RA-LSBO"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--risk-audit-only", action="store_true", help="Replay the selected late-stage risk predictions")
    mode.add_argument("--prepare-only", action="store_true",
                        help="Verify the shared prefix and initialize both external services, without a paid call")
    args = parser.parse_args()
    if args.risk_audit_only:
        from risk_audit import main as audit
        audit(args.workspace.resolve())
        return 0
    module_file = ROOT / "results/x57-47p5-five-arm-checkpoint16/launcher/run.py"
    spec = importlib.util.spec_from_file_location("workstation_launcher", module_file)
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    root = args.workspace.resolve()
    external = root / "x57-baseline-py311/Scripts/python.exe"
    physical_plan = root / "x57-qualification-v1-final/qualification_plan.json"
    if not launch.matches(physical_plan, launch.PHYSICAL_PLAN_HASH):
        raise ValueError("Unexpected physical qualification plan")
    source_bytes = (DATA / "report.json").read_bytes()
    if sha256(source_bytes).hexdigest() != REPORT_HASH:
        raise ValueError("Recovery-pilot source report hash mismatch")
    old = json.loads(source_bytes)
    env = launch.launch_environment(external, ROOT)
    code = launch.git(ROOT, "rev-parse", "HEAD").decode().strip()
    archive = launch.prepare_inputs(root / "x57-cold-start", root / "x57-cold-debug-publish/results",
                                    root / "x57-native-doe64-inputs" / launch.QUALIFICATION_SHA,
                                    launch.QUALIFICATION_SHA, launch.QUALIFICATION_INPUTS)
    output = root / ("x57-external-matched-32-" + code[:8])
    command = [sys.executable, "-B", str(ROOT / "run_x57_five_arm.py"), "--matched-external",
               "--archive", str(archive), "--initial-replay", str(DATA), "--qualification-plan", str(physical_plan),
               "--external-python", str(external), "--output", str(output), "--execute-rcaide", "--workers", "12"]
    plan_path = output / "run_plan.json"
    current = launch.read_json(output / "report.json")
    if plan_path.exists():
        plan = read(plan_path)
        if (plan.get("schema") != "x57_recovery_matched_external_v1" or plan["code_sha"] != code
                or plan["reused_shared_prefix"]["commit"] != SOURCE):
            raise ValueError("Output belongs to another comparison")
    if not current:
        status = launch.run_stage(command, ROOT, output, 8, plan_path.exists(), 12, env)
        if status:
            return status
        current = read(output / "report.json")
    if current["reused_shared_calls"] != 48:
        raise ValueError("Incomplete shared prefix")
    if args.prepare_only:
        print(f"Verified 48 shared records. No new evaluation requested. {output}", flush=True)
        return 0
    if current.get("status") != "complete" or current.get("checkpoint") != 32:
        status = launch.run_stage(command, ROOT, output, 32, True, 12, env)
        if status:
            return status
        current = read(output / "report.json")
    if (current.get("status") != "complete" or current.get("checkpoint") != 32
            or current["actual_rcaide_calls"] != 288 or current["reused_shared_calls"] != 48):
        raise ValueError("Incomplete external trajectories")
    merge_metrics(old, current, output / "five_arm_paired_metrics.csv")
    print(f"Complete. 288 external calls; five paired trajectories per seed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
