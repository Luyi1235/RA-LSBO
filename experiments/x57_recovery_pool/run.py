"""PyCharm entry point for the paired X-57 recovery-pool pilot."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "results/x57-47p5-checkpoint9-selection"
DATA = ROOT / "results/x57-47p5-five-arm-checkpoint16"
PREDICTIONS_HASH = "a221a0b84d895176275bdd91eeb1810066f4d9f0242829a487c50bddd2d60db7"
ARM = "ra_lsbo_energy_interval_recovery_pool"
PILOT_SEEDS = list(range(2026112801, 2026112807))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def check_first_move(args):
    base, seed, initial, expected = args
    sys.path.insert(0, str(ROOT))
    from ra_lsbo.x57_formal_native import FormalNative
    from ra_lsbo.x57_live import X57LiveProblem
    import torch
    torch.set_num_threads(1)
    with TemporaryDirectory() as temp:
        local = Path(temp)
        problem = X57LiveProblem(base, DATA / "distance_47.5.json", local, allow_mda=False)
        def forbid(*args, **kwargs):
            raise RuntimeError("Physical evaluation attempted in first-move replay")
        problem.evaluate = problem.problem.evaluate = forbid
        optimizer = FormalNative(problem, ARM, seed, initial, local / "native")
        point, _, info = optimizer.ask()
        native_id = expected["proposals"]["lsbo_full_budget"]["candidate_id"]
        ra_id = expected["proposals"]["ra_lsbo_energy_interval"]["candidate_id"]
        gap = expected["comparisons"]["ra_lsbo_energy_interval"]["ra_minus_lsbo_log_score"]
        winner = native_id if gap < 0 else ra_id
        if info.get("recovery_candidate_id") != native_id or problem.space.key(point) != winner:
            raise ValueError(f"Recovery candidate or final ranking did not reproduce for seed {seed}")
        return seed, winner


def write_metrics(report, output):
    with (output / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "arm", "initial_usable", "checkpoint", "first_feasible",
                         "first_47500Wh", "online_feasible", "best_feasible_Wh"])
        for seed, arms in report["per_seed"].items():
            initial = sum(r["usable"] for r in arms["shared"]["trajectory"])
            for name, row in arms.items():
                if name == "shared":
                    continue
                for end in (16, 32):
                    prefix = row["trajectory"][:end]
                    if len(prefix) != end:
                        raise ValueError(f"Incomplete trajectory: {seed}/{name}/{end}")
                    feasible = [r for r in prefix if r["usable"]]
                    hit = next((r["evaluation"] for r in feasible if r["objective_Wh"] <= 47500), None)
                    writer.writerow([seed, name, initial, end, feasible[0]["evaluation"] if feasible else None,
                                     hit, sum(r["usable"] for r in prefix[8:]), prefix[-1]["best_usable_Wh"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("D:/RA-LSBO"))
    parser.add_argument("--replay-only", action="store_true")
    cfg = parser.parse_args()
    launcher = module("workstation", DATA / "launcher/run.py")
    diag = module("diagnostic", ROOT / "tools/x57_checkpoint9_selection.py")
    os.environ.update(dict.fromkeys(diag.THREADS, "1"))
    os.environ.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    os.environ["PATH"] = str(Path(launcher.git_exe()).parent) + os.pathsep + os.environ.get("PATH", "")
    root = cfg.workspace.resolve()
    qualification = root / "x57-qualification-v1-final/qualification_plan.json"
    base = qualification.parent / "physical-base"
    if not launcher.matches(qualification, launcher.PHYSICAL_PLAN_HASH):
        raise ValueError("Qualification plan hash differs")
    reference = (DIAG / "predictions.json").read_bytes()
    if sha256(reference).hexdigest() != PREDICTIONS_HASH:
        raise ValueError("Sealed prediction archive differs")
    examples = json.loads(reference)["seeds"]
    initial, _ = diag.inspect_archive()
    git = launcher.git_exe()
    head = launcher.git(ROOT, "rev-parse", "HEAD").decode().strip()
    output = root / ("x57-recovery-pool-n8-to32-" + head[:8])
    gate = output.with_suffix(".replay.json")
    identity = {"code": head, "predictions": PREDICTIONS_HASH}
    if subprocess.check_output([git, "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Use the clean published pilot checkout")
    expected_ids = {str(r["seed"]): r["proposals"][
        "lsbo_full_budget" if r["comparisons"]["ra_lsbo_energy_interval"]["ra_minus_lsbo_log_score"] < 0
        else "ra_lsbo_energy_interval"]["candidate_id"] for r in examples}
    if launcher.read_json(gate) != {**identity, "winners": expected_ids}:
        jobs = [(base, r["seed"], initial[r["seed"]], r) for r in examples]
        with ProcessPoolExecutor(max_workers=6) as pool:
            observed = {}
            for seed, winner in pool.map(check_first_move, jobs):
                observed[str(seed)] = winner
                print(f"[replay] {seed} verified ({len(observed)}/6)", flush=True)
        if observed != expected_ids:
            raise ValueError("Incomplete first-move replay")
        temp = gate.with_suffix(".tmp")
        temp.write_text(json.dumps({**identity, "winners": observed}), encoding="utf-8")
        temp.replace(gate)
    if cfg.replay_only:
        print("Verified six historical first moves, physical calls 0.", flush=True)
        return 0
    archive = launcher.prepare_inputs(root / "x57-cold-start", root / "x57-cold-debug-publish/results",
                                      root / "x57-native-doe64-inputs" / launcher.QUALIFICATION_SHA,
                                      launcher.QUALIFICATION_SHA, launcher.QUALIFICATION_INPUTS)
    report = launcher.read_json(output / "report.json")
    if report.get("status") != "complete" or report.get("checkpoint") != 32:
        command = [sys.executable, "-B", str(ROOT / "run_x57_five_arm.py"), "--archive", str(archive),
                   "--recovery-pool-pilot", "--qualification-plan", str(qualification),
                   "--external-python", sys.executable, "--execute-rcaide", "--workers", "12",
                   "--output", str(output)]
        rc = launcher.run_stage(command, ROOT, output, 32, (output / "run_plan.json").exists(), 12,
                                os.environ.copy())
        if rc:
            return rc
        report = launcher.read_json(output / "report.json")
    plan = launcher.read_json(output / "run_plan.json")
    if (report.get("execution_code_sha") != head or report.get("actual_rcaide_calls") != 480
            or report.get("status") != "complete" or report.get("checkpoint") != 32
            or plan.get("seeds") != PILOT_SEEDS or plan.get("schema") != "x57_recovery_pool_pilot_v1"):
        raise ValueError("Unexpected final pilot report")
    write_metrics(report, output)
    print(f"Checkpoint 32 complete: {output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
