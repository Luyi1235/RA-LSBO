"""Run mock X-57 experiments or prepare a gated RCAIDE qualification."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
# Keep BLAS overhead bounded for small sequential GP and mock worker processes.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import torch

from ra_lsbo.x57_cold_campaign import ARMS, INITIAL, run_campaign
from ra_lsbo.x57_cold_qualification import N_QUALIFICATION, qualification_plan, run_qualification
from ra_lsbo.x57_cold_replay import replay_saved
from ra_lsbo.x57_cold_worker import Dispatcher, dump, restore_assets, sha256
from ra_lsbo.x57_live import X57LiveProblem

BASE = (ROOT / "runs/compact_mda_20261021T081006Z_20eb608d"
        / "qualification_recovered_20261025/frozen_base")
ORIGINAL = Path("D:/LSBO/LSBO/X57-LSBO-refactored")
CONFIG = ROOT / "configs/x57_cold_start_qualification_55.json"


def prepare_worker(args, output):
    worker = output / "physical-base"
    identity = restore_assets(BASE, args.original.resolve(), worker)
    subprocess.run([sys.executable, "-B", "-m", "problems.x57.verify_model"],
                   cwd=worker, check=True)
    return worker, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--backend", choices=("mock", "rcaide"), default="mock")
    run.add_argument("--seed", type=int, action="append")
    run.add_argument("--target-wh", type=float, required=True)
    run.add_argument("--online", type=int, default=288)
    run.add_argument("--fast-mock", action="store_true")
    run.add_argument("--mock-profile", choices=("mixed", "no-usable", "infrastructure"), default="mixed")
    run.add_argument("--execute-rcaide", action="store_true")
    qualification = sub.add_parser("qualification-plan")
    replay = sub.add_parser("replay")
    replay.add_argument("--source-run", type=Path, required=True)
    replay.add_argument("--prefix", type=int, default=8)
    replay.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("qualification-run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--execute-rcaide", action="store_true", required=True)
    for command in (run, qualification, execute):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--config", type=Path, default=CONFIG)
        command.add_argument("--original", type=Path, default=ORIGINAL)
    args = parser.parse_args()
    if args.command == "run" and args.backend == "rcaide" and (not args.execute_rcaide or args.fast_mock):
        parser.error("RCAIDE execution requires --execute-rcaide and full optimizer settings")

    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.command == "replay":
        replay_saved(BASE, args.source_run.resolve(), output, args.prefix)
        print(output / "replay_report.json")
        return
    problem = X57LiveProblem(BASE, args.config, output / "workers", allow_mda=False)
    if args.command == "qualification-plan":
        _, worker_identity = prepare_worker(args, output)
        qualification_plan(problem, output, worker_identity)
        print(output / "qualification_plan.json")
        return
    if args.command == "qualification-run":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        worker = (args.plan.resolve().parent / "physical-base")
        dump(output / "reviewed_plan.json", {"sha256": sha256(args.plan), "plan": plan})
        dispatcher = Dispatcher(problem, output, backend="rcaide", worker_root=worker,
                                timeout=problem.problem.config["runtime"]["timeout_seconds"],
                                expected_worker=plan["worker"]["worker_sha256"], max_calls=N_QUALIFICATION)
        run_qualification(problem, dispatcher, output, plan)
        print(output / "qualification_report.json")
        return
    seeds = args.seed if args.seed is not None else [0]
    if args.backend == "rcaide":
        worker, worker_identity = prepare_worker(args, output)
    else:
        worker, worker_identity = ROOT, None
    dispatcher = Dispatcher(problem, output, backend=args.backend, worker_root=worker,
                            timeout=problem.problem.config["runtime"]["timeout_seconds"] if worker_identity else 30.,
                            expected_worker=worker_identity["worker_sha256"] if worker_identity else None,
                            mock_profile=args.mock_profile,
                            max_calls=len(seeds) * (INITIAL + len(ARMS) * args.online))
    report = run_campaign(problem, dispatcher, output, seeds, args.target_wh, args.online, fast_mock=args.fast_mock)
    print(output / "report.json")
    if any(len(s["arms"]) != 3 or any(r["status"] != "complete" for r in s["arms"].values())
           for s in report["seeds"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
