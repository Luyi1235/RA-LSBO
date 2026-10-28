"""Post-hoc, paired development control using the margin study's exact initial rows."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from contextlib import redirect_stdout
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.system_experiment import append_json, save_new
from tools import run_adaptive_research as direct
from tools.run_margin_research import validate


def trial(folder, case, seed):
    torch.set_num_threads(1)
    folder = Path(folder)
    specification = load_json((folder/"plan.json").read_bytes())
    verify_sources(specification["source_sha256"])
    parent = Path(specification["paired_margin_study"])
    plan = load_json((parent/"plan.json").read_bytes())
    validate(plan)
    output = folder/case/("adaptive_full_"+str(seed))
    output.mkdir(parents=True, exist_ok=False)
    problem = direct.problem_for(case, plan["base_project"])
    with (output/"execution.log").open("x", encoding="utf-8") as stream, redirect_stdout(stream):
        return direct._trial(plan, parent, output, problem, case, "adaptive_full", seed)


def execute(parent):
    plan = load_json((parent/"plan.json").read_bytes())
    validate(plan)
    folder = parent/"direct_joint_paired_control"
    folder.mkdir(exist_ok=False)
    save_new(folder/"plan.json", {
        "schema": "ra_direct_joint_paired_supplement_v1", "additional_real_mda_calls": 0,
        "created_utc": datetime.now(timezone.utc).isoformat(), "paired_margin_study": str(parent),
        "cases": plan["cases"], "seeds": plan["seeds"], "max_controlled_online_calls": 144,
        "source_sha256": dict(plan["source_sha256"], **{
            str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}),
        "scope": "Post-hoc development control, specified after inspecting partial margin results",
        "reason": "Same seeds and exact initial/calibration rows are necessary to isolate the margin bridge",
        "unchanged": "Original direct joint learner and selection; all settings identical; all six runs retained"})
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(trial, str(folder), case, seed)
                   for case in plan["cases"] for seed in plan["seeds"]]
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            append_json(folder/"completed.jsonl", result)
            print("Paired direct control", result["case"], result["seed"],
                  result["evaluations"], result["improvement_percent"], flush=True)
    save_new(folder/"summary.json", {"status": "completed", "trials": results,
                                    "additional_real_mda_calls": 0})


if __name__ == "__main__":
    import os
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    execute(args.folder.resolve())
