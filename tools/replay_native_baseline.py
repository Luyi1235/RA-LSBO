"""Replay native LSBO proposals against settled outcomes, with MDA disabled."""

import argparse
from contextlib import redirect_stdout
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.controlled_sparse import SparseInteriorFiveModeProblem
from ra_lsbo.mechanism_system import FiveModeProblem, SystemSettings
from ra_lsbo.system_experiment import OriginalLSBOWarmStart, save_new
from ra_lsbo.x57_live import X57LiveProblem
from tools.run_system_validation import read_lines


def replay(folder, output, kind=None):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    report = load_json((folder/"report.json").read_bytes())
    audit = load_json((folder/"independent_audit.json").read_bytes())
    if report["status"] != "terminal" or audit["status"] != "passed":
        raise ValueError("Require terminal audited study")
    verify_sources(plan["source_sha256"])
    verify_sources(audit["input_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for run in [*report.get("controlled", []), *report.get("physical", [])]:
        if run["method"] != "lsbo" or (kind is not None and run["kind"] != kind):
            continue
        directory = Path(run["directory"])
        manifest = load_json((directory/"manifest.json").read_bytes())
        history = load_json(Path(manifest["history_path"]).read_bytes())
        proposals = read_lines(directory/"proposals.jsonl")
        observations = read_lines(directory/"observations.jsonl")
        target = output/(run["kind"]+"_seed_"+str(run["seed"]))
        target.mkdir()
        base = plan["physical"]["base_project"]
        if run["kind"] == "physical":
            problem = X57LiveProblem(base, plan["physical"]["config_path"], target/"unused_workers",
                                     allow_mda=False)
        else:
            case = (SparseInteriorFiveModeProblem if plan["schema"] == "ra_sparse_controlled_validation_v1"
                    else FiveModeProblem)
            problem = case(base)
        if problem.contract_id != manifest["contract_id"]:
            raise ValueError("Native replay contract differs")
        checked = []
        with (target/"native_replay.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            engine = OriginalLSBOWarmStart(
                problem, base, history, run["seed"], SystemSettings(**manifest["settings"]), target)
            for position, expected in enumerate(proposals):
                proposed, selection = engine.propose()
                actual, original = proposed["candidate"], expected["proposal"]["candidate"]
                if actual["discrete"] != original["discrete"]:
                    raise ValueError("Native replay discrete configuration differs")
                delta = float(np.max(np.abs(
                    problem.space.unit(problem.candidate_class(**actual))
                    -problem.space.unit(problem.candidate_class(**original)))))
                if (delta > 1e-8
                        or proposed["source"] != expected["proposal"]["source"]
                        or selection["original_phase"] != expected["selection"]["original_phase"]):
                    raise ValueError("Native replay proposal differs at "+str(expected["evaluation"]))
                checked.append({"evaluation": expected["evaluation"], "maximum_unit_coordinate_delta": delta,
                                "discrete_and_source_match": True})
                if position < len(observations):
                    # Feed only the outcome available after this actual submission.
                    engine.accept(observations[position], expected["selection"])
        records.append({"kind": run["kind"], "seed": run["seed"], "checked": checked,
                        "proposals_checked": len(checked), "settled_outcomes_consumed": len(observations),
                        "status": "passed", "allow_mda": False})
        print("Native replay passed", run["kind"], run["seed"], len(checked), flush=True)
    verify_sources(plan["source_sha256"])
    verify_sources(audit["input_sha256"])
    result = {
        "status": "passed", "additional_real_mda_calls": 0,
        "comparison": "Discrete configuration, source, phase and unit coordinates within 1e-8",
        "actual_outcomes_used_only_after_reconstructed_proposal": True, "runs": records,
        "input_audit_sha256": hashlib.sha256((folder/"independent_audit.json").read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_new(output/"native_replay.json", result)
    (output/"native_replay.source.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("controlled", "physical"))
    args = parser.parse_args()
    replay(args.folder.resolve(), args.output.resolve(), args.kind)
