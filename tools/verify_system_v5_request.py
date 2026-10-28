"""Validate all frozen v5 physical designs and cumulative cost with zero MDA."""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem
from tools.run_system_validation_v5 import CAP, TRAINING_CALLS, CALIBRATION_CALLS, validate_plan


def verify_candidates(plan, initial, problem):
    seen = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in initial}
    for item in plan["probes"]:
        candidate = problem.candidate_class(**item["candidate"])
        if problem.canonical(item["candidate"]) != item["candidate"]:
            raise ValueError("Noncanonical training acquisition")
        key = problem.space.key(candidate)
        if key in seen or any(not np.isfinite(v) or v > 0 for v in problem.problem.precheck(candidate).values()):
            raise ValueError("Repeated or illegal training acquisition")
        seen.add(key)
        if item["index"] > plan["carried_mda_calls"]:
            center = problem.candidate_class(**item["center_candidate"])
            if np.max(np.abs(problem.space.unit(candidate)-problem.space.unit(center))) > item["radius"]+1e-10:
                raise ValueError("Architecture stress radius exceeded")
    return len(seen)-len(initial)


def verify(folder):
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    verify_sources(plan["source_sha256"])
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                            folder/"unused_preflight", allow_mda=False)
    if problem.contract_id != plan["physical"]["contract_id"]:
        raise ValueError("Physical contract differs")
    count = verify_candidates(plan, load_json((folder/"initial_history.json").read_bytes()), problem)
    result = dict(status="passed", new_mda_calls=0, complete_mda_cap=CAP,
        prior_mda_calls=plan["carried_mda_calls"], maximum_remaining_mda_calls=CAP-plan["carried_mda_calls"],
        training_acquisition_calls=TRAINING_CALLS, fresh_calibration_calls=CALIBRATION_CALLS,
        legal_unique_probes_verified=count, physical_contract=problem.contract_id,
        label_contract_id=plan["label_contract_id"],
        plan_sha256=hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest(),
        preflight_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_new(folder/"preflight.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    verify(parser.parse_args().folder.resolve())
