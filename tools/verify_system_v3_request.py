"""Read-only physical legality and neighborhood-distance preflight; zero MDA."""

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
from tools.run_system_validation_v3 import validate_plan


def verify(folder):
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    verify_sources(plan["source_sha256"])
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                             folder/"unused_preflight", allow_mda=False)
    if problem.contract_id != plan["physical"]["contract_id"]:
        raise ValueError("Physical contract changed")
    initial = load_json((folder/"initial_history.json").read_bytes())
    seen = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in initial}
    anchors = {r["index"]: r for r in plan["carried_probes"]}
    checked = 0
    for item in plan["probes"]:
        candidate = problem.candidate_class(**item["candidate"])
        if item["candidate"] != problem.canonical(item["candidate"]):
            raise ValueError("Noncanonical physical candidate")
        key = problem.space.key(candidate)
        if key in seen or any(v > 0 for v in problem.problem.precheck(candidate).values()):
            raise ValueError("Illegal/repeated candidate")
        seen.add(key)
        if "anchor_index" in item:
            anchor = problem.candidate_class(**anchors[item["anchor_index"]]["candidate"])
            distance = np.max(np.abs(problem.space.unit(candidate)-problem.space.unit(anchor)))
            if distance > item["radius"]+1e-10:
                raise ValueError("Physical unit-coordinate neighborhood radius exceeded")
            checked += 1
    remaining = checked+72+8*3*12
    if remaining+plan["carried_mda_calls"] != 480:
        raise ValueError("Complete budget does not include prior versions")
    result = {"status": "passed", "new_mda_calls": 0, "complete_mda_cap": 480,
              "prior_mda_calls": plan["carried_mda_calls"], "maximum_remaining_mda_calls": remaining,
              "legal_unique_probes_verified": 120, "neighborhood_distances_verified": checked,
              "physical_contract": problem.contract_id,
              "plan_sha256": hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest(),
              "preflight_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_new(folder/"preflight.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    verify(args.folder.resolve())
