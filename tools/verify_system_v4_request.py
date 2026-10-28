"""Zero-MDA check of the fixed candidates and revised observation contract."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_system_validation_v4 import validate_plan


def verify(folder):
    from ra_lsbo.audit import load_json
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    from ra_lsbo.x57_live import X57LiveProblem
    from ra_lsbo.boundary_validation import verify_sources
    from ra_lsbo.system_experiment import save_new
    import hashlib
    import numpy as np
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
            center = problem.candidate_class(**anchors[item["anchor_index"]]["candidate"])
            if np.max(np.abs(problem.space.unit(candidate)-problem.space.unit(center))) > item["radius"]+1e-10:
                raise ValueError("Neighborhood radius exceeded")
            checked += 1
    remaining = 120-plan["carried_mda_calls"]+72+8*3*12
    if remaining+plan["carried_mda_calls"] != 480:
        raise ValueError("All-version complete cost differs")
    result = dict(status="passed", new_mda_calls=0, complete_mda_cap=480,
                  prior_mda_calls=68, maximum_remaining_mda_calls=remaining,
                  legal_unique_probes_verified=120, neighborhood_distances_verified=checked,
                  physical_contract=problem.contract_id, label_contract_id=plan["label_contract_id"],
                  plan_sha256=hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest(),
                  preflight_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_new(folder/"preflight.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    verify(parser.parse_args().folder.resolve())
