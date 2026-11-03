"""Plan-only iid geometry sampling. This module never imports an evaluator."""

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "57e390fc8f6280db9f201940aec14089e81c104c"
SEEDS = {"aircraft_ordinary": 21001, "aircraft_sparse": 21002}
HISTORICAL_CONTRACTS = {
    "aircraft_ordinary": "f8d3f2713792797d3d60bb2354451893b052d09f6c1bb6eaaea0c5486edb1345",
    "aircraft_sparse": "d72a3e2edc8e3c4f0020c0a8584ef13a2045c24f986dc558b216929900ebe008",
}
SOURCES = (
    "ra_lsbo/experiments/iid_qualification.py",
    "ra_lsbo/partial_bo/problems.py", "ra_lsbo/experiments/core/problems.py",
    "runs/compact_mda_20261021T081006Z_20eb608d/qualification_recovered_20261025/frozen_base/lsbo/space.py",
    "runs/compact_mda_20261021T081006Z_20eb608d/qualification_recovered_20261025/frozen_base/problems/x57/nasa_battery.py",
)
CONFIGURATIONS = tuple((layout, count, parallel)
    for layout in ("wingtip", "distributed")
    for count in ((2,) if layout == "wingtip" else (2, 4, 6))
    for parallel in (32, 40, 48))
BOUNDS = {"wing_area": (10., 18.), "aspect_ratio": (7.5, 13.5),
          "tip_diameter": (1.2, 1.9), "dep_diameter": (.45, 1.1),
          "prop_spacing": (.5, .9), "cruise_airspeed": (44., 58.)}
DEFAULTS = {"tip_diameter": 1.72, "dep_diameter": .8, "prop_spacing": .7}
REQUIRED = {"span", "packing", "stall", "root_bending", "soc", "current", "voltage", "power"}


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw(rng):
    layout, count, parallel = CONFIGURATIONS[int(rng.integers(len(CONFIGURATIONS)))]
    # Draw all six coordinates for every candidate, then canonicalize inactive ones.
    unit = rng.random(len(BOUNDS))
    continuous = {name: float(lo + value * (hi - lo))
                  for (name, (lo, hi)), value in zip(BOUNDS.items(), unit)}
    if layout == "wingtip":
        continuous["dep_diameter"] = DEFAULTS["dep_diameter"]
        continuous["prop_spacing"] = DEFAULTS["prop_spacing"]
    else:
        continuous["tip_diameter"] = DEFAULTS["tip_diameter"]
    return {"continuous": continuous, "discrete": {"prop_layout": layout,
            "propeller_count": count, "parallel_strings": parallel}}


def screen(candidate):
    c, d = candidate["continuous"], candidate["discrete"]
    span = math.sqrt(c["wing_area"] * c["aspect_ratio"])
    packing = (d["propeller_count"] / 2 * c["dep_diameter"]
               + .05 * (d["propeller_count"] / 2 - 1) + .6
               - c["prop_spacing"] * span / 2) if d["prop_layout"] == "distributed" else -1.
    values = {"span": span / 14.5 - 1, "packing": packing}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Nonfinite analytical screen")
    return values


def sample(seed, count, *, max_attempts_per_sample=1000, rng=None, precheck=screen):
    """Rejection sample iid q_A, never checking previous candidates or outcomes."""
    if type(seed) is not int or seed < 0 or type(count) is not int or count < 1:
        raise ValueError("Invalid seed or count")
    if type(max_attempts_per_sample) is not int or max_attempts_per_sample < 1:
        raise ValueError("Invalid attempt limit")
    if rng is None:
        rng = np.random.Generator(np.random.PCG64(seed))
    records, proposals, accepted = [], Counter(), Counter()
    for index in range(count):
        for attempt in range(1, max_attempts_per_sample + 1):
            candidate = draw(rng)
            key = json.dumps(candidate["discrete"], sort_keys=True)
            proposals[key] += 1
            constraints = precheck(candidate)
            if set(constraints) != {"span", "packing"} or not all(
                    isinstance(x, (int, float)) and math.isfinite(x) for x in constraints.values()):
                raise ValueError("Invalid screen contract")
            if all(x <= 0 for x in constraints.values()):
                accepted[key] += 1
                records.append({"index": index, "candidate": candidate,
                                "candidate_sha256": digest(candidate),
                                "screen": constraints, "attempts": attempt,
                                "charge_role": "one_call_per_index_even_if_duplicate"})
                break
        else:
            raise RuntimeError("Geometry rejection limit exhausted; no plan produced")
    return {"records": records, "proposed_configurations": dict(proposals),
            "accepted_configurations": dict(accepted),
            "attempts": sum(proposals.values()), "rejections": sum(proposals.values()) - count}


def outcome_contract(status, completed, objective_kind, constraints):
    """No evaluator: classify supplied evidence conservatively, never infer missing labels."""
    if status in {"unknown", "infrastructure_error", "numerical_failure", "hidden_failure"}:
        return "abort_unknown"
    if status not in {"completed", "physical_abort"}:
        raise ValueError("Unrecognized status")
    if not isinstance(constraints, dict) or not constraints:
        return "abort_unknown"
    if not set(constraints).issubset(REQUIRED):
        raise ValueError("Unknown constraint")
    values = tuple(constraints.values())
    if any(x is not True and x is not False and x is not None for x in values):
        raise ValueError("Invalid constraint evidence")
    if any(x is False for x in values):
        return "known_infeasible"
    if (status == "completed" and completed is True and objective_kind == "exact"
            and set(constraints) == REQUIRED and all(x is True for x in values)):
        return "known_feasible"
    return "abort_unknown"


def build_plan(source_commit):
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("Expected full source commit")
    for name in SOURCES[1:]:
        reference = subprocess.check_output(["git", "show", BASE_COMMIT + ":" + name], cwd=ROOT)
        local = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        if reference.replace(b"\r\n", b"\n") != local:
            raise RuntimeError("Historical physical source mismatch: " + name)
    tasks = {case: {"seed": seed, **sample(seed, 300)} for case, seed in SEEDS.items()}
    return {"schema": "iid_qualification_plan_only_v1", "source_commit": source_commit,
        "reference_aircraft_commit": BASE_COMMIT,
        "historical_20_second_contracts": HISTORICAL_CONTRACTS,
        "source_sha256": {name: file_hash(ROOT / name) for name in SOURCES},
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "platform": platform.platform(), "numpy": np.__version__,
                    "thread_environment": {key: os.environ.get(key) for key in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED")},
                    "rng": "numpy.random.Generator(PCG64(seed))"},
        "distribution": {"configurations": CONFIGURATIONS, "continuous_bounds": BOUNDS,
            "inactive_defaults": DEFAULTS, "draw": "uniform_config_then_independent_uniform_active",
            "screen": "span<=0 and packing<=0", "with_replacement": True,
            "per_sample_attempt_limit": 1000, "labels_used": False},
        "tasks": tasks, "budget": {"planned_simplified_aircraft_calls": 600,
            "actual_simplified_aircraft_calls": 0, "rcaide_calls": 0},
        "thresholds": {"ordinary_lower_95_gt": .2, "sparse_upper_95_lt": .1,
            "sparse_minimum_feasible": 1, "sparse_maximum_feasible_out_of_300": 19,
            "interval": "two-sided 95% Clopper-Pearson"},
        "stopping": ["any unknown or infrastructure/numerical failure",
                     "unsettled call", "source/environment/plan drift", "no automatic retry"],
        "seed_history_status": "incomplete_review_pending; consult separate audit",
        "execution_authorized": False}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Write a plan only; no evaluator exists here")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    if status.strip():
        raise RuntimeError("Tracked source tree must be clean")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    plan = build_plan(sha)
    path = Path(args.output)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(plan, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print("plan_sha256", file_hash(path), "source_commit", sha, "actual_calls 0")


if __name__ == "__main__":
    main()
