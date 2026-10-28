"""Prospective profiles and complete evaluation budgets for the two experiments."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import sys

from ...budget import digest
from ...partial_bo.problems import candidate_pool
from .methods import ARMS, POSTERIOR_ARMS, settings_for
from .problems import CASES, flight_scenarios, make_problem


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/"runs/compact_mda_20261021T081006Z_20eb608d/qualification_recovered_20261025/frozen_base"
PROFILES = {
    "smoke": {"name": "smoke", "cases": CASES, "seeds": (9601,), "initial_policies": ("random",),
              "initial": 6, "horizon": 2, "pool": 32, "panel": 8, "scenarios": 4,
              "encoder_epochs": 4, "native_gp_steps": 2},
    "pilot": {"name": "pilot", "cases": ("branin_sparse", "rosenbrock_sparse", "aircraft_sparse"),
              "seeds": tuple(range(9701, 9721)), "initial_policies": ("random",),
              "initial": 12, "horizon": 30, "pool": 128, "panel": 16, "scenarios": 8,
              "encoder_epochs": 30, "native_gp_steps": 15},
    "confirm": {"name": "confirm", "cases": CASES, "seeds": tuple(range(10001, 10021)),
                "initial_policies": ("random", "anchor_plus_random"),
                "initial": 16, "horizon": 60, "pool": 512, "panel": 256, "scenarios": 512,
                "encoder_epochs": 120, "native_gp_steps": 50},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget_for(profile, arms=ARMS):
    cases, seeds, policies = len(profile["cases"]), len(profile["seeds"]), profile["initial_policies"]
    aircraft = sum(case.startswith("aircraft_") for case in profile["cases"])
    initial = cases*seeds*sum(profile["initial"]-(policy == "anchor_plus_random") for policy in policies)
    result = {"qualification_anchors": 2*cases, "independent_panel": cases*profile["panel"],
              "shared_initial": initial,
              "online": cases*seeds*len(policies)*len(arms)*profile["horizon"],
              "aircraft_step_convergence": 2*aircraft,
              "held_out_flight_scenarios": aircraft*seeds*len(policies)*len(arms)*profile["scenarios"]}
    result["maximum_model_calls"] = sum(result.values())
    result["maximum_rcaide_mda_calls"] = 0
    return result


def build_plan(profile_name="smoke", suite="combined"):
    if profile_name not in PROFILES:
        raise ValueError("Unknown prospective profile")
    if suite not in {"combined", "mathematical", "simplified_aircraft"}:
        raise ValueError("Unknown experiment suite")
    profile = deepcopy(PROFILES[profile_name])
    if suite == "mathematical":
        profile["cases"] = tuple(case for case in profile["cases"] if not case.startswith("aircraft_"))
    elif suite == "simplified_aircraft":
        profile["cases"] = tuple(case for case in profile["cases"] if case.startswith("aircraft_"))
    if not profile["cases"]:
        raise ValueError(f"Profile {profile_name!r} has no cases for suite {suite!r}")
    files = sorted((ROOT/"ra_lsbo").rglob("*.py"))+sorted(BASE.rglob("*.py"))
    files += sorted((ROOT/"ra_lsbo/experiments").rglob("*.py"))
    files += [ROOT/"docs/MATHEMATICAL_EXPERIMENTS_PROTOCOL.md",
              ROOT/"docs/SIMPLIFIED_AIRCRAFT_EXPERIMENTS_PROTOCOL.md"]
    files += sorted((ROOT/"tests").glob("test_paper_experiments*.py"))
    sources = {str(path): sha(path) for path in files}
    tasks = {}
    for number, case in enumerate(profile["cases"]):
        problem = make_problem(BASE, case)
        anchors = problem.anchors()
        identities = {digest(raw) for raw in anchors}
        panel, rejected = candidate_pool(problem, 26260000+10000*number, profile["panel"], identities)
        identities.update(digest(raw) for raw in panel)
        initial = {}
        for seed in profile["seeds"]:
            initial[str(seed)] = {}
            for policy_number, policy in enumerate(profile["initial_policies"]):
                count = profile["initial"]-(policy == "anchor_plus_random")
                rows, screens = candidate_pool(problem, seed+100000*number+7000000*policy_number, count, identities)
                identities.update(digest(raw) for raw in rows)
                initial[str(seed)][policy] = {"random_candidates": rows, "exact_screen_rejections": screens,
                    "shared_anchor_index": 0 if policy == "anchor_plus_random" else None}
        tasks[case] = {"contract": problem.contract_id, "settings": problem.settings,
                       "anchors": anchors, "panel": panel, "panel_exact_screen_rejections": rejected,
                       "initial": initial}
        if case.startswith("aircraft_"):
            tasks[case]["flight_scenarios"] = {str(seed): [asdict(row) for row in flight_scenarios(
                74000000+number*100000+seed, profile["scenarios"])] for seed in profile["seeds"]}
    return {"schema": "ra_lsbo_experiments_v2", "suite": suite,
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile, "arms": ARMS, "tasks": tasks, "sources": sources, "budget": budget_for(profile),
        "settings": {arm: asdict(settings_for(arm, profile)) for arm in POSTERIOR_ARMS},
        "runtime": {"python": sys.version, "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "torch", "gpytorch", "botorch")}},
        "native_gp_steps": profile["native_gp_steps"], "new_rcaide_mda_calls": 0,
        "all_arms_share_initial_labels": True, "sequential_execution": True,
        "panel_and_scenario_labels_available_to_optimizer": False,
        "kernel_averaging": False, "risk_supervised_encoder_claim": False,
        "baseline_note": "conventional_eic is our numeric-only hierarchical one-hot adaptation; not official CBOB/SCBO",
        "initial_policy": "random draws never rejected using feasibility labels; warm/cold reported after evaluation",
        "anchor_policy": "anchors checked and reported, never used to select a case or seed; only explicit anchored policy uses them",
        "statistical_claim": "smoke/pilot exploratory; confirm is a candidate protocol, not automatic publication evidence",
        "confirmation_requires_additional_authorization": budget_for(profile)["maximum_model_calls"] > 500,
        "sparsity": "ordinary/sparse are prospective task names, actual fractions and binomial intervals must be reported",
        "flight_risk": "post-optimization validation only; nominal optimizer is not a chance-constrained physical-risk optimizer"}


def verify_plan(plan):
    if plan["schema"] != "ra_lsbo_experiments_v2" or plan["budget"] != budget_for(plan["profile"], plan["arms"]):
        raise ValueError("Unregistered schema or evaluation budget")
    suite = plan["suite"]
    if suite not in {"combined", "mathematical", "simplified_aircraft"}:
        raise ValueError("Unregistered experiment suite")
    for case in plan["profile"]["cases"]:
        if (suite == "mathematical" and case.startswith("aircraft_")
                or suite == "simplified_aircraft" and not case.startswith("aircraft_")):
            raise ValueError("Task does not belong to its registered suite")
    changed = [path for path, identity in plan["sources"].items() if sha(path) != identity]
    if changed:
        raise ValueError("Registered implementation changed: "+repr(changed))
    for case, task in plan["tasks"].items():
        if make_problem(BASE, case).contract_id != task["contract"]:
            raise ValueError("Task contract changed")
