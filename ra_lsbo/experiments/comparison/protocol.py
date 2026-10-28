"""Versioned prospective plans for external-algorithm comparisons."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path

from ..core.protocol import BASE, PROFILES, budget_for as core_budget_for, build_plan as build_core_plan
from . import ARMS
from .methods import ADAPTATIONS, METHODS

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "ra_lsbo_external_comparison_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget_for(profile, arms=ARMS):
    return core_budget_for(profile, arms)


def build_plan(profile_name="pilot", suite="mathematical"):
    if profile_name not in PROFILES or suite not in {"mathematical", "simplified_aircraft"}:
        raise ValueError("Choose a registered profile and one experiment package")
    plan = build_core_plan(profile_name, suite)
    if profile_name == "pilot":
        expected = (("branin_sparse", "rosenbrock_sparse") if suite == "mathematical"
                    else ("aircraft_sparse",))
        if tuple(plan["profile"]["cases"]) != expected:
            raise ValueError("The registered pilot must contain the sparse screening case set")
    plan.update({
        "schema": SCHEMA,
        "suite": suite,
        "confirmation_requires_additional_authorization": False,
        "arms": ARMS,
        "budget": budget_for(plan["profile"], ARMS),
        "methods": METHODS,
        "adaptations": ADAPTATIONS,
        "comparison_protocol": {
            "same_initial_labels": True,
            "same_nominal_evaluation_budget": True,
            "candidate_generation": "shared exact-screened pool; NSGA-II adapter may propose a validated offspring",
            "held_out_panels_and_flight_scenarios_are_optimizer_private": True,
            "new_rcaide_mda_calls": 0,
            "pilot_is_exploratory_only": True,
            "original_lsbo_in_primary_roster": False,
            "reason_original_lsbo_omitted": "Retained as historical/secondary reference; this screen prioritizes external algorithms.",
        },
        "registered_utc": datetime.now(timezone.utc).isoformat(),
    })
    plan["sources"][str(ROOT/"docs/ALGORITHM_COMPARISON_LITERATURE_REVIEW_20261026.md")] = sha(
        ROOT/"docs/ALGORITHM_COMPARISON_LITERATURE_REVIEW_20261026.md")
    plan["sources"][str(ROOT/"docs/EXTERNAL_ALGORITHM_COMPARISON_PROTOCOL_20261026.md")] = sha(
        ROOT/"docs/EXTERNAL_ALGORITHM_COMPARISON_PROTOCOL_20261026.md")
    test_file = ROOT/"tests/test_external_comparison.py"
    plan["sources"][str(test_file)] = sha(test_file)
    return plan


def verify_plan(plan):
    if plan.get("schema") != SCHEMA or tuple(plan.get("arms", ())) != ARMS:
        raise ValueError("Unregistered comparison schema or arm roster")
    if plan["budget"] != budget_for(plan["profile"], plan["arms"]):
        raise ValueError("Comparison budget mismatch")
    if set(plan["methods"]) != set(ARMS) or set(plan["adaptations"]) != {
            "logcei", "scbo_adapted", "sego_utb_adapted", "nsga2_adapted"}:
        raise ValueError("Comparison method definitions are incomplete")
    changed = [path for path, identity in plan["sources"].items()
               if not Path(path).is_file() or sha(path) != identity]
    if changed:
        raise ValueError("Registered comparison source changed: "+repr(changed))
    if plan["suite"] not in {"mathematical", "simplified_aircraft"}:
        raise ValueError("Comparison plan is not package-specific")
    for case, task in plan["tasks"].items():
        if task["contract"] != __import__(
                "ra_lsbo.experiments.core.problems", fromlist=["make_problem"]
            ).make_problem(BASE, case).contract_id:
            raise ValueError("Problem contract differs from prospective plan")
    return None

