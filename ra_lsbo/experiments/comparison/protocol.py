"""Versioned prospective plans for external-algorithm comparisons."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path

from ..core.protocol import BASE, PROFILES, budget_for as core_budget_for, build_plan as build_core_plan
from . import ARMS
from .methods import ADAPTATIONS, METHODS

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "ra_lsbo_external_comparison_v2"
REVISION_PROTOCOL = "docs/COMPARISON_RELIABILITY_REVISION_20261028.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget_for(profile, arms=ARMS):
    return core_budget_for(profile, arms)


def build_plan(profile_name="pilot", suite="mathematical"):
    if profile_name not in PROFILES or suite not in {"mathematical", "simplified_aircraft"}:
        raise ValueError("Choose a registered profile and one experiment package")
    if tuple(PROFILES[profile_name]["initial_policies"]) != ("random",):
        raise ValueError("External comparison supports only random initialization; "
                         "confirm/anchored profiles require a separately implemented protocol")
    plan = build_core_plan(profile_name, suite)
    if profile_name == "pilot":
        expected = (("branin_sparse", "rosenbrock_sparse") if suite == "mathematical"
                    else ("aircraft_sparse",))
        if tuple(plan["profile"]["cases"]) != expected:
            raise ValueError("The registered pilot must contain the sparse screening case set")
    plan.update({
        "schema": SCHEMA,
        "suite": suite,
        "confirmation_requires_additional_authorization": True,
        "arms": ARMS,
        "budget": budget_for(plan["profile"], ARMS),
        "methods": METHODS,
        "adaptations": ADAPTATIONS,
        "comparison_protocol": {
            "revision": "comparison_reliability_20261028",
            "same_initial_labels": True,
            "same_nominal_evaluation_budget": True,
            "candidate_generation": "shared exact-screened pool; RA adds local candidates and NSGA-II may propose a validated offspring",
            "held_out_panels_and_flight_scenarios_are_optimizer_private": True,
            "new_rcaide_mda_calls": 0,
            "pilot_is_exploratory_only": True,
            "original_lsbo_in_primary_roster": False,
            "reason_original_lsbo_omitted": "Retained as historical/secondary reference; this screen prioritizes external algorithms.",
            "explicit_execution_budget_required": True,
        },
        "registered_utc": datetime.now(timezone.utc).isoformat(),
    })
    for name in ("matplotlib", "scikit-learn"):
        plan["runtime"]["packages"][name] = importlib.metadata.version(name)
    documents = ("docs/ALGORITHM_COMPARISON_LITERATURE_REVIEW_20261026.md",
                 "docs/EXTERNAL_ALGORITHM_COMPARISON_PROTOCOL_20261026.md", REVISION_PROTOCOL)
    tests = [ROOT/"tests/test_external_comparison.py", ROOT/"tests/test_numerical_safeguards.py"]
    tests += sorted((ROOT/"tests").glob("test_comparison*.py"))
    for path in [*(ROOT/name for name in documents), *tests]:
        plan["sources"][str(path)] = sha(path)
    return plan


def verify_plan(plan):
    if plan.get("schema") != SCHEMA or tuple(plan.get("arms", ())) != ARMS:
        raise ValueError("Unregistered comparison schema or arm roster; preserve old plans unchanged")
    if tuple(plan["profile"]["initial_policies"]) != ("random",):
        raise ValueError("External comparison supports only random initialization")
    if plan["budget"] != budget_for(plan["profile"], plan["arms"]):
        raise ValueError("Comparison budget mismatch")
    if plan["methods"] != METHODS or plan["adaptations"] != ADAPTATIONS:
        raise ValueError("Comparison method definitions differ from the registered implementation")
    changed = [path for path, identity in plan["sources"].items()
               if not Path(path).is_file() or sha(path) != identity]
    if changed:
        raise ValueError("Registered comparison source changed: "+repr(changed))
    if plan["suite"] not in {"mathematical", "simplified_aircraft"}:
        raise ValueError("Comparison plan is not package-specific")
    if set(plan["tasks"]) != set(plan["profile"]["cases"]):
        raise ValueError("Comparison task roster differs from the registered profile")
    for case, task in plan["tasks"].items():
        if task["contract"] != __import__(
                "ra_lsbo.experiments.core.problems", fromlist=["make_problem"]
            ).make_problem(BASE, case).contract_id:
            raise ValueError("Problem contract differs from prospective plan")
    return None
