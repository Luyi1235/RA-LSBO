"""Prospective nested 1024/256 seed roster builder, NO aircraft execution.

All ranges are proposals only: historical seed conflicts, task qualification,
the main algorithm definition, CBOB port, physical ledger, resource limits
and explicit phase approval must be independently resolved before execution.
"""
import hashlib
import json
from pathlib import Path

from . import design

SCHEMA = "ra_lsbo_nested_confirmation_pre_registration_candidate_v1"
# Disjoint by design; uniqueness with respect to unknown historical archives is
# explicitly *not* asserted here.
PROPOSED_SEED_STARTS = {
    "aircraft_ordinary": 500001,
    "aircraft_sparse": 600001,
}
N_PRIMARY = 1024
N_AUX = 256
SUBSET_SALT = "ra-lsbo-confirmation-nested-256-prospective-v1"


def nested_subset(case, seeds, number=N_AUX):
    """Pure prespecified hash ordering, with no observations or fitness input."""
    if case not in design.CASES:
        raise ValueError("Unregistered task")
    if len(seeds) != len(set(seeds)) or not 0 < number <= len(seeds):
        raise ValueError("Invalid unique seed roster / requested subset size")
    ranked = sorted(
        seeds,
        key=lambda seed: (
            hashlib.sha256(
                f"{SUBSET_SALT}|{case}|{seed}".encode("ascii")
            ).hexdigest(),
            seed,
        ),
    )
    # Return in the original primary roster order for predictable scheduling.
    chosen = set(ranked[:number])
    return [seed for seed in seeds if seed in chosen]


def candidate_plan():
    seeds = {
        case: list(range(start, start + N_PRIMARY))
        for case, start in PROPOSED_SEED_STARTS.items()
    }
    subset = {
        case: nested_subset(case, seeds[case])
        for case in design.CASES
    }
    plan = {
        "schema": SCHEMA,
        "experiment": "nested_1024_primary_256_auxiliary",
        "seeds_by_case": seeds,
        "auxiliary_by_case": subset,
        "main_arms": list(design.MAIN),
        "auxiliary_arms": list(design.AUXILIARY),
        "budget": {
            "initial": 12,
            "online": 18,
            "verification_max": 1,
            "main_max_calls": 102400,
            "auxiliary_max_calls": 48640,
            "max_calls": 151040,
            "rcaide_max_calls": 0,
        },
        "selection": {
            "subset_rule": "sha256(SUBSET_SALT|case|seed), first 256",
            "subset_salt": SUBSET_SALT,
            "outcome_blind": True,
            "shared_initial_charged_once": True,
        },
        "historical_seed_exclusion_verified": False,
        "ordinary_task_qualified": False,
        "sparse_task_qualified": False,
        "cbob_adapted_qualified": False,
        "all_seven_algorithm_adapters_qualified": False,
        "production_truth_ledger_audited": False,
        "resource_24h_gate_passed": False,
        "phase_approval_receipt_present": False,
        "execution_authorized": False,
    }
    if not design.validate(plan)["structurally_valid"]:
        raise AssertionError("Internal campaign budget or roster mismatch")
    return plan


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def plan_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_candidate_plan(output):
    """Create a new prospective file with exclusive creation; never overwrite."""
    plan = candidate_plan()
    output = Path(output)
    with output.open("xb") as stream:
        stream.write(canonical_bytes(plan))
    return plan_sha256(plan)
