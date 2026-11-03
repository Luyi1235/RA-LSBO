"""Offline checks for the prospective 151,040-call nested plan.

No true or mocked aircraft simulation is needed to build these files.
"""
import json
import pytest

from confirm12.nested_seed_plan import (
    N_AUX, N_PRIMARY, candidate_plan, nested_subset, plan_sha256,
    write_candidate_plan,
)
from confirm12.design import validate


def test_nested_roster_and_budget_arithmetic():
    plan = candidate_plan()
    assert validate(plan)["structurally_valid"]
    assert plan["execution_authorized"] is False
    assert plan["budget"]["max_calls"] == 151040
    assert plan["budget"]["main_max_calls"] == 102400
    assert plan["budget"]["auxiliary_max_calls"] == 48640
    assert 2 * N_PRIMARY * (12 + 2 * (18 + 1)) + (
        2 * N_AUX * 5 * (18 + 1)
    ) == 151040
    all_seeds = []
    for case, seeds in plan["seeds_by_case"].items():
        assert len(seeds) == N_PRIMARY
        assert len(plan["auxiliary_by_case"][case]) == N_AUX
        assert set(plan["auxiliary_by_case"][case]).issubset(seeds)
        all_seeds.extend(seeds)
    assert len(set(all_seeds)) == 2 * N_PRIMARY
    assert all(plan[k] is False for k in (
        "historical_seed_exclusion_verified", "ordinary_task_qualified",
        "sparse_task_qualified", "cbob_adapted_qualified",
        "all_seven_algorithm_adapters_qualified",
        "production_truth_ledger_audited", "resource_24h_gate_passed",
        "phase_approval_receipt_present",
    ))


def test_subset_deterministic_and_outcome_blind():
    seeds = candidate_plan()["seeds_by_case"]["aircraft_ordinary"]
    a = nested_subset("aircraft_ordinary", seeds)
    b = nested_subset("aircraft_ordinary", list(reversed(seeds)))
    assert len(a) == N_AUX
    assert set(a) == set(b)
    assert a == nested_subset("aircraft_ordinary", seeds)


@pytest.mark.parametrize("bad", ([], [1, 1]))
def test_rejects_invalid_roster(bad):
    with pytest.raises(ValueError):
        nested_subset("aircraft_ordinary", bad)


def test_write_once_and_sha_are_stable(tmp_path):
    plan = candidate_plan()
    output = tmp_path / "plan.json"
    sha = write_candidate_plan(output)
    assert sha == plan_sha256(plan)
    assert json.loads(output.read_text(encoding="utf-8")) == plan
    with pytest.raises(FileExistsError):
        write_candidate_plan(output)
