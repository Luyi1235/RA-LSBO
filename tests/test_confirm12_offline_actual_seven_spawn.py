"""Opt-in actual seven BO one-step Windows spawn workload with fake records."""
import json
import os
from pathlib import Path
import pytest

from confirm12.offline_actual_seven_spawn import (
    SCHEMA, TEST_SEEDS, _make_plan, run,
)
from confirm12.real_candidate_engines import ALL_REQUESTED


def test_offline_plan_is_not_a_production_receipt():
    plan = _make_plan()
    assert plan["execution_authorized"] is False
    assert plan["max_calls"] == 38
    assert plan["scope"] == "RECORD_CONTRACT_TEST_ONLY"
    assert plan["note"] == SCHEMA
    assert all(len(plan["cases"][case]["arms_by_seed"][str(seed)]) == 7
               for case, seed in TEST_SEEDS.items())
    assert len(ALL_REQUESTED) == 7


def test_refuse_unsupported_workload_process_count(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path/"not_created", workers=13)
    assert not (tmp_path/"not_created").exists()


@pytest.mark.skipif(
    os.environ.get("CONFIRM12_RUN_PARALLEL_ACTUAL") != "1",
    reason="explicit real BO fitting + fake evidence multi-process opt-in",
)
def test_two_worker_real_seven_methods_auditable_offline(tmp_path):
    output=tmp_path/"trial"
    report=run(output,workers=2)
    assert report["completed_policy_steps"] == 14
    assert report["expected_policy_steps"] == 14
    assert report["workers"] == 2
    assert report["total_fake_records_charged"] == 38
    assert report["atomic_ledger_contract_passed"]
    assert not report["errors"]
    assert report["scientific_method_qualification"] is False
    assert report["production_approval"] is False
    assert report["aircraft_calls"]==report["rcaide_calls"]==0
    assert len({(r["case"],r["method"]) for r in report["results"]}) == 14
    assert all(row["source_kind"]=="OFFLINE_SYNTHETIC_NO_PHYSICS"
               for row in report["results"])
    assert (output/"offline_plan.json").is_file()
    assert (output/"offline_summary.json").is_file()
    assert json.loads((output/"offline_summary.json").read_text())["completed_policy_steps"]==14
    with pytest.raises(FileExistsError):
        run(output,workers=2)
