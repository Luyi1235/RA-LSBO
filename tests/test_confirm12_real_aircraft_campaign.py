"""Contract tests for full production *code* using in-memory fabricated records.

The tests never call the expensive actual AircraftCase.evaluate.  The real
runner invokes that method, and can be launched only from the explicit full
experiment Python entry. Scientific qualification is recorded separately.
"""
from dataclasses import replace
import sqlite3
import pytest

from confirm12 import real_call_ledger as ledger
from confirm12 import real_aircraft_campaign as campaign
from confirm12.real_candidate_engines import ALL_REQUESTED
from ra_lsbo.budget import digest
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.partial_bo.problems import candidate_pool


def _plan(case,seed):
    nominal=problem_for(case)
    fine=problem_for(case,5.)
    other="aircraft_sparse" if case=="aircraft_ordinary" else "aircraft_ordinary"
    return {
        "schema":ledger.SCHEMA,"scope":ledger.SCOPE,
        "phase":"full_seven_arm_experiment",
        "authorization_basis":"explicit_user_ide_execute",
        "execution_authorized":True,"max_calls":50,
        "initial":12,"online":18,"verification":1,
        "cases":{
            case:{
                "seeds":[seed],
                "contracts":{"20s":nominal.contract_id,"5s":fine.contract_id},
                "arms_by_seed":{str(seed):list(ALL_REQUESTED)}},
            other:{"seeds":[],"contracts":{
                "20s":problem_for(other).contract_id,
                "5s":problem_for(other,5.).contract_id},"arms_by_seed":{}},
        },
    }


def _fake_real_record(problem,candidate,ticket):
    # Fake feedback for validating record schema ONLY: not true physical work.
    outputs=dict(problem.precheck(candidate))
    outputs.update({name:-.1 for name in problem.required_constraints
                    if name not in problem.analytical_constraints})
    source=problem.record(candidate,31876.0,outputs,
                          {"source":"TEST_FAKE_DATA_NOT_PHYSICAL"})
    return replace(source,diagnostics={
        **source.diagnostics,
        "confirm12_real_call":{"call_id":ticket,
                               "evaluator":"AircraftCase.evaluate",
                               "mock_for_unit_test":True}})


def test_full_roster_is_151040_and_12_workers(monkeypatch):
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase,"evaluate",lambda *a,**k:
                        pytest.fail("Plan creation cannot evaluate airplane"))
    plan=campaign.build_plan()
    assert plan["max_calls"]==151040
    assert plan["workers"]==12
    assert plan["authorization_basis"]=="explicit_user_ide_execute"
    assert plan["scientific_status"]==campaign.SCIENCE_STATUS
    assert set(plan["cases"])==set(campaign.TASKS)
    assert set(ALL_REQUESTED)=={"ra_lsbo","full_latent","full_onehot",
                                "reverse_representation","interval_as_missing",
                                "random","cbob_adapted"}
    assert sum(len(x["seeds"]) for x in plan["cases"].values())==2048
    assert sum(len(arms) for spec in plan["cases"].values()
               for arms in spec["arms_by_seed"].values())==2*1024*2+2*256*5
    assert sum(12+len(arms)*19 for spec in plan["cases"].values()
               for arms in spec["arms_by_seed"].values())==151040


def test_production_ledger_exact_ticket_and_duplicate_guard(tmp_path):
    case="aircraft_ordinary";seed=31001
    plan=_plan(case,seed)
    path=tmp_path/"real.sqlite"
    ledger.create(path,plan)
    problem=problem_for(case)
    points,_=candidate_pool(problem,177717,12)
    for i,candidate in enumerate(points):
        ticket=ledger.start(path,case,seed,"initial","",i,candidate,problem)
        with pytest.raises(sqlite3.IntegrityError):
            ledger.start(path,case,seed,"initial","",i,candidate,problem)
        assert ledger.settle(path,ticket,_fake_real_record(problem,candidate,ticket))=="settled"
    pool,_=candidate_pool(problem,177720,128,
                          {digest(x) for x in points})
    online=pool[0]
    ticket=ledger.start(path,case,seed,"online","random",0,online,problem)
    with pytest.raises(ValueError,match="receipt"):
        ledger.settle(path,ticket,problem.record(online,3.0,{"soc":-.1},
                                                {"forged_without_ticket":True}))
    assert ledger.settle(path,ticket,_fake_real_record(problem,online,ticket))=="settled"
    report=ledger.audit(path,plan)
    assert report["ledger_contract_passed"],report
    assert report["calls"]==13
    assert report["evaluator_receipts_consistent"]
    assert report["scientific_qualification_inferred"] is False


def test_real_ledger_rejects_validation_scope(tmp_path):
    plan=_plan("aircraft_sparse",13)
    plan["execution_authorized"]=False
    with pytest.raises(ValueError,match="user-initiated"):
        ledger.create(tmp_path/"refused.sqlite",plan)


def test_wrong_feedback_preserves_started_ticket(tmp_path):
    case="aircraft_sparse";seed=42
    plan=_plan(case,seed)
    db=tmp_path/"real.sqlite"
    ledger.create(db,plan)
    problem=problem_for(case)
    candidate=problem.space.default().to_dict()
    ticket=ledger.start(db,case,seed,"initial","",0,candidate,problem)
    forged=_fake_real_record(problem,candidate,ticket)
    wrong=replace(forged,diagnostics={
        **forged.diagnostics,
        "confirm12_real_call":{"call_id":"other",
                               "evaluator":"AircraftCase.evaluate"}})
    with pytest.raises(ValueError,match="receipt"):
        ledger.settle(db,ticket,wrong)
    ledger.stop(db)
    result=ledger.audit(db,plan)
    assert result["calls"]==1
    assert not result["ledger_contract_passed"]


def test_full_run_rejects_implicit_worker_reduction(tmp_path):
    with pytest.raises(ValueError,match="exactly 12"):
        campaign.run(tmp_path/"never_created",workers=8)
    assert not (tmp_path/"never_created").exists()
