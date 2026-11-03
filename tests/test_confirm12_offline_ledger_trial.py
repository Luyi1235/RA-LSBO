"""One full real historical random optimizer decision via mock records and SQLite."""
import pytest

from confirm12 import structured_record_ledger as ledger
from confirm12.offline_ledger_trial import OfflineLedgerTrial
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.partial_bo.problems import candidate_pool


def make_plan(ordinary, sparse, seed):
    return {
        "schema":ledger.SCHEMA,"scope":ledger.SCOPE,
        "execution_authorized":False,"phase":"confirmation",
        "initial":12,"online":18,"verification":1,"max_calls":32,
        "cases":{
            "aircraft_ordinary":{"seeds":[seed],
                "contracts":{"20s":ordinary.contract_id,"5s":problem_for('aircraft_ordinary',5.).contract_id},
                "arms_by_seed":{str(seed):["random","onehot_stable"]}},
            "aircraft_sparse":{"seeds":[],
                "contracts":{"20s":sparse.contract_id,"5s":problem_for('aircraft_sparse',5.).contract_id},
                "arms_by_seed":{}},
        },
    }


def fake_record(problem, candidate):
    # No physical evaluator, GP fit or VAE training is involved in this fixture.
    return problem.record(candidate,1.0,{"soc":-.1},
                          {"offline_fixture_no_aircraft_evaluation":True})


def test_real_historical_random_from_shared_ledger(monkeypatch,tmp_path):
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase,"evaluate",
                        lambda *a,**k: pytest.fail("physical evaluation attempted"))
    ordinary=problem_for("aircraft_ordinary")
    sparse=problem_for("aircraft_sparse")
    seed=88
    plan=make_plan(ordinary,sparse,seed)
    db=tmp_path/"test.sqlite"
    ledger.create(db,plan)
    init,_=candidate_pool(ordinary,31001,12)
    for i,c in enumerate(init):
        ticket=ledger.start(db,"aircraft_ordinary",seed,"initial","",i,c,ordinary)
        ledger.settle(db,ticket,fake_record(ordinary,c))
    first=OfflineLedgerTrial(db,plan,"aircraft_ordinary",seed,"random",tmp_path/"method1")
    second=OfflineLedgerTrial(db,plan,"aircraft_ordinary",seed,"onehot_stable",tmp_path/"method2")
    assert first.policy.history_digest == second.policy.history_digest
    before=second.policy.history_digest
    pool,_=candidate_pool(ordinary,31002,32,
                          {r.observation.design_id for r in first.policy._history})
    candidate,meta=first.propose(pool)
    assert meta["method"]=="random"
    assert first.ticket is not None
    with pytest.raises(RuntimeError,match="Unsettled"):
        first.propose(pool)
    first.accept(fake_record(ordinary,candidate))
    assert first.ticket is None and second.policy.history_digest == before
    report=ledger.audit(db,plan)
    assert report["ledger_contract_passed"],report
    assert report["states"]=={"settled":13}


def test_bad_feedback_preserves_started_ticket(monkeypatch,tmp_path):
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase,"evaluate",
                        lambda *a,**k: pytest.fail("physical evaluation attempted"))
    ordinary,sparse=problem_for("aircraft_ordinary"),problem_for("aircraft_sparse")
    seed=100
    plan=make_plan(ordinary,sparse,seed)
    db=tmp_path/"test.sqlite"
    ledger.create(db,plan)
    init,_=candidate_pool(ordinary,9991,12)
    for i,c in enumerate(init):
        tid=ledger.start(db,"aircraft_ordinary",seed,"initial","",i,c,ordinary)
        ledger.settle(db,tid,fake_record(ordinary,c))
    first=OfflineLedgerTrial(db,plan,"aircraft_ordinary",seed,"random",tmp_path/"method")
    pool,_=candidate_pool(ordinary,9992,32,
                          {r.observation.design_id for r in first.policy._history})
    selected,_=first.propose(pool)
    foreign=next(c for c in pool if c!=selected)
    with pytest.raises(ValueError,match="does not match"):
        first.accept(fake_record(ordinary,foreign))
    assert first.failed
    report=ledger.audit(db,plan)
    assert report["states"]["started"]==1 and not report["ledger_contract_passed"]
    assert report["ledger_state"]=="INCOMPLETE"
