"""One full real historical random optimizer decision via mock records and SQLite."""
import json
import os
import sqlite3
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


@pytest.mark.parametrize('also_change_row_hash', [False, True])
def test_tampered_shared_observation_is_rejected_before_fitting(monkeypatch, tmp_path, also_change_row_hash):
    from confirm12.offline_ledger_trial import load_shared_initial
    from ra_lsbo.budget import digest
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase, 'evaluate', lambda *a, **k: pytest.fail('truth attempted'))
    ordinary, sparse = problem_for('aircraft_ordinary'), problem_for('aircraft_sparse')
    p = make_plan(ordinary, sparse, 123)
    db = tmp_path / 'test.sqlite'
    ledger.create(db, p)
    initial, _ = candidate_pool(ordinary, 1003, 12)
    for i, c in enumerate(initial):
        tid = ledger.start(db, 'aircraft_ordinary', 123, 'initial', '', i, c, ordinary)
        ledger.settle(db, tid, fake_record(ordinary, c))
    with sqlite3.connect(db) as connection:
        raw = connection.execute("SELECT record FROM calls WHERE role='initial' AND idx=0").fetchone()[0]
        data = json.loads(raw)
        data['observation']['objective']['value'] = 9999.0
        connection.execute("UPDATE calls SET record=? WHERE role='initial' AND idx=0", (json.dumps(data),))
        if also_change_row_hash:
            connection.execute("UPDATE calls SET record_hash=? WHERE role='initial' AND idx=0", (digest(data),))
    with pytest.raises(RuntimeError, match='digest'):
        load_shared_initial(db, p, 'aircraft_ordinary', 123)


@pytest.mark.skipif(os.environ.get('CONFIRM12_TEST_GP_ONEHOT') != '1',
                    reason='optional real historical GP computation, no truth calls')
def test_real_onehot_gp_propose_accept_through_structured_ledger(monkeypatch, tmp_path):
    from ra_lsbo.experiments.core.problems import AircraftCase
    from ra_lsbo.structure_corpus import CorpusRepresentation
    def forbidden(*args, **kwargs):
        pytest.fail('aircraft evaluation or VAE training attempted')
    monkeypatch.setattr(AircraftCase, 'evaluate', forbidden)
    monkeypatch.setattr(CorpusRepresentation, 'fit', forbidden)
    ordinary, sparse = problem_for('aircraft_ordinary'), problem_for('aircraft_sparse')
    p = make_plan(ordinary, sparse, 124)
    db = tmp_path / 'ledger.sqlite'
    ledger.create(db, p)
    pool, _ = candidate_pool(ordinary, 1004, 44)
    def record(c):
        constraints = dict(ordinary.precheck(c))
        constraints.update({name: -.1 for name in ordinary.required_constraints
                            if name not in ordinary.analytical_constraints})
        return ordinary.record(c, 32000.0 + 100 * (c['continuous']['wing_area'] - 12.) ** 2,
                               constraints, {'offline_fixture_no_aircraft_evaluation': True})
    for i, c in enumerate(pool[:12]):
        tid = ledger.start(db, 'aircraft_ordinary', 124, 'initial', '', i, c, ordinary)
        ledger.settle(db, tid, record(c))
    trial = OfflineLedgerTrial(db, p, 'aircraft_ordinary', 124, 'onehot_stable', tmp_path / 'gp')
    selected, meta = trial.propose(pool[12:])
    assert meta['method'] == 'onehot_stable'
    assert meta.get('posterior_fit_skipped') is False
    assert not meta.get('posterior_fit_failed', False)
    assert meta.get('local_enabled') is True and meta.get('local_reports')
    trial.accept(record(selected))
    result = ledger.audit(db, p)
    assert result['ledger_contract_passed'] and result['states'] == {'settled': 13}
    assert not result['production_approved'] and not result['real_physics_provenance_verified']
