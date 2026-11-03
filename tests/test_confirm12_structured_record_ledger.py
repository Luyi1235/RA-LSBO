"""No aircraft calls: tests use authentic DesignRecord objects fabricated in memory."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import sqlite3
import pytest

from confirm12 import structured_record_ledger as l
from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.risk_core.observations import RunObservation, OutputObservation


class FakeProblem:
    def __init__(self, contract="contract20_00000000"):
        self.contract_id = contract
    def canonical(self, candidate):
        return candidate
    def precheck(self, candidate):
        return {"analytic": 0.}


def plan(*, seeds=(42,), maximum=31):
    return {
        "schema":l.SCHEMA, "scope":l.SCOPE, "execution_authorized":False,
        "phase":"confirmation", "initial":12,"online":18,"verification":1,
        "max_calls":maximum,
        "cases":{
            "aircraft_ordinary":{
                "seeds":list(seeds),
                "contracts":{"20s":"contract20_00000000","5s":"contract05_00000000"},
                "arms_by_seed":{str(s):["real_random","legacy_onehot"] for s in seeds},
            },
            "aircraft_sparse":{
                "seeds":[],
                "contracts":{"20s":"sparse20_00000000","5s":"sparse05_00000000"},
                "arms_by_seed":{},
            },
        },
    }


def candidate(i):
    return {"continuous":{"x":float(i)},"discrete":{"layout":"test"}}


def record(i, contract="contract20_00000000", *, status="completed"):
    c = candidate(i)
    evid = "offline_fixture_" + str(i)
    objective = (OutputObservation("exact", evid+":f", value=float(i)) if status=="completed"
                 else OutputObservation("missing", evid+":f", reason="unknown_failure"))
    constraints = (("soc",OutputObservation("exact",evid+":soc", value=-.1)),)
    run = RunObservation(digest(c), contract, evid, status, objective, constraints)
    return DesignRecord(c, run, {"offline_fixture_no_physics":True})


def process_ticket(job):
    path, seed = job
    cid = l.start(path, 'aircraft_ordinary', seed, 'initial', '', 0, candidate(seed), FakeProblem())
    l.settle(path, cid, record(seed))
    return cid


def test_12_initials_and_first_online_use_real_record_contract(tmp_path):
    path = tmp_path/"ledger.sqlite"
    p=plan()
    l.create(path,p)
    prob=FakeProblem()
    with pytest.raises(RuntimeError, match="shared initial"):
        l.start(path,"aircraft_ordinary",42,"online","real_random",0,candidate(99),prob)
    for i in range(12):
        cid=l.start(path,"aircraft_ordinary",42,"initial","",i,candidate(i),prob)
        with pytest.raises(sqlite3.IntegrityError):
            l.start(path,"aircraft_ordinary",42,"initial","",i,candidate(i),prob)
        assert l.settle(path,cid,record(i))=="settled"
    cid=l.start(path,"aircraft_ordinary",42,"online","real_random",0,candidate(99),prob)
    with pytest.raises(RuntimeError, match="Sequential online"):
        l.start(path,"aircraft_ordinary",42,"online","real_random",1,candidate(100),prob)
    with pytest.raises(ValueError, match="does not match"):
        l.settle(path,cid,record(101))
    assert l.settle(path,cid,record(99))=="settled"
    report=l.audit(path,p)
    assert report["ledger_contract_passed"] and report["calls"]==13
    assert report["states"]=={"settled":13}
    assert not report["production_approved"]
    assert not report["real_physics_provenance_verified"]


def test_untrusted_feedback_is_charged_and_stops(tmp_path):
    path=tmp_path/"ledger.sqlite"
    p=plan(maximum=25)
    l.create(path,p)
    cid=l.start(path,"aircraft_ordinary",42,"initial","",0,candidate(0),FakeProblem())
    assert l.settle(path,cid,record(0,status="unknown"))=="unknown"
    with pytest.raises(RuntimeError,match="stopped"):
        l.start(path,"aircraft_ordinary",42,"initial","",1,candidate(1),FakeProblem())
    output=l.audit(path,p)
    assert output["states"]=={"unknown":1}
    assert not output["ledger_contract_passed"]
    assert output["ledger_state"]=="INCOMPLETE"


def test_wrong_contract_candidate_duplicate_and_tamper(tmp_path):
    path=tmp_path/"ledger.sqlite"
    p=plan(maximum=2)
    l.create(path,p)
    with pytest.raises(ValueError,match="contract"):
        l.start(path,"aircraft_ordinary",42,"initial","",0,candidate(0),
                FakeProblem("wrong"))
    c0=l.start(path,"aircraft_ordinary",42,"initial","",0,candidate(0),FakeProblem())
    with pytest.raises(ValueError,match="physics contract"):
        l.settle(path,c0,record(0,contract="wrong"))
    l.settle(path,c0,record(0))
    c1=l.start(path,"aircraft_ordinary",42,"initial","",1,candidate(1),FakeProblem())
    l.settle(path,c1,record(1))
    with pytest.raises(RuntimeError,match="cap"):
        l.start(path,"aircraft_ordinary",42,"initial","",2,candidate(2),FakeProblem())
    with sqlite3.connect(path) as db:
        db.execute("UPDATE calls SET candidate_hash='tampered' WHERE idx=0")
    assert not l.audit(path,p)["ledger_contract_passed"]


def test_12_thread_concurrent_independent_initial_tickets(tmp_path):
    path=tmp_path/"ledger.sqlite"
    p=plan(seeds=tuple(range(101,113)),maximum=12)
    l.create(path,p)
    def work(seed):
        # Each seed's own first initial candidate is charged exactly once.
        c=candidate(seed)
        cid=l.start(path,"aircraft_ordinary",seed,"initial","",0,c,FakeProblem())
        l.settle(path,cid,record(seed))
        return cid
    with ThreadPoolExecutor(max_workers=12) as executor:
        ids=list(executor.map(work,range(101,113)))
    assert len(set(ids))==12
    report=l.audit(path,p)
    assert report["ledger_contract_passed"]
    assert report["calls"]==12


def test_12_spawn_processes_charge_independent_structured_records_once(tmp_path):
    path = tmp_path / 'processes.sqlite'
    p = plan(seeds=tuple(range(201, 213)), maximum=12)
    l.create(path, p)
    with ProcessPoolExecutor(max_workers=12, mp_context=multiprocessing.get_context('spawn')) as pool:
        ids = list(pool.map(process_ticket, [(path, i) for i in range(201, 213)]))
    assert len(ids) == len(set(ids)) == 12
    result = l.audit(path, p)
    assert result['ledger_contract_passed'] and result['states'] == {'settled': 12}
    assert not result['production_approved'] and not result['real_physics_provenance_verified']


def test_validation_only_scope_rejects_production_flag(tmp_path):
    p=plan()
    p["execution_authorized"]=True
    with pytest.raises(ValueError,match="No real-campaign authorization"):
        l.create(tmp_path/"never_created.sqlite",p)


@pytest.mark.parametrize('verification_index,expected', [(0, True), (99, False)])
def test_independent_reconciliation_checks_nominal_incumbent(tmp_path, verification_index, expected):
    from confirm12 import structured_reconciliation
    path = tmp_path / 'ledger.sqlite'
    p = plan(maximum=31)
    l.create(path, p)
    for i in range(12):
        cid = l.start(path, 'aircraft_ordinary', 42, 'initial', '', i, candidate(i), FakeProblem())
        l.settle(path, cid, record(i))
    for i in range(18):
        cid = l.start(path, 'aircraft_ordinary', 42, 'online', 'real_random', i, candidate(100+i), FakeProblem())
        l.settle(path, cid, record(100+i))
    cid = l.start(path, 'aircraft_ordinary', 42, 'verification', 'real_random', 0,
                  candidate(verification_index), FakeProblem('contract05_00000000'))
    l.settle(path, cid, record(verification_index, contract='contract05_00000000'))
    assert l.audit(path, p)['ledger_contract_passed']
    result = structured_reconciliation.audit(path, p)
    assert result['structural_reconciliation_passed'] is expected
    assert not result['production_approved'] and not result['real_physics_provenance_verified']
    if not expected:
        assert 'verification differs from nominal best feasible design' in result['errors']
