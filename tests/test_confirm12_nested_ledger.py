import sqlite3
import concurrent.futures
import multiprocessing
import pytest
from confirm12 import call_ledger, nested_ledger, nested_seed_plan


def start_initial(job):
    path, case, seed, index = job
    ident = nested_ledger.start(path, case, seed, 'initial', '', index, {'x': index})
    nested_ledger.settle(path, ident, 'settled', {'scope': 'mock', 'feasible': False})
    return ident


@pytest.fixture
def prepared(tmp_path):
    proposed = nested_seed_plan.candidate_plan()
    contracts = {case: {'20s': '2' * 64, '5s': '5' * 64} for case in proposed['seeds_by_case']}
    p = nested_ledger.compile_plan(proposed, source_sha='a'*40, environment_sha='b'*64, contracts=contracts)
    path = tmp_path / 'nested.sqlite'
    nested_ledger.create(path, p)
    return path, p, proposed


def test_full_budget_subset_and_no_qualification(prepared):
    path, p, proposed = prepared
    case = 'aircraft_ordinary'
    seed = next(s for s in proposed['seeds_by_case'][case] if s not in proposed['auxiliary_by_case'][case])
    assert p['max_calls'] == 151040 and len(p['allowed_arms'][case][str(seed)]) == 2
    with pytest.raises(ValueError, match='not registered for this task'):
        nested_ledger.start(path, case, proposed['seeds_by_case']['aircraft_sparse'][0], 'initial', '', 0, {})
    with pytest.raises(ValueError, match='not registered for case and seed'):
        nested_ledger.start(path, case, seed, 'online', 'random', 0, {})
    report = nested_ledger.audit(path, p)
    assert report['integrity_passed'] and not report['campaign_complete'] and not report['production_qualified']


def test_shared_initial_once_and_5s_same_best_design(prepared):
    path, p, proposed = prepared
    case, seed, arm = 'aircraft_ordinary', proposed['seeds_by_case']['aircraft_ordinary'][0], 'ra_lsbo'
    with pytest.raises(RuntimeError, match='shared initial'):
        nested_ledger.start(path, case, seed, 'online', arm, 0, {'x': 10})
    for i in range(12):
        ident = nested_ledger.start(path, case, seed, 'initial', '', i, {'x': i})
        nested_ledger.settle(path, ident, 'settled', {'scope':'mock', 'feasible':False})
    with pytest.raises(sqlite3.IntegrityError):
        nested_ledger.start(path, case, seed, 'initial', '', 0, {'x': 0})
    for i in range(18):
        ident = nested_ledger.start(path, case, seed, 'online', arm, i, {'x': i + 100})
        nested_ledger.settle(path, ident, 'settled', {'scope':'mock', 'feasible': i == 7,
                                                     'objective': 12.0 if i == 7 else None})
    with pytest.raises(ValueError, match='nominal incumbent'):
        nested_ledger.start(path, case, seed, 'verification', arm, 0, {'x': 108})
    ident = nested_ledger.start(path, case, seed, 'verification', arm, 0, {'x': 107})
    nested_ledger.settle(path, ident, 'settled', {'scope':'mock', 'feasible':True, 'objective':12.2})
    report = nested_ledger.audit(path, p)
    assert report['integrity_passed'] and report['charged_calls'] == 31
    assert report['scope'] == 'SIMULATED_ONLY' and not report['production_qualified']


def test_unknown_remains_charged_and_blocks(prepared):
    path, p, proposed = prepared
    seed = proposed['seeds_by_case']['aircraft_ordinary'][0]
    ident = nested_ledger.start(path, 'aircraft_ordinary', seed, 'initial', '', 0, {'x': 0})
    nested_ledger.settle(path, ident, 'unknown', {'scope':'mock', 'error':'infrastructure'})
    with pytest.raises(RuntimeError):
        nested_ledger.start(path, 'aircraft_ordinary', seed, 'initial', '', 1, {'x': 1})
    report = nested_ledger.audit(path, p)
    assert not report['integrity_passed'] and report['charged_calls'] == 1


def test_real_evidence_and_approval_cannot_be_faked(prepared):
    path, p, proposed = prepared
    seed = proposed['seeds_by_case']['aircraft_ordinary'][0]
    ident = nested_ledger.start(path, 'aircraft_ordinary', seed, 'initial', '', 0, {})
    with pytest.raises(ValueError, match='only explicit mock'):
        nested_ledger.settle(path, ident, 'settled', {'scope':'aircraft'})
    assert nested_ledger.audit(path, p)['states'] == {'started': 1}


def test_twelve_spawn_workers_unique_shared_tickets(prepared):
    path, p, proposed = prepared
    seed = proposed['seeds_by_case']['aircraft_ordinary'][0]
    jobs = [(path, 'aircraft_ordinary', seed, i) for i in range(12)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        identities = list(pool.map(start_initial, jobs))
    assert len(set(identities)) == 12
    report = nested_ledger.audit(path, p)
    assert report['charged_calls'] == 12 and report['states'] == {'settled': 12}
    assert report['integrity_passed'] and not report['campaign_complete']


def test_candidate_subset_and_authorization_are_frozen():
    proposed = nested_seed_plan.candidate_plan()
    contracts = {case: {'20s': '2' * 64, '5s': '5' * 64} for case in proposed['seeds_by_case']}
    case = 'aircraft_ordinary'
    proposed['auxiliary_by_case'][case].reverse()
    with pytest.raises(ValueError):
        nested_ledger.compile_plan(proposed, source_sha='a'*40, environment_sha='b'*64, contracts=contracts)
    proposed = nested_seed_plan.candidate_plan()
    proposed['execution_authorized'] = True
    with pytest.raises(ValueError):
        nested_ledger.compile_plan(proposed, source_sha='a'*40, environment_sha='b'*64, contracts=contracts)


def test_independent_audit_rejects_bypassed_non_mock_settlement(prepared):
    path, p, proposed = prepared
    seed = proposed['seeds_by_case']['aircraft_ordinary'][0]
    ident = nested_ledger.start(path, 'aircraft_ordinary', seed, 'initial', '', 0, {})
    call_ledger.settle(path, ident, 'settled', {'scope': 'aircraft'})
    assert not nested_ledger.audit(path, p)['integrity_passed']
