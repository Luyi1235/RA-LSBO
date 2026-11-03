import concurrent.futures
import multiprocessing
import os
import sqlite3
import pytest
from confirm12 import call_ledger as ledger


def plan(limit=10):
    return {'cases': ['A'], 'seeds': [101], 'arms': ['one', 'two'], 'initial': 2,
            'online': 1, 'verification': 1, 'max_calls': limit}


def race(job):
    path, index = job
    try:
        return ledger.start(path, 'A', 101, 'initial', '', index % 2, {'value': index})
    except (RuntimeError, sqlite3.IntegrityError):
        return None


def crash_after_started(path):
    ledger.start(path, 'A', 101, 'initial', '', 0, {'crash': True})
    os._exit(29)


def test_shared_initial_identity_and_budget_race(tmp_path):
    path = tmp_path / 'ledger.db'
    p = plan(1)
    ledger.create(path, p)
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(race, [(path, i) for i in range(24)]))
    assert sum(x is not None for x in results) == 1
    audit = ledger.audit(path, p)
    assert audit['charged_calls'] == 1 and audit['states'] == {'started': 1}
    assert not audit['integrity_passed'] and not audit['campaign_complete']


def test_shared_history_gates_and_no_duplicate_charges(tmp_path):
    path = tmp_path / 'ledger.db'
    p = plan()
    ledger.create(path, p)
    with pytest.raises(RuntimeError):
        ledger.start(path, 'A', 101, 'online', 'one', 0, {'x': 1})
    for i in range(2):
        ident = ledger.start(path, 'A', 101, 'initial', '', i, {'x': i})
        ledger.settle(path, ident, 'settled', {'y': i})
    with pytest.raises(sqlite3.IntegrityError):
        ledger.start(path, 'A', 101, 'initial', '', 0, {'x': 999})
    for arm in p['arms']:
        ident = ledger.start(path, 'A', 101, 'online', arm, 0, {'arm': arm})
        ledger.settle(path, ident, 'settled', {'arm': arm})
    audit = ledger.audit(path, p)
    assert audit['integrity_passed'] and audit['charged_calls'] == 4
    assert not audit['campaign_complete']


@pytest.mark.parametrize('state', ['unknown', 'failed'])
def test_terminal_failure_stops_dispatch_preserves_cost(tmp_path, state):
    path = tmp_path / 'ledger.db'
    p = plan()
    ledger.create(path, p)
    ident = ledger.start(path, 'A', 101, 'initial', '', 0, {'x': 0})
    ledger.settle(path, ident, state, {'detail': 'injected'})
    with pytest.raises(RuntimeError):
        ledger.start(path, 'A', 101, 'initial', '', 1, {'x': 1})
    report = ledger.audit(path, p)
    assert not report['integrity_passed'] and report['status'] == 'INCOMPLETE'
    assert report['states'][state] == 1 and report['charged_calls'] == 1
    with pytest.raises(RuntimeError):
        ledger.settle(path, ident, 'settled', {'x': 0})


def test_ledger_and_plan_tampering_detected(tmp_path):
    path = tmp_path / 'ledger.db'
    p = plan()
    ledger.create(path, p)
    ident = ledger.start(path, 'A', 101, 'initial', '', 0, {'x': 0})
    ledger.settle(path, ident, 'settled', {'v': 1})
    changed = dict(p, max_calls=999)
    assert not ledger.audit(path, changed)['integrity_passed']
    with sqlite3.connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE events SET payload_hash='bad'")
        db.execute("UPDATE calls SET outcome='{}'")
    assert not ledger.audit(path, p)['integrity_passed']


def test_process_crash_preserves_unsettled_and_never_retries(tmp_path):
    path = tmp_path / 'crashed.db'
    p = plan()
    ledger.create(path, p)
    process = multiprocessing.get_context('spawn').Process(target=crash_after_started, args=(path,))
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 29
    report = ledger.audit(path, p)
    assert report['states'] == {'started': 1} and report['charged_calls'] == 1
    assert not report['integrity_passed'] and not report['campaign_complete']
    with pytest.raises(sqlite3.IntegrityError):
        ledger.start(path, 'A', 101, 'initial', '', 0, {'crash': True})
