import json
import concurrent.futures
import multiprocessing
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from confirm12 import ledger, simulation
from run_ra_lsbo_confirm_12workers import MAIN_MAX, AUX_MAX


def race_for_budget(job):
    db, index = job
    try:
        ledger.start(db, 'race%d' % index, 'initial', {'index': index})
        return True
    except RuntimeError:
        return False


def test_budget_arithmetic():
    assert (MAIN_MAX, AUX_MAX, MAIN_MAX + AUX_MAX) == (102400, 48640, 151040)


def test_atomic_budget_and_unsettled(tmp_path):
    db = tmp_path / 'ledger.sqlite'
    plan = {'kind': 'SYNTHETIC_ONLY'}
    ledger.create(db, 1, plan)
    ledger.start(db, 'call1', 'initial', {'x': 1})
    with pytest.raises(RuntimeError):
        ledger.start(db, 'call2', 'initial', {'x': 2})
    assert not ledger.audit(db, plan, 1)['passed']
    ledger.settle(db, 'call1', {'status': 'ok'})
    assert ledger.audit(db, plan, 1)['passed']
    with pytest.raises(RuntimeError):
        ledger.settle(db, 'call1', {'status': 'ok'})


def test_tamper_detected(tmp_path):
    db = tmp_path / 'ledger.sqlite'
    ledger.create(db, 1, {'plan': 1})
    ledger.start(db, 'id', 'initial', {'x': 1})
    ledger.settle(db, 'id', {'x': 2})
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE calls SET outcome='{}'")
    assert not ledger.audit(db, {'plan': 1}, 1)['passed']
    assert not ledger.audit(db, {'plan': 2}, 1)['passed']


def test_twelve_process_budget_race(tmp_path):
    db = tmp_path / 'race.sqlite'
    ledger.create(db, 1, {'plan': 'race'})
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        wins = list(pool.map(race_for_budget, [(str(db), n) for n in range(24)]))
    assert sum(wins) == 1
    report = ledger.audit(db, {'plan': 'race'}, 1)
    assert report['started'] == 1 and report['settled'] == 0 and not report['passed']


def test_injected_unknown_stops_without_settlement(tmp_path):
    db = tmp_path / 'stopped.sqlite'
    ledger.create(db, 3, {'plan': 'failure'})
    with pytest.raises(RuntimeError):
        simulation._evaluate(str(db), 'bad', 'online', {'x': 1}, fail=True)
    with pytest.raises(RuntimeError):
        ledger.start(db, 'next', 'online', {'x': 2})
    report = ledger.audit(db, {'plan': 'failure'}, 1)
    assert report['status'] == 'INCOMPLETE' and report['started'] == 1 and report['settled'] == 0


def test_twelve_process_end_to_end(tmp_path):
    output = tmp_path / 'run'
    result = simulation.run(output, seeds=(7,), workers=12)
    assert result['passed'] and result['settled'] == 290
    assert result['aircraft_calls'] == result['rcaide_calls'] == 0
    with pytest.raises(RuntimeError):
        ledger.start(output / 'ledger.sqlite', 'after_complete', 'online', {'x': 1})
    for case in simulation.CASES:
        initial = json.loads((output / case / '7/shared/history.json').read_text())
        for arm in simulation.ARMS:
            arm_data = json.loads((output / case / '7' / arm / 'result.json').read_text())
            assert arm_data['initial_hash'] == ledger.digest(initial)
            assert len(arm_data['history']) == 30
    with pytest.raises(FileExistsError):
        simulation.run(output, seeds=(7,), workers=12)


def test_default_is_zero_call_preflight_and_real_execution_requires_explicit_mode():
    root = Path(__file__).resolve().parents[1]
    script = root / 'run_ra_lsbo_confirm_12workers.py'
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert run.returncode == 0 and '"total_maximum": 151040' in run.stdout
    assert '"new_aircraft_calls": 0' in run.stdout
    # NEVER invoke --mode execute in synthetic-only CI: this is now the genuine
    # user-initiated 151040-call model route, not a zero-truth exception test.
    source = script.read_text(encoding='utf8')
    assert "default='preflight'" in source
    assert "if args.mode == 'execute':" in source
    assert 'from confirm12.real_aircraft_campaign import run as true_campaign' in source
