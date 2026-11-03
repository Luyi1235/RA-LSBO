import json
import pytest
from confirm12 import call_ledger, ledger_simulation


def test_twelve_worker_ledger_simulation(tmp_path):
    out = tmp_path / 'sim'
    report = ledger_simulation.run(out)
    assert report['integrity_passed'] and report['campaign_complete']
    assert report['charged_calls'] == 290 and report['states'] == {'settled': 290}
    assert report['scope'] == 'SIMULATED_ONLY'
    for case in ledger_simulation.CASES:
        for arm in ledger_simulation.ARMS:
            assert (out / case / '1' / arm / 'result.json').is_file()
    target = out / ledger_simulation.CASES[0] / '1' / ledger_simulation.ARMS[0] / 'result.json'
    target.write_text('{}', encoding='utf-8')
    assert not ledger_simulation.audit(out)['campaign_complete']


def test_timeout_preserves_incomplete_simulation(tmp_path):
    out = tmp_path / 'timed'
    with pytest.raises(TimeoutError):
        ledger_simulation.run(out, max_wall_seconds=0.000001)
    plan = json.loads((out / 'plan.json').read_text())
    report = call_ledger.audit(out / 'calls.sqlite', plan)
    assert report['status'] == 'INCOMPLETE' and not report['campaign_complete']
    assert (out / 'error.txt').is_file()
