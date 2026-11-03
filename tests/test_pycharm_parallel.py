from pathlib import Path
import subprocess
import sys

import pytest
from offline_parallel import schedule


def tasks(checkpoint, fail=False):
    return [{'id': 'job%d' % i, 'seed': i, 'checkpoint': str(checkpoint),
             'rounds': 10, 'fail': fail and i == 0} for i in range(4)]


def test_spawn_deterministic_and_isolated(tmp_path):
    checkpoint = tmp_path / 'state.npz'
    checkpoint.write_bytes(b'historical checkpoint fixture')
    one = schedule(tasks(checkpoint), tmp_path / 'one', 1, 30)
    two = schedule(tasks(checkpoint), tmp_path / 'two', 2, 30)
    assert one['status'] == two['status'] == 'complete'
    assert [r['digest'] for r in one['completed']] == [r['digest'] for r in two['completed']]
    assert all((tmp_path / 'two' / ('job%d' % i) / 'result.json').is_file() for i in range(4))
    assert one['truth_calls'] == two['truth_calls'] == 0


def test_failure_preserves_evidence_and_never_retries(tmp_path):
    checkpoint = tmp_path / 'state.npz'
    checkpoint.write_bytes(b'state')
    result = schedule(tasks(checkpoint, fail=True), tmp_path / 'out', 1, 30)
    assert result['status'] == 'incomplete'
    assert result['dispatched'] == 1 and result['undispatched'] == 3
    assert (tmp_path / 'out/job0/error.txt').exists()
    with pytest.raises(FileExistsError):
        schedule(tasks(checkpoint), tmp_path / 'out', 1, 30)


def test_duplicate_refused_before_output(tmp_path):
    task = {'id': 'same', 'seed': 1, 'checkpoint': 'unused', 'rounds': 1}
    with pytest.raises(ValueError):
        schedule([task, task], tmp_path / 'out', 1, 30)
    assert not (tmp_path / 'out').exists()


def test_deadline_leaves_unstarted_tasks(tmp_path):
    checkpoint = tmp_path / 'state.npz'
    checkpoint.write_bytes(b'state')
    result = schedule(tasks(checkpoint), tmp_path / 'out', 1, 0.000001)
    assert result['status'] == 'incomplete'
    assert result['dispatched'] < 4


def test_default_and_execution_are_fail_closed():
    root = Path(__file__).resolve().parents[1]
    entry = root / 'run_ra_lsbo_pycharm.py'
    ok = subprocess.run([sys.executable, str(entry)], cwd=root, capture_output=True, text=True)
    assert ok.returncode == 0 and '"truth_calls": 0' in ok.stdout
    refused = subprocess.run([sys.executable, str(entry), '--mode', 'execute'], cwd=root, capture_output=True, text=True)
    assert refused.returncode != 0 and 'execution disabled' in refused.stderr
