from pathlib import Path
import pytest
from confirm12 import qualification


def test_registered_candidates_reproduce_without_truth():
    root = Path(__file__).resolve().parents[1]
    report = qualification.inspect(root / 'results/iid-qualification-preflight-20261103/plan.json')
    assert [report['tasks'][x]['samples'] for x in ('aircraft_ordinary', 'aircraft_sparse')] == [300, 300]
    assert report['new_aircraft_calls'] == report['rcaide_calls'] == 0
    assert not report['qualification_ready'] and 'incomplete' in report['seed_history_status']


def test_modified_plan_refused(tmp_path):
    path = tmp_path / 'plan.json'
    path.write_text('{}')
    with pytest.raises(ValueError):
        qualification.inspect(path)
