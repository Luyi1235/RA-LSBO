import pytest
from confirm12.adapters import QUALIFICATION, Trial, create


@pytest.mark.parametrize('arm', list(QUALIFICATION))
def test_each_simulated_arm_isolated_budget_and_blocked_real(arm):
    trial = Trial('ordinary', 77, arm)
    with pytest.raises(RuntimeError):
        create(arm, trial, scope='real')
    adapter = create(arm, trial, scope='simulation')
    initial = [{'case': 'ordinary', 'seed': 77, 'role': 'initial', 'arm': ''} for _ in range(12)]
    candidate, metadata = adapter.propose(initial, [{'x': 0}, {'x': 1}])
    assert candidate in ({'x': 0}, {'x': 1}) and metadata['simulated_only']
    foreign = initial + [{'case': 'ordinary', 'seed': 77, 'role': 'online', 'arm': 'foreign'}]
    with pytest.raises(ValueError):
        adapter.propose(foreign, [{'x': 0}])
    full = initial + [{'case': 'ordinary', 'seed': 77, 'role': 'online', 'arm': arm} for _ in range(18)]
    with pytest.raises(RuntimeError):
        adapter.propose(full, [{'x': 0}])
