import copy
import pytest
from confirm12 import design


def fixture_plan():
    seeds = {'aircraft_ordinary': list(range(10000, 11024)),
             'aircraft_sparse': list(range(20000, 21024))}
    return {'seeds_by_case': seeds, 'auxiliary_by_case': {k: v[:256] for k, v in seeds.items()},
            'main_arms': list(design.MAIN), 'auxiliary_arms': list(design.AUXILIARY),
            'budget': {'initial': 12, 'online': 18, 'verification_max': 1,
                       'main_max_calls': 102400, 'auxiliary_max_calls': 48640,
                       'max_calls': 151040, 'rcaide_max_calls': 0}}


def test_design_shape_is_not_seed_clearance_or_approval():
    report = design.validate(fixture_plan())
    assert report['structurally_valid'] and not report['execution_ready']
    assert not report['seed_history_independence_verified']
    assert set(report['qualification_status'].values()) == {'BLOCKED'}


@pytest.mark.parametrize('change', ['subset', 'duplicate', 'cross_task', 'budget', 'arm'])
def test_reject_bad_nested_design(change):
    plan = fixture_plan()
    if change == 'subset':
        plan['auxiliary_by_case']['aircraft_ordinary'][0] = -1
    elif change == 'duplicate':
        plan['seeds_by_case']['aircraft_ordinary'][0] = plan['seeds_by_case']['aircraft_ordinary'][1]
    elif change == 'cross_task':
        plan['seeds_by_case']['aircraft_sparse'][300] = plan['seeds_by_case']['aircraft_ordinary'][300]
    elif change == 'budget':
        plan['budget']['max_calls'] = 151041
    else:
        plan['main_arms'].reverse()
    with pytest.raises(ValueError):
        design.validate(plan)
