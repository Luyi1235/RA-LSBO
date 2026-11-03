"""Validate proposed nested confirmation design; never assign seeds or approve runs."""
from . import adapters

CASES = ('aircraft_ordinary', 'aircraft_sparse')
MAIN = ('ra_lsbo', 'full_latent')
AUXILIARY = ('full_onehot', 'reverse_representation', 'interval_as_missing', 'random', 'cbob_adapted')


def validate(plan):
    if set(plan.get('seeds_by_case', {})) != set(CASES) or set(plan.get('auxiliary_by_case', {})) != set(CASES):
        raise ValueError('both task seed lists and subsets must be frozen')
    combined = []
    for case in CASES:
        seeds = plan['seeds_by_case'][case]
        subset = plan['auxiliary_by_case'][case]
        if (len(seeds) != 1024 or len(set(seeds)) != 1024 or
                any(type(x) is not int or x < 0 for x in seeds)):
            raise ValueError('exactly 1024 distinct nonnegative seeds per task required')
        if (len(subset) != 256 or len(set(subset)) != 256 or
                any(type(x) is not int for x in subset) or not set(subset).issubset(seeds)):
            raise ValueError('exactly 256 distinct preselected seeds required')
        combined.extend(seeds)
    if len(set(combined)) != len(combined):
        raise ValueError('task seed lists overlap; independence is unverified')
    if plan.get('main_arms') != list(MAIN) or plan.get('auxiliary_arms') != list(AUXILIARY):
        raise ValueError('method list or ordering differs from requested design')
    if plan.get('budget') != {'initial': 12, 'online': 18, 'verification_max': 1,
                               'main_max_calls': 102400, 'auxiliary_max_calls': 48640,
                               'max_calls': 151040, 'rcaide_max_calls': 0}:
        raise ValueError('budget mismatch')
    return {'structurally_valid': True, 'seed_history_independence_verified': False,
            'qualification_status': {arm: adapters.QUALIFICATION[arm][0] for arm in (*MAIN, *AUXILIARY)},
            'execution_ready': False, 'blockers': ['seed history and independent sampling review required',
                'physical qualification and seven real adapters not qualified',
                'real call accounting, resource qualification and approval receipt not certified']}
