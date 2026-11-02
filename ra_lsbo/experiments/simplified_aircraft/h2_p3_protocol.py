"""Prospective fixed-versus-causal H2 mixture ablation; no evaluations."""
from copy import deepcopy
from . import h2_protocol as h2

SCHEMA = 'ra_lsbo_aircraft_h2_p3_mixture_ablation_v1'
DOCUMENT = 'docs/RA_LSBO_H2_P3_FIXED_MIXTURE_ABLATION_20261102.md'
CASES = h2.CASES
ARMS = ('h2_dynamic', 'h2_fixed50')
CORPUS_ARMS = ARMS
REFERENCE = 'h2_dynamic'
PROFILES = {
    'smoke': {'name': 'smoke', 'seeds': [12070], 'initial': 6, 'horizon': 2,
              'pool': 32, 'encoder_epochs': 30, 'native_gp_steps': 2},
    'pilot': {'name': 'pilot', 'seeds': list(range(12071, 12079)), 'initial': 12,
              'horizon': 18, 'pool': 128, 'encoder_epochs': 30, 'native_gp_steps': 15},
}


def profile_for(name):
    if name not in PROFILES:
        raise ValueError('Only the registered smoke and pilot profiles are supported')
    return deepcopy(PROFILES[name])


def budget_for(profile):
    n = len(CASES) * len(profile['seeds']) * len(ARMS)
    roles = {'shared_initial': len(CASES) * len(profile['seeds']) * profile['initial'],
             'online': n * profile['horizon'], 'incumbent_step_verification': n}
    return {**roles, 'maximum_model_calls': sum(roles.values()), 'maximum_rcaide_mda_calls': 0}


def model_settings(method, profile):
    if method not in ARMS:
        raise ValueError('Unregistered P3 arm')
    return h2.model_settings('h2_causal', profile)


def method_spec(method, profile):
    if method not in ARMS:
        raise ValueError('Unregistered P3 arm')
    spec = h2.method_spec('h2_causal', profile)
    spec['weight_policy'] = 'causal' if method == 'h2_dynamic' else 'fixed_0.5'
    return spec


def method_order(profile, case, seed):
    offset = (profile['seeds'].index(seed) + CASES.index(case)) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]
