"""Fixed five-arm prospective closed-loop protocol; no evaluation on import."""
from copy import deepcopy
from dataclasses import asdict, replace

from ..core.methods import settings_for
from ..discovery_stages.policies import COVERAGE_PERIOD, NEAR_TOLERANCE

SCHEMA = 'ra_lsbo_aircraft_corpus_pilot_v1'
DOCUMENT = 'docs/AIRCRAFT_CORPUS_PILOT_PROTOCOL_20261031.md'
CASES = ('aircraft_ordinary', 'aircraft_sparse')
ARMS = ('random', 'old30', 'old480', 'corpus_persistent', 'onehot')
REFERENCE = 'corpus_persistent'
PROFILES = {
    'smoke': {'name': 'smoke', 'seeds': [9930], 'initial': 6, 'horizon': 2,
              'pool': 32, 'encoder_epochs': 30, 'native_gp_steps': 2},
    'pilot': {'name': 'pilot', 'seeds': list(range(9931, 9939)), 'initial': 12,
              'horizon': 30, 'pool': 128, 'encoder_epochs': 30, 'native_gp_steps': 15},
}


def profile_for(name):
    if name not in PROFILES:
        raise ValueError('Only smoke or exploratory pilot; no resume/confirm/MDA')
    return deepcopy(PROFILES[name])


def budget_for(profile):
    trials = len(CASES)*len(profile['seeds'])*len(ARMS)
    roles = {'shared_initial': len(CASES)*len(profile['seeds'])*profile['initial'],
             'online': trials*profile['horizon'], 'incumbent_step_verification': trials}
    return {**roles, 'maximum_model_calls': sum(roles.values()), 'maximum_rcaide_mda_calls': 0}


def model_settings(method, profile):
    if method not in set(ARMS)-{'random'}:
        raise ValueError('No posterior settings for this arm')
    return replace(settings_for('ra_lsbo', profile), latent=method != 'onehot',
                   encoder_epochs=480 if method == 'old480' else 30)


def method_spec(method, profile):
    if method not in ARMS:
        raise ValueError('Unregistered corpus comparison arm')
    if method == 'random':
        return {'engine': 'unchanged_external_v2',
                'description': 'Uniform random search over the exact-screened candidate pool',
                'adaptation': None}
    result = {'engine': 'phase_discovery_v1', 'cold_global': True, 'protect': True,
              'feedback': 'interval', 'random_discovery': False, 'coverage_scope': 'cold',
              'settings': asdict(model_settings(method, profile)), 'near_tolerance': NEAR_TOLERANCE,
              'coverage_period': COVERAGE_PERIOD, 'transfer_mode': None}
    result['representation_policy'] = (
        {'kind': 'persistent_corpus_v1', 'draws': 2000, 'sampling': 'uniform_canonical_with_replacement',
         'initial_epochs': 30, 'update_epochs': 20, 'update_every': 8,
         'initial_learning_rate': .001, 'update_learning_rate': .0004,
         'fit_before_coverage': True, 'keep_duplicates': True,
         'save_numeric_checkpoints': True, 'normalization': 'unchanged_prefix_padding',
         'automatic_quality_fallback': False, 'optimizer_state_retained': False}
        if method == 'corpus_persistent' else
        {'kind': 'hierarchical_onehot'} if method == 'onehot' else
        {'kind': 'reset_enumerated_grammar', 'epochs': 480 if method == 'old480' else 30})
    return result


def method_order(profile, case, seed):
    """Cyclic arm order, predeclared without outcome-dependent selection.

    Method RNGs and candidate streams depend on method/seed/iteration, not wall
    order. No claim of perfectly balanced timing in eight seeds / five methods.
    """
    index = profile['seeds'].index(seed)
    shift = (index+CASES.index(case)) % len(ARMS)
    return ARMS[shift:]+ARMS[:shift]
