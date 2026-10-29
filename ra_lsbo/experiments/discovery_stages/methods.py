"""Frozen controls and explicitly named new adapters, not paper reproductions."""
from dataclasses import asdict, replace

import numpy as np

from ...mechanism_system import SystemSettings
from ...system_experiment import OriginalLSBOWarmStart
from ..comparison import methods as old
from ..core.methods import settings_for
from ..core.runner import NATIVE_STOP
from .optimizer import DiscoveryBO
from .policies import COVERAGE_PERIOD, NEAR_TOLERANCE, optimistic_bound

STAGES = {
    'phase': ('random', 'ra_v2', 'ra_coverage', 'ra_discovery',
              'ra_guard_only', 'ra_phase_coverage', 'ra_phase_discovery'),
    'discovery': ('random', 'ra_v2', 'ra_cold_global', 'ra_coverage', 'ra_discovery'),
    'representation': ('random', 'random_then_local', 'latent_exact', 'latent_sign',
                       'latent_interval', 'onehot_exact', 'onehot_sign', 'onehot_interval'),
    'baselines': ('random', 'ra_discovery', 'original_lsbo', 'logcei', 'scbo_adapted',
                  'sego_conservative_v2', 'sego_utb_v3', 'nsga2_adapted'),
}
REFERENCE = {'phase': 'ra_phase_discovery', 'discovery': 'ra_discovery', 'representation': 'latent_interval', 'baselines': 'ra_discovery'}


def specification(method, profile):
    settings = settings_for('ra_lsbo', profile)
    if method in {'ra_guard_only', 'ra_phase_coverage', 'ra_phase_discovery'}:
        return {'engine': 'phase_discovery_v1',
                'cold_global': method == 'ra_phase_discovery', 'protect': True,
                'feedback': 'interval', 'random_discovery': False,
                'coverage_scope': 'none' if method == 'ra_guard_only' else 'cold',
                'settings': asdict(settings), 'near_tolerance': NEAR_TOLERANCE,
                'coverage_period': COVERAGE_PERIOD}
    if method in {'ra_cold_global', 'ra_coverage', 'ra_discovery', 'random_then_local'} or method.startswith(('latent_', 'onehot_')):
        feedback = method.split('_', 1)[1] if method.startswith(('latent_', 'onehot_')) else 'interval'
        if method == 'random_then_local':
            feedback = 'sign'
        latent = not (method.startswith('onehot_') or method == 'random_then_local')
        return {'engine': 'discovery_bo_v1', 'cold_global': method != 'ra_coverage',
                'protect': method != 'ra_cold_global', 'feedback': feedback,
                'random_discovery': method == 'random_then_local',
                'settings': asdict(replace(settings, latent=latent)),
                'near_tolerance': NEAR_TOLERANCE, 'coverage_period': COVERAGE_PERIOD}
    if method == 'ra_v2':
        return {'engine': 'unchanged_ra_v2', 'settings': asdict(settings)}
    if method == 'original_lsbo':
        return {'engine': 'frozen_original_warm_start_adapter',
                'native_gp_steps': profile['native_gp_steps'],
                'native_stop_retained': True, 'native_pool_not_matched': True,
                'partial_feedback': 'legacy exact-only adapter; native stop not retried'}
    if method in {'sego_conservative_v2', 'sego_utb_v3'}:
        return {'engine': 'sego_adapter', 'constraint_sign': 'g<=0',
                'bound': 'mu+beta*sigma' if method.endswith('v2') else 'mu-beta*sigma',
                'beta': 1.96, 'other_settings': old.ADAPTATIONS['sego_utb_adapted'],
                'official_implementation': False}
    if method in old.METHODS:
        return {'engine': 'unchanged_external_v2', 'description': old.METHODS[method],
                'adaptation': old.ADAPTATIONS.get(method)}
    raise ValueError('Unregistered staged method')


class OptimisticUTB(old.ExternalOptimizer):
    """Change ONLY UTB sign; retains fixed kernels, completion/sign weighting, etc."""
    def _sego_utb(self, records, pool):
        feasible = [r.observation.objective.value for r in records if r.usable_feasible]
        probability = old._joint_probability(self.problem, records, pool)
        if not feasible:
            return np.log(probability), {'phase': 'feasibility_search', 'utb_beta': 1.96}
        objective = old._regressor(self.problem, records, '__objective__')
        mean, std = old._predict(objective, self.problem, pool, min(feasible))
        log_ei = old.log_expected_improvement(mean, std, min(feasible))
        violation, details = np.zeros(len(pool)), {}
        for name in sorted({name for r in records for name, _ in r.observation.constraints}):
            regressor = old._regressor(self.problem, records, name)
            if regressor is None:
                continue
            mean, std = old._predict(regressor, self.problem, pool)
            bound = optimistic_bound(mean, std)
            details[name] = bound.tolist()
            violation += np.maximum(bound, 0.)
        eligible = violation <= 1e-12
        if eligible.any():
            scores = np.where(eligible, log_ei+np.log(probability), -np.inf)
            if not np.isfinite(scores).any():
                scores = np.where(eligible, np.log(probability), -np.inf)
            phase = 'optimistic_utb_feasible_set'
        else:
            least = np.isclose(violation, violation.min(), rtol=1e-10, atol=1e-12)
            scores = np.where(least, np.log(probability), -np.inf)
            phase = 'optimistic_utb_restoration'
        return scores, {'phase': phase, 'utb_beta': 1.96, 'eligible': eligible.tolist(),
                        'joint_probability': probability.tolist(), 'optimistic_bounds': details,
                        'optimistic_violation': violation.tolist(), 'constraint_sign': 'g<=0'}


class Labelled:
    def __init__(self, engine, method):
        self.engine, self.method = engine, method
        self.stop_reason = None

    def propose(self, history, pool, iteration):
        candidate, metadata = self.engine.propose(history, pool, iteration)
        return candidate, {**metadata, 'method': self.method}

    def accept(self, record):
        self.engine.accept(record)


class NativeLSBO:
    def __init__(self, problem, base, seed, profile, output):
        self.problem, self.base, self.seed = problem, base, seed
        self.profile, self.output = profile, output
        self.engine, self.pending, self.stop_reason = None, None, None
        self.iteration = 0

    def propose(self, history, pool, iteration):
        if self.pending is not None or self.stop_reason or iteration != self.iteration:
            raise ValueError('Native state does not match the trial')
        if self.engine is None:
            self.engine = OriginalLSBOWarmStart(self.problem, self.base,
                [r.legacy_row(i) for i,r in enumerate(history)], self.seed,
                SystemSettings(calls=self.profile['horizon'], failure_cap=self.profile['horizon'],
                               gp_steps=self.profile['native_gp_steps']), self.output)
        proposal, selection = self.engine.propose()
        self.pending = (len(history), selection)
        return proposal['candidate'], {**selection, 'method': 'original_lsbo',
                                       'native_settings': self.engine.settings}

    def accept(self, record):
        if self.pending is None:
            raise ValueError('No native proposal to settle')
        length, selection = self.pending
        try:
            self.engine.accept(record.legacy_row(length), selection)
        except RuntimeError as error:
            if str(error) != NATIVE_STOP:
                raise
            self.stop_reason = NATIVE_STOP
        self.pending, self.iteration = None, self.iteration+1


def optimizer_for(problem, base, seed, method, profile, output):
    spec = specification(method, profile)
    if spec['engine'] in {'discovery_bo_v1', 'phase_discovery_v1'}:
        settings = replace(settings_for('ra_lsbo', profile), latent=spec['settings']['latent'])
        engine = DiscoveryBO(problem, base, seed, settings, **{key: spec[key] for key in
                             ('cold_global', 'protect', 'feedback', 'random_discovery')},
                             coverage_scope=spec.get('coverage_scope', 'all'))
    elif method == 'original_lsbo':
        return NativeLSBO(problem, base, seed, profile, output)
    elif method == 'sego_utb_v3':
        engine = OptimisticUTB(problem, seed, 'sego_utb_adapted', profile)
    else:
        original = {'ra_v2': 'ra_lsbo', 'sego_conservative_v2': 'sego_utb_adapted'}.get(method, method)
        engine = old.optimizer_for(problem, base, seed, original, profile)
    return Labelled(engine, method)
