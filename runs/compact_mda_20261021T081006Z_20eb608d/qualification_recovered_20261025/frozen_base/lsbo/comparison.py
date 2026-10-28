"""Paired initialization and independent mixed-variable comparison algorithms.

Only the evaluation journal, engineering prechecks and result schema are shared
with LSBO. Baselines do not train a VAE or call LSBO's proposal/refinement policy.
"""
from copy import copy
from collections import Counter
import hashlib
import json
import time

import numpy as np
import torch
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .optimizer import Optimizer
from .records import Evaluation, Observation
from .space import Candidate
from .surrogate import fit_gp
from .candidate_pool import rank_observations, mixed_candidate_pool, SEARCH_POLICY_VERSION
from .risk_selection import select_candidate


METHODS = ('lsbo', 'random', 'onehot_gp_bo', 'smac_rf_bo', 'mixed_ga')
LABELS = dict(zip(METHODS, ('LSBO', 'Random Search', 'One-hot GP-BO',
                           'SMAC-style RF-BO', 'Mixed-GA')))
BASELINE_SETTINGS = {
    'candidate_pool': 'settings.proposal_pool_size', 'global_pool_fraction': .5,
    'random_exploration_probability': .1,
    'rf_trees': 128, 'rf_min_leaf': 2, 'rf_uncertainty': 'between_tree_std',
    'ga_population': 40, 'ga_tournament': 3, 'ga_crossover_probability': .9,
    'ga_sbx_eta': 15., 'ga_mutation_eta': 20.,
    'ga_mutation_probability': '1 / number_of_variables',
    'constraint_policy': 'true_objective_and_separate_constraints_no_big_penalty',
    'implementation': 'local_smac_style_not_official_SMAC3',
}


def paired_initial_design(optimizer):
    """Freeze geometry-screened grammar initialization before observing any MDA.

    Use the existing v49-style uniform/stratified initializer, not historical
    results. Synthetic records below count strata only; they never train a model
    and are never written to the evaluation journal.
    """
    planner = copy(optimizer)
    planner.rng = np.random.default_rng(optimizer.settings.seed)
    planner.observations, planner.seen = [], set()
    planner.precheck_rejections = Counter()
    planner.solver_failures = 0
    design = []
    while len(design) < planner.settings.initial:
        if not design and planner.accepted(planner.space.default()):
            candidate = planner.space.default()
        else:
            candidate, _ = planner.initial_candidate()
        design.append(candidate)
        planner.seen.add(planner.space.key(candidate))
        planner.observations.append(Observation(len(design), candidate,
                                   Evaluation(None, False), 'planning'))
    data = [c.to_dict() for c in design]
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return design, digest, planner.rng.bit_generator.state


def features(space, candidates):
    """Normalized active continuous values, masks and active category one-hots."""
    rows = []
    for c in candidates:
        c = space.canonical(c)
        mask = space.mask(c.discrete)
        row = list(space.unit(c) * mask) + list(mask)
        for v in space.discrete:
            active = space.active(v, c.discrete)
            row.extend(float(active and c.discrete[v.name] == value) for value in v.values)
            row.append(float(active))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def expected_improvement(mean, std, best):
    std = np.maximum(std, 1e-12)
    delta = best - mean
    z = delta / std
    return np.maximum(delta * norm.cdf(z) + std * norm.pdf(z), 0.)


class BaselineOptimizer(Optimizer):
    def __init__(self, problem, settings, output_dir, method):
        if method not in METHODS[1:]:
            raise ValueError(f'Unknown baseline: {method}')
        self.method = method
        super().__init__(problem, settings, output_dir)

    def update_refinement_state(self, observation, improved):
        # Baselines never use the LSBO trust-region/refinement state.
        return

    def run(self):
        print(f'[{LABELS[self.method]}] independent optimizer; budget={self.settings.budget}', flush=True)
        try:
            for candidate in self.initial_design:
                self.evaluate(candidate, 'initial', {'source': 'paired_initial_design',
                              'method': self.method, 'initial_design_sha256': self.initial_design_sha256})
            while len(self.observations) < self.settings.budget:
                started = time.perf_counter()
                candidate, info = self.propose('search')
                info['proposal_seconds'] = time.perf_counter() - started
                self.evaluate(candidate, 'search', dict(info, method=self.method))
        except Exception:
            self.save_summary('stopped_with_error')
            raise
        self.save_summary('complete')
        return self.elites()[0] if self.elites() else None

    def ranking(self):
        """Feasibility first; normalized violation ranks infeasible GA parents."""
        return rank_observations(self.observations, self.problem.required_constraints)

    def candidate_pool(self):
        pool = mixed_candidate_pool(self, self.settings.proposal_pool_size)
        if not pool:
            raise RuntimeError('No candidate passed shared engineering prechecks')
        return pool

    def prediction(self, x, y, query, name):
        y = np.asarray(y, dtype=float)
        center, scale = float(y.mean()), max(float(y.std()), 1e-8)
        target = (y - center) / scale
        if self.method == 'onehot_gp_bo':
            with torch.random.fork_rng():
                torch.manual_seed(self.settings.seed + len(self.observations))
                model = fit_gp(torch.tensor(x), torch.tensor(target[:, None]),
                               self.settings.gp_steps, name=name)
            with torch.no_grad():
                posterior = model.posterior(torch.tensor(query))
                mean = posterior.mean.squeeze(-1).numpy()
                std = posterior.variance.squeeze(-1).clamp_min(1e-12).sqrt().numpy()
        else:
            model = RandomForestRegressor(n_estimators=BASELINE_SETTINGS['rf_trees'],
                    min_samples_leaf=BASELINE_SETTINGS['rf_min_leaf'], max_features=1.,
                    random_state=self.settings.seed + len(self.observations), n_jobs=1)
            model.fit(x, target)
            samples = np.asarray([tree.predict(query) for tree in model.estimators_])
            mean, std = samples.mean(0), np.maximum(samples.std(0), 1e-6)
        return center + scale * mean, scale * std

    def bo_candidate(self):
        usable = [o for o in self.observations if o.result.has_objective()]
        if len(usable) < 3 or self.rng.random() < BASELINE_SETTINGS['random_exploration_probability']:
            return self.random_candidate(), {'source': self.method + '_random_exploration'}
        pool = self.candidate_pool()
        query = features(self.space, pool)
        x = features(self.space, [o.candidate for o in usable])
        y = np.array([o.result.objective for o in usable])
        mean, std = self.prediction(x, y, query, 'objective')
        feasible = [o.result.objective for o in usable if o.result.feasible(self.problem.required_constraints)]
        score = (np.log(np.maximum(expected_improvement(mean, std, min(feasible)), 1e-300))
                 if feasible else .05 * np.log(np.maximum(std, 1e-12)))
        for name in self.problem.required_constraints:
            if name in getattr(self.problem, 'analytical_constraints', ()):
                continue
            known = [o for o in self.observations if o.result.solver_success
                     and o.result.constraints.get(name) is not None and np.isfinite(o.result.constraints[name])]
            if len(known) < 3:
                score += np.log(.5)
                continue
            cx = features(self.space, [o.candidate for o in known])
            cm, cs = self.prediction(cx, [o.result.constraints[name] for o in known], query, name)
            score += log_ndtr(-cm / np.maximum(cs, 1e-12))
        labels = np.array([int(o.result.has_objective()) for o in self.observations])
        probabilities = {'success': np.full(len(pool), (labels.sum() + 1) / (len(labels) + 2))}
        if len(set(labels)) > 1:
            classifier = RandomForestClassifier(n_estimators=64, min_samples_leaf=2,
                          random_state=self.settings.seed, n_jobs=1)
            classifier.fit(features(self.space, [o.candidate for o in self.observations]), labels)
            probabilities['success'] = classifier.predict_proba(query)[:, list(classifier.classes_).index(1)]
        score += np.log(np.clip(probabilities['success'], 1e-6, 1.))
        metric = getattr(self.problem, 'solver_domain_metric', None) if self.settings.use_solver_domain else None
        known = [o for o in self.observations if metric and o.result.metrics.get(metric) is not None
                 and np.isfinite(o.result.metrics[metric])]
        if len(known) >= 3:
            dx = features(self.space, [o.candidate for o in known])
            dm, ds = self.prediction(dx, [o.result.metrics[metric] for o in known], query, 'solver_discharge_domain')
            z = dm / np.maximum(ds, 1e-12)
            score += log_ndtr(z)
            probabilities['domain'] = ndtr(z)
        if not np.isfinite(score).all():
            raise FloatingPointError('Nonfinite baseline acquisition scores')
        screening = {'mode': 'disabled'}
        if self.settings.screen_solver_risk:
            prefix = self.search_phase()
            thresholds = {name: getattr(self.settings, f'{prefix}_{name}_probability') for name in ('domain', 'success')}
            index, screening = select_candidate(score, probabilities, thresholds)
        else:
            index = int(np.argmax(score))
        return pool[index], {'source': self.method + '_constrained_ei',
                             'candidate_count': len(pool), 'log_acquisition': float(score[index]),
                             'raw_pool_count': len(pool), 'raw_pool_limit': self.settings.proposal_pool_size,
                             'search_policy_version': SEARCH_POLICY_VERSION, 'solver_risk_screen': screening}

    def ga_candidate(self):
        population = self.ranking()[:BASELINE_SETTINGS['ga_population']]
        dimension = len(self.space.variables)
        probability = 1. / dimension
        for _ in range(self.settings.max_proposal_attempts):
            parents = []
            for _ in range(2):
                indices = self.rng.integers(len(population), size=BASELINE_SETTINGS['ga_tournament'])
                parents.append(population[int(min(indices))].candidate)
            a, b = parents
            xa, xb = self.space.unit(a), self.space.unit(b)
            child = xa.copy()
            d = dict(a.discrete)
            if self.rng.random() < BASELINE_SETTINGS['ga_crossover_probability']:
                u = self.rng.random(len(child))
                eta = BASELINE_SETTINGS['ga_sbx_eta']
                beta = np.where(u <= .5, (2*u)**(1/(eta+1)),
                                (1/(2*(1-u)))**(1/(eta+1)))
                sign = self.rng.choice([-1., 1.], size=len(child))
                child = np.clip(.5 * ((xa+xb) + sign*beta*(xa-xb)), 0., 1.)
                d = {v.name: (a if self.rng.random() < .5 else b).discrete[v.name]
                     for v in self.space.discrete}
            eta = BASELINE_SETTINGS['ga_mutation_eta']
            for j in range(len(child)):
                if self.rng.random() < probability:
                    u = self.rng.random()
                    delta = ((2*u+(1-2*u)*(1-child[j])**(eta+1))**(1/(eta+1))-1
                             if u < .5 else
                             1-(2*(1-u)+2*(u-.5)*child[j]**(eta+1))**(1/(eta+1)))
                    child[j] = np.clip(child[j]+delta, 0., 1.)
            for v in self.space.discrete:
                if self.rng.random() < probability:
                    d[v.name] = v.values[int(self.rng.integers(len(v.values)))]
            candidate = self.space.from_unit(child, d)
            if self.accepted(candidate):
                return candidate, {'source': 'mixed_ga_offspring', 'population_size': len(population)}
        # Explicit immigrant if repeated offspring cannot pass geometry/novelty.
        return self.random_candidate(), {'source': 'mixed_ga_random_immigrant'}

    def propose(self, phase):
        if self.method == 'random':
            return self.random_candidate(), {'source': 'random_search'}
        if self.method == 'mixed_ga':
            return self.ga_candidate()
        return self.bo_candidate()
