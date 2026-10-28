from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from botorch.optim import optimize_acqf

from .records import Evaluation, Observation
from .space import Candidate
from .representation import Representation
from .surrogate import JointAcquisition, Surrogate
from .risk_selection import select_candidate
from .candidate_pool import mixed_candidate_pool, SEARCH_POLICY_VERSION
from .feasible_search import fixed_branch_search


RECOVERY_POLICY_VERSION = 'margin_ranked_anchor_backoff_v1'


class NoNeighborhoodCandidate(RuntimeError):
    pass


@dataclass
class Settings:
    budget: int = 120
    initial: int = 24
    seed: int = 0
    latent_dim: int = 4
    pretrain_epochs: int = 60
    update_epochs: int = 20
    update_every: int = 8
    gp_steps: int = 45
    branches: int = 12
    points_per_branch: int = 10
    restarts: int = 4
    raw_samples: int = 64
    local_branches: int = 2
    refinement_fraction: float = 0.2
    exploration_probability: float = 0.1
    use_prior: bool = True
    max_proposal_attempts: int = 3000
    max_consecutive_solver_failures: int = 5
    initial_local_fraction: float = 0.0
    initial_local_radius: float = 0.03
    stratified_initialization: bool = True
    coverage_interval: int = 5
    use_solver_domain: bool = True
    refinement_radius: float = 0.03
    adaptive_refinement: bool = True
    refinement_min_radius: float = 0.001
    screen_solver_risk: bool = True
    bo_domain_probability: float = 0.80
    refinement_domain_probability: float = 0.95
    bo_success_probability: float = 0.50
    refinement_success_probability: float = 0.70
    proposal_pool_size: int = 512
    coverage_min_evaluations: int = 4
    refinement_challenger_patience: int = 8
    refinement_challenger_interval: int = 5
    constrained_local_search: bool = True

    def validate(self):
        if not 4 <= self.initial <= self.budget:
            raise ValueError("Require 4 <= initial <= budget")
        if not 0 <= self.refinement_fraction < 1 or not 0 <= self.exploration_probability <= 1:
            raise ValueError("Invalid allocation fractions")
        for name in ("latent_dim", "pretrain_epochs", "update_epochs", "update_every", "gp_steps",
                     "branches", "points_per_branch", "restarts", "raw_samples", "max_proposal_attempts",
                     "max_consecutive_solver_failures"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.local_branches < 0 or self.raw_samples < self.restarts:
            raise ValueError("Invalid acquisition optimization configuration")
        for name in ('proposal_pool_size', 'coverage_min_evaluations',
                     'refinement_challenger_patience', 'refinement_challenger_interval'):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not 0 <= self.initial_local_fraction <= 1 or not 0 < self.initial_local_radius <= .5:
            raise ValueError('Invalid initialization neighborhood settings')
        if isinstance(self.coverage_interval, bool) or int(self.coverage_interval) != self.coverage_interval or self.coverage_interval < 0:
            raise ValueError('coverage_interval must be a nonnegative integer (0 disables scheduled coverage)')
        if not 0 < self.refinement_radius <= .5:
            raise ValueError('refinement_radius must be in (0, .5]')
        if not 0 < self.refinement_min_radius <= self.refinement_radius:
            raise ValueError('Invalid refinement minimum radius')
        for name in ('bo_domain_probability', 'refinement_domain_probability',
                     'bo_success_probability', 'refinement_success_probability'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')


def select_branches(space, elites, decoded, random_branches, limit):
    """Reserve elite positions before filling other slots, never append then truncate."""
    result, seen = [], set()
    for d in list(elites) + list(decoded) + list(random_branches):
        key = space.branch_key(d)
        if key not in seen:
            result.append(dict(d))
            seen.add(key)
        if len(result) == limit:
            break
    return result


class Optimizer:
    def __init__(self, problem, settings, output_dir):
        settings.validate()
        self.problem, self.space, self.settings = problem, problem.space, settings
        self.log_label = getattr(self, 'method', 'lsbo').upper()
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "evaluations.jsonl").exists():
            raise FileExistsError("Refusing to overwrite an existing evaluation journal")
        self.rng = np.random.default_rng(settings.seed)
        self.representation = (Representation(self.space, settings.seed, settings.latent_dim)
                               if getattr(self, 'method', 'lsbo') == 'lsbo' else None)
        self.observations, self.seen = [], set()
        self.precheck_rejections = Counter()
        self.radius, self.branch_stalls = {}, Counter()
        self.refinement_radii, self.refinement_stalls = {}, Counter()
        self.last_representation_update = 0
        self.surrogate_errors = 0
        self.solver_failures = 0
        self.resume_failure_limit = None
        self.continuation = None
        self.coverage_variables = tuple(getattr(problem, 'coverage_variables', ()))
        self.coverage_strata = []
        if self.coverage_variables:
            if not set(self.coverage_variables).issubset(v.name for v in self.space.discrete):
                raise ValueError('Coverage variables must identify discrete design variables')
            unique = {}
            for d in self.space.configurations():
                key = tuple(d[name] for name in self.coverage_variables)
                unique.setdefault(key, {name: d[name] for name in self.coverage_variables})
            self.coverage_strata = list(unique.values())

    def coverage_counts(self):
        return [sum(all(o.candidate.discrete[k] == v for k, v in stratum.items())
                    for o in self.observations) for stratum in self.coverage_strata]

    def coverage_candidate(self, stratum):
        # A feasibility-only initialization policy, independent of mission energy.
        d = dict(self.space.default().discrete, **stratum)
        default = self.space.canonical(Candidate({}, d))
        if self.accepted(default):
            return default
        center, mask = self.space.unit(default), self.space.mask(d)
        candidates = []
        for index in range(min(256, self.settings.max_proposal_attempts)):
            c = (self.space.from_unit(center + self.rng.normal(0, .15, len(center)) * mask, d)
                 if index % 2 == 0 else self.space.sample(self.rng, d))
            if self.accepted(c):
                candidates.append(c)
        if not candidates:
            return self.random_candidate(d)
        return min(candidates, key=lambda c: np.linalg.norm((self.space.unit(c) - center) * mask))

    def accepted(self, candidate):
        candidate = self.space.canonical(candidate)
        if self.space.key(candidate) in self.seen:
            return False
        checks = self.problem.precheck(candidate)
        rejected = [k for k, value in checks.items() if value is None or not np.isfinite(value) or value > 0]
        self.precheck_rejections.update(rejected)
        return not rejected

    def random_candidate(self, discrete=None):
        for _ in range(self.settings.max_proposal_attempts):
            candidate = self.space.sample(self.rng, discrete)
            if self.accepted(candidate):
                return candidate
        raise RuntimeError("No novel candidate passed cheap constraints; inspect bounds/constraints")

    def elites(self):
        ranked = sorted((o for o in self.observations if o.result.feasible(self.problem.required_constraints)),
                        key=lambda o: o.result.objective)
        unique = {}
        for o in ranked:
            unique.setdefault(self.space.branch_key(o.candidate.discrete), o)
        return list(unique.values())

    def local_refinement_radius(self, key):
        if not self.settings.adaptive_refinement:
            return min(self.radius.get(key, .2), self.settings.refinement_radius)
        return self.refinement_radii.get(key, self.settings.refinement_radius)

    def search_phase(self):
        fraction = (len(self.observations) - self.settings.initial) / max(1, self.settings.budget - self.settings.initial)
        return 'refinement' if fraction >= 1 - self.settings.refinement_fraction else 'bo'

    def coverage_target(self, phase):
        counts, interval = self.coverage_counts(), self.settings.coverage_interval
        if (phase == 'refinement' or not counts or not interval
                or min(counts) >= self.settings.coverage_min_evaluations
                or (len(self.observations) - self.settings.initial) % interval):
            return None
        return self.coverage_strata[int(np.argmin(counts))]

    def global_stagnation(self):
        best, last = float('inf'), len(self.observations)
        for i, observation in enumerate(self.observations, 1):
            if observation.result.feasible(self.problem.required_constraints) and observation.result.objective < best:
                best, last = observation.result.objective, i
        return len(self.observations) - last

    def challenger_scores(self, scores, candidates, probabilities, thresholds, phase):
        """Occasional alternative-configuration check after refinement stagnates."""
        elites = self.elites()
        if (phase != 'refinement' or not elites
                or self.global_stagnation() < self.settings.refinement_challenger_patience
                or (len(self.observations) - self.settings.initial) % self.settings.refinement_challenger_interval):
            return scores, False
        incumbent = self.space.branch_key(elites[0].candidate.discrete)
        eligible = np.isfinite(scores) & np.array([
            self.space.branch_key(c.discrete) != incumbent for c in candidates])
        for name, values in probabilities.items():
            eligible &= values >= thresholds[name]
        if not eligible.any():
            return scores, False
        return np.where(eligible, scores, -np.inf), True

    def update_refinement_state(self, observation, improved):
        if (not self.settings.adaptive_refinement or observation.phase != 'refinement'
                or observation.proposal.get('source') not in
                ('elite_local_rescore', 'elite_neighborhood_recovery')):
            return
        key = self.space.branch_key(observation.candidate.discrete)
        radius = self.local_refinement_radius(key)
        feasible = observation.result.feasible(self.problem.required_constraints)
        if improved:
            self.refinement_stalls[key] = 0
            radius *= 1.2
        elif not feasible:
            self.refinement_stalls[key] = 0
            radius *= .5
        else:
            self.refinement_stalls[key] += 1
            if self.refinement_stalls[key] % 3 == 0:
                radius *= .7
        self.refinement_radii[key] = float(np.clip(radius, self.settings.refinement_min_radius,
                                                   self.settings.refinement_radius))

    def neighborhood_candidate(self, anchor, radius, max_attempts=None):
        center, mask = self.space.unit(anchor), self.space.mask(anchor.discrete)
        attempts = self.settings.max_proposal_attempts if max_attempts is None else max_attempts
        for _ in range(attempts):
            unit = np.clip(center + self.rng.uniform(-radius, radius, len(center)) * mask, 0., 1.)
            candidate = self.space.from_unit(unit, anchor.discrete)
            if self.accepted(candidate):
                return candidate
        raise NoNeighborhoodCandidate('No novel candidate in the feasible-anchor neighborhood')

    def recovery_candidate(self, phase):
        # Energy elites can sit on a solver boundary. Keep all feasible anchors,
        # including higher-energy points discarded by the per-branch elite list.
        anchors = [o for o in self.observations
                   if o.result.feasible(self.problem.required_constraints)]
        if not anchors:
            raise RuntimeError('Recovery requires an evaluated feasible anchor')
        recent_failures = []
        for observation in reversed(self.observations):
            if observation.result.has_objective():
                break
            recent_failures.append(observation)
        anchor_uses = Counter(o.proposal.get('anchor_evaluation') for o in recent_failures)
        branch_uses = Counter(self.space.branch_key(o.candidate.discrete) for o in recent_failures)
        metric = getattr(self.problem, 'solver_domain_metric', None)

        def margin(observation):
            value = observation.result.metrics.get(metric) if metric else None
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) and value >= 0 else None

        def priority(observation):
            value = margin(observation)
            return (value is None, branch_uses[self.space.branch_key(observation.candidate.discrete)],
                    anchor_uses[observation.index], -(value if value is not None else 0.),
                    observation.result.objective, observation.index)

        anchors.sort(key=priority)
        radius = self.settings.initial_local_radius * 2. ** (-max(1, self.solver_failures - 1))
        if phase == 'refinement':
            radius = min(radius, self.settings.refinement_radius)
        # Bound cheap search across anchors, rather than spend MDA on a repeat.
        pool = anchors[:min(8, self.settings.max_proposal_attempts)]
        attempts = self.settings.max_proposal_attempts // len(pool)
        for anchor in pool:
            try:
                candidate = self.neighborhood_candidate(anchor.candidate, radius, attempts)
            except NoNeighborhoodCandidate:
                continue
            return candidate, {
                'source': 'initial_feasible_recovery' if phase == 'initial' else 'feasible_neighborhood_recovery',
                'recovery_policy': RECOVERY_POLICY_VERSION,
                'radius': radius, 'anchor_evaluation': anchor.index,
                'anchor_domain_margin': margin(anchor), 'anchor_domain_metric': metric,
                'consecutive_failures_before': self.solver_failures,
                'failed_anchor_uses': anchor_uses[anchor.index],
                'failed_branch_uses': branch_uses[self.space.branch_key(anchor.candidate.discrete)]}
        raise RuntimeError('No novel recovery candidate passed cheap constraints around feasible anchors')

    def initial_candidate(self):
        counts = self.coverage_counts()
        elites = self.elites()
        slot = max(0, len(self.observations) - 1)
        fraction = self.settings.initial_local_fraction
        local_slot = np.ceil((slot + 1) * fraction) > np.ceil(slot * fraction)
        recovery = self.solver_failures >= 2
        if elites and recovery:
            return self.recovery_candidate('initial')
        if elites and local_slot:
            radius = self.settings.initial_local_radius
            anchor = elites[0].candidate
            return self.neighborhood_candidate(anchor, radius), {
                'source': 'initial_feasible_neighborhood',
                'radius': radius, 'anchor_evaluation': elites[0].index}
        target = None
        if self.settings.stratified_initialization and counts:
            least_sampled = np.flatnonzero(np.asarray(counts) == min(counts))
            target = self.coverage_strata[int(self.rng.choice(least_sampled))]
        candidate, attempts = self.grammar_initial_candidate(target)
        return candidate, {'source': 'stratified_initial' if target is not None else 'grammar_random_initial',
                           'sampling_policy': 'v49_grammar_uniform_geometry_screened_v1',
                           'coverage_target': target, 'geometry_screening_attempts': attempts}

    def grammar_initial_candidate(self, target=None):
        """v49-style grammar traversal and uniform draws, screened before MDA."""
        configurations = [d for d in self.space.configurations()
                          if target is None or all(d[k] == v for k, v in target.items())]
        counts = Counter(self.space.branch_key(o.candidate.discrete) for o in self.observations)
        # Randomize ties; reuse a complete configuration only after less-visited
        # configurations have been tried. No energy ranking in the initial design.
        order = [configurations[int(i)] for i in self.rng.permutation(len(configurations))]
        order.sort(key=lambda d: counts[self.space.branch_key(d)])
        attempts = 0
        for d in order:
            for _ in range(min(64, self.settings.max_proposal_attempts - attempts)):
                attempts += 1
                candidate = self.space.sample(self.rng, d)
                if self.accepted(candidate):
                    return candidate, attempts
            if attempts >= self.settings.max_proposal_attempts:
                break
        raise RuntimeError('No initial candidate passed geometry screening in the requested grammar; '
                           'inspect bounds/constraints rather than silently changing the sampling policy')

    def evaluate(self, candidate, phase, proposal):
        candidate = self.space.canonical(candidate)
        if len(self.observations) >= self.settings.budget:
            raise RuntimeError("Evaluation budget exhausted")
        if not self.accepted(candidate):
            raise ValueError("Attempt to evaluate duplicate or rejected candidate")
        key = self.space.branch_key(candidate.discrete)
        previous = [o.result.objective for o in self.observations
                    if self.space.branch_key(o.candidate.discrete) == key and o.result.feasible(self.problem.required_constraints)]
        print(f"\n[{self.log_label}] START {len(self.observations) + 1:04d}/{self.settings.budget} "
              f"phase={phase} source={proposal.get('source', 'unspecified')}", flush=True)
        print(f"[{self.log_label}] Configuration: " + json.dumps(candidate.discrete), flush=True)
        print(f"[{self.log_label}] Continuous: " + json.dumps(candidate.continuous), flush=True)
        if proposal.get('recovery_policy'):
            print(f'[LSBO] Recovery: policy={proposal["recovery_policy"]}, '
                  f'anchor={proposal["anchor_evaluation"]}, radius={proposal["radius"]:.6g}, '
                  f'anchor_margin={proposal["anchor_domain_margin"]}, '
                  f'failed_branch_uses={proposal["failed_branch_uses"]}', flush=True)
        start = time.perf_counter()
        try:
            result = self.problem.evaluate(candidate)
            if not isinstance(result, Evaluation):
                raise TypeError("Problem must return Evaluation")
        except Exception as error:
            result = Evaluation(None, False, status=f"exception:{type(error).__name__}", metrics={"message": str(error)})
        result.elapsed_seconds = time.perf_counter() - start
        observation = Observation(len(self.observations) + 1, candidate, result, phase, proposal)
        self.observations.append(observation)
        self.seen.add(self.space.key(candidate))
        with (self.output / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(observation.to_dict(self.problem.required_constraints), allow_nan=False) + "\n")
            stream.flush()
        improved = result.feasible(self.problem.required_constraints) and (not previous or result.objective < min(previous))
        radius = self.radius.get(key, 0.2)
        if improved:
            self.branch_stalls[key] = 0
            self.radius[key] = min(0.5, radius * 1.1)
        else:
            self.branch_stalls[key] += 1
            if self.branch_stalls[key] % 3 == 0:
                self.radius[key] = max(0.025, radius * 0.7)
        self.update_refinement_state(observation, improved)
        self.solver_failures = 0 if result.has_objective() else self.solver_failures + 1
        if result.has_objective():
            self.resume_failure_limit = None
        self.save_summary("running")
        print(f"{observation.index:04d}/{self.settings.budget} {phase:10s} objective={result.objective} "
              f"feasible={result.feasible(self.problem.required_constraints)} status={result.status} "
              f"elapsed={result.elapsed_seconds:.1f}s", flush=True)
        best = self.elites()
        feasible_count = sum(o.result.feasible(self.problem.required_constraints) for o in self.observations)
        print(f"[{self.log_label}] Best feasible objective={best[0].result.objective if best else 'none'}; "
              f"feasible evaluations={feasible_count}/{len(self.observations)}", flush=True)
        violations = {name: result.constraints.get(name) for name in self.problem.required_constraints
                      if result.constraints.get(name) is None or not np.isfinite(result.constraints[name])
                      or result.constraints[name] > 0}
        if violations and result.has_objective():
            print(f"[{self.log_label}] Violated/missing constraints: " + str(violations), flush=True)
        if not result.has_objective():
            failure_limit = self.resume_failure_limit or self.settings.max_consecutive_solver_failures
            print(f'[{self.log_label}] Recorded failed candidate with no objective; consecutive failures='
                  f'{self.solver_failures}/{failure_limit}. '
                  f'Worker details: {result.metrics.get("worker_dir", "see evaluation record")}', flush=True)
        if self.solver_failures >= (self.resume_failure_limit or self.settings.max_consecutive_solver_failures):
            raise RuntimeError("Repeated missing/failed objective returns; stopped before wasting remaining budget")

    def _optimize(self, acquisition, bounds):
        with torch.random.fork_rng():
            torch.manual_seed(self.settings.seed + 1009 * (len(self.observations) + 1))
            point, _ = optimize_acqf(acquisition, bounds=bounds, q=1, num_restarts=self.settings.restarts,
                                    raw_samples=self.settings.raw_samples,
                                    options={"maxiter": 60, "batch_limit": self.settings.restarts})
        return point[0].detach()

    def propose(self, phase):
        usable = sum(o.result.has_objective() for o in self.observations)
        elites = self.elites()
        if elites and self.solver_failures >= 2:
            return self.recovery_candidate(phase)
        if elites and usable < 3:
            radius = self.settings.initial_local_radius
            if phase == 'refinement':
                key = self.space.branch_key(elites[0].candidate.discrete)
                radius = self.local_refinement_radius(key)
            return self.neighborhood_candidate(elites[0].candidate, radius), {
                'source': 'feasible_neighborhood_recovery', 'radius': radius,
                'anchor_evaluation': elites[0].index}
        if usable < 3:
            return self.random_candidate(), {"source": "cold_start"}
        target = self.coverage_target(phase)
        if phase != 'refinement' and target is None and self.rng.random() < self.settings.exploration_probability:
            return self.random_candidate(), {"source": "global_exploration"}
        rep = self.representation
        if rep.version == 0 or len(self.observations) - self.last_representation_update >= self.settings.update_every:
            print(f"[LSBO] Training VAE: observations={len(self.observations)}, valid objectives={usable}", flush=True)
            started = time.perf_counter()
            rep.fit(self.observations, self.settings.pretrain_epochs if rep.version == 0 else self.settings.update_epochs)
            self.last_representation_update = len(self.observations)
            torch.save(rep.checkpoint(), self.output / "representation.pt")
            print(f"[LSBO] VAE ready: version={rep.version}, elapsed={time.perf_counter() - started:.1f}s", flush=True)
        print("[LSBO] Fitting objective/constraint surrogates", flush=True)
        started = time.perf_counter()
        surrogate = Surrogate(self.observations, rep, self.problem.required_constraints,
                              self.settings.seed + len(self.observations), self.settings.gp_steps, self.settings.use_prior,
                              analytical_constraints=getattr(self.problem, 'analytical_constraints', ()),
                              solver_domain_metric=(getattr(self.problem, 'solver_domain_metric', None)
                                                    if self.settings.use_solver_domain else None))
        print(f"[LSBO] Surrogates ready ({time.perf_counter() - started:.1f}s); searching latent/continuous space", flush=True)
        print("[LSBO] GP checks: " + ", ".join(
            f"{name}: noise={info['noise_variance']:.3g}, attempts={info['attempts']}"
            for name, info in surrogate.fit_diagnostics.items()), flush=True)
        elite_obs = self.elites()
        anchor = elite_obs[0].candidate if elite_obs else self.observations[-1].candidate
        if target is not None:
            matching = [o for o in elite_obs if all(o.candidate.discrete[k] == v for k, v in target.items())]
            anchor = matching[0].candidate if matching else self.coverage_candidate(target)
        n = len(self.space.continuous)
        z = rep.latent(anchor.discrete)
        latent = rep.features([o.candidate for o in self.observations])[:, 2 * n:]
        low, high = latent.min(0).values - 0.5, latent.max(0).values + 0.5
        bounds = torch.stack([torch.cat([torch.zeros(n), low]), torch.cat([torch.ones(n), high])]).double()
        relaxed = (torch.cat([torch.as_tensor(self.space.unit(anchor), dtype=torch.double), z])
                   if phase == 'refinement' and elite_obs else
                   self._optimize(JointAcquisition(surrogate, self.space.mask(anchor.discrete), z), bounds))
        temp = 1.0 - 0.7 * len(self.observations) / self.settings.budget
        decoded = [rep.decode(relaxed[n:], self.rng, temp)]
        decoded += [rep.decode(relaxed[n:], self.rng, temp, True) for _ in range(self.settings.branches)]
        random_d = [self.space.sample(self.rng).discrete for _ in range(self.settings.branches * 3)]
        if target is not None:
            eligible = [d for d in self.space.configurations() if all(d[k] == v for k, v in target.items())]
            random_d = [eligible[int(self.rng.integers(len(eligible)))] for _ in random_d]
            decoded = [d for d in decoded if all(d[k] == v for k, v in target.items())]
        # Reserve exploration slots even when many decoded/elite branches are available.
        reserve = max(1, self.settings.branches // 4)
        elites = [o.candidate.discrete for o in elite_obs[:max(1, self.settings.branches // 3)]]
        if target is not None:
            elites = [d for d in elites if all(d[k] == v for k, v in target.items())]
        branches = select_branches(self.space, elites, decoded, random_d, max(1, self.settings.branches - reserve))
        branches = select_branches(self.space, branches, [], random_d, self.settings.branches)
        if phase == 'refinement' and elite_obs:
            branches = [o.candidate.discrete for o in elite_obs[:self.settings.branches]]
        centers = {self.space.branch_key(o.candidate.discrete): o.candidate for o in elite_obs}
        candidates, candidate_keys = [], set()
        pool_limit = self.settings.proposal_pool_size
        # BO combines structured candidates with the shared mixed pool. Refinement
        # retains branch-local bounds and fills the pool across distinct elites.
        structured_limit = pool_limit if phase == 'refinement' else max(1, pool_limit // 2)
        optimization_events, local_search_reports = [], []
        def add(c, optimized=False):
            if not optimized and len(candidates) >= structured_limit:
                return
            c = self.space.canonical(c)
            key = self.space.key(c)
            if optimized:
                checks = self.problem.precheck(c)
                violations = {k: v for k, v in checks.items()
                              if v is None or not np.isfinite(v) or v > 0}
                outcome = ('duplicate_pool' if key in candidate_keys else
                           'duplicate_history' if key in self.seen else
                           'precheck_rejected' if violations else 'accepted')
                optimization_events.append(dict(outcome=outcome, violations=violations,
                                                candidate=c.to_dict()))
            if key not in candidate_keys and self.accepted(c):
                candidates.append(c)
                candidate_keys.add(key)
        for d in branches:
            key = self.space.branch_key(d)
            center = centers.get(key)
            radius = self.radius.get(key, .2)
            if phase == 'refinement' and center is not None:
                radius = self.local_refinement_radius(key)
            else:
                add(self.space.from_unit(relaxed[:n].numpy(), d))
            points = max(self.settings.points_per_branch, int(np.ceil(structured_limit / max(1, len(branches)))))
            for j in range(points):
                local = center is not None and (phase == 'refinement' or j % 2 == 0)
                if local:
                    delta = (self.rng.uniform(-radius, radius, n) if phase == 'refinement'
                             else self.rng.normal(0, radius, n))
                    unit = self.space.unit(center) + delta
                    add(self.space.from_unit(unit, d))
                else:
                    add(self.space.sample(self.rng, d))
        structured_count = len(candidates)
        if phase != 'refinement':
            shared = mixed_candidate_pool(self, pool_limit - len(candidates), target, candidate_keys)
            candidates.extend(shared)
            candidate_keys.update(self.space.key(c) for c in shared)
        raw_pool_count = len(candidates)
        if not candidates:
            if phase == 'refinement' and elite_obs:
                radius = self.local_refinement_radius(self.space.branch_key(elite_obs[0].candidate.discrete))
                return self.neighborhood_candidate(elite_obs[0].candidate, radius), {
                    'source': 'elite_neighborhood_recovery', 'anchor_evaluation': elite_obs[0].index,
                    'refinement_radius': radius}
            return (self.coverage_candidate(target) if target is not None else self.random_candidate()), {
                'source': 'precheck_recovery', 'coverage_target': target}
        scores = surrogate.score(rep.features(candidates))
        selected_branches = set()
        for index in np.argsort(-scores):
            if len(selected_branches) >= self.settings.local_branches:
                break
            candidate = candidates[int(index)]
            key = self.space.branch_key(candidate.discrete)
            if key in selected_branches:
                continue
            selected_branches.add(key)
            mask, unit = self.space.mask(candidate.discrete), self.space.unit(candidate)
            active = np.where(mask > 0)[0]
            if not len(active):
                continue
            lb, ub = np.zeros(len(active)), np.ones(len(active))
            if phase == "refinement" and key in centers:
                center = self.space.unit(centers[key])[active]
                radius = self.local_refinement_radius(key)
                lb, ub = np.maximum(0, center - radius), np.minimum(1, center + radius)
            acq = JointAcquisition(surrogate, mask, rep.latent(candidate.discrete), unit, active_only=True)
            local_bounds = torch.tensor(np.stack([lb, ub]), dtype=torch.double)
            if self.settings.constrained_local_search:
                optimized, report = fixed_branch_search(acq, local_bounds, candidate,
                                                         self.space, self.problem.precheck)
                local_search_reports.append(dict(discrete=candidate.discrete, **report))
            else:
                optimized = self._optimize(acq, local_bounds)
            unit[active] = optimized.numpy()
            add(self.space.from_unit(unit, candidate.discrete), optimized=True)
            if phase == 'refinement' and key in centers:
                anchor_unit = self.space.unit(centers[key])
                for fraction in (.25, .5, .75):
                    add(self.space.from_unit(anchor_unit + fraction*(unit-anchor_unit), candidate.discrete), optimized=True)
        features = rep.features(candidates)
        scores = surrogate.score(features)
        if not np.isfinite(scores).any():
            raise RuntimeError("All candidate acquisition scores are non-finite")
        screening = {'mode': 'disabled'}
        thresholds, probabilities = {}, {}
        if self.settings.screen_solver_risk:
            prefix = 'refinement' if phase == 'refinement' else 'bo'
            thresholds = {name: getattr(self.settings, f'{prefix}_{name}_probability')
                          for name in ('domain', 'success')}
            probabilities = surrogate.solver_probabilities(features)
        selection_scores, challenger = self.challenger_scores(scores, candidates, probabilities, thresholds, phase)
        if self.settings.screen_solver_risk:
            index, screening = select_candidate(selection_scores, probabilities, thresholds)
        else:
            index = int(np.argmax(np.where(np.isfinite(selection_scores), selection_scores, -np.inf)))
        return candidates[index], {"source": 'elite_local_rescore' if phase == 'refinement' and elite_obs else 'latent_decode_rescore',
                                    "score": float(scores[index]),
                                    "coverage_target": target,
                                    "refinement_radius": self.local_refinement_radius(self.space.branch_key(candidates[index].discrete)) if phase == 'refinement' else None,
                                    "solver_risk_screen": screening,
                                    "representation_version": rep.version, "candidate_count": len(candidates),
                                    "search_policy_version": SEARCH_POLICY_VERSION,
                                    "raw_pool_count": raw_pool_count, "raw_pool_limit": pool_limit,
                                    "structured_candidates": structured_count,
                                    "local_optimized_candidates": len(candidates) - raw_pool_count,
                                    "local_search_reports": local_search_reports,
                                    "local_optimization_events": optimization_events,
                                    "refinement_challenger": challenger,
                                    "global_stagnation": self.global_stagnation(),
                                    "head_y_mean": rep.y_mean, "head_y_std": rep.y_std,
                                    "gp_diagnostics": surrogate.fit_diagnostics}

    def run(self):
        print(f"[LSBO] problem={self.problem.name} budget={self.settings.budget} "
              f"initial={self.settings.initial} seed={self.settings.seed}", flush=True)
        try:
            if not self.observations and hasattr(self, 'initial_design'):
                for candidate in self.initial_design:
                    self.evaluate(candidate, 'initial', {'source': 'paired_initial_design',
                                  'initial_design_sha256': self.initial_design_sha256})
            if not self.observations and self.accepted(self.space.default()):
                self.evaluate(self.space.default(), "initial", {"source": "configured_reference"})
            while len(self.observations) < self.settings.initial:
                candidate, info = self.initial_candidate()
                self.evaluate(candidate, "initial", info)
            while len(self.observations) < self.settings.budget:
                phase = self.search_phase()
                print(f"\n[LSBO] Proposing evaluation {len(self.observations) + 1}/{self.settings.budget}, phase={phase}", flush=True)
                proposal_started = time.perf_counter()
                try:
                    candidate, info = self.propose(phase)
                    info['proposal_seconds'] = time.perf_counter() - proposal_started
                    self.surrogate_errors = 0
                    print(f"[LSBO] Candidate ready in {time.perf_counter() - proposal_started:.1f}s", flush=True)
                except Exception:
                    with (self.output / "model_errors.log").open("a", encoding="utf-8") as stream:
                        stream.write(traceback.format_exc() + "\n")
                    self.surrogate_errors += 1
                    print(f"[LSBO] Model/proposal error #{self.surrogate_errors}; see {self.output / 'model_errors.log'}", flush=True)
                    raise RuntimeError("Model/proposal failed after internal GP retries; no random MDA fallback. "
                                       "See model_errors.log")
                self.evaluate(candidate, phase, info)
        except Exception:
            self.save_summary("stopped_with_error")
            raise
        self.save_summary("complete")
        return self.elites()[0] if self.elites() else None

    def save_summary(self, status):
        best = self.elites()
        summary = {"status": status, "problem": self.problem.name, "settings": asdict(self.settings),
                   "method": getattr(self, 'method', 'lsbo'),
                   "initial_design_sha256": getattr(self, 'initial_design_sha256', None),
                   "recovery_policy": RECOVERY_POLICY_VERSION,
                   "consecutive_solver_failures": self.solver_failures,
                   "resume_failure_limit": self.resume_failure_limit,
                   "continuation": self.continuation,
                   "last_representation_update": self.last_representation_update,
                   "coverage": [dict(stratum=stratum, evaluations=count) for stratum, count
                                in zip(self.coverage_strata, self.coverage_counts())],
                   "evaluations": len(self.observations), "feasible_evaluations": sum(o.result.feasible(self.problem.required_constraints) for o in self.observations),
                   "precheck_rejections": dict(self.precheck_rejections),
                   "proposal_sources": dict(Counter(o.proposal.get("source", "unspecified") for o in self.observations)),
                   "best": best[0].to_dict(self.problem.required_constraints) if best else None,
                   "rng_state": self.rng.bit_generator.state}
        temporary = self.output / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.output / "summary.json")
