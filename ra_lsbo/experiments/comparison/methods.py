"""Common-pool external baselines with explicit mixed/partial-feedback adapters."""
import zlib

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

from ...budget import digest
from ...numerics import log_expected_improvement
from ...soc_revision.optimizer import RiskRefiningBO
from ..core.methods import settings_for


METHODS = {
    "ra_lsbo": "RA-LSBO (current risk-aware structured latent method)",
    "random": "Uniform random search over the exact-screened candidate pool",
    "logcei": "logCEI, independent one-hot GP models with completion/constraint classifiers",
    "scbo_adapted": "SCBO-style constrained Thompson sampling with mixed-variable trust region",
    "sego_utb_adapted": "SEGO-UTB-style EI over a GP upper-confidence feasible set",
    "nsga2_adapted": "NSGA-II-style mixed-variable evolutionary search with feasibility-first ranking",
}

ADAPTATIONS = {
    "logcei": "Stable log-EI; objective GP uses all trusted completed exact objectives, including "
              "explicitly infeasible designs. Constraint/completion GPCs use sign-resolved evidence. "
              "Overlapping intervals and unknown outputs are omitted; kernel hyperparameters are fixed.",
    "scbo_adapted": "Single-point candidate-pool adapter with exact categorical equality and active "
                    "continuous trust-region distances. Incumbent includes initial labels and never "
                    "regresses. Bernoulli feasibility sampling is not reference SCBO constraint sampling.",
    "sego_utb_adapted": "Independent exact-value GPs, fixed 1.96 upper-confidence feasible set, "
                       "completion/sign evidence weighting and explicit restoration. Missing constraints "
                       "are not fabricated as numbers. Not a full SEGO-UTB reproduction.",
    "nsga2_adapted": "Sequential mixed-variable evolutionary adapter with declared evidence-first "
                     "constraint ordering then observed feasible objective. Missing objectives stay "
                     "missing. Not an official NSGA-II implementation or exact reproduction.",
}


def _features(problem, candidates):
    rows = []
    for candidate in candidates:
        row = []
        for variable in problem.space.discrete:
            row.extend(float(candidate["discrete"][variable.name] == value) for value in variable.values)
            row.append(float(problem.space.active(variable, candidate["discrete"])))
        for variable in problem.space.continuous:
            low, high = variable.bounds
            active = problem.space.active(variable, candidate["discrete"])
            value = candidate["continuous"][variable.name]
            row.extend((float(active), float((value-low)/(high-low)) if active else 0.))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _class_label(observation):
    return observation.constraint_satisfaction()


def _classifier(problem, records, name=None):
    x, y = [], []
    for record in records:
        if record.observation.status in {"unknown", "infrastructure_error"}:
            continue
        if name == "__completion__":
            label = record.observation.completion_label()
        else:
            try:
                label = _class_label(record.observation.constraint(name))
            except KeyError:
                label = None
        if label is not None:
            x.append(record.candidate)
            y.append(int(label))
    if not y:
        return None, .5
    rate = (sum(y)+1.)/(len(y)+2.)
    if len(set(y)) < 2 or len(y) < 3:
        return None, rate
    model = GaussianProcessClassifier(
        kernel=ConstantKernel(1., (1e-2, 1e2))*RBF(length_scale=1.),
        optimizer=None, random_state=0, max_iter_predict=50)
    model.fit(_features(problem, x), np.asarray(y))
    return model, rate


def _probability(model, rate, x):
    if model is None:
        return np.full(len(x), rate, dtype=float)
    probabilities = model.predict_proba(x)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(x))


def _regressor(problem, records, output):
    x, y = [], []
    for record in records:
        if record.observation.status in {"unknown", "infrastructure_error"}:
            continue
        if output == "__objective__":
            observation = record.observation.objective
            valid = record.observation.status == "completed"
        else:
            try:
                observation = record.observation.constraint(output)
            except KeyError:
                continue
            valid = observation.kind == "exact"
        if valid and observation.kind == "exact":
            x.append(record.candidate)
            y.append(float(observation.value))
    if len(y) < 2:
        return None
    model = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (1e-2, 1e2))*Matern(length_scale=1., nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed"),
        alpha=1e-6, normalize_y=True, optimizer=None, random_state=0)
    model.fit(_features(problem, x), np.asarray(y))
    return model


def _predict(model, problem, pool, fallback=0.):
    x = _features(problem, pool)
    if model is None:
        return np.full(len(pool), fallback), np.full(len(pool), 1.)
    mean, std = model.predict(x, return_std=True)
    return mean, np.maximum(std, 1e-9)


def _constraint_models(problem, records):
    names = sorted({name for row in records for name, _ in row.observation.constraints})
    return {name: _classifier(problem, records, name) for name in names}


def _joint_probability(problem, records, pool):
    completion = _classifier(problem, records, "__completion__")
    probability = _probability(completion[0], completion[1], _features(problem, pool))
    for name, (model, rate) in _constraint_models(problem, records).items():
        probability *= _probability(model, rate, _features(problem, pool))
    return np.clip(probability, 1e-9, 1.)


def _constraint_key(record):
    """Declared evidence ordering, NOT an imputed physical violation magnitude.

    Usable feasible < completed known-infeasible < unresolved < hidden failure
    < unknown/infrastructure. Within known-infeasible records compare violation
    count, unresolved count, then observed numeric lower bound. In particular,
    (0,+inf) remains one known violation even though its infimum equals zero.
    """
    status = record.observation.status
    if status in {"unknown", "infrastructure_error"}:
        return (4, 0, 0, 0.)
    if record.usable_feasible:
        return (0, 0, 0, 0.)
    if status == "hidden_failure":
        return (3, 0, 0, 0.)
    violations, unresolved, lower_bound = 0, 0, 0.
    for _, observation in record.observation.constraints:
        label = observation.constraint_satisfaction()
        unresolved += int(label is None)
        if label is False:
            violations += 1
            bound = observation.value if observation.kind == "exact" else observation.lower
            if bound is not None:
                lower_bound += max(0., float(bound))
    return (1 if violations else 2, violations, unresolved, lower_bound)


class ExternalOptimizer:
    def __init__(self, problem, seed, method, profile):
        self.problem, self.seed, self.method = problem, int(seed), method
        self.profile, self.horizon, self.pool_size = profile, profile["horizon"], profile["pool"]
        self.pending, self.next_iteration = None, 0
        self.rng = np.random.default_rng(seed+zlib.crc32(method.encode("utf-8")))
        self.radius, self.successes, self.failures = .5, 0, 0
        self.best_before = None

    def _center(self, records):
        feasible = [row for row in records if row.usable_feasible]
        if feasible:
            return min(feasible, key=lambda row: row.observation.objective.value)
        return min(records, key=_constraint_key) if records else None

    def _within_scbo_region(self, center, candidates):
        if center is None:
            return np.ones(len(candidates), dtype=bool)
        mask = []
        for candidate in candidates:
            same = candidate["discrete"] == center.candidate["discrete"]
            distances = []
            for variable in self.problem.space.continuous:
                if (self.problem.space.active(variable, candidate["discrete"])
                        and self.problem.space.active(variable, center.candidate["discrete"])):
                    low, high = variable.bounds
                    distances.append(abs(candidate["continuous"][variable.name]
                                         -center.candidate["continuous"][variable.name])/(high-low))
            mask.append(same and (not distances or max(distances) <= self.radius))
        return np.asarray(mask, dtype=bool)

    def _logcei(self, records, pool):
        objective = _regressor(self.problem, records, "__objective__")
        probability = _joint_probability(self.problem, records, pool)
        feasible_values = [row.observation.objective.value for row in records if row.usable_feasible]
        if not feasible_values:
            return np.log(probability), {"phase": "feasibility_search", "joint_probability": probability.tolist()}
        mean, std = _predict(objective, self.problem, pool, min(feasible_values))
        log_ei = log_expected_improvement(mean, std, min(feasible_values))
        score = log_ei+np.log(probability)
        if not np.isfinite(score).any():
            score = np.log(probability)
        return score, {"phase": "log_constrained_ei",
                       "log_ei": [float(v) if np.isfinite(v) else None for v in log_ei],
                       "joint_probability": probability.tolist()}

    def _scbo(self, records, pool, iteration):
        center = self._center(records)
        indices = np.flatnonzero(self._within_scbo_region(center, pool))
        if not len(indices):
            indices = np.arange(len(pool))
        local = [pool[i] for i in indices]
        objective = _regressor(self.problem, records, "__objective__")
        if objective is None:
            probability = _joint_probability(self.problem, records, local)
            return indices, np.log(probability), {"phase": "feasibility_search", "radius": self.radius}
        x = _features(self.problem, local)
        draws = objective.sample_y(x, n_samples=1, random_state=self.seed+iteration).reshape(-1)
        probability = _joint_probability(self.problem, records, local)
        random = np.random.default_rng(self.seed+iteration+50000)
        sampled_feasible = random.random(len(local)) < probability
        if sampled_feasible.any():
            scores = np.where(sampled_feasible, -draws, -np.inf)
        else:
            scores = -np.log(np.maximum(1.-probability, 1e-9))
        return indices, scores, {"phase": "constrained_thompson", "radius": self.radius,
            "sampled_feasible_candidates": int(sampled_feasible.sum()),
            "has_feasible_incumbent": any(row.usable_feasible for row in records)}

    def _sego_utb(self, records, pool):
        feasible_values = [row.observation.objective.value for row in records if row.usable_feasible]
        objective = _regressor(self.problem, records, "__objective__")
        probability = _joint_probability(self.problem, records, pool)
        if not feasible_values:
            return np.log(probability), {"phase": "feasibility_search", "utb_beta": 1.96}
        mean, std = _predict(objective, self.problem, pool, min(feasible_values))
        log_ei = log_expected_improvement(mean, std, min(feasible_values))
        upper_violation = np.zeros(len(pool))
        names = sorted({name for row in records for name, _ in row.observation.constraints})
        details = {}
        for name in names:
            regressor = _regressor(self.problem, records, name)
            if regressor is None:
                continue
            constraint_mean, constraint_std = _predict(regressor, self.problem, pool)
            upper = constraint_mean+1.96*constraint_std
            upper_violation += np.maximum(upper, 0.)
            details[name] = upper.tolist()
        eligible = upper_violation <= 1e-12
        if eligible.any():
            score = np.where(eligible, log_ei+np.log(probability), -np.inf)
            if not np.isfinite(score).any():
                score = np.where(eligible, np.log(probability), -np.inf)
            phase = "utb_feasible_set"
        else:
            least = np.isclose(upper_violation, upper_violation.min(), rtol=1e-10, atol=1e-12)
            score = np.where(least, np.log(probability), -np.inf)
            phase = "utb_support_restoration"
        return score, {"phase": phase, "utb_beta": 1.96, "eligible": eligible.tolist(),
            "joint_probability": probability.tolist(), "upper_violation": upper_violation.tolist(),
            "upper_bounds": details}

    def _nsga2(self, records, pool):
        if not records:
            return {"phase": "random_initial_population", "population": 0, "crowding_order": []}
        keys = [_constraint_key(row) for row in records]
        objectives = np.asarray([row.observation.objective.value if row.usable_feasible else np.nan
                                 for row in records], float)
        values = np.column_stack((np.asarray(keys, float), objectives))

        def better(i, j):
            if keys[i] != keys[j]:
                return keys[i] < keys[j]
            return (records[i].usable_feasible and records[j].usable_feasible
                    and objectives[i] < objectives[j])

        count = len(records)
        domination_count = np.zeros(count, dtype=int)
        dominates = [[] for _ in range(count)]
        fronts, first = [], []
        for i in range(count):
            for j in range(count):
                if i == j:
                    continue
                if better(i, j):
                    dominates[i].append(j)
                elif better(j, i):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                first.append(i)
        fronts.append(first)
        rank = np.full(count, count, dtype=int)
        for front_index, front in enumerate(fronts):
            next_front = []
            for i in front:
                rank[i] = front_index
                for j in dominates[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
        crowding = np.zeros(count, dtype=float)
        for front in fronts:
            if len(front) <= 2:
                crowding[front] = np.inf
                continue
            for objective in range(values.shape[1]):
                if not np.isfinite(values[front, objective]).all():
                    continue
                ordered = sorted(front, key=lambda index: values[index, objective])
                low, high = values[ordered[0], objective], values[ordered[-1], objective]
                if high <= low or not np.isfinite(high-low):
                    continue
                crowding[ordered[0]] = crowding[ordered[-1]] = np.inf
                for position in range(1, len(ordered)-1):
                    crowding[ordered[position]] += (
                        values[ordered[position+1], objective]-values[ordered[position-1], objective]
                    )/(high-low)
        return {"phase": "parent_nondominated_sorting", "population": count,
            "front_sizes": [len(front) for front in fronts], "best_constraint_key": list(min(keys)),
            "best_observed_objective": min((row.observation.objective.value for row in records
                if row.usable_feasible), default=None),
            "crowding_order": sorted(range(count), key=lambda index: (rank[index], -crowding[index]))}

    def _evolutionary_candidate(self, records, pool, iteration):
        if len(records) < 2:
            return None
        selection = self._nsga2(records, pool)
        ordered = [records[index] for index in selection["crowding_order"]]
        parents = ordered[:max(2, min(8, len(ordered)))]
        seen = {row.observation.design_id for row in records}
        for _ in range(48):
            first = parents[int(self.rng.integers(len(parents)))]
            second = parents[int(self.rng.integers(len(parents)))]
            discrete = dict(first.candidate["discrete"])
            for variable in self.problem.space.discrete:
                if self.rng.random() < .5:
                    discrete[variable.name] = second.candidate["discrete"][variable.name]
            continuous = dict(first.candidate["continuous"])
            for variable in self.problem.space.continuous:
                if not self.problem.space.active(variable, discrete):
                    continue
                low, high = variable.bounds
                if self.rng.random() < .5 and self.problem.space.active(variable, second.candidate["discrete"]):
                    continuous[variable.name] = .5*(first.candidate["continuous"][variable.name]
                                                   +second.candidate["continuous"][variable.name])
                if self.rng.random() < .35:
                    continuous[variable.name] = float(np.clip(
                        continuous[variable.name]+self.rng.normal(0., .15*(high-low)), low, high))
            try:
                candidate = self.problem.canonical({"continuous": continuous, "discrete": discrete})
            except (TypeError, ValueError):
                continue
            if (digest(candidate) not in seen and not any(not np.isfinite(value) or value > 0
                    for value in self.problem.precheck(candidate).values())):
                return candidate
        return None

    def propose(self, records, pool, iteration):
        if self.pending is not None or iteration != self.next_iteration:
            raise ValueError("Settle each comparison proposal before continuing")
        seen = {row.observation.design_id for row in records}
        if (not pool or len({digest(candidate) for candidate in pool}) != len(pool)
                or any(self.problem.canonical(candidate) != candidate or digest(candidate) in seen
                       or any(not np.isfinite(value) or value > 0
                              for value in self.problem.precheck(candidate).values()) for candidate in pool)):
            raise ValueError("Candidate pool must be canonical, unique, screened, unseen and nonempty")
        if self.method == "scbo_adapted":
            self.best_before = min((row.observation.objective.value for row in records
                                    if row.usable_feasible), default=None)
        metadata = {"method": self.method, "adaptation": ADAPTATIONS.get(self.method)}
        if self.method == "random":
            index = int(self.rng.integers(len(pool)))
        elif self.method == "logcei":
            scores, details = self._logcei(records, pool)
            metadata.update(details)
            index = int(np.argmax(scores))
        elif self.method == "scbo_adapted":
            indices, scores, details = self._scbo(records, pool, iteration)
            metadata.update(details)
            index = int(indices[int(np.argmax(scores))])
        elif self.method == "sego_utb_adapted":
            scores, details = self._sego_utb(records, pool)
            metadata.update(details)
            index = int(np.argmax(scores))
        elif self.method == "nsga2_adapted":
            offspring = self._evolutionary_candidate(records, pool, iteration)
            if offspring is not None:
                candidate, index = offspring, None
                metadata["phase"] = "mixed_variable_crossover_mutation"
            else:
                index = int(self.rng.integers(len(pool)))
                metadata["phase"] = "screened_pool_fallback"
        elif self.method == "ra_lsbo":
            optimizer = getattr(self, "_ra_optimizer", None)
            if optimizer is None:
                optimizer = RiskRefiningBO(self.problem, self.base_project, self.seed,
                                          settings_for("ra_lsbo", self.profile))
                self._ra_optimizer = optimizer
            candidate, info = optimizer.propose(records, pool, iteration)
            metadata.update(info)
            self.pending = (candidate, optimizer, "ra")
            return candidate, metadata
        else:
            raise ValueError("Unknown comparison method: "+self.method)
        candidate = pool[index] if index is not None else candidate
        self.pending = (candidate, None, self.method)
        return candidate, metadata

    def accept(self, record):
        if (self.pending is None or record.candidate != self.pending[0]
                or record.observation.contract_id != self.problem.contract_id):
            raise ValueError("Feedback does not match the pending comparison proposal")
        _, optimizer, _ = self.pending
        if optimizer is not None:
            optimizer.accept(record)
        if self.method == "scbo_adapted":
            incumbent = self.best_before
            value = record.observation.objective.value if record.usable_feasible else None
            if value is not None and (incumbent is None or value < incumbent):
                self.successes += 1
                self.failures = 0
                if self.successes >= 3:
                    self.radius = min(1., self.radius*1.5)
                    self.successes = 0
            else:
                self.failures += 1
                self.successes = 0
                if self.failures >= 4:
                    self.radius = max(.05, self.radius*.5)
                    self.failures = 0
            if value is not None:
                self.best_before = value if incumbent is None else min(incumbent, value)
        self.pending = None
        self.next_iteration += 1


def optimizer_for(problem, base_project, seed, method, profile):
    if method not in METHODS:
        raise ValueError("Unknown external comparison method")
    optimizer = ExternalOptimizer(problem, seed, method, profile)
    optimizer.base_project = base_project
    return optimizer
