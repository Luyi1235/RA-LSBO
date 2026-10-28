"""LSBO-style branch refinement driven by a sealed partial-observation GP."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize

from ..budget import digest
from ..partial_bo.selector import PartialBOSettings, choose_acquisition
from .posterior import RefinementPosterior, pointwise_acquisition


@dataclass(frozen=True)
class RefinementSettings:
    horizon: int = 12
    latent: bool = True
    kernel: str = "grouped"
    local: bool = True
    encoder_epochs: int = 120
    continuous_grid: tuple = (.05, .15, .4, 1.)
    discrete_grid: tuple = (.5, 1.5)
    local_branches: int = 4
    local_starts: int = 2
    local_points: int = 12
    local_maxiter: int = 25
    initial_radius: float = .2
    minimum_radius: float = .025
    maximum_radius: float = .5
    boundary_fraction: float = .2

    def __post_init__(self):
        for name in ("horizon", "encoder_epochs", "local_branches", "local_starts", "local_points", "local_maxiter"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer search settings required")
        if self.kernel not in {"grouped", "legacy"} or type(self.latent) is not bool or type(self.local) is not bool:
            raise ValueError("Unknown representation, kernel or search setting")
        for grid in (self.continuous_grid, self.discrete_grid):
            if not grid or any(not np.isfinite(value) or value <= 0 for value in grid):
                raise ValueError("Positive finite kernel grid required")
        if not 0 < self.minimum_radius <= self.initial_radius <= self.maximum_radius <= 1:
            raise ValueError("Invalid branch refinement radii")
        if not 0 <= self.boundary_fraction < 1:
            raise ValueError("Invalid boundary allocation")


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    return value


class RefiningLatentBO:
    def __init__(self, problem, base_project, seed=0, settings=None):
        self.problem, self.base, self.seed = problem, base_project, seed
        self.settings = settings or RefinementSettings()
        self.radii, self.stalls = {}, {}
        self.pending, self.expected_prefix, self.next_iteration = None, None, 0

    def _candidate(self, unit, discrete):
        return self.problem.canonical(self.problem.space.from_unit(unit, discrete).to_dict())

    def _unit(self, candidate):
        return self.problem.space.unit(self.problem.candidate_class(**candidate))

    def _branch(self, candidate):
        return digest(candidate["discrete"])

    def _augment(self, posterior, records, global_pool, metadata):
        settings = self.settings
        seen = {record.observation.design_id for record in records}
        pool, keys, rejected = list(global_pool), {digest(candidate) for candidate in global_pool}, {}
        reports = []

        def add(candidate):
            key = digest(candidate)
            reason = ("seen" if key in seen else "duplicate" if key in keys else
                      "exact_precheck" if any(value > 0 for value in self.problem.precheck(candidate).values()) else None)
            if reason:
                rejected[reason] = rejected.get(reason, 0)+1
            else:
                pool.append(candidate)
                keys.add(key)

        mode = metadata["mode"]
        restoration = str(metadata["fallback"]).startswith("no_")
        if restoration:
            mode = "posterior_support_restoration"
        scores, _ = pointwise_acquisition(posterior.predict(global_pool), posterior.best, mode,
            metadata["kappa"], .5)
        by_branch = {}
        priority = np.lexsort((-scores, ~np.asarray(metadata["eligible"], bool)))
        for index in priority:
            candidate = global_pool[int(index)]
            by_branch.setdefault(self._branch(candidate), []).append(candidate)
        anchors = {}
        for record in records:
            key = self._branch(record.candidate)
            if record.usable_feasible and (key not in anchors or
                    record.observation.objective.value < anchors[key].observation.objective.value):
                anchors[key] = record
        random = np.random.default_rng(self.seed*10000+self.next_iteration+800000000)
        for key, branch_pool in list(by_branch.items())[:settings.local_branches]:
            discrete = branch_pool[0]["discrete"]
            active = np.asarray(self.problem.space.mask(discrete), bool)
            radius = self.radii.get(key, settings.initial_radius)
            center = self._unit(anchors[key].candidate) if key in anchors else None
            low, high = (np.maximum(0., center-radius), np.minimum(1., center+radius)) if center is not None else (
                np.zeros(len(active)), np.ones(len(active)))
            for _ in range(settings.local_points):
                add(self._candidate(random.uniform(low, high), discrete))
            starts = [anchors[key].candidate] if key in anchors else []
            starts.extend(branch_pool)
            start_keys, chosen_starts = set(), []
            for candidate in starts:
                unit = np.clip(self._unit(candidate), low, high)
                identity = tuple(unit[active])
                if identity not in start_keys:
                    start_keys.add(identity)
                    chosen_starts.append(unit)
                if len(chosen_starts) == settings.local_starts:
                    break
            for unit in chosen_starts:
                if not active.any():
                    continue
                cache = {}

                def forecast(values):
                    identity = tuple(np.round(values, 14))
                    if identity not in cache:
                        complete = unit.copy()
                        complete[active] = values
                        candidate = self._candidate(complete, discrete)
                        score, support = pointwise_acquisition(posterior.predict([candidate]), posterior.best,
                            mode, metadata["kappa"], .5)
                        geometry = np.asarray([-value for value in self.problem.precheck(candidate).values()])
                        margins = geometry if restoration else np.r_[support[0], geometry]
                        cache[identity] = candidate, float(score[0]), margins
                    return cache[identity]

                result = minimize(lambda values: -forecast(values)[1], unit[active], method="SLSQP",
                    bounds=list(zip(low[active], high[active])),
                    constraints={"type": "ineq", "fun": lambda values: forecast(values)[2]},
                    options={"maxiter": settings.local_maxiter, "ftol": 1e-7})
                endpoint = forecast(result.x)
                add(endpoint[0])
                eligible = [entry for entry in cache.values() if np.all(entry[2] >= -1e-7)]
                if eligible:
                    add(max(eligible, key=lambda entry: entry[1])[0])
                reports.append({"branch": discrete, "radius": radius, "has_feasible_center": center is not None,
                    "start": self._candidate(unit, discrete), "endpoint": endpoint[0],
                    "active_dimensions": np.flatnonzero(active).tolist(),
                    "bounds": [low[active].tolist(), high[active].tolist()],
                    "success": bool(result.success), "status": int(result.status), "message": str(result.message),
                    "nfev": int(result.nfev), "unique_posterior_queries": len(cache),
                    "endpoint_raw_score": endpoint[1], "endpoint_support": endpoint[2].tolist(),
                    "local_mode": mode})
        return pool, {"local_reports": reports, "local_rejections": rejected,
                      "global_pool_size": len(global_pool), "augmented_pool_size": len(pool)}

    def propose(self, records, global_pool, iteration):
        if self.pending is not None:
            raise RuntimeError("Settle the pending proposal before another fit")
        if iteration != self.next_iteration or not 0 <= iteration < self.settings.horizon:
            raise ValueError("Iteration does not match the settled optimizer state")
        prefix = digest([record.to_dict() for record in records])
        if self.expected_prefix is not None and prefix != self.expected_prefix:
            raise ValueError("History differs from the previously settled prefix")
        seen = {record.observation.design_id for record in records}
        if not global_pool or len({digest(candidate) for candidate in global_pool}) != len(global_pool):
            raise ValueError("A nonempty unique global pool is required")
        for candidate in global_pool:
            if digest(candidate) in seen or candidate != self.problem.canonical(candidate) or any(
                    value > 0 for value in self.problem.precheck(candidate).values()):
                raise ValueError("Invalid or previously evaluated global candidate")
        posterior = RefinementPosterior(self.problem, self.base, records, self.seed, self.settings)
        acquisition = PartialBOSettings(horizon=self.settings.horizon, boundary_fraction=self.settings.boundary_fraction)
        predictions = posterior.predict(global_pool)
        _, before = choose_acquisition(predictions, posterior.best, iteration, acquisition)
        candidates, search = (self._augment(posterior, records, global_pool, before) if self.settings.local else
            (global_pool, {"local_reports": [], "local_rejections": {},
                           "global_pool_size": len(global_pool), "augmented_pool_size": len(global_pool)}))
        predictions = posterior.predict(candidates)
        index, selection = choose_acquisition(predictions, posterior.best, iteration, acquisition)
        candidate = candidates[index]
        prior = min((record.observation.objective.value for record in records
                     if record.usable_feasible and self._branch(record.candidate) == self._branch(candidate)), default=None)
        self.pending = {"design": digest(candidate), "contract": self.problem.contract_id,
                        "records": list(records), "prefix_digest": prefix,
                        "prior_branch_best": prior, "branch": self._branch(candidate)}
        return candidate, {**posterior.diagnostics(), **selection, **search,
            "pool": candidates, "forecasts": serial(predictions), "settings": asdict(self.settings),
            "branch_radii": dict(self.radii), "branch_stalls": dict(self.stalls),
            "initial_pool_mode": before["mode"], "initial_pool_fallback": before["fallback"],
            "selected_from_local_augmentation": index >= len(global_pool)}

    def accept(self, record):
        if self.pending is None or (record.observation.design_id, record.observation.contract_id) != (
                self.pending["design"], self.pending["contract"]):
            raise ValueError("Feedback does not match the pending design and task")
        if digest([row.to_dict() for row in self.pending["records"]]) != self.pending["prefix_digest"]:
            raise ValueError("Training records mutated after the proposal")
        branch, best = self.pending["branch"], self.pending["prior_branch_best"]
        improved = record.usable_feasible and (best is None or record.observation.objective.value < best)
        radius = self.radii.get(branch, self.settings.initial_radius)
        if improved:
            self.stalls[branch] = 0
            radius = min(self.settings.maximum_radius, radius*1.1)
        else:
            self.stalls[branch] = self.stalls.get(branch, 0)+1
            if self.stalls[branch] % 3 == 0:
                radius = max(self.settings.minimum_radius, radius*.7)
        self.radii[branch] = radius
        self.expected_prefix = digest([row.to_dict() for row in self.pending["records"]+[record]])
        self.next_iteration += 1
        self.pending = None
