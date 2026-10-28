"""Isolated local-search revision; frozen LSBO geometry and radius rules remain."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize

from ..budget import digest
from ..numerics import InvalidLocalPoint, bounded_point, record_slsqp_clipping
from ..partial_bo.records import without_intervals
from ..partial_bo.selector import PartialBOSettings
from ..partial_refinement.optimizer import RefinementSettings, RefiningLatentBO, serial
from .acquisition import choose_acquisition, pointwise_acquisition
from .posterior import RiskPosterior


@dataclass(frozen=True)
class RevisionSettings(RefinementSettings):
    average_kernel: bool = True
    probability_exploitation: bool = True
    use_intervals: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.kernel != "grouped" or any(type(getattr(self, name)) is not bool for name in (
                "average_kernel", "probability_exploitation", "use_intervals")):
            raise ValueError("The isolated revision requires a grouped kernel and boolean switches")


class RiskRefiningBO(RefiningLatentBO):
    def __init__(self, problem, base_project, seed=0, settings=None):
        super().__init__(problem, base_project, seed, settings or RevisionSettings())

    def _augment(self, posterior, records, global_pool, metadata):
        settings = self.settings
        seen = {record.observation.design_id for record in records}
        pool, keys, rejected = list(global_pool), {digest(candidate) for candidate in global_pool}, {}
        reports = []

        def add(candidate):
            key = digest(candidate)
            geometry = np.asarray(list(self.problem.precheck(candidate).values()), float)
            reason = ("seen" if key in seen else "duplicate" if key in keys else
                      "invalid_candidate" if candidate != self.problem.canonical(candidate) else
                      "exact_precheck" if not np.isfinite(geometry).all() or np.any(geometry > 0) else None)
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
                bound_projections, max_bound_excess = 0, 0.

                def forecast(values):
                    nonlocal bound_projections, max_bound_excess
                    values, excess = bounded_point(values, low[active], high[active])
                    bound_projections += int(excess > 0)
                    max_bound_excess = max(max_bound_excess, excess)
                    identity = tuple(values)
                    if identity not in cache:
                        complete = unit.copy()
                        complete[active] = values
                        candidate = self._candidate(complete, discrete)
                        score, support = pointwise_acquisition(posterior.predict([candidate]), posterior.best,
                            mode, metadata["kappa"], .5)
                        geometry = np.asarray([-value for value in self.problem.precheck(candidate).values()])
                        margins = geometry if restoration else np.r_[support[0], geometry]
                        if not np.isfinite(score[0]) or not np.isfinite(margins).all():
                            raise InvalidLocalPoint("Nonfinite local acquisition or support")
                        cache[identity] = candidate, float(score[0]), margins
                    return cache[identity]

                # Local search generates surrogate candidates, never model calls.
                # Preserve valid cached points and the global pool on local failure.
                result, endpoint, local_error = None, None, None
                with record_slsqp_clipping() as clipping:
                    try:
                        result = minimize(lambda values: -forecast(values)[1], unit[active], method="SLSQP",
                            bounds=list(zip(low[active], high[active])),
                            constraints={"type": "ineq", "fun": lambda values: forecast(values)[2]},
                            options={"maxiter": settings.local_maxiter, "ftol": 1e-7})
                        endpoint = forecast(result.x)
                    except InvalidLocalPoint as error:
                        local_error = str(error)
                if endpoint is not None:
                    add(endpoint[0])
                eligible = [entry for entry in cache.values() if np.all(entry[2] >= -1e-7)]
                if eligible:
                    add(max(eligible, key=lambda entry: entry[1])[0])
                reports.append({"branch": discrete, "radius": radius, "has_feasible_center": center is not None,
                    "start": self._candidate(unit, discrete),
                    "endpoint": None if endpoint is None else endpoint[0],
                    "active_dimensions": np.flatnonzero(active).tolist(),
                    "bounds": [low[active].tolist(), high[active].tolist()],
                    "success": bool(result is not None and result.success and local_error is None),
                    "status": None if result is None else int(result.status),
                    "message": local_error if result is None else str(result.message),
                    "nfev": 0 if result is None else int(result.nfev),
                    "unique_posterior_queries": len(cache),
                    "endpoint_raw_score": None if endpoint is None else endpoint[1],
                    "endpoint_support": None if endpoint is None else endpoint[2].tolist(),
                    "local_error": local_error, "bound_projections": bound_projections,
                    "max_callback_bound_excess": max_bound_excess, **clipping,
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
                    not np.isfinite(value) or value > 0 for value in self.problem.precheck(candidate).values()):
                raise ValueError("Invalid or previously evaluated global candidate")
        model_records = records if self.settings.use_intervals else [without_intervals(record) for record in records]
        posterior = RiskPosterior(self.problem, self.base, model_records, self.seed, self.settings)
        acquisition = PartialBOSettings(horizon=self.settings.horizon, boundary_fraction=self.settings.boundary_fraction)
        predictions = posterior.predict(global_pool)
        _, before = choose_acquisition(predictions, posterior.best, iteration, acquisition,
            self.settings.probability_exploitation)
        candidates, search = (self._augment(posterior, records, global_pool, before) if self.settings.local else
            (global_pool, {"local_reports": [], "local_rejections": {},
                           "global_pool_size": len(global_pool), "augmented_pool_size": len(global_pool)}))
        predictions = posterior.predict(candidates)
        index, selection = choose_acquisition(predictions, posterior.best, iteration, acquisition,
            self.settings.probability_exploitation)
        candidate = candidates[index]
        prior = min((record.observation.objective.value for record in records
                     if record.usable_feasible and self._branch(record.candidate) == self._branch(candidate)), default=None)
        self.pending = {"design": digest(candidate), "contract": self.problem.contract_id,
                        "records": list(records), "prefix_digest": prefix,
                        "prior_branch_best": prior, "branch": self._branch(candidate)}
        return candidate, {**posterior.diagnostics(), **selection, **search,
            "pool": candidates, "forecasts": serial(predictions), "settings": asdict(self.settings),
            "raw_history_digest": prefix, "branch_radii": dict(self.radii), "branch_stalls": dict(self.stalls),
            "initial_pool_mode": before["mode"], "initial_pool_fallback": before["fallback"],
            "selected_from_local_augmentation": index >= len(global_pool)}
