"""Transfer the fixed EP hierarchy and completion-weighted LSBO search to X-57."""

from copy import copy

import numpy as np
from scipy.linalg import cho_solve
import torch

from .budget import digest
from .partial_refinement.posterior import group_divisors
from .risk_core.completion_refinement import RiskWeightedRefinement
from .risk_core.mixed_gp import MixedGPSettings, MixedObservationGP
from .x57_cold_feedback import risk_observation
from .x57_cold_sampling import LAYOUT, LAYOUTS
from .x57_energy_surrogate import (
    EnergyAcquisition, engineering_log_probability, feature_bounds, fit_cold_constraints,
)


def _kernel(left, right, divisor):
    delta = left.unsqueeze(-2) - right
    radius = 5.**.5 * torch.linalg.vector_norm(delta[..., :-3] / divisor, dim=-1)
    latent = (1 + radius + radius.square() / 3) * torch.exp(-radius)
    ordinal = torch.exp(-.5 * delta[..., -3].square())
    same_layout = (delta[..., -2] == 0).to(latent.dtype)
    same_branch = (delta[..., -1] == 0).to(latent.dtype)
    return .5 * latent + .25 * (ordinal + .5 * (same_layout + same_branch))


class X57TimingGP(MixedObservationGP):
    def _kernel(self, left, right):
        return _kernel(torch.from_numpy(left), torch.from_numpy(right),
                       torch.as_tensor(self._scales[:-3])).numpy()


class CompletionPosterior:
    def __init__(self, rep, candidates, evidence, seed):
        space = rep.space
        configurations = space.configurations()
        self.grammar_features = torch.cat((torch.tensor(np.stack([space.mask(d) for d in configurations])),
                                           torch.stack([rep.latent(d) for d in configurations])), dim=-1)
        self.indices = {space.branch_key(d): i for i, d in enumerate(configurations)}
        self.metadata = torch.tensor([
            (d["propeller_count"] / 4 if d[LAYOUT] == LAYOUTS[1] else 0.,
             LAYOUTS.index(d[LAYOUT]), i)
            for i, d in enumerate(configurations)
        ], dtype=torch.double)
        self.rep = rep
        self.n_continuous = len(space.continuous)
        n_latent = self.grammar_features.shape[-1] - self.n_continuous
        # The fixed reference head scales by these groups before using length 0.5.
        self.divisor = .5 * torch.as_tensor(group_divisors(
            2 * self.n_continuous + n_latent, self.n_continuous, .4, 1.5))
        model = X57TimingGP(self.features(candidates).numpy(), evidence, seed=seed, settings=MixedGPSettings(
            length_scale=tuple(self.divisor.tolist() + [1., 1., 1.]), stable_tail_moments=True))
        if not model.diagnostics["converged"]:
            raise FloatingPointError("Completion EP did not converge")
        self.interval_inputs = torch.as_tensor(model.interval_x)
        self.factor = torch.as_tensor(np.tril(model._interval_factor[0]))
        self.mean_weights = torch.as_tensor(cho_solve(
            model._interval_factor, model._posterior_mean - model._interval_base_mean))
        self.covariance = torch.as_tensor(model._posterior_covariance)

    def features(self, candidates):
        indices = [self.indices[self.rep.space.branch_key(c.discrete)] for c in candidates]
        return torch.cat((self.rep.features(candidates), self.metadata[indices]), dim=-1)

    def log_probability(self, features):
        # The native hook receives only mask and latent, including relaxed points.
        # Project those coordinates to the nearest canonical grammar for the hierarchy term.
        grammar = features[..., self.n_continuous:]
        with torch.no_grad():
            distance = (grammar.unsqueeze(-2) - self.grammar_features).square().sum(-1)
            metadata = self.metadata[distance.argmin(dim=-1)]
        return self._log_probability(torch.cat((features, metadata), dim=-1))

    def _log_probability(self, augmented):
        cross = _kernel(augmented, self.interval_inputs, self.divisor)
        projection = torch.cholesky_solve(cross.unsqueeze(-1), self.factor).squeeze(-1)
        mean = cross @ self.mean_weights
        variance = 1. - (projection * cross).sum(-1)
        variance += ((projection @ self.covariance) * projection).sum(-1)
        return torch.special.log_ndtr(-mean / variance.clamp_min(1e-14).sqrt())

    def probabilities(self, candidates):
        return np.exp(self.log_probabilities(candidates))

    def log_probabilities(self, candidates):
        with torch.no_grad():
            return self._log_probability(self.features(candidates)).numpy()


RISK_ERRORS = (np.linalg.LinAlgError, FloatingPointError, RuntimeError)


def latent_pool(native, phase, mixed_candidate_pool, limit):
    rep, rng, space = native.representation, native.rng, native.space
    target = native.coverage_target(phase)
    latent = np.vstack([rep.latent(o.candidate.discrete).numpy() for o in native.observations])
    lower, upper = latent.min(axis=0) - .5, latent.max(axis=0) + .5
    points, keys = [], set()
    for _ in range(limit):
        if len(points) >= limit // 2:
            break
        discrete = rep.decode(rng.uniform(lower, upper), rng, stochastic=True)
        if target and any(discrete[name] != value for name, value in target.items()):
            continue
        point = space.sample(rng, discrete)
        key = space.key(point)
        if key not in keys and native.accepted(point):
            points.append(point)
            keys.add(key)
    return points + mixed_candidate_pool(native, limit - len(points), target, keys)


def propose(native, history, arm, mixed_candidate_pool, risk_pool_size, *, fit_energy=None, recovery=None):
    phase = native.search_phase()
    if arm == "lsbo_full_budget":
        candidate, info = native.propose(phase)
        return candidate, phase, info

    usable_designs = sum(row["feedback"]["usable"] for row in history)
    exact_objectives = sum(obs.result.has_objective() for obs in native.observations)
    # Native LogEI needs a feasible incumbent and the native surrogate needs
    # three valid objectives. Completed designs can supply objectives even
    # when they violate an engineering constraint.
    stage = "performance" if usable_designs and exact_objectives >= 3 else "completion"
    evidence = [(obs.candidate, value) for obs, row in zip(native.observations, history)
                if (value := risk_observation(row, arm == "structured_interval")) is not None]
    info = {"risk_stage": stage, "risk_training_rows": len(evidence),
            "usable_designs": usable_designs, "exact_objectives": exact_objectives,
            "n_finite_information_rows": sum(
                (v.lower is not None and v.lower > 0.) or (v.upper is not None and v.upper > 0.)
                for _, v in evidence)}

    def fit(rep):
        candidates, values = zip(*evidence)
        return CompletionPosterior(rep, candidates, values, native.settings.seed)

    def fallback(reason):
        point, native_info = native.propose(phase)
        return point, phase, {**native_info, **info, "risk_fallback": reason}

    def with_energy(surrogate, rep):
        head = fit_energy(rep, (surrogate.offset, surrogate.scale))
        info["engineering_energy_fit"] = head.audit()
        return EnergyAcquisition(surrogate, head)

    if not evidence and fit_energy is None:
        return fallback("no_supported_completion_feedback")
    if stage == "completion":
        rep = native.representation
        if not rep.version or len(history) - native.last_representation_update >= native.settings.update_every:
            epochs = native.settings.update_epochs if rep.version else native.settings.pretrain_epochs
            rep.fit(native.observations, epochs)
            native.last_representation_update = len(history)
            torch.save(rep.checkpoint(), native.output / "representation.pt")
        pool = latent_pool(native, phase, mixed_candidate_pool, risk_pool_size)
        if recovery is not None:
            neighbor, metadata = recovery
            neighbor_key = native.space.key(neighbor)
            info["recovery_candidate_id"] = neighbor_key
            if all(native.space.key(p) != neighbor_key for p in pool):
                pool.append(neighbor)
        if not pool:
            return fallback("no_screened_cold_candidates")
        log_completion = np.zeros(len(pool))
        if evidence:
            try:
                log_completion = fit(rep).log_probabilities(pool)
                if not np.isfinite(log_completion).all():
                    raise FloatingPointError("Nonfinite completion probabilities")
            except RISK_ERRORS:
                if fit_energy is None:
                    return fallback("invalid_completion_posterior")
                log_completion = np.zeros(len(pool))
                info["completion_fallback"] = "invalid_completion_posterior"
        else:
            info["completion_fallback"] = "no_supported_completion_feedback"
        scores = log_completion
        if fit_energy:
            try:
                bounds = feature_bounds(rep)
                head = fit_energy(rep, bounds)
                names, models = fit_cold_constraints(native, rep, bounds)
                with torch.no_grad():
                    features = rep.features(pool)
                    log_engineering = engineering_log_probability(features, head, names, models, bounds).numpy()
                    energy_probabilities = head.log_probability(features).exp().numpy()
                scores = log_completion + log_engineering
                if not np.isfinite(scores).all():
                    raise FloatingPointError("Nonfinite joint feasibility scores")
            except RISK_ERRORS:
                return fallback("invalid_engineering_posterior")
            info["engineering_energy_fit"] = head.audit()
            info["modeled_engineering_constraints"] = ["usable_energy", *models]
        index = max(range(len(pool)), key=lambda i: (scores[i], native.space.key(pool[i])))
        if fit_energy:
            info["p_usable_energy"] = float(energy_probabilities[index])
            info["engineering_log_probability"] = float(log_engineering[index])
        selected = {"source": "lsbo_latent_completion_risk"}
        if recovery is not None and native.space.key(pool[index]) == neighbor_key:
            selected = metadata
        return pool[index], phase, {**info, **selected,
                                    "candidate_count": len(pool),
                                    "candidate_list_sha256": digest([p.to_dict() for p in pool]),
                                    "p_complete": None if "completion_fallback" in info else
                                    float(np.exp(log_completion[index]))}

    state = {"risk_mode": "native_recovery_or_exploration"}
    posterior = None
    ranking = None

    def weighted(surrogate, rep):
        nonlocal posterior, ranking
        if fit_energy:
            try:
                surrogate = with_energy(surrogate, rep)
            except RISK_ERRORS:
                state["engineering_fallback"] = "invalid_engineering_posterior"
        if not evidence:
            state["completion_fallback"] = "no_supported_completion_feedback"
            return surrogate
        try:
            posterior = fit(rep)
        except RISK_ERRORS:
            state.update(risk_mode="native_fallback", risk_fallback="invalid_completion_posterior")
            return surrogate
        if fit_energy:
            # The structured completion model replaces both native numerical
            # heuristics. The native objective and constraint factors stay in place.
            surrogate = copy(surrogate)
            raw = copy(surrogate.native) if isinstance(surrogate, EnergyAcquisition) else surrogate
            raw.domain_model = None
            raw.success_model = None
            raw.fit_diagnostics = {k: v for k, v in raw.fit_diagnostics.items()
                                   if k != "solver_discharge_domain"}
            if isinstance(surrogate, EnergyAcquisition):
                surrogate.native = raw
                surrogate.fit_diagnostics = {k: v for k, v in surrogate.fit_diagnostics.items()
                                             if k != "solver_discharge_domain"}
        state["risk_mode"] = "risk_weighted_native_search"
        ranking = RiskWeightedRefinement(surrogate, posterior)
        return ranking

    point, native_info = native.propose(phase, refine_surrogate=weighted)
    if recovery is not None:
        neighbor, metadata = recovery
        state["recovery_candidate_id"] = native.space.key(neighbor)
        if ranking is not None and native.space.key(point) != state["recovery_candidate_id"]:
            with torch.no_grad():
                pair = ranking.score(native.representation.features([point, neighbor])).reshape(-1).tolist()
            if not np.isfinite(pair).all():
                raise ValueError("Nonfinite final recovery comparison")
            state["recovery_comparison"] = {"ra_candidate_id": native.space.key(point),
                                            "ra_log_score": pair[0], "neighbor_log_score": pair[1]}
            if pair[1] > pair[0]:
                point, native_info = neighbor, metadata
    if posterior is not None:
        state["p_complete"] = float(posterior.probabilities([point])[0])
    return point, phase, {**native_info, **info, **state}
