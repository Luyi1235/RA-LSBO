"""Mixed-observation surrogate for the X-57 usable-terminal-energy constraint."""

from copy import copy
from dataclasses import asdict

import numpy as np
import torch

from .risk_core.mixed_gp import MixedGPSettings, MixedObservationGP
from .risk_core.observations import OutputObservation, finite
from .x57_cold_feedback import usable_energy_feedback

CONSTRAINT = "usable_energy"
TERMINAL_CAP_WH = 47000.
SETTINGS = MixedGPSettings(signal_variance=.15**2, exact_noise=1e-4,
                           stable_tail_moments=True, max_sweeps=120)


def energy_training_row(row, evidence_id, distance, prefix_energy, use_interval):
    feedback = row["feedback"]
    value = feedback["constraints"].get(CONSTRAINT)
    if feedback["completion"] is True and finite(value):
        return OutputObservation("exact", evidence_id, value=value), None
    metrics = row["worker_result"].get("metrics") or {}
    if (feedback["outcome"] != "numerical_nonconvergence" or value is not None
            or metrics.get("mda_evaluated") is not True
            or metrics.get("usable_terminal_energy_Wh") != TERMINAL_CAP_WH
            or (metrics.get("mission_profile") or {}).get("cruise_distance_nmi") != distance):
        return OutputObservation("missing", evidence_id, reason="no_verified_energy_prefix"), None
    lower, _ = prefix_energy(metrics.get("mission_solver_segments", []))
    # The observed prefix is a lower bound on a nonnegative discharge sum.
    return usable_energy_feedback(evidence_id, lower, TERMINAL_CAP_WH,
                                  use_interval=use_interval), lower


def feature_bounds(rep):
    n = 2 * len(rep.space.continuous)
    with torch.no_grad():
        grammar, _ = rep.model.encode(rep.grammar)
    offset = torch.cat((torch.zeros(n), grammar.min(0).values - .5)).double()
    upper = torch.cat((torch.ones(n), grammar.max(0).values + .5)).double()
    return offset, (upper - offset).clamp_min(1e-8)


class EnergyPosterior:
    def __init__(self, rep, candidates, evidence, seed, bounds=None):
        self.offset, self.scale = feature_bounds(rep) if bounds is None else bounds
        inputs = (rep.features(candidates) - self.offset) / self.scale
        model = self.model = MixedObservationGP(inputs.numpy(), evidence, seed=seed, settings=SETTINGS)
        if not model.diagnostics["converged"]:
            raise FloatingPointError("Usable-energy EP did not converge")
        self.exact = torch.from_numpy(model.exact_x)
        self.intervals = torch.from_numpy(model.interval_x)
        if len(self.exact):
            self.exact_factor = torch.from_numpy(np.tril(model._exact_factor[0]))
            self.alpha = torch.from_numpy(model._exact_alpha)
            self.exact_to_interval = self.kernel(self.exact, self.intervals)
        if len(self.intervals):
            self.interval_factor = torch.from_numpy(np.tril(model._interval_factor[0]))
            self.mean_shift = torch.from_numpy(model._posterior_mean - model._interval_base_mean)
            self.covariance = torch.from_numpy(model._posterior_covariance)

    def kernel(self, x, y):
        radius = (5.**.5 / SETTINGS.length_scale) * torch.linalg.vector_norm(x.unsqueeze(-2) - y, dim=-1)
        return SETTINGS.signal_variance * (1 + radius + radius.square() / 3) * torch.exp(-radius)

    def log_probability(self, features):
        # Mirror MixedObservationGP.predict with differentiable candidate features.
        x = (features - self.offset) / self.scale
        mean = x.new_full(x.shape[:-1], SETTINGS.prior_mean)
        variance = x.new_full(x.shape[:-1], SETTINGS.signal_variance)
        if len(self.exact):
            cross = self.kernel(x, self.exact)
            weights = torch.cholesky_solve(cross.unsqueeze(-1), self.exact_factor).squeeze(-1)
            mean = mean + cross @ self.alpha
            variance = variance - (cross * weights).sum(-1)
        if len(self.intervals):
            cross = self.kernel(x, self.intervals)
            if len(self.exact):
                cross = cross - weights @ self.exact_to_interval
            projection = torch.cholesky_solve(cross.unsqueeze(-1), self.interval_factor).squeeze(-1)
            mean = mean + projection @ self.mean_shift
            variance = variance - (projection * cross).sum(-1)
            variance = variance + ((projection @ self.covariance) * projection).sum(-1)
        return torch.special.log_ndtr(-mean / variance.clamp_min(1e-14).sqrt())

    def audit(self):
        return {"diagnostics": self.model.diagnostics,
                "partial_observations": [asdict(obs) for obs in self.model.observations
                                         if obs.kind == "interval"]}


class EnergyAcquisition:
    """Replace exactly one log-feasibility factor in the native acquisition."""

    def __init__(self, native, energy):
        self.native = copy(native)
        self.native.required_constraints = tuple(k for k in native.required_constraints if k != CONSTRAINT)
        self.model = native.model
        self.fit_diagnostics = {k: v for k, v in native.fit_diagnostics.items() if k != CONSTRAINT}
        self.energy = energy

    def log_acquisition(self, features):
        return self.native.log_acquisition(features) + self.energy.log_probability(features).squeeze(-1)

    def solver_probabilities(self, features):
        return self.native.solver_probabilities(features)

    def score(self, features):
        with torch.no_grad():
            score = self.log_acquisition(features.unsqueeze(-2)).numpy()
        return score + np.log(np.clip(self.solver_probabilities(features)["success"], 1e-6, 1))

    def engineering_probability(self, features):
        with torch.no_grad():
            log_p = self.energy.log_probability(features)
            x = (features - self.native.offset) / self.native.scale
            for name in self.native.required_constraints:
                pair = self.native.constraints.get(name)
                if pair is None:
                    log_p = log_p + np.log(.5)
                else:
                    model, threshold = pair
                    post = model.posterior(x)
                    z = (threshold - post.mean) / post.variance.clamp_min(1e-12).sqrt()
                    log_p = log_p + torch.special.log_ndtr(z).squeeze(-1)
            return log_p.exp().numpy()
