"""Use the incremental evidence from energy lower bounds in native LSBO refinement."""

from dataclasses import replace

import numpy as np
from scipy.linalg import cho_solve
import torch

from aircraft_timing_energy_value import energy_evidence
from ra_lsbo.risk_core.completion_refinement import LOG_MIN_COMPLETION
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings, MixedObservationGP


class LogEnergyImprovement:
    """Differentiable probability of improvement for the fitted mixed GP."""

    def __init__(self, model, divisor):
        self.model = model
        scales = model._scales
        self.divisor = torch.as_tensor(divisor * scales)
        self.exact = torch.as_tensor(model.exact_x / scales)
        self.intervals = torch.as_tensor(model.interval_x / scales)
        self.exact_alpha = torch.as_tensor(model._exact_alpha)
        unit = np.eye(len(self.exact))
        self.exact_precision = torch.as_tensor(cho_solve(model._exact_factor, unit))
        if len(self.intervals):
            self.cross = torch.as_tensor(model._kernel(model.exact_x, model.interval_x))
            self.interval_precision = torch.as_tensor(cho_solve(
                model._interval_factor, np.eye(len(self.intervals))))
            self.interval_weights = self.interval_precision @ torch.as_tensor(
                model._posterior_mean - model._interval_base_mean)
            self.interval_covariance = torch.as_tensor(model._posterior_covariance)

    def kernel(self, features, centres):
        radius = 5.**.5 * torch.linalg.vector_norm(features.unsqueeze(-2) - centres, dim=-1)
        return self.model.settings.signal_variance * (1. + radius + radius.square()/3.) * torch.exp(-radius)

    def log_probability(self, features):
        features = features / self.divisor
        exact = self.kernel(features, self.exact)
        exact_projection = exact @ self.exact_precision
        mean = self.model.settings.prior_mean + exact @ self.exact_alpha
        variance = self.model.settings.signal_variance - (exact * exact_projection).sum(-1)
        if len(self.intervals):
            cross = self.kernel(features, self.intervals) - exact_projection @ self.cross
            projection = cross @ self.interval_precision
            mean = mean + cross @ self.interval_weights
            variance = variance - (cross * projection).sum(-1)
            variance = variance + (projection @ self.interval_covariance * projection).sum(-1)
        return torch.special.log_ndtr(-mean / variance.clamp_min(1e-14).sqrt())


def fit_energy_update(history, features, divisor, seed):
    evidence = [energy_evidence(record) for record in history]
    exact = [i for i, item in enumerate(evidence) if item is not None and item[0].kind == "exact"]
    bounds = [i for i, item in enumerate(evidence) if item is not None and item[0].kind == "interval"]
    if not exact or not bounds:
        return None
    best = min(record.observation.objective.value for record in history if record.usable_feasible)
    numeric = [replace(evidence[i][0], value=np.log(evidence[i][0].value / best)) for i in exact]
    censored = [replace(evidence[i][0], lower=np.log(evidence[i][0].lower / best)) for i in bounds]
    settings = MixedGPSettings(stable_tail_moments=True)
    prior = MixedObservationGP(features[exact] / divisor, numeric, seed=seed, settings=settings)
    updated = MixedObservationGP(features[exact + bounds] / divisor, numeric + censored,
                                 seed=seed, settings=settings)
    return LogEnergyImprovement(prior, divisor), LogEnergyImprovement(updated, divisor), len(bounds)


class CensoredEnergyRefinement:
    """Adjust the native acquisition by the log ratio of improvement probabilities."""

    def __init__(self, baseline, prior, updated):
        self.baseline, self.prior, self.updated = baseline, prior, updated
        self.model, self.fit_diagnostics = baseline.model, baseline.fit_diagnostics

    def log_acquisition(self, features):
        prior = self.prior.log_probability(features).clamp_min(LOG_MIN_COMPLETION)
        updated = self.updated.log_probability(features).clamp_min(LOG_MIN_COMPLETION)
        return self.baseline.log_acquisition(features) + (updated - prior).squeeze(-1)

    def score(self, features):
        with torch.no_grad():
            return self.log_acquisition(features.unsqueeze(-2)).numpy()

    def solver_probabilities(self, features):
        return self.baseline.solver_probabilities(features)
