"""Use a differentiable interval completion posterior in LSBO's native search."""

import numpy as np
from scipy.linalg import cho_solve
import torch


LOG_MIN_COMPLETION = float(np.log(1e-6))


class IntervalCompletionProbability:
    """Evaluate the fitted Matern-5/2 EP posterior and preserve gradients in z."""

    def __init__(self, head):
        model = head.model
        if len(model.exact_x) or not len(model.interval_x):
            raise ValueError("Refinement expects a fitted interval-only completion model")
        self.inputs = torch.as_tensor(model.interval_x)
        self.divisor = torch.as_tensor(head.divisor * model._scales)
        self.scaled_inputs = self.inputs / torch.as_tensor(model._scales)
        self.mean = model.settings.prior_mean
        self.signal = model.settings.signal_variance
        self.factor = torch.as_tensor(np.tril(model._interval_factor[0]))
        self.mean_weights = torch.as_tensor(cho_solve(
            model._interval_factor, model._posterior_mean - model._interval_base_mean))
        self.covariance = torch.as_tensor(model._posterior_covariance)
        self.variance_floor = 1e-14 / head.scale**2

    def log_probability(self, features):
        distance = torch.linalg.vector_norm(
            features.unsqueeze(-2) / self.divisor - self.scaled_inputs, dim=-1)
        radius = 5.**.5 * distance
        kernel = self.signal * (1. + radius + radius.square() / 3.) * torch.exp(-radius)
        projection = torch.cholesky_solve(kernel.unsqueeze(-1), self.factor).squeeze(-1)
        mean = self.mean + kernel @ self.mean_weights
        variance = self.signal - (projection * kernel).sum(-1)
        variance = variance + ((projection @ self.covariance) * projection).sum(-1)
        return torch.special.log_ndtr(-mean / variance.clamp_min(self.variance_floor).sqrt())


class RiskWeightedRefinement:
    """Replace LSBO's legacy completion factor wherever it scores improvement."""

    def __init__(self, native, completion):
        self.native = native
        self.model = native.model
        self.fit_diagnostics = native.fit_diagnostics
        self.completion = completion

    def log_acquisition(self, features):
        log_completion = self.completion.log_probability(features).squeeze(-1).clamp_min(LOG_MIN_COMPLETION)
        return self.native.log_acquisition(features) + log_completion

    def solver_probabilities(self, features):
        with torch.no_grad():
            completion = self.completion.log_probability(features).exp().numpy()
        return {**self.native.solver_probabilities(features), "success": completion}

    def score(self, features):
        with torch.no_grad():
            return self.log_acquisition(features.unsqueeze(-2)).numpy()
