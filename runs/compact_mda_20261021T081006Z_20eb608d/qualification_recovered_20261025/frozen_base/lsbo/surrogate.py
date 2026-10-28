from __future__ import annotations

from copy import deepcopy
import math
import warnings
import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.means import Mean
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.ensemble import RandomForestClassifier
from linear_operator.utils.errors import NotPSDError, NanError


MIN_GP_NOISE = 1e-6


class GPTrainingError(RuntimeError):
    """All bounded attempts to fit a usable GP failed."""


def validate_gp(model, x):
    """Check the final optimizer step before acquisition uses the posterior."""
    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        noise = model.likelihood.noise
        if not torch.isfinite(noise).all() or (noise < MIN_GP_NOISE).any():
            raise FloatingPointError("Invalid GP noise variance")
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Non-finite GP parameters")
        covariance = model.likelihood(model.forward(x)).covariance_matrix
        _, info = torch.linalg.cholesky_ex(covariance)
        if not torch.isfinite(covariance).all() or (info != 0).any():
            raise NotPSDError("Final noisy training covariance is not positive definite")
        probe = torch.cat([x, x.mean(0, keepdim=True), x.amin(0, keepdim=True), x.amax(0, keepdim=True)])
        posterior = model.posterior(probe)
        if not torch.isfinite(posterior.mean).all() or not torch.isfinite(posterior.variance).all():
            raise FloatingPointError("Non-finite GP posterior")
        if (posterior.variance < 0).any():
            raise FloatingPointError("Negative GP posterior variance")
    return float(noise.min())


class FrozenPerformanceMean(Mean):
    """Freeze weights, not input derivatives; retain the head's fitted scaler."""

    def __init__(self, representation, gp_mean, gp_std, offset=None, scale=None):
        super().__init__()
        self.head = deepcopy(representation.model.performance).eval()
        self.head.requires_grad_(False)
        self.head_mean, self.head_std = representation.y_mean, representation.y_std
        self.gp_mean, self.gp_std = gp_mean, gp_std
        dimensions = self.head[0].in_features
        self.register_buffer("offset", torch.zeros(dimensions, dtype=torch.double) if offset is None else offset)
        self.register_buffer("scale", torch.ones(dimensions, dtype=torch.double) if scale is None else scale)

    def forward(self, x):
        energy = self.head(x * self.scale + self.offset).squeeze(-1) * self.head_std + self.head_mean
        return (-energy - self.gp_mean) / self.gp_std


def fit_gp(x, target, steps, mean_module=None, name="objective"):
    if x.ndim != 2 or target.shape != (len(x), 1) or len(x) < 3 or steps < 1:
        raise ValueError("GP requires N x D inputs, N x 1 targets, N >= 3 and positive steps")
    if not torch.isfinite(x).all() or not torch.isfinite(target).all():
        raise ValueError("GP training data must be finite")
    x, target = x.detach().double(), target.detach().double()
    errors = []
    # The default BoTorch 0.10 likelihood disables its noise transform.
    # Adam needs a positive transform, not just nominal parameter bounds.
    for attempt, (learning_rate, initial_noise) in enumerate(((0.04, 0.01), (0.01, 0.05)), 1):
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(MIN_GP_NOISE),
                                        noise_prior=GammaPrior(1.1, 0.05)).to(x)
        likelihood.noise = initial_noise
        model = SingleTaskGP(x, target, likelihood=likelihood, mean_module=deepcopy(mean_module),
                             outcome_transform=None,
                             covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=x.shape[-1]))).to(x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        try:
            for _ in range(steps):
                optimizer.zero_grad()
                loss = -mll(model(x), model.train_targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite GP marginal likelihood")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError("Non-finite GP training gradient")
                torch.nn.utils.clip_grad_norm_(parameters, 10)
                optimizer.step()
            noise = validate_gp(model, x)
            model.fit_diagnostics = {"name": name, "attempts": attempt, "noise_variance": noise,
                                     "learning_rate": learning_rate, "training_points": len(x),
                                     "previous_errors": errors.copy()}
            return model
        except (NotPSDError, NanError, FloatingPointError) as error:
            errors.append(f"{type(error).__name__}: {error}")
            warnings.warn(f"GP {name} fit attempt {attempt}/2 failed: {error}", RuntimeWarning)
    raise GPTrainingError(f"GP {name} failed validation after 2 attempts: {'; '.join(errors)}")


class Surrogate:
    def __init__(self, observations, representation, required_constraints, seed, steps=45, use_prior=True,
                 analytical_constraints=(), solver_domain_metric=None):
        usable = [o for o in observations if o.result.has_objective()]
        if len(usable) < 3:
            raise ValueError("At least three valid objective observations are needed")
        if not set(analytical_constraints).issubset(required_constraints):
            raise ValueError('Analytical constraints must also be required constraints')
        # Known geometry is enforced by candidate prechecks, not a noisy GP.
        self.required_constraints = tuple(name for name in required_constraints if name not in analytical_constraints)
        self.representation_version = representation.version
        x = representation.features([o.candidate for o in usable])
        n = len(representation.space.continuous)
        with torch.no_grad():
            grammar_z, _ = representation.model.encode(representation.grammar)
        self.offset = torch.cat([torch.zeros(2 * n), grammar_z.min(0).values - 0.5]).double()
        upper = torch.cat([torch.ones(2 * n), grammar_z.max(0).values + 0.5]).double()
        self.scale = (upper - self.offset).clamp_min(1e-8)
        x = (x - self.offset) / self.scale
        target = -torch.tensor([o.result.objective for o in usable], dtype=torch.double)
        self.mean, self.std = float(target.mean()), float(target.std().clamp_min(1e-8))
        y = ((target - self.mean) / self.std).unsqueeze(-1)
        prior = FrozenPerformanceMean(representation, self.mean, self.std, self.offset, self.scale) if use_prior and representation.performance_fitted else None
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = fit_gp(x, y, steps, prior)
            self.constraints = {}
            for name in self.required_constraints:
                known = [o for o in observations if o.result.solver_success and o.result.constraints.get(name) is not None
                         and math.isfinite(o.result.constraints[name])]
                if len(known) >= 3:
                    cx = (representation.features([o.candidate for o in known]) - self.offset) / self.scale
                    cy = torch.tensor([o.result.constraints[name] for o in known], dtype=torch.double)
                    center, scale = float(cy.mean()), float(cy.std().clamp_min(1e-6))
                    self.constraints[name] = (fit_gp(cx, ((cy - center) / scale).unsqueeze(-1), steps,
                                                     name=name), -center / scale)
        self.fit_diagnostics = {"objective": self.model.fit_diagnostics,
                                **{name: pair[0].fit_diagnostics for name, pair in self.constraints.items()}}
        self.domain_model = None
        domain_rows = [o for o in observations if solver_domain_metric and
                       o.result.metrics.get(solver_domain_metric) is not None and
                       math.isfinite(o.result.metrics[solver_domain_metric])]
        if len(domain_rows) >= 3:
            dx = (representation.features([o.candidate for o in domain_rows]) - self.offset) / self.scale
            dy = torch.tensor([o.result.metrics[solver_domain_metric] for o in domain_rows], dtype=torch.double)
            center, scale = float(dy.mean()), float(dy.std().clamp_min(1.))
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.domain_model = (fit_gp(dx, ((dy-center)/scale).unsqueeze(-1), steps,
                                            name='solver_discharge_domain'), -center/scale)
            self.fit_diagnostics['solver_discharge_domain'] = self.domain_model[0].fit_diagnostics
        feasible_y = [(-o.result.objective - self.mean) / self.std for o in usable
                      if o.result.feasible(required_constraints)]
        self.improvement = LogExpectedImprovement(self.model, max(feasible_y)) if feasible_y else None
        labels = np.array([int(o.result.has_objective()) for o in observations])
        self.success_model = None
        self.success_rate = float((labels.sum() + 1) / (len(labels) + 2))
        if len(set(labels)) > 1:
            self.success_model = RandomForestClassifier(n_estimators=64, min_samples_leaf=2, random_state=seed, n_jobs=1)
            self.success_model.fit(representation.features([o.candidate for o in observations]).numpy(), labels)

    def log_acquisition(self, features):
        features = (features - self.offset) / self.scale
        if self.improvement is None:
            posterior = self.model.posterior(features)
            value = 0.05 * torch.log(posterior.variance.squeeze(-1).squeeze(-1).clamp_min(1e-12))
        else:
            value = self.improvement(features)
        if self.domain_model is not None:
            model, threshold = self.domain_model
            post = model.posterior(features)
            # Advisory probability of a nonnegative-load root. It never changes
            # the physical feasibility label or invents a failed mission target.
            z = (post.mean-threshold) / post.variance.clamp_min(1e-12).sqrt()
            value = value + torch.special.log_ndtr(z).squeeze(-1).squeeze(-1)
        for name in self.required_constraints:
            pair = self.constraints.get(name)
            if pair is None:
                value = value + math.log(0.5)
            else:
                model, threshold = pair
                post = model.posterior(features)
                z = (threshold - post.mean) / post.variance.clamp_min(1e-12).sqrt()
                value = value + torch.special.log_ndtr(z).squeeze(-1).squeeze(-1)
        return value

    def solver_probabilities(self, features):
        """Separate numerical-success predictions from engineering constraints."""
        if self.success_model is not None:
            index = list(self.success_model.classes_).index(1)
            probability = self.success_model.predict_proba(features.numpy())[:, index]
        else:
            probability = np.full(len(features), self.success_rate)
        result = {'success': probability}
        if self.domain_model is not None:
            model, threshold = self.domain_model
            with torch.no_grad():
                post = model.posterior((features - self.offset) / self.scale)
                z = (post.mean - threshold) / post.variance.clamp_min(1e-12).sqrt()
                result['domain'] = torch.distributions.Normal(0., 1.).cdf(z).reshape(-1).numpy()
        return result

    def score(self, features):
        with torch.no_grad():
            score = self.log_acquisition(features.unsqueeze(-2)).numpy()
        return score + np.log(np.clip(self.solver_probabilities(features)['success'], 1e-6, 1))


class JointAcquisition(AcquisitionFunction):
    """Latent proposal or fixed-branch active-subspace optimization."""

    def __init__(self, surrogate, mask, latent, unit=None, active_only=False):
        super().__init__(surrogate.model)
        self.surrogate = surrogate
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.double))
        self.register_buffer("latent", latent.detach().double())
        self.register_buffer("unit", torch.as_tensor(unit if unit is not None else mask * 0.5, dtype=torch.double))
        self.active = torch.where(self.mask > 0)[0]
        self.active_only = active_only

    def forward(self, x):
        if self.active_only:
            unit = self.unit.expand(*x.shape[:-1], -1).clone()
            unit[..., self.active] = x
            latent = self.latent.expand(*x.shape[:-1], -1)
        else:
            unit, latent = x[..., :len(self.mask)], x[..., len(self.mask):]
        mask = self.mask.expand_as(unit)
        features = torch.cat([unit * mask, mask, latent], -1)
        return self.surrogate.log_acquisition(features)
