"""GPML EP evidence selection and prefix-only structured representations."""

from dataclasses import asdict, replace

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import log_ndtr

from ..budget import digest
from ..features import load_base
from ..risk_core.mixed_gp import MixedGPSettings, MixedObservationGP
from ..risk_core.observations import OutputObservation, validated_runs
from ..structured_encoder import (
    EncoderSettings, StructuredRepresentation, hierarchy_space, state_digest)


def log_interval_probability(lower, upper):
    if lower >= upper:
        raise ValueError("Empty standardized interval")
    if lower == -np.inf:
        return float(log_ndtr(upper))
    if upper == np.inf:
        return float(log_ndtr(-lower))
    if lower >= 0:
        log_large, log_small = log_ndtr(-lower), log_ndtr(-upper)
    else:
        log_large, log_small = log_ndtr(upper), log_ndtr(lower)
    return float(log_large+np.log(-np.expm1(log_small-log_large)))


class EvidenceGP(MixedObservationGP):
    """Expose the existing EP site's normalizer without modifying its source."""

    def _site_posterior(self, root, base_mean, tau, nu):
        self.site_tau, self.site_nu = tau.copy(), nu.copy()
        return MixedObservationGP._site_posterior(root, base_mean, tau, nu)

    def log_evidence(self):
        exact_log = 0.
        if self._exact_factor is not None:
            values = np.asarray([value.value for value in self.observations if value.kind == "exact"])
            centered = values-self.settings.prior_mean
            exact_log = float(-.5*centered@self._exact_alpha
                              -np.log(np.diag(self._exact_factor[0])).sum()
                              -.5*len(values)*np.log(2*np.pi))
        if not len(self.interval_x):
            return exact_log
        base_mean = self._interval_base_mean
        covariance = self._baseline_covariance(self.interval_x, self.interval_x)
        covariance = .5*(covariance+covariance.T)
        covariance.flat[::len(covariance)+1] += self.settings.jitter
        root = cholesky(covariance, lower=True)
        tau, nu = self.site_tau, self.site_nu
        precision = np.eye(len(tau))+root.T@(tau[:, None]*root)
        factor = cho_factor(precision, lower=True)
        linear = root.T@(nu-tau*base_mean)
        log_integral = (-np.log(np.diag(factor[0])).sum()
                        -.5*np.dot(tau, base_mean**2)+np.dot(nu, base_mean)
                        +.5*linear@cho_solve(factor, linear))
        log_sites = 0.
        for index, (lower, upper) in enumerate(self._bounds):
            marginal_var = self._posterior_covariance[index, index]
            cavity_precision = 1/marginal_var-tau[index]
            if cavity_precision <= 1e-6*max(1/marginal_var, tau[index]):
                other_tau, other_nu = tau.copy(), nu.copy()
                other_tau[index], other_nu[index] = 0., 0.
                cavity_mean_all, cavity_cov = MixedObservationGP._site_posterior(
                    root, base_mean, other_tau, other_nu)
                cavity_var = cavity_cov[index, index]
                cavity_mean = cavity_mean_all[index]
            else:
                cavity_var = 1/cavity_precision
                cavity_mean = cavity_var*(self._posterior_mean[index]/marginal_var-nu[index])
            sd = np.sqrt(cavity_var+self.settings.interval_noise**2)
            log_true = log_interval_probability((lower-cavity_mean)/sd, (upper-cavity_mean)/sd)
            denominator = 1+tau[index]*cavity_var
            log_gaussian_site = (-.5*np.log(denominator)
                +(cavity_var*nu[index]**2+2*cavity_mean*nu[index]
                  -tau[index]*cavity_mean**2)/(2*denominator))
            log_sites += log_true-log_gaussian_site
        result = float(exact_log+log_integral+log_sites)
        if not np.isfinite(result):
            raise FloatingPointError("Nonfinite EP marginal likelihood")
        return result


class LearnedMixedHead:
    """Choose kernel scale by current-prefix approximate marginal likelihood."""

    def __init__(self, inputs, observations, *, objective=False, completion=False,
                 seed=0, length_grid=(.25, .5, 1., 2.), reference_scale=1.):
        inputs = np.asarray(inputs, float)
        values = np.asarray([value.value for value in observations if value.kind == "exact"])
        self.center = float(values.mean()) if objective and len(values) else 0.
        self.scale = (max(float(values.std()), reference_scale*.01) if objective and len(values) > 1
                      else max(float(np.median(np.abs(values))), .05) if len(values)
                      else float(reference_scale))
        if completion:
            self.center, self.scale = 0., 1.
        normalized = []
        for value in observations:
            if value.kind == "exact":
                value = replace(value, value=(value.value-self.center)/self.scale)
            elif value.kind == "interval":
                value = replace(value,
                    lower=None if value.lower is None else (value.lower-self.center)/self.scale,
                    upper=None if value.upper is None else (value.upper-self.center)/self.scale)
            normalized.append(value)
        grid = tuple(float(length) for length in length_grid)
        if not grid or any(not np.isfinite(length) or length <= 0 for length in grid):
            raise ValueError("Positive finite kernel scale candidates required")
        self.candidates, models = [], []
        for length in grid:
            settings = MixedGPSettings(
                length_scale=length*np.sqrt(inputs.shape[1]),
                exact_noise=1e-3, interval_noise=1. if completion else 0., jitter=1e-8)
            try:
                model = EvidenceGP(inputs, normalized, seed=seed, settings=settings)
                evidence = model.log_evidence()
                self.candidates.append({"length": length, "log_evidence": evidence, "status": "ok"})
                models.append(model)
            except (np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
                self.candidates.append({"length": length, "log_evidence": None,
                                        "status": type(error).__name__})
                models.append(None)
        valid = [index for index, model in enumerate(models) if model is not None]
        if not valid:
            raise RuntimeError("No kernel candidate produced a valid EP posterior")
        selected = max(valid, key=lambda index: self.candidates[index]["log_evidence"])
        self.model, self.selected = models[selected], self.candidates[selected]
        self.identity = digest({"model": self.model.to_dict(), "center": self.center,
                                "scale": self.scale, "grid": self.candidates})

    def predict(self, inputs):
        prediction = self.model.predict(inputs)
        return {"mean": self.center+self.scale*prediction["mean"],
                "variance": self.scale**2*prediction["variance"]}

    def diagnostics(self):
        return {"identity": self.identity, "center": self.center, "scale": self.scale,
                "selected": self.selected, "candidates": self.candidates,
                "counts": self.model.diagnostics,
                "evidence_interpretation": "EP approximation; not a safety certificate"}


class PrefixFeatures:
    """Train the existing paper encoder directly on explicit output evidence."""

    def __init__(self, problem, base, records, seed, latent=True, epochs=120):
        if not records:
            raise ValueError("A nonempty training prefix is required")
        validated_runs(record.observation for record in records)
        if any(record.observation.contract_id != problem.contract_id for record in records):
            raise ValueError("Training problem identity differs")
        self.modules, _ = load_base(base)
        variables = [asdict(variable) for variable in problem.space.variables]
        self.space = hierarchy_space(self.modules, variables)
        self.problem, self.latent, self.encoder = problem, latent, None
        if latent:
            settings = EncoderSettings(latent_dim=4, initial_epochs=epochs, joint_view="latent")
            self.encoder = StructuredRepresentation(self.space, settings, seed)
            observations = []
            for index, record in enumerate(records, 1):
                run = record.observation
                exact = run.objective.kind == "exact"
                usable = run.status not in {"infrastructure_error", "unknown"}
                evaluation = self.modules["records"].Evaluation(
                    run.objective.value if exact and usable else None, exact and usable,
                    {name: value.value if value.kind == "exact" and usable else None
                     for name, value in run.constraints}, run.status)
                observations.append(self.modules["records"].Observation(
                    index, self.candidate(record.candidate), evaluation, "current_prefix"))
            self.encoder.fit(observations)
        raw = self._raw([record.candidate for record in records])
        self.offset, self.scale = np.zeros(raw.shape[1]), np.ones(raw.shape[1])
        if latent:
            self.offset[-4:] = raw[:, -4:].min(0)-.5
            self.scale[-4:] = raw[:, -4:].max(0)+.5-self.offset[-4:]
        self.state_hash = None if self.encoder is None else state_digest(self.encoder)
        self.identity = digest({"prefix": [record.to_dict() for record in records],
                                "schema": variables, "latent": latent, "encoder": self.state_hash,
                                "offset": self.offset.tolist(), "scale": self.scale.tolist()})

    def candidate(self, raw):
        return self.space.canonical(self.modules["space"].Candidate(**raw))

    def _raw(self, candidates):
        values = [self.candidate(candidate) for candidate in candidates]
        units = np.asarray([self.space.unit(candidate) for candidate in values])
        masks = np.asarray([self.space.mask(candidate.discrete) for candidate in values])
        parts = [units*masks, masks]
        if self.latent:
            with torch.no_grad():
                encoded, _ = self.encoder.model.encode(self.encoder._indices(values))
            parts.append(encoded.numpy())
        else:
            for variable in self.space.discrete:
                active = np.asarray([self.space.active(variable, candidate.discrete) for candidate in values])
                indicator = np.zeros((len(values), len(variable.values)))
                indicator[np.arange(len(values)), [variable.values.index(candidate.discrete[variable.name])
                                                  for candidate in values]] = 1.
                parts.extend([indicator*active[:, None], active[:, None].astype(float)])
        return np.hstack(parts)

    def transform(self, candidates):
        if self.encoder is not None and state_digest(self.encoder) != self.state_hash:
            raise RuntimeError("Encoder changed after the GP training prefix was sealed")
        result = (self._raw(candidates)-self.offset)/self.scale
        if not np.isfinite(result).all():
            raise FloatingPointError("Invalid transformed design")
        return result
