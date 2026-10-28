"""Prefix-only group metrics without changing the frozen mixed GP likelihood."""

import numpy as np
from scipy.special import ndtr

from ..budget import digest
from ..partial_bo.models import LearnedMixedHead, PrefixFeatures
from ..partial_bo.selector import completion_support, feasibility_probability
from ..risk_core.mixed_gp import eicb_factor
from ..risk_core.observations import OutputObservation
from ..risk_core.optimizer import expected_improvement


def group_divisors(dimension, continuous_count, continuous_length, discrete_length):
    if (type(dimension) is not int or type(continuous_count) is not int
            or continuous_count < 1 or dimension <= 2*continuous_count
            or not np.isfinite(continuous_length) or continuous_length <= 0
            or not np.isfinite(discrete_length) or discrete_length <= 0):
        raise ValueError("Invalid structured kernel groups")
    return np.r_[np.full(continuous_count, continuous_length*np.sqrt(continuous_count)),
                 np.full(continuous_count, np.sqrt(continuous_count)),
                 np.full(dimension-2*continuous_count,
                         discrete_length*np.sqrt(dimension-2*continuous_count))]


class GroupedMixedHead:
    """Select continuous/discrete block lengths by the same prefix EP evidence."""

    def __init__(self, inputs, observations, *, continuous_count, seed=0,
                 continuous_grid=(.05, .15, .4, 1.), discrete_grid=(.5, 1.5), **kwargs):
        inputs = np.asarray(inputs, float)
        self.candidates, fitted = [], []
        for continuous in continuous_grid:
            for discrete in discrete_grid:
                divisor = group_divisors(inputs.shape[1], continuous_count, continuous, discrete)
                entry = {"continuous_length": continuous, "discrete_length": discrete}
                try:
                    head = LearnedMixedHead(inputs/divisor, observations, seed=seed,
                        length_grid=(1/np.sqrt(inputs.shape[1]),), **kwargs)
                    entry.update(status="ok", log_evidence=head.selected["log_evidence"])
                    fitted.append((head, divisor))
                except (np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
                    entry.update(status=type(error).__name__, log_evidence=None)
                    fitted.append(None)
                self.candidates.append(entry)
        valid = [index for index, head in enumerate(fitted) if head is not None]
        if not valid:
            raise RuntimeError("No grouped kernel produced a valid EP posterior")
        selected = max(valid, key=lambda index: self.candidates[index]["log_evidence"])
        self.head, self.divisor = fitted[selected]
        self.selected = self.candidates[selected]
        self.scale = self.head.scale
        self.identity = digest({"head": self.head.identity, "divisor": self.divisor.tolist(),
                                "grid": self.candidates})

    def predict(self, inputs):
        return self.head.predict(np.asarray(inputs)/self.divisor)

    def diagnostics(self):
        return {"identity": self.identity, "center": self.head.center, "scale": self.scale,
                "selected": self.selected, "candidates": self.candidates,
                "divisors": self.divisor.tolist(), "counts": self.head.model.diagnostics,
                "kernel": "Matern-5/2 with independent normalized continuous/mask/discrete block metrics",
                "evidence_interpretation": "prefix EP approximation, no safety guarantee"}


class RefinementPosterior:
    def __init__(self, problem, base, records, seed, settings):
        self.problem = problem
        self.features = PrefixFeatures(problem, base, records, seed, settings.latent, settings.encoder_epochs)
        inputs = self.features.transform([record.candidate for record in records])
        packets = [record.observation for record in records]

        def fit(observations, **kwargs):
            if settings.kernel == "grouped":
                return GroupedMixedHead(inputs, observations,
                    continuous_count=len(problem.space.continuous),
                    continuous_grid=settings.continuous_grid, discrete_grid=settings.discrete_grid, **kwargs)
            return LearnedMixedHead(inputs, observations, **kwargs)

        def missing(packet, name):
            return OutputObservation("missing", packet.evidence_id+":"+name, reason="untrusted_terminal_status")

        def supported(packet, value, name):
            return value if packet.status not in {"unknown", "infrastructure_error"} else missing(packet, name)

        labels = []
        for packet in packets:
            label = packet.completion_label()
            labels.append(missing(packet, "completion") if label is None else
                OutputObservation("interval", packet.evidence_id+":completion", upper=0.) if label else
                OutputObservation("interval", packet.evidence_id+":completion", lower=0., lower_inclusive=False))
        self.completion = fit(labels, seed=seed, completion=True)
        self.constraints = {name: fit([supported(packet, packet.constraint(name), name) for packet in packets],
                seed=seed+number+1) for number, name in enumerate(problem.required_constraints)
                if name not in problem.analytical_constraints}
        observations = [supported(packet, packet.objective, "objective") for packet in packets]
        self.objective = fit(observations, seed=seed+100, objective=True,
            reference_scale=getattr(problem, "objective_reference_scale", 1.)) if any(
                value.kind != "missing" for value in observations) else None
        self.best = min((record.observation.objective.value for record in records if record.usable_feasible), default=None)
        self.history_digest = digest([record.to_dict() for record in records])

    def predict(self, candidates):
        inputs = self.features.transform(candidates)
        return {"completion": self.completion.predict(inputs),
                "constraints": {name: {**head.predict(inputs), "scale": head.scale}
                                for name, head in self.constraints.items()},
                "objective": None if self.objective is None else self.objective.predict(inputs)}

    def diagnostics(self):
        heads = {"completion": self.completion.diagnostics(),
                 **{name: head.diagnostics() for name, head in self.constraints.items()}}
        if self.objective is not None:
            heads["objective"] = self.objective.diagnostics()
        return {"feature_identity": self.features.identity, "history_digest": self.history_digest,
                "heads": heads, "best_observed": self.best}


def pointwise_acquisition(prediction, best, mode, kappa, completion_threshold):
    """A batch-independent smooth objective and support margins for SLSQP."""
    completed = prediction["completion"]
    mean, variance = np.asarray(completed["mean"]), np.asarray(completed["variance"])
    success = ndtr(-mean/np.sqrt(1+variance))
    upper = completion_support(mean, variance, kappa)
    margins = [upper-completion_threshold]
    score, joint, uncertainty = np.log(np.maximum(success, 1e-300)), success.copy(), np.zeros(len(mean))
    violation = np.maximum(completion_threshold-upper, 0.)
    for forecast in prediction["constraints"].values():
        constraint_mean = np.asarray(forecast["mean"])
        constraint_variance = np.asarray(forecast["variance"])
        sd = np.sqrt(np.maximum(constraint_variance, 0.))
        margin = (kappa*sd-constraint_mean)/forecast["scale"]
        margins.append(margin)
        violation += np.maximum(-margin, 0.)
        score += np.log(np.maximum(eicb_factor(constraint_mean, constraint_variance), 1e-300))
        joint *= feasibility_probability(constraint_mean, constraint_variance)
        uncertainty += sd/forecast["scale"]
    if mode == "posterior_support_restoration":
        score = -violation
    elif mode == "budgeted_boundary_information":
        probability = np.clip(joint, 1e-12, 1-1e-12)
        score = -probability*np.log(probability)-(1-probability)*np.log(1-probability)
        score += .01*uncertainty/(1+uncertainty)
    elif mode == "balanced_feasible_improvement" and prediction["objective"] is not None and best is not None:
        objective = prediction["objective"]
        score += np.log(np.maximum(expected_improvement(
            np.asarray(objective["mean"]), np.asarray(objective["variance"]), best), 1e-300))
    return np.asarray(score), np.stack(margins, axis=1)
