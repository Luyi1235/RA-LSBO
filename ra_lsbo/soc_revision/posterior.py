"""Prefix EP evidence averaging; these posteriors are not safety certificates."""

import numpy as np

from ..budget import digest
from ..partial_bo.models import LearnedMixedHead, PrefixFeatures
from ..partial_refinement.posterior import (
    GroupedMixedHead, RefinementPosterior, group_divisors)
from ..risk_core.observations import OutputObservation


class AveragedGroupedHead:
    """Average the frozen grouped heads under a uniform discrete kernel prior.

    Diagnostics retain every grid entry, including failed fits with zero weight.
    Prediction components contain every valid fit in grid order, even when its
    weight underflows to zero. No failed fit supplies a numeric prediction.
    """

    def __init__(self, inputs, observations, *, continuous_count, seed=0,
                 continuous_grid=(.05, .15, .4, 1.), discrete_grid=(.5, 1.5), **kwargs):
        inputs = np.asarray(inputs, float)
        if inputs.ndim != 2 or not np.isfinite(inputs).all():
            raise ValueError("Finite two-dimensional training inputs required")
        continuous_grid, discrete_grid = tuple(continuous_grid), tuple(discrete_grid)
        if not continuous_grid or not discrete_grid:
            raise ValueError("Nonempty grouped kernel grids required")
        observations = tuple(observations)
        self.candidates, self.heads, self.divisors = [], [], []
        for continuous in continuous_grid:
            for discrete in discrete_grid:
                divisor = group_divisors(inputs.shape[1], continuous_count, continuous, discrete)
                entry = {"continuous_length": continuous, "discrete_length": discrete}
                head = None
                try:
                    head = LearnedMixedHead(inputs/divisor, observations, seed=seed,
                        length_grid=(1/np.sqrt(inputs.shape[1]),), **kwargs)
                    evidence = float(head.selected["log_evidence"])
                    if (not np.isfinite(evidence) or not np.isfinite(head.center)
                            or not np.isfinite(head.scale) or head.scale <= 0):
                        raise FloatingPointError("Nonfinite EP evidence or invalid output normalization")
                    entry.update(status="ok", log_evidence=evidence)
                except (np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
                    head = None
                    entry.update(status=type(error).__name__, log_evidence=None,
                                 error=str(error))
                self.heads.append(head)
                self.divisors.append(divisor)
                self.candidates.append(entry)
        self.valid_indices = np.asarray([
            index for index, head in enumerate(self.heads) if head is not None], dtype=int)
        if not len(self.valid_indices):
            raise RuntimeError("No grouped kernel produced a valid EP posterior")
        evidence = np.asarray([self.candidates[index]["log_evidence"]
                               for index in self.valid_indices])
        with np.errstate(over="ignore", under="ignore"):
            relative = np.exp(evidence-evidence.max())
        self.weights = np.zeros(len(self.heads), dtype=float)
        self.weights[self.valid_indices] = relative/relative.sum()
        for entry, weight in zip(self.candidates, self.weights):
            entry["weight"] = float(weight)
        selected = int(self.valid_indices[np.argmax(evidence)])
        self.head, self.divisor = self.heads[selected], self.divisors[selected]
        self.selected = self.candidates[selected]
        self.center, self.scale = self.head.center, self.head.scale
        self.n_features = inputs.shape[1]
        self.identity = digest({
            "method": "approximate EP evidence averaging", "prior": "uniform discrete",
            "heads": [None if head is None else head.identity for head in self.heads],
            "divisors": [divisor.tolist() for divisor in self.divisors],
            "grid": self.candidates, "weights": self.weights.tolist()})

    def predict(self, inputs):
        inputs = np.asarray(inputs, float)
        if (inputs.ndim != 2 or inputs.shape[1] != self.n_features
                or not np.isfinite(inputs).all()):
            raise ValueError("Finite prediction inputs matching training dimension required")
        means, variances = [], []
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for index in self.valid_indices:
                prediction = self.heads[index].predict(inputs/self.divisors[index])
                mean = np.asarray(prediction["mean"], float)
                variance = np.asarray(prediction["variance"], float)
                if mean.shape != (len(inputs),) or variance.shape != (len(inputs),):
                    raise ValueError("Each component prediction must have shape (N,)")
                if not np.isfinite(mean).all() or not np.isfinite(variance).all():
                    raise FloatingPointError("Nonfinite grouped component prediction")
                tolerance = 64*np.finfo(float).eps*max(1., self.heads[index].scale**2)
                if np.any(variance < -tolerance):
                    raise FloatingPointError("Materially negative component predictive variance")
                means.append(mean)
                variances.append(np.maximum(variance, 0.))
            means, variances = np.stack(means), np.stack(variances)
            weights = self.weights[self.valid_indices].copy()
            mean = np.sum(weights[:, None]*means, axis=0)
            variance = np.sum(weights[:, None]*(variances+(means-mean)**2), axis=0)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise FloatingPointError("Nonfinite averaged predictive moments")
        return {"mean": mean, "variance": variance,
                "components": {"weights": weights, "mean": means, "variance": variances}}

    def diagnostics(self):
        return {"identity": self.identity, "center": self.center, "scale": self.scale,
                "counts": self.head.model.diagnostics, "selected": self.selected,
                "candidates": self.candidates, "weights": self.weights.tolist(),
                "divisors": [divisor.tolist() for divisor in self.divisors],
                "kernel": "Matern-5/2 with independent normalized continuous/mask/discrete block metrics",
                "evidence_interpretation": (
                    "approximate EP evidence averaging under a uniform discrete prior; "
                    "not a safety certificate")}


class RiskPosterior(RefinementPosterior):
    """Use the frozen evidence protocol with optional grouped kernel averaging."""

    def __init__(self, problem, base, records, seed, settings):
        self.problem = problem
        self.features = PrefixFeatures(problem, base, records, seed, settings.latent, settings.encoder_epochs)
        inputs = self.features.transform([record.candidate for record in records])
        packets = [record.observation for record in records]

        def fit(observations, **kwargs):
            head = AveragedGroupedHead if settings.average_kernel else GroupedMixedHead
            return head(inputs, observations, continuous_count=len(problem.space.continuous),
                continuous_grid=settings.continuous_grid, discrete_grid=settings.discrete_grid, **kwargs)

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
        self.best = min((record.observation.objective.value for record in records
                         if record.usable_feasible and record.observation.objective.kind == "exact"),
                        default=None)
        self.history_digest = digest([record.to_dict() for record in records])
