"""Explicit discovery, feasible improvement and limited boundary acquisition."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import ndtr

from ..budget import digest
from ..risk_core.mixed_gp import eicb_factor
from ..risk_core.observations import OutputObservation
from ..risk_core.optimizer import expected_improvement
from .models import LearnedMixedHead, PrefixFeatures


@dataclass(frozen=True)
class PartialBOSettings:
    horizon: int = 14
    latent: bool = True
    encoder_epochs: int = 120
    boundary_fraction: float = .2
    completion_threshold: float = .5
    initial_kappa: float = 2.
    final_kappa: float = 1.
    length_grid: tuple = (.25, .5, 1., 2.)
    learn_kernel: bool = True
    trust_gate: bool = True

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in (self.horizon, self.encoder_epochs)):
            raise ValueError("Positive integer iteration limits required")
        if not 0 <= self.boundary_fraction < 1 or not 0 < self.completion_threshold < 1:
            raise ValueError("Invalid boundary fraction or completion threshold")
        if not 0 <= self.final_kappa <= self.initial_kappa or not np.isfinite(self.initial_kappa):
            raise ValueError("Invalid posterior support schedule")


def feasibility_probability(mean, variance):
    mean, variance = np.asarray(mean), np.asarray(variance)
    sd = np.sqrt(np.maximum(variance, 1e-24))
    return np.where(variance <= 1e-16, (mean <= 0).astype(float), ndtr(-mean/sd))


def completion_support(mean, variance, kappa):
    return ndtr(-np.asarray(mean)+kappa*np.sqrt(np.maximum(variance, 0.)))


def choose_acquisition(predictions, best, iteration, settings):
    """All inputs are prefix posterior forecasts; this function has no oracle."""
    completion = predictions["completion"]
    mean, variance = np.asarray(completion["mean"]), np.asarray(completion["variance"])
    success = ndtr(-mean/np.sqrt(1+variance))
    kappa = settings.initial_kappa+(settings.final_kappa-settings.initial_kappa)*(
        iteration/max(1, settings.horizon-1))
    upper = completion_support(mean, variance, kappa)
    supported = upper >= settings.completion_threshold if settings.trust_gate else np.ones(len(mean), bool)
    optimistic, violation = np.ones(len(mean), bool), np.zeros(len(mean))
    log_balanced, joint = np.log(np.maximum(success, 1e-300)), success.copy()
    uncertainty = np.zeros(len(mean))
    for forecast in predictions["constraints"].values():
        constraint_mean = np.asarray(forecast["mean"])
        constraint_variance = np.asarray(forecast["variance"])
        sd = np.sqrt(np.maximum(constraint_variance, 0.))
        lower = constraint_mean-kappa*sd
        optimistic &= lower <= 0
        violation += np.maximum(lower/forecast["scale"], 0.)
        factor = eicb_factor(constraint_mean, constraint_variance)
        log_balanced += np.log(np.maximum(factor, 1e-300))
        joint *= feasibility_probability(constraint_mean, constraint_variance)
        uncertainty += sd/forecast["scale"]
    fallback = None
    if not supported.any():
        supported = np.isclose(upper, upper.max(), rtol=1e-10, atol=1e-12)
        fallback = "no_viability_support_use_best_posterior_upper_probability"
    eligible = supported & optimistic
    if not eligible.any():
        minimum = violation[supported].min()
        eligible = supported & np.isclose(violation, minimum, rtol=1e-10, atol=1e-12)
        fallback = "no_joint_optimistic_support_minimize_normalized_optimistic_violation"
    allocated_before = int(np.floor(iteration*settings.boundary_fraction))
    allocated_after = int(np.floor((iteration+1)*settings.boundary_fraction))
    boundary = allocated_after > allocated_before
    score, mode = log_balanced.copy(), "feasible_discovery"
    if boundary:
        probability = np.clip(joint, 1e-12, 1-1e-12)
        score = -probability*np.log(probability)-(1-probability)*np.log(1-probability)
        score += .01*uncertainty/(1+uncertainty)
        mode = "budgeted_boundary_information"
    elif best is not None and predictions["objective"] is not None:
        objective = predictions["objective"]
        improvement = expected_improvement(
            np.asarray(objective["mean"]), np.asarray(objective["variance"]), best)
        if np.any(improvement[eligible] > 0):
            score += np.log(np.maximum(improvement, 1e-300))
            mode = "balanced_feasible_improvement"
        else:
            fallback = "zero_expected_improvement_return_to_feasible_discovery"
    score = np.where(eligible, score, -np.inf)
    index = int(np.argmax(score))
    return index, {"mode": mode, "fallback": fallback, "kappa": float(kappa),
                   "boundary_slot": boundary, "boundary_slots_through_iteration": allocated_after,
                   "completion_probability": success.tolist(), "completion_upper": upper.tolist(),
                   "joint_probability_independence_proxy": joint.tolist(),
                   "optimistic_violation": violation.tolist(), "eligible": eligible.tolist(),
                   "scores": [float(value) if np.isfinite(value) else None for value in score],
                   "probability_guarantee": False}


class PartialLatentBO:
    def __init__(self, problem, base_project, seed=0, settings=None):
        self.problem, self.base, self.seed = problem, base_project, seed
        self.settings = settings or PartialBOSettings()
        self.pending = None

    def propose(self, records, candidates, iteration):
        if self.pending is not None:
            raise RuntimeError("Settle the previous proposal before updating its prefix")
        if type(iteration) is not int or not 0 <= iteration < self.settings.horizon:
            raise ValueError("Iteration outside the registered horizon")
        if not candidates or len({digest(candidate) for candidate in candidates}) != len(candidates):
            raise ValueError("Nonempty unique candidate pool required")
        if any(digest(candidate) in {record.observation.design_id for record in records}
               for candidate in candidates):
            raise ValueError("Previously evaluated design in candidate pool")
        for candidate in candidates:
            if self.problem.canonical(candidate) != candidate or any(
                    value > 0 for value in self.problem.precheck(candidate).values()):
                raise ValueError("Pool contains a noncanonical or analytically invalid design")
        features = PrefixFeatures(self.problem, self.base, records, self.seed,
                                  self.settings.latent, self.settings.encoder_epochs)
        training = features.transform([record.candidate for record in records])
        query = features.transform(candidates)
        packets = [record.observation for record in records]
        missing = lambda packet, name: OutputObservation(
            "missing", packet.evidence_id+":"+name, reason="untrusted_terminal_status")
        usable = lambda packet: packet.status not in {"infrastructure_error", "unknown"}
        completion_observations = []
        for packet in packets:
            label = packet.completion_label()
            completion_observations.append(missing(packet, "completion") if label is None else
                OutputObservation("interval", packet.evidence_id+":completion", upper=0.) if label else
                OutputObservation("interval", packet.evidence_id+":completion", lower=0., lower_inclusive=False))
        grid = self.settings.length_grid if self.settings.learn_kernel else (.5,)
        completion_head = LearnedMixedHead(
            training, completion_observations, completion=True, seed=self.seed, length_grid=grid)
        prediction = {"completion": completion_head.predict(query), "constraints": {}, "objective": None}
        diagnostics = {"completion": completion_head.diagnostics()}
        for number, name in enumerate(self.problem.required_constraints):
            if name in self.problem.analytical_constraints:
                continue
            observations = [packet.constraint(name) if usable(packet) else missing(packet, name)
                            for packet in packets]
            head = LearnedMixedHead(training, observations, seed=self.seed+number+1, length_grid=grid)
            prediction["constraints"][name] = {**head.predict(query), "scale": head.scale}
            diagnostics[name] = head.diagnostics()
        objective_observations = [packet.objective if usable(packet) else missing(packet, "objective")
                                  for packet in packets]
        if any(value.kind != "missing" for value in objective_observations):
            head = LearnedMixedHead(training, objective_observations, objective=True,
                reference_scale=getattr(self.problem, "objective_reference_scale", 1.),
                seed=self.seed+100, length_grid=grid)
            prediction["objective"] = head.predict(query)
            diagnostics["objective"] = head.diagnostics()
        best = min((record.observation.objective.value for record in records if record.usable_feasible),
                   default=None)
        index, acquisition = choose_acquisition(prediction, best, iteration, self.settings)
        self.pending = digest(candidates[index]), self.problem.contract_id
        forecasts = {"completion": {key: value.tolist() for key, value in prediction["completion"].items()},
                     "constraints": {name: {key: value.tolist() if isinstance(value, np.ndarray) else value
                                            for key, value in forecast.items()}
                                     for name, forecast in prediction["constraints"].items()},
                     "objective": None if prediction["objective"] is None else
                         {key: value.tolist() for key, value in prediction["objective"].items()}}
        return candidates[index], {"index": index, "best_observed": best,
            "settings": asdict(self.settings), "feature_identity": features.identity,
            "history_digest": digest([record.to_dict() for record in records]),
            "heads": diagnostics, "forecasts": forecasts, **acquisition}

    def accept(self, record):
        identity = record.observation.design_id, record.observation.contract_id
        if self.pending is None or identity != self.pending:
            raise ValueError("Settled design or contract differs from the pending proposal")
        self.pending = None
