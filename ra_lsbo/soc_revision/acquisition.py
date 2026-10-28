"""Keep exploration bonuses out of incumbent-directed feasible improvement."""

import numpy as np
from scipy.special import ndtr

from ..partial_bo.selector import feasibility_probability
from ..risk_core.mixed_gp import eicb_factor
from ..risk_core.optimizer import expected_improvement


def components(forecast):
    if "components" in forecast:
        values = forecast["components"]
        return (np.asarray(values["weights"]), np.asarray(values["mean"]),
                np.asarray(values["variance"]))
    return np.ones(1), np.asarray(forecast["mean"])[None, :], np.asarray(forecast["variance"])[None, :]


def average(forecast, function):
    weights, mean, variance = components(forecast)
    return weights @ function(mean, variance)


def quantities(prediction, best, kappa, threshold):
    completion = prediction["completion"]
    success = average(completion, lambda mean, variance: ndtr(-mean/np.sqrt(1+variance)))
    upper = average(completion, lambda mean, variance: ndtr(-mean+kappa*np.sqrt(np.maximum(variance, 0.))))
    margins, probabilities, balanced = [upper-threshold], {}, {}
    joint, uncertainty = success.copy(), np.zeros(len(success))
    violation = np.maximum(threshold-upper, 0.)
    for name, forecast in prediction["constraints"].items():
        mean, variance = np.asarray(forecast["mean"]), np.asarray(forecast["variance"])
        margin = (kappa*np.sqrt(np.maximum(variance, 0.))-mean)/forecast["scale"]
        probability = average(forecast, feasibility_probability)
        factor = average(forecast, eicb_factor)
        margins.append(margin)
        probabilities[name], balanced[name] = probability, factor
        joint *= probability
        violation += np.maximum(-margin, 0.)
        uncertainty += np.sqrt(np.maximum(variance, 0.))/forecast["scale"]
    improvement = (average(prediction["objective"], lambda mean, variance:
        expected_improvement(mean, variance, best))
        if best is not None and prediction["objective"] is not None else None)
    return {"success": success, "upper": upper, "margins": np.stack(margins, axis=1),
            "probabilities": probabilities, "balanced": balanced, "joint": joint,
            "violation": violation, "uncertainty": uncertainty, "ei": improvement}


def mode_score(values, mode):
    if mode == "posterior_support_restoration":
        return -values["violation"]
    if mode == "budgeted_boundary_information":
        probability = np.clip(values["joint"], 1e-12, 1-1e-12)
        return (-probability*np.log(probability)-(1-probability)*np.log(1-probability)
                +.01*values["uncertainty"]/(1+values["uncertainty"]))
    score = np.log(np.maximum(values["success"], 1e-300))
    factors = values["probabilities"] if mode == "probability_weighted_feasible_improvement" else values["balanced"]
    for probability in factors.values():
        score += np.log(np.maximum(probability, 1e-300))
    if mode in {"balanced_feasible_improvement", "probability_weighted_feasible_improvement"}:
        score += np.log(np.maximum(values["ei"], 1e-300))
    return score


def pointwise_acquisition(prediction, best, mode, kappa, threshold):
    values = quantities(prediction, best, kappa, threshold)
    return mode_score(values, mode), values["margins"]


def choose_acquisition(prediction, best, iteration, settings, probability_exploitation):
    kappa = settings.initial_kappa+(settings.final_kappa-settings.initial_kappa)*(
        iteration/max(1, settings.horizon-1))
    values = quantities(prediction, best, kappa, settings.completion_threshold)
    supported = values["upper"] >= settings.completion_threshold
    optimistic = np.all(values["margins"][:, 1:] >= 0., axis=1)
    fallback = None
    if not supported.any():
        supported = np.isclose(values["upper"], values["upper"].max(), rtol=1e-10, atol=1e-12)
        fallback = "no_viability_support_use_best_posterior_upper_probability"
    eligible = supported & optimistic
    if not eligible.any():
        # Match the frozen fallback: among supported candidates only response violations matter.
        response_violation = np.maximum(-values["margins"][:, 1:], 0.).sum(axis=1)
        minimum = response_violation[supported].min()
        eligible = supported & np.isclose(response_violation, minimum, rtol=1e-10, atol=1e-12)
        fallback = "no_joint_optimistic_support_minimize_normalized_optimistic_violation"
    allocated_before = int(np.floor(iteration*settings.boundary_fraction))
    allocated_after = int(np.floor((iteration+1)*settings.boundary_fraction))
    boundary = allocated_after > allocated_before
    mode = "budgeted_boundary_information" if boundary else "feasible_discovery"
    if not boundary and values["ei"] is not None:
        if np.any(values["ei"][eligible] > 0):
            mode = ("probability_weighted_feasible_improvement" if probability_exploitation
                    else "balanced_feasible_improvement")
        else:
            fallback = "zero_expected_improvement_return_to_feasible_discovery"
    score = np.where(eligible, mode_score(values, mode), -np.inf)
    index = int(np.argmax(score))
    return index, {"mode": mode, "fallback": fallback, "kappa": float(kappa),
        "boundary_slot": boundary, "boundary_slots_through_iteration": allocated_after,
        "completion_probability": values["success"].tolist(), "completion_upper": values["upper"].tolist(),
        "joint_probability_independence_proxy": values["joint"].tolist(),
        "optimistic_violation": np.maximum(-values["margins"][:, 1:], 0.).sum(axis=1).tolist(),
        "eligible": eligible.tolist(), "scores": [float(value) if np.isfinite(value) else None for value in score],
        "per_constraint_probability": {name: value.tolist() for name, value in values["probabilities"].items()},
        "per_constraint_balanced_factor": {name: value.tolist() for name, value in values["balanced"].items()},
        "probability_guarantee": False,
        "kernel_probability": "componentwise mixture CDF and EI, not moment-matched Gaussian probabilities"}
