"""Independent literature-inspired selectors with explicit cold-start extensions."""

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

from ..budget import digest
from .adapter import CoreFeatures, packet
from .boundary_ensemble import BoundaryEnsemble
from .mixed_gp import MixedGPSettings, MixedObservationGP, eicb_factor
from .observations import OutputObservation
from .online_calibration import OnlineResidualCalibrator


FAMILIES = ("mixed_eicb", "ensemble_boundary", "calibrated_eicb")


def probability(mean, variance):
    return ndtr(-np.asarray(mean)/np.sqrt(np.maximum(variance, 1e-12)))


def expected_improvement(mean, variance, best):
    sd = np.sqrt(np.maximum(variance, 1e-12))
    z = (best-mean)/sd
    return np.maximum((best-mean)*ndtr(z)+sd*norm.pdf(z), 0.)


def sign_observation(label, key):
    if label is None:
        return OutputObservation("missing", key, reason="unobserved_completion")
    return (OutputObservation("interval", key, upper=0.) if label else
            OutputObservation("interval", key, lower=0., lower_inclusive=False))


def response_gp(x, observations, seed, objective=False):
    values = np.asarray([o.value for o in observations if o.kind == "exact"])
    center = float(values.mean()) if objective and len(values) else 0.
    if objective:
        amplitude = (max(float(values.std()), 1e-3) if len(values) > 1 else
                     max(abs(center)*.1, 1.))
    else:
        amplitude = max(float(np.median(np.abs(values))), .1) if len(values) else 1.
    settings = MixedGPSettings(
        prior_mean=center, signal_variance=amplitude**2,
        exact_noise=amplitude*1e-6, jitter=amplitude**2*1e-10)
    return MixedObservationGP(x, observations, seed=seed, settings=settings)


def observed_best(packets):
    values = [p.objective.value for p in packets
              if p.status == "completed" and p.objective.kind == "exact"
              and all(o.constraint_satisfaction() is True for _, o in p.constraints)]
    return None if not values else min(values)


class CoreSelector:
    def __init__(self, family, problem, base, seed, latent=False, stages=False):
        if family not in FAMILIES:
            raise ValueError("Unknown independently implemented method")
        self.family, self.problem, self.base, self.seed = family, problem, base, seed
        self.latent, self.stages = latent, stages
        self.calibrators = {
            n: OnlineResidualCalibrator(target_alpha=.2, learning_rate=.05, window=64)
            for n in problem.required_constraints if n not in problem.analytical_constraints}
        self.pending = None

    def _completion(self, x, query, packets):
        labels = [p.completion_label() for p in packets]
        if self.stages:
            names = sorted({s for p in packets for s in p.passed_stages}
                           | {p.failed_stage for p in packets if p.failed_stage is not None})
            if names:
                score = np.ones(len(query))
                for name in names:
                    values = [1 if name in p.passed_stages else
                              0 if p.failed_stage == name else None for p in packets]
                    model = MixedObservationGP(
                        x, [sign_observation(v, p.evidence_id+":"+name)
                            for p, v in zip(packets, values)], seed=self.seed)
                    prediction = model.predict(query)
                    score *= probability(prediction["mean"], prediction["variance"])
                return score, {"kind": "conditional_stage_survival", "heads": names}
        if self.family == "ensemble_boundary":
            model = BoundaryEnsemble(x, labels, seed=self.seed)
            result = model.predict(query)
            return result["probability"], {"kind": "ensemble_completion"}
        model = MixedObservationGP(
            x, [sign_observation(v, p.evidence_id+":completion")
                for p, v in zip(packets, labels)], seed=self.seed)
        result = model.predict(query)
        return probability(result["mean"], result["variance"]), {"kind": "aggregate_completion_gp"}

    def propose(self, rows, candidates, iteration):
        if self.pending is not None:
            raise RuntimeError("The previous forecast must be settled before proposing again")
        features = CoreFeatures(self.problem, self.base, rows, self.seed, self.latent)
        x = features.transform([r["candidate"] for r in rows])
        query = features.transform(candidates)
        packets = [packet(r) for r in rows]
        completion, completion_info = self._completion(x, query, packets)
        factor, mean_probability = completion.copy(), completion.copy()
        boundary_eligible = np.ones(len(query), dtype=bool)
        calibrated_eligible = np.ones(len(query), dtype=bool)
        forecasts, counts = {}, {}
        for number, name in enumerate(sorted(self.calibrators)):
            observations = [p.constraint(name) for p in packets]
            counts[name] = {kind: sum(o.kind == kind for o in observations)
                            for kind in ("exact", "interval", "missing")}
            model = response_gp(x, observations, seed=self.seed+number)
            prediction = model.predict(query)
            mu, var = np.asarray(prediction["mean"]), np.asarray(prediction["variance"])
            p = probability(mu, var)
            mean_probability *= p
            forecasts[name] = {"mean": mu, "sd": np.sqrt(np.maximum(var, 1e-12))}
            if self.family == "ensemble_boundary":
                classifier = BoundaryEnsemble(
                    x, [None if o.constraint_satisfaction() is None else
                        int(o.constraint_satisfaction()) for o in observations],
                    seed=self.seed+101+number)
                band = classifier.predict(query)
                boundary_eligible &= np.asarray(band["upper"]) >= .5
                factor *= np.asarray(band["probability"])
            else:
                factor *= eicb_factor(mu, var, beta=1.96)
            if self.family == "calibrated_eicb":
                q = self.calibrators[name].current_quantile()["quantile"]
                calibrated_eligible &= (mu+q*forecasts[name]["sd"]) <= 0
        best = observed_best(packets)
        objective_observations = [p.objective for p in packets]
        objective = None
        if any(o.kind == "exact" for o in objective_observations):
            objective = response_gp(
                x, objective_observations, seed=self.seed+200, objective=True)
        if best is None or objective is None:
            score, mode = factor.copy(), "cold_start_balanced_feasibility"
        else:
            response = objective.predict(query)
            ei = expected_improvement(
                np.asarray(response["mean"]), np.asarray(response["variance"]), best)
            if self.family == "ensemble_boundary":
                score, mode = ei*completion, "completion_weighted_ei_under_ensemble_upper_band"
            else:
                score, mode = ei*factor, "balanced_expected_feasible_improvement"
        fallback = None
        if self.family == "ensemble_boundary" and best is not None:
            if boundary_eligible.any():
                score = np.where(boundary_eligible, score, -np.inf)
            else:
                score, fallback = factor.copy(), "no_upper_band_candidate_discovery"
        if self.family == "calibrated_eicb" and best is not None:
            if calibrated_eligible.any():
                score = np.where(calibrated_eligible, score, -np.inf)
            else:
                fallback = "no_residual_certified_candidate_raw_policy_no_guarantee"
        if not np.isfinite(score).any():
            raise FloatingPointError("No finite acquisition candidate")
        index = int(np.argmax(score))
        forecast = {
            name: {"mean": float(values["mean"][index]), "sd": float(values["sd"][index])}
            for name, values in forecasts.items()}
        calibration = {}
        key = digest([self.seed, iteration, candidates[index]])
        if self.family == "calibrated_eicb":
            for name, values in forecast.items():
                token = self.calibrators[name].predict(values["mean"], values["sd"], key+":"+name)
                # Infinity denotes unsupported certification, not a JSON number.
                calibration[name] = {
                    k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                    for k, v in token.items() if k != "token"}
        self.pending = {"key": key, "forecast": forecast, "candidate_key": digest(candidates[index])}
        metadata = {
            "family": self.family, "latent": self.latent, "stages": self.stages,
            "feature_identity": features.identity, "best_observed": best,
            "index": index, "mode": mode, "fallback": fallback,
            "candidate_key": digest(candidates[index]), "prefix_digest": digest(rows),
            "observation_counts": counts, "completion_model": completion_info,
            "selected_completion_probability": float(completion[index]),
            "raw_independent_feasibility": float(mean_probability[index]),
            "selected_forecast": forecast, "calibration_forecast": calibration,
            "calibration_state_before": {n: c.state_dict() for n, c in self.calibrators.items()},
            "acquisition_scores": [float(v) if np.isfinite(v) else None for v in score],
            "raw_completion": completion.tolist(), "raw_feasibility": mean_probability.tolist(),
            "boundary_eligible": boundary_eligible.tolist(),
            "calibrated_eligible": calibrated_eligible.tolist(),
            "probability_guarantee": False,
        }
        return candidates[index], metadata

    def accept(self, observed):
        if self.pending is None:
            raise RuntimeError("No saved forecast to settle")
        p = packet(observed)
        if p.design_id != self.pending["candidate_key"]:
            raise ValueError("Observed design differs from the saved selection")
        if self.family == "calibrated_eicb":
            for name, calibrator in self.calibrators.items():
                o = p.constraint(name)
                calibrator.observe(self.pending["key"]+":"+name,
                                   o.value if o.kind == "exact" else None)
        self.pending = None
