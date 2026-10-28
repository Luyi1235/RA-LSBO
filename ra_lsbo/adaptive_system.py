"""Experimental online risk learning and adaptive local candidate selection."""

from dataclasses import asdict, dataclass, replace

import numpy as np

from .adaptive_risk import JointRiskEnsemble, JointRiskSettings, ReferenceTemperature
from .budget import digest
from .features import history_identity, validate_history
from .selection import candidate_pool, incumbent, log_expected_improvement, normalized
from .structured_encoder import StructuredFeatures
from .structured_models import StructuredResponseHeads


ADAPTIVE_METHODS = ("adaptive_full", "adaptive_aggregate", "adaptive_onehot", "adaptive_no_auxiliary",
                    "adaptive_no_calibration", "adaptive_no_information", "adaptive_no_trust")


class AdaptiveLearning:
    def __init__(self, method, settings, reference_ids, seed=0, risk_settings=None):
        if method not in ADAPTIVE_METHODS:
            raise ValueError("Unknown experimental method")
        self.method, self.settings, self.seed = method, settings, seed
        self.reference_ids = frozenset(reference_ids)
        if not self.reference_ids:
            raise ValueError("Declare a fixed held-out reference before learning")
        self.risk_settings = replace(
            risk_settings or JointRiskSettings(), aggregate=method == "adaptive_aggregate",
            latent=method != "adaptive_onehot", auxiliary=method != "adaptive_no_auxiliary")
        self.features, self.reference_identity = None, None

    def fit(self, rows, variables, base_project):
        validate_history(rows)
        reference = [r for r in rows if r["evaluation"] in self.reference_ids]
        training = [r for r in rows if r["evaluation"] not in self.reference_ids]
        if {r["evaluation"] for r in reference} != self.reference_ids:
            raise ValueError("Fixed calibration reference is missing")
        identity = history_identity(reference)
        if self.reference_identity is not None and identity != self.reference_identity:
            raise ValueError("Calibration reference was modified")
        self.reference_identity = identity
        update = "initial"
        if self.features is None:
            self.features = StructuredFeatures(base_project, variables, self.seed,
                                               self.settings.encoder()).fit(training)
        else:
            self.features.update(training)
            update = "append_all_available_nonreference_labels"
        risks = JointRiskEnsemble(training, self.features, self.seed, self.risk_settings)
        responses = StructuredResponseHeads(training, self.features, self.seed, self.settings.gp_steps)
        # Fit the same transform in the ablation; only its application is switched off.
        calibrated = ReferenceTemperature(risks, reference)
        metadata = {
            "method": self.method, "training_indices": self.features.training_indices,
            "training_history_id": self.features.history_id, "reference_indices": sorted(self.reference_ids),
            "reference_history_id": identity, "encoder_update": update,
            "representation_snapshot": self.features.snapshot_id, "model_id": risks.model_id,
            "risk_settings": asdict(self.risk_settings), "risk_diagnostics": risks.diagnostics,
            "response_diagnostics": responses.diagnostics, "calibration": calibrated.to_dict(),
            "calibration_id": calibrated.calibration_id,
            "apply_calibration": self.method != "adaptive_no_calibration"}
        return (risks, responses, calibrated), metadata


@dataclass
class LocalRegion:
    width: float = .4
    successes: int = 0
    failures: int = 0
    restarts: int = 0

    def observe(self, previous_best, risk):
        best = risk["objective"] if risk["feasible_label"] is True else None
        improved = best is not None and (
            previous_best is None or best < previous_best-1e-6*max(abs(previous_best), 1.))
        if improved:
            self.successes += 1
            self.failures = 0
            if self.successes >= 2:
                self.width, self.successes = min(1., self.width*1.5), 0
        else:
            self.failures += 1
            self.successes = 0
            if self.failures >= 4:
                self.width, self.failures = self.width/2., 0
        if self.width < .025:
            self.width, self.restarts = .4, self.restarts+1


def local_pool(problem, rng, count, seen, rows, region, enabled=True):
    """Three local candidates per global candidate, with canonical duplicate keys."""
    if not enabled:
        return candidate_pool(problem.space, rng, count, seen, rows, problem.precheck)
    feasible = sorted((r for r in rows if r["risk"]["feasible_label"] is True),
                      key=lambda r: r["risk"]["objective"])
    result, keys = [], set(seen)
    rejected = {"duplicate": 0, "cheap_constraint": 0}
    for attempt in range(count*100):
        if len(result) >= count:
            break
        if feasible and attempt % 4:
            anchor = problem.candidate_class(**feasible[0]["candidate"])
            unit = problem.space.unit(anchor)
            lower, upper = np.maximum(0., unit-region.width/2), np.minimum(1., unit+region.width/2)
            # Category moves are proposed without assigning fictitious coordinates to inactive children.
            discrete = (problem.space.sample(rng).discrete if attempt % 7 == 0 else anchor.discrete)
            candidate = problem.space.from_unit(rng.uniform(lower, upper), discrete)
            source = "adaptive_local"
        else:
            candidate, source = problem.space.sample(rng), "global_coverage"
        raw = problem.canonical(candidate.to_dict())
        key = digest(raw)
        if key in keys:
            rejected["duplicate"] += 1
            continue
        if any(not np.isfinite(v) or v > 0 for v in problem.precheck(raw).values()):
            rejected["cheap_constraint"] += 1
            continue
        result.append({"candidate": raw, "source": source})
        keys.add(key)
    return result, rejected


def select_adaptive(pool, rows, models, iteration, method, settings, observed_failures,
                    expected_risk, information_used):
    if not pool:
        return None, {"reason": "empty_pool"}
    risks, responses, calibrated = models
    candidates = [p["candidate"] for p in pool]
    forecast = risks.predict(candidates)
    raw = np.asarray(forecast["probabilities"])
    transformed = calibrated.transform(raw, risks.model_id)
    probability = (raw if method == "adaptive_no_calibration" else transformed)[:, 0]
    log_ei = log_expected_improvement(responses, candidates, incumbent(rows))
    if np.isfinite(log_ei).any():
        relative_ei = np.exp(log_ei-np.max(log_ei))
    else:
        predicted = responses.predict(candidates).get("objective")
        relative_ei = (np.ones(len(pool)) if predicted is None else normalized(predicted["std"]))
    remaining = settings.calls-iteration
    expected_allowance = settings.calls/2.
    # This is a soft price, separate from the hard call and observed-failure limits.
    expected_price = remaining/max(.5, expected_allowance-expected_risk)
    observed_price = remaining/max(1., settings.failure_cap-observed_failures)
    price = float(np.clip(.25*max(expected_price, observed_price), .25, 8.))
    development = relative_ei*probability-price*(1.-probability)
    information = np.asarray(forecast["information"])/np.log(len(risks.events))
    explore = (method != "adaptive_no_information" and iteration % 4 == 0
               and iteration < (settings.calls+1)//2 and information_used < settings.boundary_cap)
    if explore:
        utility = information*(.25+relative_ei+.25*probability)-.1*price*(1.-probability)
        mode = "information"
    else:
        utility, mode = development, "development"
    index = int(np.argmax(utility))
    proposal = dict(pool[index], mode=mode)
    return proposal, {
        "reason": "experimental_soft_risk_price", "mode": mode, "index": index,
        "pool_size": len(pool), "candidate_key": digest(candidates[index]),
        "model_id": risks.model_id, "calibration_id": calibrated.calibration_id,
        "history_id": history_identity(rows), "utility": float(utility[index]),
        "raw_events": raw[index].tolist(), "transformed_events": transformed[index].tolist(),
        "raw_final": float(raw[index, 0]), "used_final": float(probability[index]),
        "entropy": forecast["total_entropy"][index], "information": forecast["information"][index],
        "estimated_risk_debit": float(1.-probability[index]), "risk_price": price,
        "estimated_risk_before": expected_risk, "estimated_risk_allowance": expected_allowance,
        "calibration_applied": method != "adaptive_no_calibration",
        "calibration_status": calibrated.status, "unsupported_events": forecast["unsupported_events"],
        "risk_guarantee": False}
