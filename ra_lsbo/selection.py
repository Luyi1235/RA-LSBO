"""Finite candidate-pool acquisition with final-candidate budget approval."""

from collections import Counter

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement

from .budget import digest
from .records import finite_number


def incumbent(rows):
    valid = [row["risk"]["objective"] for row in rows if row["risk"]["feasible_label"] is True]
    return min(valid) if valid else None


def log_expected_improvement(responses, candidates, best):
    if best is None or "objective" not in responses.models:
        return np.full(len(candidates), -np.inf)
    model, center, scale, _ = responses.models["objective"]
    x = (responses.features.transform(candidates) - responses.offset) / responses.scale
    with torch.no_grad():
        log_ei = LogExpectedImprovement(model, best_f=(-best-center)/scale)(x.unsqueeze(-2))
    return log_ei.numpy() + np.log(scale)


def normalized(values):
    values = np.asarray(values, dtype=float)
    maximum = values.max() if len(values) else 0.
    return values / maximum if maximum > 0 else np.zeros_like(values)


def candidate_pool(space, rng, count, seen, rows, precheck, max_attempts=10000):
    elite = sorted((row for row in rows if row["risk"]["feasible_label"] is True),
                   key=lambda row: row["risk"]["objective"])[:4]
    result, keys, rejected = [], set(seen), Counter()
    for attempt in range(max_attempts):
        if len(result) >= count:
            break
        if elite and attempt % 2:
            anchor = elite[int(rng.integers(len(elite)))]["candidate"]
            candidate_class = type(space.default())
            anchor = space.canonical(candidate_class(**anchor))
            unit = np.clip(space.unit(anchor) + rng.normal(0, .1, len(space.continuous)), 0., 1.)
            candidate = space.from_unit(unit, anchor.discrete)
            source = "elite_local"
        else:
            candidate, source = space.sample(rng), "global_coverage"
        raw = candidate.to_dict()
        key = digest(raw)
        if key in keys:
            rejected["duplicate"] += 1
            continue
        checks = precheck(raw)
        if any(not finite_number(value) or value > 0 for value in checks.values()):
            rejected["cheap_constraint"] += 1
            continue
        keys.add(key)
        result.append({"candidate": raw, "source": source})
    return result, dict(rejected)


def constraint_acquisition(raw, predicted, log_ei):
    # An independence approximation for ranking, never calibrated evidence.
    feasibility = np.asarray(raw["hidden_success"]).copy()
    for name, values in predicted.items():
        if name.startswith("constraint:"):
            feasibility *= .5 if values is None else np.asarray(values["p_satisfied_raw"])
    if np.isfinite(log_ei).any():
        return log_ei + np.log(np.clip(feasibility, 1e-12, 1.))
    return feasibility


def select_candidate(pool, rows, risks, responses, calibrated, ledger, iteration,
                     algorithm="ra", boundary_every=4, discovery_rule="risk"):
    if algorithm not in ("ra", "ra_no_boundary", "cbo", "unconstrained_ei", "random"):
        raise ValueError("Unknown algorithm")
    if type(boundary_every) is not int or boundary_every < 1:
        raise ValueError("Positive boundary schedule required")
    if discovery_rule not in ("risk", "constraint_ei"):
        raise ValueError("Unknown discovery acquisition rule")
    if not pool:
        return None, {"reason": "empty_candidate_pool"}
    candidates = [item["candidate"] for item in pool]
    raw = risks.predict(candidates)
    predicted = responses.predict(candidates)
    cal = calibrated.predict(candidates) if calibrated is not None else None
    probabilities = None if cal is None else cal["risk_probabilities_calibrated"]["final_feasibility"]
    best = incumbent(rows)
    log_ei = log_expected_improvement(responses, candidates, best)
    relative_ei = np.exp(log_ei - np.max(log_ei)) if np.isfinite(log_ei).any() else np.zeros(len(pool))
    objective = predicted.get("objective")
    sd = np.zeros(len(pool)) if objective is None else np.asarray(objective["std"])
    boundary = np.zeros(len(pool))
    for name, values in predicted.items():
        if name.startswith("constraint:") and values is not None:
            p = np.asarray(values["p_satisfied_raw"])
            boundary = np.maximum(boundary, 4*p*(1-p))
    forest = risks.heads["final_feasibility"].model
    disagreement = np.zeros(len(pool))
    if forest is not None:
        x = risks.features.transform(candidates).numpy()
        disagreement = np.std([tree.predict_proba(x)[:, 1] for tree in forest.estimators_], axis=0)
    p_raw = np.asarray(raw["final_feasibility"])
    development_score = log_ei
    discovery_score = p_raw + .15*normalized(disagreement) + .05*normalized(sd)
    if discovery_rule == "constraint_ei":
        discovery_score = constraint_acquisition(raw, predicted, log_ei)
    # A heuristic information score, not mutual information or a safety bound.
    boundary_score = ((boundary + normalized(disagreement)) *
                      (.25 + relative_ei + .25*normalized(sd)))
    if algorithm in ("cbo", "unconstrained_ei", "random"):
        if algorithm == "random":
            scores = -np.arange(len(pool), dtype=float)
        elif algorithm == "unconstrained_ei":
            scores = log_ei if np.isfinite(log_ei).any() else sd
        else:
            scores = constraint_acquisition(raw, predicted, log_ei)
        modes = [("baseline", scores)]
    elif probabilities is None or best is None:
        modes = [("discovery", discovery_score)]
    else:
        modes = [("development", development_score)]
        if algorithm != "ra_no_boundary":
            boundary_mode = ("boundary", boundary_score)
            modes = [boundary_mode] + modes if iteration % boundary_every == 0 else modes + [boundary_mode]
    rejections = Counter()
    for mode, scores in modes:
        for index in np.argsort(-scores, kind="stable"):
            if not np.isfinite(scores[index]):
                rejections["nonfinite_acquisition"] += 1
                continue
            probability = None if probabilities is None else float(probabilities[index])
            evidence = {
                "basis": "raw" if probability is None else "empirical_calibrated",
                "probability": float(p_raw[index]) if probability is None else probability,
                "model_id": risks.model_id,
                "calibration_id": None if calibrated is None else calibrated.calibration_id,
                "candidate_key": digest(candidates[index]),
            }
            item = pool[index]
            assessment = ledger.assess(item["candidate"], mode, evidence, item["source"])
            if assessment["allowed"]:
                return dict(item, mode=mode, evidence=evidence), {
                    "reason": "approved", "algorithm": algorithm, "mode": mode,
                    "score": float(scores[index]), "ei": float(np.exp(log_ei[index])),
                    "log_ei": float(log_ei[index]) if np.isfinite(log_ei[index]) else None,
                    "constraint_boundary": float(boundary[index]),
                    "forest_disagreement": float(disagreement[index]),
                    "incumbent": best, "rejections": dict(rejections),
                    "risk_charge": assessment["charge"],
                    "discovery_rule": discovery_rule if mode == "discovery" else None,
                }
            rejections[assessment["reason"]] += 1
    return None, {"reason": "no_budget_approved_candidate", "rejections": dict(rejections)}
