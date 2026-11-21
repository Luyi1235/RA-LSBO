"""Metrics for reaching multiple usable designs and a shared completion audit."""

from collections import Counter, defaultdict
from itertools import combinations
from statistics import mean

import numpy as np
from scipy.special import ndtr

from aircraft_risk_discovery import label as outcome_label
from aircraft_structured_risk import STRUCTURED_ARMS, HierarchicalTimingHead, with_configuration
from aircraft_timing_information import supports_timing
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.partial_bo.models import PrefixFeatures
from run_aircraft_coverage_factorial import FixedTimingHead
from run_aircraft_feedback_replication import paired
from run_aircraft_risk_cold_start import discovery_outcome
from run_aircraft_timing_interval_value import effect


PROFILE = "multi-feasible-pilot"
TARGET = 3
AUDIT_SIZE = 16
AUDIT_CHECKPOINTS = (8, 12, 16)


def acquisition_stage(history, target):
    return sum(record.usable_feasible for record in history) < target


def accumulation_metrics(history, initial, target):
    valid = [i for i, row in enumerate(history, 1) if row.usable_feasible]
    hit = valid[target-1] if len(valid) >= target else None
    failures = Counter(discovery_outcome(row.to_dict()) for row in history[initial:hit]
                       if not row.usable_feasible)
    configurations = {digest(row.candidate["discrete"]) for row in history[:hit] if row.usable_feasible}
    return {"target_call": hit, "capped_target_call": hit if hit is not None else len(history)+1,
            "nonusable_online_calls_through_target": sum(failures.values()),
            "nonusable_outcomes": dict(sorted(failures.items())),
            "distinct_usable_configurations_through_target": len(configurations)}


def score_audit(panel, labels, probabilities):
    """Score exact panel labels equally by observed discrete configuration."""
    groups = defaultdict(list)
    for point, truth, p in zip(panel, labels, probabilities):
        key = tuple(sorted(point["discrete"].items()))
        groups[key].append((float(truth), float(p)))
    by_configuration = [
        {"configuration": dict(key), "n": len(values), "observed_completion_rate": mean(y for y, _ in values),
         "mean_p_complete": mean(p for _, p in values),
         "brier": mean((p-y)**2 for y, p in values)}
        for key, values in sorted(groups.items())
    ]
    matches = []
    for left, right in combinations(by_configuration, 2):
        actual = left["observed_completion_rate"] - right["observed_completion_rate"]
        predicted = left["mean_p_complete"] - right["mean_p_complete"]
        if actual != 0.:
            matches.append(.5 if abs(predicted) <= 1e-12 else float(actual * predicted > 0.))
    return {"n": len(panel), "configuration_balanced_brier": mean(g["brier"] for g in by_configuration),
            "comparable_configuration_pairs": len(matches),
            "configuration_rank_agreement": mean(matches) if matches else None,
            "configurations": by_configuration}


def completion_audit(problem, seed, histories, heldout, checkpoints, initial):
    """Assess histories with the same explicit feature map and interval GP."""
    common = next(iter(histories.values()))[:initial]
    features = PrefixFeatures(problem, BASE, common, seed, latent=False)
    results = {}
    for checkpoint in checkpoints:
        used = {row.observation.design_id for history in histories.values() for row in history[:checkpoint]}
        labeled = [(row.candidate, outcome_label(discovery_outcome(row.to_dict()), "mission"))
                   for row in heldout if row.observation.design_id not in used]
        labeled = [(candidate, label) for candidate, label in labeled if label is not None]
        if not labeled:
            results[str(checkpoint)] = {}
            continue
        panel, labels = zip(*labeled)
        query = features.transform(panel)
        scores = {}
        for arm, history in histories.items():
            trusted = [row for row in history[:checkpoint] if supports_timing(row)]
            if not trusted:
                scores[arm] = {"unavailable": "no_supported_timing"}
                continue
            try:
                inputs = features.transform([row.candidate for row in trusted])
                if arm in STRUCTURED_ARMS:
                    inputs = with_configuration(inputs, [row.candidate for row in trusted])
                    head = HierarchicalTimingHead(problem, inputs, trusted, seed, use_interval=True)
                    moments = head.predict(with_configuration(query, panel))
                else:
                    head = FixedTimingHead(problem, inputs, trusted, seed, use_interval=True)
                    moments = head.predict(query)
                probabilities = ndtr(-moments["mean"]/np.sqrt(np.maximum(moments["variance"], 1e-14)))
                if not np.isfinite(probabilities).all():
                    raise FloatingPointError("Nonfinite audit forecast")
                scores[arm] = score_audit(panel, labels, probabilities)
            except (np.linalg.LinAlgError, FloatingPointError, RuntimeError):
                scores[arm] = {"unavailable": "numerical_failure"}
        results[str(checkpoint)] = scores
    return results


def compare_accumulation(rows, left, right, checkpoints):
    pairs = [(row["arms"][left]["accumulation"], row["arms"][right]["accumulation"]) for row in rows]
    result = {
        "target_hits": {arm: sum(row["arms"][arm]["accumulation"]["target_call"] is not None for row in rows)
                        for arm in (left, right)},
        **{key + "_delta": paired([a[key]-b[key] for a, b in pairs])
           for key in ("capped_target_call", "nonusable_online_calls_through_target")},
        "distinct_usable_configurations_through_target_delta": effect([
            a["distinct_usable_configurations_through_target"]-b["distinct_usable_configurations_through_target"]
            for a, b in pairs]),
    }
    audit = {}
    for step in map(str, checkpoints):
        comparable = [(row["completion_audit"][step].get(left, {}),
                       row["completion_audit"][step].get(right, {})) for row in rows]
        audit[step] = {
            # Report lower-is-better deltas for both scores, so negative rank error is better.
            name: paired([direction*(a[key]-b[key]) for a, b in comparable
                          if a.get(key) is not None and b.get(key) is not None])
            for name, key, direction in (("configuration_balanced_brier_delta", "configuration_balanced_brier", 1),
                                         ("configuration_rank_error_delta", "configuration_rank_agreement", -1))
        }
    result["heldout_completion"] = audit
    return result
