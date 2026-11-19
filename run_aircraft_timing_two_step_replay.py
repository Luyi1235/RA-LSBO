"""Cross stopping feedback and two-step acquisition on sealed PR128 candidate pools."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.special import ndtr

from aircraft_risk_acquisition import posterior_cross_covariance, sign_covariance
from ra_lsbo.budget import digest
from run_aircraft_initial_budget import paired_interval, save, select
from run_aircraft_initial_timing_diagnostic import fit_completion_heads
from run_aircraft_risk_cold_start import discovery_outcome, file_hash
from run_aircraft_timing_online import QUERY_BUDGET
from run_aircraft_timing_pool_audit import load_archive, load_supplement, plan, reconcile


SUPPLEMENT_SHA = "d8135723f782be2e8fb7f51141f3e2d7ffe37f4b"
SUPPLEMENT_RESULT_SHA = "9d59c38ea89d80b70d10d397c679335b04b6716e"
SUPPLEMENT_SEALS = {
    "smoke": "543bdade9b51bd056833bec53ac395b74d88650b2cb414be394897cd0bbb78ca",
    "headwind-online": "b6cbaa475ac10c6a5a90f7b7228fdbf3e21e66b440deb5edfa48bab88859d1e6",
}
FEEDBACK = ("timing_sign", "timing_interval")
ACQUISITIONS = ("greedy", "two_step")
TIE_TOLERANCE = 1e-12


def two_step_scores(probability, z, correlation):
    """Probability of succeeding now or on the best remaining point after a stop."""
    p = np.asarray(probability, dtype=float)
    if len(p) < 2:
        return p.copy()
    both = p[:, None]*p[None, :] + sign_covariance(z, z, correlation)
    both = np.clip(both, np.maximum(0., p[:, None]+p[None, :]-1.),
                   np.minimum(p[:, None], p[None, :]))
    follow_up = p[None, :] - both
    np.fill_diagonal(follow_up, -1.)
    return np.minimum(1., p + follow_up.max(axis=1))


def forecast(problem, observed, candidates, seed, method):
    heads, query, _ = fit_completion_heads(problem, observed, candidates, seed, methods=(method,))
    head = heads[method]
    moments = head.predict(query)
    sd = np.sqrt(np.maximum(moments["variance"], 1e-14))
    z = -moments["mean"]/sd
    p = ndtr(z)
    raw_query = query/head.divisor
    covariance = posterior_cross_covariance(head.model, raw_query, raw_query)*head.scale**2
    correlation = covariance / sd[:, None] / sd[None, :]
    if not np.isfinite(correlation).all() or np.any(np.abs(correlation) > 1.00001):
        return p, None
    return p, two_step_scores(p, z, correlation)


def simulate(problem, unit, panel, method, acquisition, cache):
    history = list(unit.history)
    remaining = list(panel)
    trace = []
    for step in range(1, QUERY_BUDGET+1):
        if not remaining:
            break
        designs = [row.candidate for row in remaining]
        key = (method, tuple(row.observation.design_id for row in history))
        if key not in cache:
            cache[key] = forecast(problem, history, designs, unit.seed, method)
        probabilities, future_scores = cache[key]
        index = select(designs, probabilities)
        reason = "greedy"
        if acquisition == "two_step" and step < QUERY_BUDGET and len(remaining) > 1:
            if future_scores is None:
                reason = "numerical_greedy_fallback"
            else:
                near_best = [i for i, score in enumerate(future_scores)
                             if score >= future_scores.max()-TIE_TOLERANCE]
                index = near_best[select([designs[i] for i in near_best], probabilities[near_best])]
                reason = "two_step"
        row = remaining.pop(index)
        trace.append({"design_id": row.observation.design_id, "reason": reason,
                      "completion_probability": float(probabilities[index]),
                      "outcome": discovery_outcome(row.to_dict())})
        if row.usable_feasible:
            return {"discovered": True, "capped_first": step, "trace": trace}
        history.append(row)
    return {"discovered": False, "capped_first": QUERY_BUDGET+1, "trace": trace}


def contrast(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": mean(values), "paired_95": paired_interval(values),
            "negative": sum(x < 0 for x in values), "positive": sum(x > 0 for x in values)}


def summarize(rows):
    methods = {}
    effects = {}
    for method in FEEDBACK:
        methods[method] = {}
        for mode in ACQUISITIONS:
            arms = [row["methods"][method][mode] for row in rows]
            methods[method][mode] = {
                "discovered": sum(x["discovered"] for x in arms),
                "mean_capped_first": mean(x["capped_first"] for x in arms) if arms else None,
                "numerical_greedy_fallback": sum(step["reason"] == "numerical_greedy_fallback"
                                                 for x in arms for step in x["trace"]),
            }
        effects[method] = [
            row["methods"][method]["two_step"]["capped_first"]
            - row["methods"][method]["greedy"]["capped_first"] for row in rows]
    return {
        "n": len(rows), "methods": methods,
        "two_step_minus_greedy": {method: contrast(effect) for method, effect in effects.items()},
        "feedback_acquisition_interaction": contrast([a-b for a, b in zip(
            effects["timing_interval"], effects["timing_sign"])]),
    }


def replay(original, supplement):
    spec, problem, seeds = load_archive(original)
    name = spec["name"]
    if file_hash(supplement / "seal.json") != SUPPLEMENT_SEALS[name]:
        raise ValueError("Supplement does not match the frozen PR128 archive")
    protocol = plan(spec, problem, seeds)
    calls = load_supplement(protocol, problem, supplement, SUPPLEMENT_SHA)
    rows, used = [], 0
    for unit in seeds:
        panel, used = reconcile(unit, problem, calls, used)
        outcomes = Counter(discovery_outcome(row.to_dict()) for row in panel)
        if set(outcomes) - {"abort", "usable"}:
            raise ValueError("Two-step completion replay requires screened abort/usable outcomes")
        cache = {}
        methods = {
            method: {mode: simulate(problem, unit, panel, method, mode, cache)
                     for mode in ACQUISITIONS} for method in FEEDBACK
        }
        if any(methods[method]["greedy"]["discovered"] != unit.discovered[method]
               for method in FEEDBACK):
            raise ValueError("Greedy replay does not reproduce the original discovery")
        rows.append({"seed": unit.seed, "usable": outcomes["usable"], "methods": methods})
        print(f"{name} {unit.seed} replayed", flush=True)
    if used != len(calls):
        raise ValueError("Unregistered pool-completion call")
    return {
        "status": "COMPLETE_TIMING_TWO_STEP_REPLAY", "profile": name,
        "original_result_sha": protocol["input_result_sha"], "supplement_result_sha": SUPPLEMENT_RESULT_SHA,
        "query_budget": QUERY_BUDGET, "new_aircraft_calls": 0, "new_rcaide_mda_calls": 0,
        "cold": summarize(rows), "pools_with_usable": summarize([row for row in rows if row["usable"]]),
        "per_seed": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Sealed PR127 profile")
    parser.add_argument("--supplement", type=Path, required=True, help="Sealed PR128 profile")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(folder.resolve() in args.output.resolve().parents for folder in (args.input, args.supplement)):
        parser.error("Write the report outside the sealed archives")
    report = replay(args.input, args.supplement)
    save(args.output, report)
    print(json.dumps({"status": report["status"], "profile": report["profile"],
                      "cold": report["cold"]["n"], "new_calls": 0}))
