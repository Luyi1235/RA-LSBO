"""Outcome semantics and probability of first usable discovery in two calls."""
import math

from ra_lsbo.risk_core.observations import OutputObservation

STAGE_HEADS = ("mission", "feasible_given_mission", "usable_factored", "usable_direct")
PHYSICAL_STOPS = {
    "physical_soc_reserve_abort", "physical_battery_power_domain", "physical_discharge_domain",
}
SHORTLIST_PER_SCORE = 12
ROLLOUT_RULES = {"collapsed_lookahead", "staged_lookahead"}


def outcome(record, joint_label):
    """Classify trusted feedback using mission evidence, not journal status."""
    packet, diagnostics = record["observation"], record["diagnostics"]
    if packet["status"] in {"unknown", "infrastructure_error"} or diagnostics.get("numerical_failure"):
        return "unknown"
    mission = diagnostics.get("mission_completed")
    if mission is False or diagnostics.get("termination_reason") in PHYSICAL_STOPS:
        return "abort"
    if mission is True and packet["status"] == "completed" and packet["objective"]["kind"] == "exact":
        if joint_label is not None:
            return "usable" if joint_label else "violation"
        return "complete_unknown"
    return "unknown"


def label(result, head):
    if result == "unknown":
        return None
    if head == "mission":
        return result != "abort"
    if result == "complete_unknown" or (head == "feasible_given_mission" and result == "abort"):
        return None
    return result == "usable"


def fit_stage_heads(history, past_x, outcomes, seed, continuous_count):
    from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import StableGroupedHead

    heads = {}
    for name in ("mission", "feasible_given_mission", "usable_direct"):
        observations = []
        for row, result in zip(history, outcomes):
            evidence = row.observation.evidence_id + ":" + name
            target = label(result, name)
            if target is None:
                obs = OutputObservation("missing", evidence, reason="stage_label_unobserved")
            elif target:
                obs = OutputObservation("interval", evidence, upper=0.)
            else:
                obs = OutputObservation("interval", evidence, lower=0., lower_inclusive=False)
            observations.append(obs)
        heads[name] = StableGroupedHead(past_x, observations, seed=seed, name=name,
                                       continuous_count=continuous_count, completion=True)
    return heads


def stage_forecast(heads, query):
    import numpy as np
    from scipy.special import ndtr

    moments = {name: head.predict(query) for name, head in heads.items()}
    z = {name: -pred["mean"] / np.sqrt(1 + pred["variance"]) for name, pred in moments.items()}
    p = {name: ndtr(value) for name, value in z.items()}
    p["usable_factored"] = p["mission"] * p["feasible_given_mission"]
    return p, moments, z


def shortlist(probabilities, candidate_ids):
    mission, feasible = probabilities["mission"], probabilities["feasible_given_mission"]
    usable = probabilities["usable_factored"]
    uncertainty = [p*(1-p) + p*q*(1-q) for p, q in zip(mission, feasible)]
    ids = range(len(candidate_ids))
    return sorted(set().union(*(sorted(ids, key=lambda i: (values[i], candidate_ids[i]),
                                      reverse=True)[:SHORTLIST_PER_SCORE]
                                for values in (usable, uncertainty))))


def rollout_components(heads, query, probabilities, moments, z, indices):
    import numpy as np
    from aircraft_risk_acquisition import posterior_cross_covariance, sign_covariance

    joint = {}
    for name in ("mission", "feasible_given_mission"):
        head = heads[name]
        scale = np.sqrt(1 + moments[name]["variance"])
        covariance = posterior_cross_covariance(head.model, query[indices]/head.divisor, query/head.divisor)
        rho = covariance * head.scale**2 / scale[indices, None] / scale[None, :]
        if not np.isfinite(rho).all() or np.any(np.abs(rho) > 1.00001):
            raise FloatingPointError("Invalid future probit-label correlation")
        p = probabilities[name]
        both = p[indices, None]*p[None, :] + sign_covariance(z[name][indices], z[name], rho)
        joint[name] = np.clip(both, np.maximum(0., p[indices, None]+p[None, :]-1.),
                              np.minimum(p[indices, None], p[None, :]))
    c, f = probabilities["mission"], probabilities["feasible_given_mission"]
    after_abort = (c[None, :] - joint["mission"]) * f[None, :]
    after_violation = joint["mission"] * (f[None, :] - joint["feasible_given_mission"])
    after_abort[np.arange(len(indices)), indices] = 0.
    after_violation[np.arange(len(indices)), indices] = 0.
    return [[i, float(a), float(v), float(collapsed)] for i, a, v, collapsed in
            zip(indices, after_abort.max(axis=1), after_violation.max(axis=1),
                (after_abort+after_violation).max(axis=1))]


def discovery_scores(rule, receipt, candidate_ids, remaining_calls):
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Discovery candidates must be unique")
    values, components = receipt["probabilities"], receipt["rollout"]
    heads = {*STAGE_HEADS, "joint", "coverage"}
    if (set(receipt) != {"probabilities", "rollout", "outcomes"}
            or set(values) != heads
            or any(len(arr) != len(candidate_ids) for arr in values.values())
            or any(not math.isfinite(v) or v < 0 or (head != "coverage" and v > 1)
                   for head, arr in values.items() for v in arr)):
        raise ValueError("Invalid discovery forecast")
    c, f, usable = (values[key] for key in ("mission", "feasible_given_mission", "usable_factored"))
    if any(not math.isclose(x*y, p, abs_tol=1e-13) for x, y, p in zip(c, f, usable)):
        raise ValueError("Invalid conditional feasibility product")
    if rule in ROLLOUT_RULES and remaining_calls >= 2:
        if [row[0] for row in components] != shortlist(values, candidate_ids):
            raise ValueError("Rollout shortlist does not match the frozen rule")
        scores = [-1.] * len(candidate_ids)
        for i, after_abort, after_violation, collapsed in components:
            if (not 0 <= after_abort <= 1-c[i]+1e-12
                    or not 0 <= after_violation <= c[i]*(1-f[i])+1e-12
                    or not max(after_abort, after_violation)-1e-12 <= collapsed <= after_abort+after_violation+1e-12):
                raise ValueError("Invalid rollout branch probability")
            followup = collapsed if rule == "collapsed_lookahead" else after_abort+after_violation
            scores[i] = min(usable[i] + followup, 1.)
        return scores
    if components:
        raise ValueError("Unexpected rollout continuations")
    key = {"coverage": "coverage", "constraints": "joint", "direct": "usable_direct",
           "staged": "usable_factored", "staged_lookahead": "usable_factored",
           "collapsed_lookahead": "usable_factored"}[rule]
    return values[key]


def discovery_pairs(trajectories, horizon):
    comparisons = (
        ("discovery_lookahead", "discovery_collapsed"),
        ("discovery_lookahead", "discovery_staged"),
        ("discovery_staged", "discovery_direct"),
        ("discovery_staged", "discovery_constraints"),
        ("discovery_lookahead", "discovery_coverage"),
    )
    index = {(row["seed"], row["arm"]): row for row in trajectories}
    pairs = []
    for seed in sorted({seed for seed, _ in index}):
        for arm, baseline in comparisons:
            left, right = index[seed, arm], index[seed, baseline]
            if left["initial_outcomes"] != right["initial_outcomes"]:
                raise ValueError("Discovery arms did not share the initial outcomes")
            first, second = (row["first_usable_feasible_call"] for row in (left, right))
            a, b = (value if value is not None else horizon+1 for value in (first, second))
            pairs.append({"seed": seed, "arm": arm, "baseline": baseline,
                          "initial_usable": left["initial_usable"],
                          "arm_first_usable_call": first, "baseline_first_usable_call": second,
                          "arm_discovered": first is not None, "baseline_discovered": second is not None,
                          "capped_calls_delta": a-b,
                          "outcome": "win" if a < b else "loss" if a > b else "tie"})
    return pairs
