"""Exact, finite-hypothesis two-step planning with censored feedback.

A verification model, NOT a fitted GP, aircraft experiment, or novelty claim.
Delivered value is a declared terminal utility, never an imputed objective.
The planner receives the hypothesis ensemble, never the realized world index.
"""
from dataclasses import dataclass
import math
from typing import Optional, Tuple


@dataclass(frozen=True)
class Outcome:
    feasible: bool
    delivered_value: Optional[float]
    evidence: Tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.feasible) is not bool:
            raise ValueError("Feasibility must be known in the finite test")
        if self.feasible:
            if (type(self.delivered_value) not in (int, float) or
                    not math.isfinite(self.delivered_value) or self.delivered_value < 0):
                raise ValueError("A delivered design needs finite nonnegative utility")
        elif self.delivered_value is not None:
            raise ValueError("Failure cannot disclose a missing objective/utility")
        if not isinstance(self.evidence, tuple) or any(type(x) is not str for x in self.evidence):
            raise ValueError("Evidence must be an immutable tuple of observable tokens")

    def token(self, feedback):
        if feedback not in ("full", "without_failure_evidence"):
            raise ValueError("Unknown observation ablation")
        evidence = self.evidence if self.feasible or feedback == "full" else ()
        # The ablation retains the known failure event; it drops extra evidence.
        return self.feasible, self.delivered_value, evidence


class FinitePlanner:
    """Enumerate observable outcomes and update a discrete posterior exactly."""

    def __init__(self, worlds, costs):
        if not worlds or not costs:
            raise ValueError("Nonempty finite worlds and actions required")
        self.actions = tuple(sorted(costs))
        if any(set(w) != set(self.actions) for w in worlds):
            raise ValueError("Worlds must describe the identical action space")
        if any(not isinstance(o, Outcome) for w in worlds for o in w.values()):
            raise ValueError("Use observable Outcome objects, not hidden truth arrays")
        if any(type(c) not in (int, float) or not math.isfinite(c) or c <= 0
               for c in costs.values()):
            raise ValueError("Every evaluation, including success, has positive cost")
        self.worlds = tuple(dict(w) for w in worlds)
        self.costs = dict(costs)

    def _weights(self, weights):
        if (len(weights) != len(self.worlds) or
                any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in weights) or sum(weights) <= 0):
            raise ValueError("Invalid finite-hypothesis posterior")
        total = sum(weights)
        return tuple(v / total for v in weights)

    def groups(self, weights, action, feedback):
        weights = self._weights(weights)
        groups = {}
        for i, (weight, world) in enumerate(zip(weights, self.worlds)):
            if weight:
                token = world[action].token(feedback)
                groups.setdefault(token, []).append((i, weight))
        for token, rows in groups.items():
            mass = sum(w for _, w in rows)
            conditional = [0.] * len(weights)
            for i, weight in rows:
                conditional[i] = weight / mass
            yield token, mass, tuple(conditional)

    def update(self, weights, action, observation, feedback="full"):
        token = observation.token(feedback)
        for candidate, _, posterior in self.groups(weights, action, feedback):
            if candidate == token:
                return posterior
        raise ValueError("Observation has zero mass under this finite test model")

    def choose(self, weights, remaining, incumbent=0., used=(), *,
               depth=2, feedback="full"):
        if type(remaining) is not int or remaining < 0 or depth not in (1, 2):
            raise ValueError("Nonnegative remaining calls and depth 1/2 required")
        if (not math.isfinite(incumbent) or incumbent < 0 or
                len(set(used)) != len(used) or not set(used) <= set(self.actions)):
            raise ValueError("Invalid incumbent or evaluated actions")
        if feedback not in ("full", "without_failure_evidence"):
            raise ValueError("Unknown feedback")
        weights = self._weights(weights)
        h = min(remaining, depth)
        best_action, best_value = None, float(incumbent)  # STOP costs zero
        scores = {}
        if h:
            for action in self.actions:
                if action in used:
                    continue
                q = -self.costs[action]
                for token, probability, posterior in self.groups(weights, action, feedback):
                    feasible, delivered, _ = token
                    next_best = max(incumbent, delivered) if feasible else incumbent
                    future = self.choose(posterior, h - 1, next_best,
                                         used + (action,), depth=depth, feedback=feedback)
                    q += probability * future["value"]
                scores[action] = q
                # On equality, prefer STOP; then deterministic action-name order.
                if q > best_value + 1e-12:
                    best_action, best_value = action, q
        return {"action": best_action, "value": best_value, "scores": scores,
                "stop_value": incumbent, "remaining": remaining,
                "lookahead_depth": h, "feedback": feedback}


def fixture(informative=True, cost=1.):
    """Four possible feasible islands; a failed probe can identify the island.

    Probe region evidence is censored g_region>0; its measured auxiliary
    response can be retained despite the abort. No failed objective is exposed.
    This deliberately constructed test is not a general optimization benchmark.
    """
    worlds = []
    for world in range(4):
        rows = {
            "probe": Outcome(False, None, ("region:(0,+inf)",
                "aux:%s" % world if informative else "aux:constant")),
            "blind_probe": Outcome(False, None, ("region:(0,+inf)", "aux:constant")),
            "safe": Outcome(True, 5., ("region:-1",)),
        }
        for island in range(4):
            rows["target_%d" % island] = (
                Outcome(True, 10., ("region:-1",)) if island == world else
                Outcome(False, None, ("region:(0,+inf)",)))
        worlds.append(rows)
    return FinitePlanner(worlds, {a: cost for a in worlds[0]})


def run_mechanisms():
    rows = []
    settings = (("myopic", 1, "full"), ("two_step_full", 2, "full"),
                ("two_step_without_failure_evidence", 2, "without_failure_evidence"))
    for scenario, informative, cost in (("informative_abort", True, 1.),
                                       ("uninformative_abort", False, 1.),
                                       ("expensive_evaluations", True, 7.)):
        model = fixture(informative, cost)
        for horizon in (1, 2):
            for arm, depth, feedback in settings:
                for realized_world in range(4):
                    weights, incumbent, used, paid = (.25,) * 4, 0., (), 0.
                    trace = []
                    for step in range(horizon):
                        decision = model.choose(weights, horizon - step, incumbent,
                                                used, depth=depth, feedback=feedback)
                        action = decision["action"]
                        if action is None:
                            trace.append({"step": step, "decision": decision,
                                          "actual_action": "stop", "outcome": None,
                                          "evaluation_cost": 0.})
                            break
                        # The true world is read ONLY after the policy has decided.
                        observation = model.worlds[realized_world][action]
                        trace.append({"step": step, "decision": decision,
                                      "actual_action": "evaluate",
                                      "outcome": observation.token(feedback),
                                      "evaluation_cost": model.costs[action]})
                        weights = model.update(weights, action, observation, feedback)
                        used += (action,)
                        paid += model.costs[action]
                        if observation.feasible:
                            incumbent = max(incumbent, observation.delivered_value)
                    rows.append({"scenario": scenario, "horizon": horizon,
                                 "arm": arm, "world": realized_world, "trace": trace,
                                 "calls": len(used), "cost": paid,
                                 "delivered_utility": incumbent,
                                 "net_utility": incumbent - paid})
    return {"status": "FINITE_MECHANISM_VERIFICATION_ONLY", "rows": rows,
            "gp_fits": 0, "analytical_problem_calls": 0, "aircraft_calls": 0,
            "interpretation": "Exhaustive constructed finite-world checks; no BO superiority or safety claim."}
