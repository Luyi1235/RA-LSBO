"""Two-step constrained improvement with observable, not omniscient, fantasies.

Cold: maximize probability of at least one feasible delivery in at most two calls.
Warm: maximize expected improvement of the best observed feasible objective.
All experimental arms actually use the same fixed call budget. No stopping or
arbitrary failure-price scale is inferred from these acquisitions.
"""
from dataclasses import dataclass
import math
import time
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
from .protocol import FIRST_ACTIONS, FANTASIES_PER_BRANCH


class FantasyFitError(RuntimeError):
    """Numerical fantasy refit failure, not an evaluator failure or software bug."""


@dataclass(frozen=True)
class Outcome:
    aborted: bool
    region: object
    resource: float
    objective: object

    def __post_init__(self):
        if type(self.aborted) is not bool or not math.isfinite(self.resource):
            raise ValueError("Invalid observable fantasy")
        if self.aborted:
            if self.region is not None or self.objective is not None:
                raise ValueError("Abort exposes only region>0, never exact region or hidden objective")
        elif self.region is None or not math.isfinite(self.region) or self.region > 0:
            raise ValueError("Completed region requires a nonpositive exact value")
        if self.objective is not None and not math.isfinite(self.objective):
            raise ValueError("Invalid observed objective")

    @property
    def feasible(self):
        return not self.aborted and self.resource <= 0

    def packet(self):
        return {"aborted": self.aborted,
                "region": {"kind": "interval", "lower": 0., "lower_inclusive": False} if self.aborted else
                          {"kind": "exact", "value": float(self.region)},
                "resource": {"kind": "exact", "value": float(self.resource)},
                "objective": {"kind": "missing"} if self.objective is None else
                             {"kind": "exact", "value": float(self.objective)}}


def moments(pred):
    m, v = np.asarray(pred["mean"], float), np.asarray(pred["variance"], float)
    if m.ndim != 1 or m.shape != v.shape or not len(m) or not np.isfinite(m).all() or not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError("Finite aligned Gaussian mean/variance required")
    return m, np.sqrt(v)


def probability(pred):
    m, s = moments(pred)
    z = np.divide(-m, s, out=np.zeros_like(m), where=s > 0)
    return np.where(s > 0, ndtr(z), (m <= 0).astype(float))


def one_step(forecast, best):
    p = probability(forecast["constraints"]["region"]) * probability(forecast["constraints"]["resource"])
    if best is None:
        return p
    if not math.isfinite(best) or forecast.get("objective") is None:
        raise ValueError("Warm improvement requires a real objective posterior/incumbent")
    m, s = moments(forecast["objective"])
    d = best - m
    z = np.divide(d, s, out=np.zeros_like(m), where=s > 0)
    ei = np.where(s > 0, d * ndtr(z) + s * np.exp(-0.5 * z*z) / math.sqrt(2*math.pi), np.maximum(d, 0.))
    if p.shape != ei.shape:
        raise ValueError("Misaligned objective and constraint forecasts")
    return p * np.maximum(ei, 0.)


def shortlist(forecast, base, limit=FIRST_ACTIONS):
    """Numerical first-action screen. Includes full-pool myopic winner.

    Two top one-step values, boundary ambiguity and region variance; fill any
    duplicate slots from the myopic ordering. Second actions use the full pool.
    These heuristic screen rules are shared by both two-step arms, not a claim
    that uncertainty itself equals information value.
    """
    n = len(base)
    if n < 2 or limit < 1:
        raise ValueError("Need at least two novel candidates")
    p = probability(forecast["constraints"]["region"]) * probability(forecast["constraints"]["resource"])
    order = sorted(range(n), key=lambda i: (-float(base[i]), i))
    picks = order[:2] + [int(np.argmax(p*(1-p))), int(np.argmax(forecast["constraints"]["region"]["variance"]))] + order
    unique = []
    for i in picks:
        if i not in unique:
            unique.append(i)
        if len(unique) == min(n, limit):
            break
    return unique


def nodes(seed, iteration):
    key = (int(seed) + 730000000 + 1009 * int(iteration)) % (2**32)
    m = int(math.log2(FANTASIES_PER_BRANCH))
    if 2**m != FANTASIES_PER_BRANCH:
        raise ValueError("Power-of-two Sobol rule required")
    return np.clip(qmc.Sobol(d=3, scramble=True, seed=key).random_base2(m), 1e-12, 1-1e-12)


def outcomes(forecast, index, unit_nodes, warm):
    """Split region-abort/completion analytically; never draw hidden abort f.

    Independent output heads are an explicit surrogate approximation. Exact
    resource is observable in both branches, exact objective only on completion.
    In cold discovery mode objective is not needed and is not fantasized.
    """
    unit_nodes = np.asarray(unit_nodes, float)
    if unit_nodes.ndim != 2 or unit_nodes.shape[1] != 3 or len(unit_nodes) < 1 or np.any(unit_nodes <= 0) or np.any(unit_nodes >= 1):
        raise ValueError("Interior three-dimensional quadrature nodes required")
    rm, rs = moments(forecast["constraints"]["region"])
    cm, cs = moments(forecast["constraints"]["resource"])
    p = float(probability(forecast["constraints"]["region"])[index])
    if warm:
        fm, fs = moments(forecast["objective"])
    for abort, weight in ((True, 1-p), (False, p)):
        if weight == 0:
            continue
        for u in unit_nodes:
            resource = float(cm[index] + cs[index] * ndtri(u[1]))
            if abort:
                outcome = Outcome(True, None, resource, None)
            else:
                region = float(rm[index]) if rs[index] == 0 else float(rm[index] + rs[index] * ndtri(max(np.finfo(float).tiny, p*u[0])))
                region = min(region, 0.)  # round-off at the conditioned boundary only
                objective = float(fm[index] + fs[index] * ndtri(u[2])) if warm else None
                outcome = Outcome(False, region, resource, objective)
            yield weight / len(unit_nodes), outcome


def choose(forecast, best, remaining, factory, *, arm, seed, iteration):
    """Factory conditions real GP heads on an observable Outcome only.

    One-step immediate value is integrated analytically; only continuation is
    quadrature-approximated. Numerical failure of ANY required fantasy discards
    the entire lookahead decision and uses the logged full-pool myopic choice.
    Programming errors propagate rather than being disguised as low utility.
    """
    if arm not in ("cei_myopic", "cei_two_step", "cei_two_step_no_abort_update") or type(remaining) is not int or remaining < 1:
        raise ValueError("Invalid decision arm/remaining actual evaluations")
    start = time.perf_counter()
    base = one_step(forecast, best)
    if not np.isfinite(base).all() or np.any(base < 0):
        raise ValueError("Nonfinite one-step acquisition")
    myopic = int(np.argmax(base))
    common = {"arm": arm, "remaining_actual_calls": remaining,
              "depth": 1 if arm == "cei_myopic" or remaining == 1 else 2,
              "criterion": "first_feasible_probability" if best is None else "expected_feasible_objective_improvement",
              "myopic_index": myopic, "myopic_scores": base.tolist(),
              "first_action_indices": [], "fantasies": [], "fallback": None,
              "actual_feedback_always_full": True,
              "quadrature_nodes_per_region_branch": FANTASIES_PER_BRANCH,
              "probability_guarantee": False, "stop_action_in_experiment": False}
    if common["depth"] == 1:
        common.update(selected_index=myopic, scores=base.tolist(), planner_seconds=time.perf_counter()-start)
        return myopic, common
    unit_nodes = nodes(seed, iteration)
    candidates = shortlist(forecast, base)
    common["first_action_indices"] = candidates
    scores = [None] * len(base)
    use_abort = arm != "cei_two_step_no_abort_update"
    common["condition_on_abort_interval"] = use_abort
    try:
        for i in candidates:
            continuation = 0.
            for weight, outcome in outcomes(forecast, i, unit_nodes, best is not None):
                # Once a feasible design is found the cold-stage delivery event
                # is achieved. There is no fabricated f or objective GP here.
                row = {"first_index": i, "weight": weight, "observation": outcome.packet(),
                       "conditioned": False, "abort_interval_used": bool(use_abort and outcome.aborted)}
                if best is None and outcome.feasible:
                    value, second = 0., None
                else:
                    updated = factory.condition(i, outcome, use_abort_interval=use_abort)
                    row["conditioned"] = True
                    next_best = min(best, outcome.objective) if best is not None and outcome.feasible else best
                    future = one_step(updated, next_best)
                    future[i] = -np.inf  # no repeated candidate evaluation
                    second = int(np.argmax(future))
                    value = float(future[second])
                    if not math.isfinite(value) or value < 0:
                        raise FantasyFitError("Invalid fantasy continuation")
                row.update(second_index=second, continuation=value)
                common["fantasies"].append(row)
                continuation += weight * value
            scores[i] = float(base[i] + continuation)
    except FantasyFitError as error:
        common.update(fallback="fantasy_numerical_failure_full_myopic", error=str(error),
                      selected_index=myopic, scores=base.tolist(),
                      partial_lookahead_scores_not_used=scores,
                      planner_seconds=time.perf_counter()-start)
        return myopic, common
    selected = max(candidates, key=lambda i: (scores[i], -i))
    common.update(selected_index=selected, scores=scores, planner_seconds=time.perf_counter()-start)
    return selected, common
