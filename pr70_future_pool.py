"""PR70: change ONLY the continuation candidate support of the PR68 planner.

The future candidate stream is a registered, outcome-independent function of
case, seed, iteration and past candidate coordinates. It never sees future
observations. The four-node rule, current first-action screen, EP kernels and
real GP/VAE updates are unchanged. Availability is not an optimality guarantee.
"""
from copy import deepcopy
import math
import time
import numpy as np

from ra_lsbo.budget import digest
from ra_lsbo.experiments.risk_budget63_runner import _candidates
from ra_lsbo.experiments.discovery_stages.policies import filter_near
from ra_lsbo.experiments.information_value_stage3.native import FrozenFantasyFactory, FantasyValueBO
from ra_lsbo.experiments.information_value_stage3.planner import (
    FantasyFitError, choose as parent_choose, one_step, probability, shortlist, nodes, outcomes)


class CandidateOnly:
    """Coordinate-only view: no observation, feasibility or objective exists."""
    __slots__ = ("candidate",)

    def __init__(self, candidate):
        self.candidate = deepcopy(candidate)


def future_candidates(problem, case, seed, iteration, history, first):
    """The EXACT next raw and protected model pools for a hypothetical first x."""
    prefix = [CandidateOnly(r.candidate) for r in history] + [CandidateOnly(first)]
    raw = _candidates(problem, case, seed, iteration + 1, prefix)
    model, _, _ = filter_near(problem, raw, prefix)
    if not model or first in model or len({digest(c) for c in model}) != len(model):
        raise ValueError("Future pool is empty, duplicated or retains the first action")
    return raw, model


def trim_first(forecast):
    def trim(head):
        if head is None:
            return None
        return {**head, "mean": np.asarray(head["mean"])[1:].copy(),
                "variance": np.asarray(head["variance"])[1:].copy()}
    return {"constraints": {k: trim(v) for k, v in forecast["constraints"].items()},
            "objective": trim(forecast.get("objective"))}


class NextPoolFactory:
    """Reuse the unmodified real EP fantasy adapter on [first]+future queries."""
    def __init__(self, posterior, candidates, problem, case, seed, iteration, records):
        self.posterior, self.candidates = posterior, list(candidates)
        self.problem, self.case, self.seed, self.iteration = problem, case, seed, iteration
        self.records = list(records)
        self.supports, self.factories = {}, {}

    def support(self, index):
        if index not in self.supports:
            raw, model = future_candidates(self.problem, self.case, self.seed, self.iteration,
                                           self.records, self.candidates[index])
            self.supports[index] = {"iteration": self.iteration + 1,
                "raw_pool_sha256": digest(raw), "model_pool_sha256": digest(model),
                "model_candidates": model,
                "first_candidate_sha256": digest(self.candidates[index]),
                "generation_uses_observations": False}
        return self.supports[index]

    def condition(self, index, outcome, *, use_abort_interval=True):
        if not use_abort_interval:
            raise ValueError("PR70 does not introduce another information ablation")
        if index not in self.factories:
            query = [self.candidates[index]] + self.support(index)["model_candidates"]
            forecast = self.posterior.predict(query)
            self.factories[index] = FrozenFantasyFactory(self.posterior, query, forecast)
        return trim_first(self.factories[index].condition(0, outcome, use_abort_interval=True))

    def diagnostics(self):
        stats = [f.diagnostics() for f in self.factories.values()]
        counts = {}
        for s in stats:
            for k, v in s["head_counts"].items():
                counts[k] = counts.get(k, 0) + v
        return {"head_fits_started": sum(s["head_fits_started"] for s in stats),
                "head_fits_completed": sum(s["head_fits_completed"] for s in stats),
                "head_fit_seconds": sum(s["head_fit_seconds"] for s in stats),
                "head_counts": counts, "real_evaluator_calls": 0,
                "hyperparameters_reselected": False, "representation_trained_on_fantasies": False,
                "prior_model_mutated": False}


def choose_next(forecast, best, remaining, factory, *, seed, iteration):
    """PR68 Q1 + expected continuation; only the continuation support differs.

    Root samples and immediate gain are still from the CURRENT pool. A second
    index is local to that first action's future pool, NOT the current pool.
    Any necessary numerical failure discards the entire lookahead decision.
    """
    started = time.perf_counter()
    if type(remaining) is not int or remaining < 1:
        raise ValueError("Positive actual remaining calls required")
    if remaining == 1:
        index, receipt = parent_choose(forecast, best, remaining, factory,
                                       arm="cei_two_step", seed=seed, iteration=iteration)
        receipt.update(arm="cei_two_step_next_pool", continuation_support="actual_next_pool",
                       future_supports={}, selected_future_support=None)
        return index, receipt
    base = one_step(forecast, best)
    if not np.isfinite(base).all() or np.any(base < 0):
        raise ValueError("Invalid root one-step values")
    myopic = int(np.argmax(base))
    firsts = shortlist(forecast, base)
    unit_nodes = nodes(seed, iteration)
    receipt = {"arm": "cei_two_step_next_pool", "depth": 2,
        "remaining_actual_calls": remaining,
        "criterion": "first_feasible_probability" if best is None else "expected_feasible_objective_improvement",
        "myopic_index": myopic, "myopic_scores": base.tolist(),
        "first_action_indices": firsts, "fantasies": [], "fallback": None,
        "actual_feedback_always_full": True, "condition_on_abort_interval": True,
        "quadrature_nodes_per_region_branch": len(unit_nodes),
        "probability_guarantee": False, "stop_action_in_experiment": False,
        "continuation_support": "actual_next_pool"}
    # Geometry-only supports are defined before any hypothetical observation.
    supports = {str(i): deepcopy(factory.support(i)) for i in firsts}
    scores = [None] * len(base)
    selected = myopic
    try:
        for i in firsts:
            continuation = 0.
            for weight, outcome in outcomes(forecast, i, unit_nodes, best is not None):
                row = {"first_index": i, "weight": weight, "observation": outcome.packet(),
                       "conditioned": False, "abort_interval_used": bool(outcome.aborted)}
                if best is None and outcome.feasible:
                    value, second, second_hash = 0., None, None
                else:
                    updated = factory.condition(i, outcome, use_abort_interval=True)
                    row["conditioned"] = True
                    next_best = min(best, outcome.objective) if best is not None and outcome.feasible else best
                    future = one_step(updated, next_best)
                    support = supports[str(i)]["model_candidates"]
                    if len(future) != len(support):
                        raise ValueError("Future prediction/support length mismatch")
                    second = int(np.argmax(future))
                    value = float(future[second])
                    second_hash = digest(support[second])
                    if not math.isfinite(value) or value < 0:
                        raise FantasyFitError("Invalid future continuation")
                row.update(second_index=second, second_candidate_sha256=second_hash, continuation=value)
                receipt["fantasies"].append(row)
                continuation += weight * value
            scores[i] = float(base[i] + continuation)
        selected = max(firsts, key=lambda i: (scores[i], -i))
    except FantasyFitError as error:
        receipt.update(fallback="fantasy_numerical_failure_full_myopic", error=str(error),
                       partial_lookahead_scores_not_used=scores)
        scores = base.tolist()
    receipt.update(selected_index=selected, scores=scores, future_supports=supports,
                   selected_future_support=supports[str(selected)],
                   planner_seconds=time.perf_counter()-started)
    return selected, receipt


class NextPoolBO(FantasyValueBO):
    def __init__(self, problem, base, seed, output, *, case, horizon):
        super().__init__(problem, base, seed, output, arm="cei_two_step", horizon=horizon)
        self.next_case = case

    def _choose_acquisition(self, predictions, best, iteration, acquisition, records):
        candidates = self._predicted_candidates
        if self._posterior is None or not candidates:
            raise ValueError("Missing bound current posterior")
        key = digest([iteration, best, candidates, [r.to_dict() for r in records]])
        if key in self._decision_cache:
            return deepcopy(self._decision_cache[key])
        factory = NextPoolFactory(self._posterior, candidates, self.problem, self.next_case,
                                  self.seed, iteration, records)
        index, receipt = choose_next(predictions, best, self.settings.horizon-iteration,
                                    factory, seed=self.seed, iteration=iteration)
        pf = probability(predictions["constraints"]["region"]) * probability(predictions["constraints"]["resource"])
        receipt.update(candidate_pool_hash=digest(candidates),
            pre_outcome_history_hash=digest([r.to_dict() for r in records]),
            selected_candidate_hash=digest(candidates[index]),
            selected_predicted_failure=float(1-pf[index]),
            fantasy_fits=factory.diagnostics(), actual_action="evaluate_fixed_budget")
        meta = {"mode": "cei_two_step_next_pool:" + receipt["criterion"],
                "eligible": [True]*len(candidates), "scores": receipt["scores"],
                "fallback": receipt["fallback"], "kappa": 0.,
                "feasibility_proxy_no_status": pf.tolist(),
                "information_value_decision": receipt, "probability_guarantee": False}
        self._decision_cache[key] = deepcopy((index, meta))
        return index, meta
