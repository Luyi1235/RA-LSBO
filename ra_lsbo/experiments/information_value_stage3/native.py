"""Adapter to the frozen latent-objective / explicit-constraint EP-GP stack.

Real histories refit the original VAE/GP. Fantasies append only observable
responses to independent scalar GP heads; freeze their CURRENT selected kernels,
normalizations and features. No true evaluator, benchmark formula, or old result
is queried by a fantasy. This is EP with fixed hyperparameters, not exact Bayes.
"""
from copy import deepcopy
from dataclasses import replace
import time
import numpy as np

from ...budget import digest
from ...partial_bo.models import EvidenceGP
from ...risk_core.observations import OutputObservation
from ..simplified_aircraft.dualview_optimizer import StableCorpusDiscoveryBO
from ..risk_budget54_engine import settings_for_math54
from .planner import FantasyFitError, choose, probability


class FrozenFantasyFactory:
    def __init__(self, posterior, candidates, forecast):
        if set(posterior.constraints) != {"region", "resource"}:
            raise ValueError("PR68 observation adapter is only for the registered mathematical cases")
        self.posterior = posterior
        self.candidates = list(candidates)
        self.forecast = forecast
        self.risk_x = posterior.risk_features.transform(candidates)
        self.objective_x = posterior.objective_features.transform(candidates)
        self.started = self.completed = 0
        self.fit_seconds = 0.
        self.head_counts = {}

    def _update(self, name, head, inputs, i, observation, old):
        if observation is None:
            return old
        if head is None:
            raise ValueError("Cannot invent an objective posterior to condition")
        if observation.kind == "exact":
            normalized = replace(observation, value=(observation.value-head.center)/head.scale)
        elif observation.kind == "interval":
            normalized = replace(observation,
                lower=None if observation.lower is None else (observation.lower-head.center)/head.scale,
                upper=None if observation.upper is None else (observation.upper-head.center)/head.scale)
        else:
            raise ValueError("Only observable exact/interval responses enter fantasy refits")
        x = np.vstack((head.model.x, np.asarray(inputs[i], float)/head.divisor))
        observations = tuple(head.model.observations) + (normalized,)
        self.started += 1
        self.head_counts[name] = self.head_counts.get(name, 0) + 1
        start = time.perf_counter()
        try:
            model = EvidenceGP(x, observations, seed=head.model.seed, settings=head.model.settings)
            prediction = model.predict(np.asarray(inputs, float)/head.divisor)
            mean = head.center + head.scale*np.asarray(prediction["mean"])
            variance = head.scale**2 * np.asarray(prediction["variance"])
            if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0):
                raise FloatingPointError("Invalid fantasy Gaussian moments")
        except (np.linalg.LinAlgError, FloatingPointError, RuntimeError) as error:
            raise FantasyFitError("%s: %s: %s" % (name, type(error).__name__, error)) from error
        finally:
            self.fit_seconds += time.perf_counter() - start
        self.completed += 1
        return {"mean": mean, "variance": variance, "scale": head.scale}

    def condition(self, index, outcome, *, use_abort_interval=True):
        if not 0 <= index < len(self.candidates):
            raise ValueError("Fantasy candidate index not in bound pool")
        identity = "fantasy:" + digest(self.candidates[index])
        region = (OutputObservation("interval", identity+":region", lower=0., lower_inclusive=False)
                  if outcome.aborted and use_abort_interval else
                  None if outcome.aborted else
                  OutputObservation("exact", identity+":region", value=outcome.region))
        resource = OutputObservation("exact", identity+":resource", value=outcome.resource)
        objective = (None if outcome.objective is None else
                     OutputObservation("exact", identity+":objective", value=outcome.objective))
        updated = {name: self._update(name, self.posterior.constraints[name], self.risk_x,
                   index, obs, self.forecast["constraints"][name])
                   for name, obs in (("region", region), ("resource", resource))}
        obj = self._update("objective", self.posterior.objective, self.objective_x,
                           index, objective, self.forecast.get("objective"))
        return {"constraints": updated, "objective": obj}

    def diagnostics(self):
        heads = dict(self.posterior.constraints)
        if self.posterior.objective is not None:
            heads["objective"] = self.posterior.objective
        return {"head_fits_started": self.started, "head_fits_completed": self.completed,
                "head_fit_seconds": self.fit_seconds, "head_counts": dict(self.head_counts),
                "fixed_head_identities": {name: head.identity for name, head in heads.items()},
                "fixed_normalization": {name: {"center": head.center, "scale": head.scale} for name, head in heads.items()},
                "hyperparameters_reselected": False, "representation_trained_on_fantasies": False,
                "real_evaluator_calls": 0, "prior_model_mutated": False}


class _BoundPosterior:
    def __init__(self, owner, raw):
        self.owner, self.raw = owner, raw

    def predict(self, candidates):
        self.owner._predicted_candidates = list(candidates)
        return self.raw.predict(candidates)

    @property
    def best(self):
        return self.raw.best

    @best.setter
    def best(self, value):
        self.raw.best = value

    def __getattr__(self, name):
        return getattr(self.raw, name)


class FantasyValueBO(StableCorpusDiscoveryBO):
    def __init__(self, problem, base, seed, output, *, arm, horizon):
        if arm not in ("cei_myopic", "cei_two_step", "cei_two_step_no_abort_update") or horizon not in (2, 6):
            raise ValueError("Unregistered GP information-value arm/horizon")
        settings = replace(settings_for_math54(), horizon=horizon, local=False)
        super().__init__(problem, base, seed, settings, output, allocation="onehot_risk")
        # The three matched arms share this single integrated action choice;
        # no calendar boundary/coverage selection overrides it. The inherited
        # numerical MODEL-failure coverage remains explicit and fully charged.
        self.coverage_scope = "none"
        self.iv_arm = arm
        self._posterior = None
        self._predicted_candidates = None
        self._decision_cache = {}

    def _build_posterior(self, records):
        raw = super()._build_posterior(records)
        self._posterior = raw
        self._decision_cache = {}
        return _BoundPosterior(self, raw)

    def _choose_acquisition(self, predictions, best, iteration, acquisition, records):
        candidates = self._predicted_candidates
        if self._posterior is None or not candidates:
            raise ValueError("Missing bound current posterior/candidates")
        key = digest([iteration, best, candidates, [r.to_dict() for r in records]])
        if key in self._decision_cache:
            return deepcopy(self._decision_cache[key])
        factory = FrozenFantasyFactory(self._posterior, candidates, predictions)
        index, receipt = choose(predictions, best, self.settings.horizon-iteration, factory,
                                arm=self.iv_arm, seed=self.seed, iteration=iteration)
        pf = probability(predictions["constraints"]["region"]) * probability(predictions["constraints"]["resource"])
        receipt.update(candidate_pool_hash=digest(candidates),
                       pre_outcome_history_hash=digest([r.to_dict() for r in records]),
                       selected_candidate_hash=digest(candidates[index]),
                       selected_predicted_failure=float(1-pf[index]),
                       fantasy_fits=factory.diagnostics(), actual_action="evaluate_fixed_budget")
        meta = {"mode": self.iv_arm + ":" + receipt["criterion"], "eligible": [True]*len(candidates),
                "scores": receipt["scores"], "fallback": receipt["fallback"], "kappa": 0.,
                "feasibility_proxy_no_status": pf.tolist(),
                "information_value_decision": receipt, "probability_guarantee": False}
        self._decision_cache[key] = deepcopy((index, meta))
        return index, meta

    def propose(self, records, global_pool, iteration):
        candidate, meta = super().propose(records, global_pool, iteration)
        if meta.get("coverage_reason") is not None:
            meta.pop("information_value_decision", None)
            meta["new_policy_model_forecast_available"] = False
        else:
            r = meta.get("information_value_decision")
            if not r or r["selected_candidate_hash"] != digest(candidate):
                raise ValueError("Information-value receipt not bound to dispatched design")
            meta["new_policy_model_forecast_available"] = True
        return candidate, meta
