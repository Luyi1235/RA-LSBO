"""RA-LSBO-H2 opt-in causally updated SOC risk-head controller.

The *decision at iteration t* uses ONLY previously settled prequential
Bernoulli feasibility Brier scores. Both experts predict the SAME chosen
design before its evaluation; the next observation then updates the score.

This is a response-head choice controller, not a VAE/GP, a probability
calibrator, a safety certificate, or a post-hoc oracle. Censored intervals
with a resolvable g<=0 sign are legitimate Brier labels. Ambiguous intervals,
missing outcomes, scheduled coverage and model failure do not fabricate
training targets or update expert weights.

Registered fixed hyperparameters are intentionally outcome-independent.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Optional

from scipy.special import log_ndtr

from ...risk_core.observations import OutputObservation


VIEWS=("onehot","latent")
SCHEMA="ra_lsbo_h2_causal_risk_controller_v1"


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class H2Settings:
    prior_onehot: float=.75
    learning_rate: float=1.
    minimum_resolved_pairs: int=3
    log_odds_clip: float=4.
    max_steps: int=48

    def __post_init__(self):
        if (not 0 < self.prior_onehot < 1 or
            not math.isfinite(self.prior_onehot) or
            not math.isfinite(self.learning_rate) or self.learning_rate<=0 or
            not math.isfinite(self.log_odds_clip) or self.log_odds_clip<=0 or
            type(self.minimum_resolved_pairs) is not int or self.minimum_resolved_pairs<1 or
            type(self.max_steps) is not int or self.max_steps<1):
            raise ValueError("Invalid preregistered H2 causal risk settings")


@dataclass(frozen=True)
class Gaussian:
    mean: float
    variance: float

    def __post_init__(self):
        if (not math.isfinite(self.mean) or not math.isfinite(self.variance)
            or self.variance<=0):
            raise ValueError("A valid GP view needs finite mean and positive variance")

    def probability_feasible(self):
        # Phi(-mu/sigma) = 1/2 erfc(mu/(sqrt(2)*sigma)).
        return .5*math.erfc(self.mean/math.sqrt(2*self.variance))


def _interval_logmass(lower,upper):
    if not lower<upper:
        raise ValueError("Empty/invalid standardized SOC interval")
    if lower==-math.inf:return float(log_ndtr(upper))
    if upper==math.inf:return float(log_ndtr(-lower))
    if lower>=0:
        large,small=float(log_ndtr(-lower)),float(log_ndtr(-upper))
    else:
        large,small=float(log_ndtr(upper)),float(log_ndtr(lower))
    return float(large+math.log(-math.expm1(small-large)))


def proper_evidence_score(gaussian,observation):
    """Diagnostic only; NEVER drives H2 weight updates or future-kind choice.

    Exact continuous density NLL and interval probability-mass NLL have
    different measures and must never be combined into a single score.
    """
    if gaussian is None or observation.kind=="missing":
        return {"kind":observation.kind,"nll":None,"score_type":None}
    sd=math.sqrt(gaussian.variance)
    if observation.kind=="exact":
        z=(observation.value-gaussian.mean)/sd
        nll=.5*math.log(2*math.pi*gaussian.variance)+.5*z*z
        score="exact_density_nll"
    elif observation.kind=="interval":
        low=-math.inf if observation.lower is None else (observation.lower-gaussian.mean)/sd
        high=math.inf if observation.upper is None else (observation.upper-gaussian.mean)/sd
        nll=-_interval_logmass(low,high)
        score="interval_mass_nll"
    else:
        raise ValueError("Unknown SOC evidence kind")
    return {"kind":observation.kind,"nll":nll if math.isfinite(nll) else None,
            "score_type":score,"numerically_finite":math.isfinite(nll)}


class CausalRiskController:
    """Precommit/settle journal with strictly forward causal weight updates.

    No control can start the next iteration while a forecast is pending.
    Choose one GP *head* for the entire iteration; never choose a view
    per candidate using unobserved prediction scores. On all-head failure
    it asks its caller to use separately registered deterministic coverage.
    """
    def __init__(self,settings=None):
        self.settings=settings or H2Settings()
        self._events=[]
        self._pending=None
        self._loss_onehot=0.
        self._loss_latent=0.
        self._paired_resolved=0

    def _log_odds(self):
        s=self.settings
        prior=math.log(s.prior_onehot/(1-s.prior_onehot))
        logit=prior-s.learning_rate*(self._loss_onehot-self._loss_latent)
        return max(-s.log_odds_clip,min(s.log_odds_clip,logit))

    def state(self):
        x=self._log_odds()
        w=1/(1+math.exp(-x))
        return {"next_step":len(self._events),
                "resolved_pairs":self._paired_resolved,
                "onehot_cumulative_brier":self._loss_onehot,
                "latent_cumulative_brier":self._loss_latent,
                "onehot_weight":w,
                "currently_preferred":"onehot" if (
                    self._paired_resolved<self.settings.minimum_resolved_pairs
                    or w>=.5) else "latent",
                "awaiting_observation":self._pending is not None}

    def begin(self,step,candidate_id,prefix_digest,forecasts):
        if self._pending is not None:
            raise RuntimeError("Cannot forecast next design with unsettled SOC feedback")
        if (type(step) is not int or step!=len(self._events)
            or step>=self.settings.max_steps):
            raise ValueError("H2 step is not next chronologically registered iteration")
        if not isinstance(candidate_id,str) or not candidate_id:
            raise ValueError("Canonical candidate identity required")
        if not isinstance(prefix_digest,str) or not prefix_digest:
            raise ValueError("History prefix identity required")
        if not isinstance(forecasts,dict) or set(forecasts)!=set(VIEWS):
            raise ValueError("Both registered SOC expert keys must be present")
        if any(v is not None and not isinstance(v,Gaussian) for v in forecasts.values()):
            raise ValueError("Invalid Gaussian response-head envelope")
        before=self.state()
        live=[k for k,v in forecasts.items() if v is not None]
        if not live:
            selected=None
            reason="no_valid_soc_head_registered_coverage_required"
        elif len(live)==1:
            selected=live[0]
            reason="one_valid_head_no_brier_pair"
        else:
            selected=before["currently_preferred"]
            reason=("onehot_cold_shrinkage" if before["resolved_pairs"]<
                    self.settings.minimum_resolved_pairs else
                    "past_brier_online_head_choice")
        probabilities={name:(None if forecasts[name] is None
                             else forecasts[name].probability_feasible())
                       for name in VIEWS}
        commitment={
            "schema":SCHEMA,"step":step,"candidate_id":candidate_id,
            "prefix_digest":prefix_digest,"selected_view":selected,"reason":reason,
            "decision_state_before":before,
            "expert_moments":{name:(None if forecasts[name] is None
                else asdict(forecasts[name])) for name in VIEWS},
            "expert_feasibility_probabilities":probabilities,
        }
        commitment["commitment_sha256"]=digest(commitment)
        self._pending=commitment
        return json.loads(json.dumps(commitment,sort_keys=True,allow_nan=False))

    def settle(self,step,candidate_id,observation):
        pending=self._pending
        if pending is None:
            raise RuntimeError("No prior prediction to settle")
        if (type(step) is not int or step!=len(self._events)
            or pending["step"]!=step or candidate_id!=pending["candidate_id"]):
            raise ValueError("Physical feedback does not belong to the pending design/step")
        if not isinstance(observation,OutputObservation):
            raise ValueError("Only explicit SOC evidence can settle H2")
        resolved=observation.constraint_satisfaction()
        p=pending["expert_feasibility_probabilities"]
        paired=resolved is not None and all(p[k] is not None for k in VIEWS)
        brier={k:((p[k]-float(resolved))**2 if resolved is not None and p[k] is not None
                   else None) for k in VIEWS}
        if paired:
            self._loss_onehot+=brier["onehot"]
            self._loss_latent+=brier["latent"]
            self._paired_resolved+=1
        diagnostics={
            name:proper_evidence_score(
                None if pending["expert_moments"][name] is None
                else Gaussian(**pending["expert_moments"][name]),observation)
            for name in VIEWS}
        event={
            "step":step,"candidate_id":candidate_id,
            "commitment_sha256":pending["commitment_sha256"],
            "selected_view":pending["selected_view"],
            "reason":pending["reason"],
            "resolved_feasible":resolved,
            "observation_kind":observation.kind,
            "observation_evidence_id":observation.evidence_id,
            "feedback_pair_scored":paired,"brier_by_view":brier,
            "proper_evidence_score_diagnostic":diagnostics,
        }
        self._events.append(event)
        self._pending=None
        event["state_after"]=self.state()
        return json.loads(json.dumps(event,sort_keys=True,allow_nan=False))

    def journal(self):
        return {
            "schema":SCHEMA,"settings":asdict(self.settings),
            "events":json.loads(json.dumps(self._events,sort_keys=True,allow_nan=False)),
            "pending":json.loads(json.dumps(self._pending,sort_keys=True,allow_nan=False))
                       if self._pending is not None else None,
            "state":self.state(),
        }
