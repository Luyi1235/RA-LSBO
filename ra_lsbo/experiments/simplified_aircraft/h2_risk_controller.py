"""RA-LSBO-H2: bounded prefix-causal SOC view controller, opt-in only."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.special import log_ndtr
from ...risk_core.observations import OutputObservation


def stamp(obj):
    return sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class H2Settings:
    prior_onehot: float = .5
    min_onehot: float = .1
    max_onehot: float = .9
    warmup_scored: int = 4
    eta: float = .35
    delta_clip: float = 4.

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in (self.prior_onehot,
                self.min_onehot,self.max_onehot,self.eta,self.delta_clip))
            or not 0 < self.min_onehot <= self.prior_onehot <= self.max_onehot < 1
            or self.eta <= 0 or self.delta_clip <= 0
            or type(self.warmup_scored) is not int or self.warmup_scored < 1):
            raise ValueError("Invalid registered controller parameters")


def checked(p):
    if not isinstance(p,dict) or set(p)!={"mean","variance"}:
        raise ValueError("Exactly one Gaussian mean/variance pair required")
    mu,var=float(p["mean"]),float(p["variance"])
    if not math.isfinite(mu) or not math.isfinite(var) or var<=0:
        raise ValueError("Nonfinite or degenerate SOC Gaussian")
    return {"mean":mu,"variance":var}


def log_interval(lo,hi):
    if not lo<hi:
        raise ValueError("Empty or reversed censored interval")
    if lo==-math.inf:return float(log_ndtr(hi))
    if hi==math.inf:return float(log_ndtr(-lo))
    if lo>=0:
        large,small=log_ndtr(-lo),log_ndtr(-hi)
    else:
        large,small=log_ndtr(hi),log_ndtr(lo)
    return float(large+np.log(-np.expm1(small-large)))


def proper_nll(pred,evidence):
    """Exact density and interval mass remain separate observation types."""
    mu,var=(v for v in checked(pred).values())
    sd=math.sqrt(var)
    if not isinstance(evidence,OutputObservation):
        raise TypeError("Expected original immutable SOC evidence")
    if evidence.kind=="missing":return None
    if evidence.kind=="exact":
        z=(evidence.value-mu)/sd
        nll=.5*math.log(2*math.pi*var)+.5*z*z
        kind="exact_density"
    elif evidence.kind=="interval":
        lo=-math.inf if evidence.lower is None else (evidence.lower-mu)/sd
        hi=math.inf if evidence.upper is None else (evidence.upper-mu)/sd
        nll=-log_interval(lo,hi)
        kind="interval_mass"
    else:raise ValueError("Unsupported SOC observation")
    if not math.isfinite(nll):
        raise FloatingPointError("Nonfinite exact/interval proper score")
    return {"type":kind,"nll":float(nll)}


def component_mixture(onehot,latent,weight,scale):
    """Use exact mixture components for acquisition CDFs; margins use moments."""
    w=float(weight)
    if not math.isfinite(w) or not 0<=w<=1 or not math.isfinite(scale) or scale<=0:
        raise ValueError("Invalid mixed SOC weight or scale")
    a,b=np.asarray(onehot["mean"],float),np.asarray(latent["mean"],float)
    av,bv=np.asarray(onehot["variance"],float),np.asarray(latent["variance"],float)
    if (a.ndim!=1 or a.shape!=b.shape or a.shape!=av.shape or a.shape!=bv.shape
        or not all(np.isfinite(x).all() for x in (a,b,av,bv))
        or (av<=0).any() or (bv<=0).any()):
        raise ValueError("Mismatched or invalid component SOC predictions")
    weights=np.array([w,1-w])
    means=np.stack([a,b])
    vars_=np.stack([av,bv])
    mean=weights@means
    var=weights@(vars_+means**2)-mean**2
    if not np.isfinite(var).all() or (var<=0).any():
        raise FloatingPointError("Invalid mixture moments")
    return {"mean":mean,"variance":var,"scale":float(scale),
            "components":{"weights":weights,"mean":means,"variance":vars_},
            "h2_onehot_weight":w}


class CausalSOCController:
    """Prediction commit -> settle -> bounded update, with zero future access."""

    def __init__(self,settings=None):
        self.settings=settings or H2Settings()
        self.events=[]
        self.pending=None
        self.n_scored=0
        self.sum_clipped_delta=0.

    @property
    def weight(self):
        s=self.settings
        if self.n_scored<s.warmup_scored:return s.prior_onehot
        z=math.log(s.prior_onehot/(1-s.prior_onehot))
        z+=s.eta*self.sum_clipped_delta/math.sqrt(self.n_scored)
        z=min(60.,max(-60.,z))
        return min(s.max_onehot,max(s.min_onehot,1/(1+math.exp(-z))))

    def snapshot(self):
        return {"schema":"ra_h2_soc_causal_bounded_v1",
                "settings":asdict(self.settings),"settled":len(self.events),
                "scored":self.n_scored,"clipped_sum":self.sum_clipped_delta,
                "onehot_weight":self.weight,"event_chain":stamp(self.events),
                "pending":self.pending is not None}

    def register(self,design_id,prefix,onehot=None,latent=None,skip=None):
        if self.pending is not None:raise RuntimeError("Unsettled forecast exists")
        if not all(isinstance(x,str) and x for x in (design_id,prefix)):
            raise ValueError("Design and history identity required")
        if (skip is None)!=(onehot is not None and latent is not None):
            raise ValueError("Two prior forecasts or an explicit skip reason required")
        if skip is not None and skip not in (
            "coverage","model_failure","no_forecast"):
            raise ValueError("Unknown skip reason")
        p={"step":len(self.events),"design_id":design_id,"prefix":prefix,
           "onehot":checked(onehot) if onehot is not None else None,
           "latent":checked(latent) if latent is not None else None,
           "skip":skip,"weight_at_proposal":self.weight,
           "prior_chain":stamp(self.events)}
        p["commitment"]=stamp(p)
        self.pending=p
        return dict(p)

    def settle(self,design_id,prefix,observation):
        p=self.pending
        if p is None:raise RuntimeError("No registered forecast")
        if (p["design_id"]!=design_id or p["prefix"]!=prefix or
            stamp({k:v for k,v in p.items() if k!="commitment"})!=p["commitment"]):
            raise ValueError("Mismatched settlement or modified prediction")
        if not isinstance(observation,OutputObservation):
            raise TypeError("Settlement requires actual SOC evidence")
        if abs(self.weight-p["weight_at_proposal"])>1e-14:
            raise ValueError("Future data changed the committed weight")
        event={"step":p["step"],"design_id":design_id,"prefix":prefix,
               "forecast_commitment":p["commitment"],
               "weight_before":self.weight,"observation_kind":observation.kind,
               "onehot_nll":None,"latent_nll":None,
               "delta_raw":None,"delta_capped":None,"skipped":p["skip"]}
        if p["skip"] is None and observation.kind!="missing":
            a,b=proper_nll(p["onehot"],observation),proper_nll(p["latent"],observation)
            if a["type"]!=b["type"]:raise ValueError("Mismatched observation type")
            delta=b["nll"]-a["nll"]
            cap=max(-self.settings.delta_clip,min(self.settings.delta_clip,delta))
            event.update(onehot_nll=a["nll"],latent_nll=b["nll"],
                         score_type=a["type"],delta_raw=delta,delta_capped=cap)
            self.n_scored+=1
            self.sum_clipped_delta+=cap
        elif observation.kind=="missing" and p["skip"] is None:
            event["skipped"]="missing_SOC"
        event["weight_after"]=self.weight
        event["scored_after"]=self.n_scored
        event["id"]=stamp(event)
        self.events.append(event)
        self.pending=None
        return dict(event)
