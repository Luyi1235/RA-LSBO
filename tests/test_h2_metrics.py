"""Scoring exact H2 Gaussian mixture, including censored observations."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import ndtr,logsumexp

from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.experiments.simplified_aircraft.h2_metrics import (
    score_mixture,h2_prequential_soc)


def pair():
    return {"weights":[.65,.35],
            "mean":[[-.2],[.3]],
            "variance":[[.04],[.01]]}


def test_mixture_exact_density_is_not_moment_matched():
    obs=OutputObservation("exact","unit",value=-.05)
    data=score_mixture(pair(),0,obs)
    comps=pair()
    densities=[]
    for w,mu,var in zip(comps["weights"],[-.2,.3],[.04,.01]):
        densities.append(math.log(w)-.5*math.log(2*math.pi*var)-
                         .5*((obs.value-mu)**2/var))
    assert data["nll"]==-logsumexp(densities)
    assert data["score_kind"]=="exact_density"
    cdf=.65*ndtr((obs.value+.2)/.2)+.35*ndtr((obs.value-.3)/.1)
    assert data["covered95"]==(.025<=cdf<=.975)


def test_one_sided_censor_score_is_exact_mixture_mass():
    obs=OutputObservation("interval","unit",lower=0.)
    result=score_mixture(pair(),0,obs)
    mass=.65*(1-ndtr((0+.2)/.2))+.35*(1-ndtr((0-.3)/.1))
    assert result["nll"]==pytest.approx(-math.log(mass),abs=1e-12)
    assert result["score_kind"]=="interval_mass"
    assert result["covered95"] is None
    assert result["brier"] is not None


def test_mixture_prequential_preserves_unfitted_coverage():
    def item(kind):
        candidate={"x":1}
        o=OutputObservation("exact","e",value=-.12) if kind=="exact" else \
          OutputObservation("interval","i",lower=.05)
        proposal={"candidate":candidate,"metadata":{
            "posterior_fit_skipped":False,"selected_index":0,
            "forecasts":{"constraints":{"soc":{"components":pair()}}}}}
        record=SimpleNamespace(candidate=candidate,
            observation=SimpleNamespace(constraint=lambda _name:o))
        return proposal,record
    p1,r1=item("exact")
    p2,r2=item("interval")
    skipped={"candidate":{"x":1},"metadata":{"posterior_fit_skipped":True}}
    report=h2_prequential_soc([p1,p2,skipped],[r1,r2,r1])
    assert report["exact"]==1
    assert report["interval"]==1
    assert report["skipped_no_forecast"]==1
    assert report["resolved_feasibility_brier_n"]==2
    assert report["uncertainty_semantics"].startswith("exact")


def test_invalid_component_distribution_does_not_fall_back():
    p=pair()
    p["variance"][0][0]=-.1
    with pytest.raises(ValueError):
        score_mixture(p,0,OutputObservation("exact","e",value=0.))
