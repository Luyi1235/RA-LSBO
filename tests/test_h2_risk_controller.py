"""Causal risk controller adversarial and exact/interval tests. No evaluator."""
import math

import numpy as np
import pytest
from scipy.special import ndtr

from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.experiments.simplified_aircraft.h2_risk_controller import (
    H2Settings,CausalSOCController,proper_nll,component_mixture)


def exact(v):
    return OutputObservation("exact","test-exact",value=v)


def interval(lower=None,upper=None):
    return OutputObservation("interval","test-interval",lower=lower,upper=upper)


def test_proper_scoring_exact_and_censored_distinct():
    p={"mean":0.,"variance":1.}
    assert proper_nll(p,exact(.1))["nll"]==pytest.approx(.5*math.log(2*math.pi)+.005)
    assert proper_nll(p,interval(lower=0.))["nll"]==pytest.approx(math.log(2))
    assert proper_nll(p,interval(upper=0.))["nll"]==pytest.approx(math.log(2))
    assert proper_nll(p,interval(lower=-1.,upper=1.))["nll"]==pytest.approx(
        -math.log(ndtr(1)-ndtr(-1)))
    assert proper_nll(p,OutputObservation("missing","test-m",reason="not_evaluated")) is None


def test_no_future_observation_before_committed_proposal_and_one_settlement():
    c=CausalSOCController(H2Settings(warmup_scored=1))
    p1={"mean":-2.,"variance":.5}
    p2={"mean":2.,"variance":.5}
    before=c.snapshot()
    commitment=c.register("design0","prefix0",p1,p2)
    assert c.snapshot()["onehot_weight"]==before["onehot_weight"]==.5
    assert c.pending["commitment"]==commitment["commitment"]
    with pytest.raises(RuntimeError):
        c.register("design1","prefix1",p1,p2)
    with pytest.raises(ValueError):
        c.settle("other_design","prefix0",exact(-2.))
    with pytest.raises(ValueError):
        c.settle("design0","wrong_prefix",exact(-2.))
    event=c.settle("design0","prefix0",exact(-2.))
    assert event["delta_raw"]>0
    assert c.weight>.5
    assert event["weight_before"]==.5
    assert event["weight_after"]==c.weight
    with pytest.raises(RuntimeError):
        c.settle("design0","prefix0",exact(-2.))


def test_single_extreme_censored_value_is_clipped_and_weight_bounded():
    c=CausalSOCController(H2Settings(warmup_scored=1,delta_clip=2.))
    for i in range(50):
        c.register(f"d{i}",f"p{i}",
                   {"mean":-10.,"variance":.01},
                   {"mean":10.,"variance":.01})
        event=c.settle(f"d{i}",f"p{i}",interval(upper=-3.))
        assert event["delta_capped"]==2.
        assert .1<=c.weight<=.9
    assert c.weight==pytest.approx(.9)
    assert c.n_scored==50


def test_new_observation_type_cannot_choose_view_before_prediction():
    prior={"mean":-.5,"variance":.1}
    competing={"mean":.5,"variance":.1}
    a,b=CausalSOCController(H2Settings(warmup_scored=1)),CausalSOCController(H2Settings(warmup_scored=1))
    p=a.register("c","prefix",prior,competing)
    q=b.register("c","prefix",prior,competing)
    assert p==q
    assert p["weight_at_proposal"]==.5
    a.settle("c","prefix",exact(-.4))
    b.settle("c","prefix",interval(lower=0.))
    assert a.weight!=b.weight
    assert a.events[0]["score_type"]=="exact_density"
    assert b.events[0]["score_type"]=="interval_mass"


def test_scheduled_coverage_and_missing_outcomes_cannot_generate_scores():
    c=CausalSOCController(H2Settings(warmup_scored=1))
    before=c.snapshot()["onehot_weight"]
    c.register("x","p",skip="coverage")
    e=c.settle("x","p",exact(-1.))
    assert e["onehot_nll"] is None
    assert c.n_scored==0 and c.weight==before
    c.register("y","p2",{"mean":-.2,"variance":1.},{"mean":.2,"variance":1.})
    e2=c.settle("y","p2",OutputObservation("missing","no-target",reason="simulation_unavailable"))
    assert e2["skipped"]=="missing_SOC"
    assert c.n_scored==0


def test_exact_gaussian_mixture_CDF_is_not_moment_match():
    one={"mean":np.array([-.3,-.1]),"variance":np.array([.01,.03])}
    lat={"mean":np.array([.4,.25]),"variance":np.array([.03,.03])}
    mixed=component_mixture(one,lat,.75,scale=.3)
    components=mixed["components"]
    p=(components["weights"]@ndtr(-components["mean"]/np.sqrt(components["variance"])))
    expected=.75*ndtr(-one["mean"]/np.sqrt(one["variance"])) + .25*ndtr(
        -lat["mean"]/np.sqrt(lat["variance"]))
    np.testing.assert_allclose(p,expected,rtol=0,atol=1e-13)
    assert not np.allclose(p,ndtr(-mixed["mean"]/np.sqrt(mixed["variance"])))
    assert np.all(mixed["variance"]>0)
    assert mixed["scale"]==.3


def test_predictive_component_validation_and_no_arbitrary_fallback():
    a={"mean":np.array([0.]),"variance":np.array([1.])}
    b={"mean":np.array([0.,1.]),"variance":np.array([1.,1.])}
    with pytest.raises(ValueError):
        component_mixture(a,b,.5,1.)
    with pytest.raises(ValueError):
        component_mixture(a,a,float("nan"),1.)
    with pytest.raises(ValueError):
        CausalSOCController().register("c","prefix",skip="bad_skip")
    with pytest.raises(ValueError):
        H2Settings(prior_onehot=1.)


def test_cumulative_state_is_replay_deterministic():
    def exercise():
        c=CausalSOCController(H2Settings(warmup_scored=2))
        for i in range(4):
            c.register(str(i),f"p{i}",{"mean":-.15,"variance":.02},
                       {"mean":.35,"variance":.04})
            c.settle(str(i),f"p{i}",exact(-.1) if i%2 else interval(lower=0.))
        return c
    a,b=exercise(),exercise()
    assert a.snapshot()==b.snapshot()
    assert a.events==b.events
    assert a.snapshot()["event_chain"]!=CausalSOCController().snapshot()["event_chain"]
