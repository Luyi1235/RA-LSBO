"""No truth calls: fixed H2-M six-arm aircraft protocol and controller replay."""
from types import SimpleNamespace

import pytest

from ra_lsbo.budget import digest
from ra_lsbo.risk_core.observations import OutputObservation,RunObservation
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.experiments.simplified_aircraft import h2_protocol,h2_pilot
from ra_lsbo.experiments.simplified_aircraft.h2_risk_controller import (
    CausalSOCController,H2Settings)


def test_new_seed_roster_capped_48_1008():
    pr=h2_protocol
    assert pr.ARMS==("random","onehot_stable","corpus_stable",
                     "hybrid_onehot","hybrid_guard","h2_causal")
    smoke=pr.profile_for("smoke")
    pilot=pr.profile_for("pilot")
    assert smoke["seeds"]==[11061] and pilot["seeds"]==[11062,11063,11064,11065]
    assert pr.budget_for(smoke)=={
        "shared_initial":12,"online":24,"incumbent_step_verification":12,
        "maximum_model_calls":48,"maximum_rcaide_mda_calls":0}
    assert pr.budget_for(pilot)=={
        "shared_initial":96,"online":864,"incumbent_step_verification":48,
        "maximum_model_calls":1008,"maximum_rcaide_mda_calls":0}
    assert len(pr.method_order(pilot,pr.CASES[0],11062))==6


def test_h2_model_registry_uses_exact_component_mixture():
    spec=h2_protocol.method_spec("h2_causal",h2_protocol.profile_for("pilot"))
    assert spec["engine"]=="ra_lsbo_h2_causal_risk_v1"
    assert spec["controller"]["prior_onehot"]==.5
    assert spec["controller"]["delta_clip"]==4.
    assert spec["prequential_scoring"]=="exact_two_component_mixture_log_likelihood"
    assert spec["gp_stable_tail_moments"] is True
    assert spec["gp_grid_continuous"]==[.05,.15,.4,1.]
    assert spec["gp_grid_discrete"]==[.5,1.5]
    assert spec["response_policy"]["soc_mixture"]=="fixed_two_view_prior_components"


def test_h2_saved_coverage_commit_replays_without_any_GP_or_VAE():
    candidate={"x":.5}
    prefix=digest([])
    c=CausalSOCController(H2Settings())
    snapshot=c.snapshot()
    prior=c.register(digest(candidate),prefix,skip="coverage")
    proposal={"candidate":candidate,"prefix_digest":prefix,
        "metadata":{
            "method":"h2_causal","raw_history_digest":prefix,
            "posterior_fit_skipped":True,
            "coverage_reason":"scheduled_coverage",
            "H2_controller_before":snapshot,
            "H2_prior_forecast_commitment":prior}}
    packet=RunObservation(design_id=digest(candidate),contract_id="synthetic",
        evidence_id="e",status="completed",
        objective=OutputObservation("exact","o",value=1.),
        constraints=(("soc",OutputObservation("exact","soc",value=-.1)),))
    record=DesignRecord(candidate,packet,{"synthetic_only":True})
    report=h2_pilot.replay_h2_controller([proposal],[record])
    assert report["settled_events"]==1
    assert report["scored_events"]==0
    assert report["final_state"]["onehot_weight"]==.5
    proposal["metadata"]["H2_controller_before"]["scored"]=9
    with pytest.raises(ValueError,match="controller"):
        h2_pilot.replay_h2_controller([proposal],[record])


def test_h2_saved_fitted_selected_gaussian_pair_replays_one_step():
    candidate={"m":"A","x":.2}
    prefix=digest([])
    c=CausalSOCController(H2Settings(warmup_scored=4))
    snap=c.snapshot()
    mix={
        "weights":[.5,.5],
        "mean":[[-.2],[.3]],
        "variance":[[.02],[.06]]}
    original=c.register(digest(candidate),prefix,
        onehot={"mean":-.2,"variance":.02},
        latent={"mean":.3,"variance":.06})
    proposal={"candidate":candidate,"prefix_digest":prefix,
       "metadata":{"method":"h2_causal","raw_history_digest":prefix,
           "posterior_fit_skipped":False,"coverage_reason":None,
           "selected_index":0,"forecasts":{"constraints":{"soc":{
               "components":mix}}},
           "H2_controller_before":snap,"H2_prior_forecast_commitment":original}}
    record=DesignRecord(candidate,RunObservation(
        design_id=digest(candidate),contract_id="synthetic",
        evidence_id="e",status="completed",
        objective=OutputObservation("exact","o",value=1.),
        constraints=(("soc",OutputObservation("interval","soc",lower=.1)),)),
        {"synthetic_only":True})
    audited=h2_pilot.replay_h2_controller([proposal],[record])
    assert audited["scored_events"]==1
    assert audited["interval_scores"]==1
    assert audited["new_GP_fits"]==0
