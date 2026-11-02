"""Tests for H2 algorithm integration and synthetic hierarchical benchmark.

All aircraft model evaluations and GP/VAE training are prohibited here.
"""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from scipy.special import ndtr

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.simplified_aircraft import h2_optimizer
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.experiments.simplified_aircraft.dualview_protocol import model_settings,profile_for
from ra_lsbo.experiments.simplified_aircraft.corpus_optimizer import CorpusDiscoveryBO
from ra_lsbo.experiments.simplified_aircraft.h2_risk_controller import component_mixture
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.risk_core.observations import OutputObservation,RunObservation
from ra_lsbo.soc_revision.acquisition import quantities
from tools.run_h2_hierarchical_synthetic import SCENARIOS,STEPS,SEEDS,run,summarize


def test_mixture_probability_is_propagated_as_components_to_real_RA_acquisition():
    a={"mean":np.array([-.4,.2]),"variance":np.array([.01,.03])}
    b={"mean":np.array([.3,-.2]),"variance":np.array([.03,.04])}
    mix=component_mixture(a,b,.75,scale=.15)
    pred={"completion":{"mean":np.array([-1.,-1.]),
                         "variance":np.array([.05,.05])},
          "constraints":{"soc":mix},"objective":None}
    response=quantities(pred,best=None,kappa=1.96,threshold=.5)
    expected=.75*ndtr(-a["mean"]/np.sqrt(a["variance"]))+ \
             .25*ndtr(-b["mean"]/np.sqrt(b["variance"]))
    np.testing.assert_allclose(response["probabilities"]["soc"],expected,atol=1e-12)
    assert response["margins"].shape==(2,2)


def test_parent_proposal_commit_and_settle_updates_only_after_real_feedback():
    problem=problem_for("aircraft_ordinary")
    from ra_lsbo.partial_bo.problems import candidate_pool
    pool,_=candidate_pool(problem,711711,2)
    candidate=pool[0]
    engine=h2_optimizer.CausalH2CorpusBO(problem,BASE,17,
        model_settings("hybrid_onehot",profile_for("smoke")),
        "/tmp/unit-only-h2-never-created")
    prediction=component_mixture(
        {"mean":np.array([-.2]),"variance":np.array([.03])},
        {"mean":np.array([.35]),"variance":np.array([.03])},
        .5,scale=.2)
    meta={"selected_index":0,
          "posterior_fit_skipped":False,
          "coverage_reason":None,
          "forecasts":{"constraints":{"soc":{"components":{
               "weights":prediction["components"]["weights"].tolist(),
               "mean":prediction["components"]["mean"].tolist(),
               "variance":prediction["components"]["variance"].tolist()}}}}}
    def fake_parent_propose(*_args):
        engine.pending={"design":digest(candidate),"contract":problem.contract_id,
                        "records":[],"prefix_digest":digest([]),
                        "branch":engine._branch(candidate),"prior_branch_best":None}
        return candidate,meta
    with patch.object(CorpusDiscoveryBO,"propose",side_effect=fake_parent_propose):
        selected,reported=engine.propose([],pool,0)
    assert selected==candidate
    assert engine.h2.n_scored==0 and engine.h2.weight==.5
    assert reported["H2_controller_before"]["scored"]==0
    assert reported["H2_prior_forecast_commitment"]["onehot_prior"]=={
        "mean":-.2,"variance":.03}
    result=RunObservation(
        design_id=digest(candidate),contract_id=problem.contract_id,
        evidence_id="synthetic_only",
        status="completed",
        objective=OutputObservation("exact","o",value=10000.),
        constraints=tuple((name,OutputObservation("exact",name,value=-.05))
                          for name in problem.required_constraints))
    row=DesignRecord(candidate,result,{"synthetic":True})
    engine.accept(row)
    assert engine.h2.n_scored==1
    assert engine.h2.pending is None and len(engine.controller_events())==1
    assert engine.h2.weight==.5  # warmup_scored=4


def test_h2_cold_coverage_does_not_invent_GP_predictions():
    problem=problem_for("aircraft_ordinary")
    from ra_lsbo.partial_bo.problems import candidate_pool
    pool,_=candidate_pool(problem,711712,3)
    engine=h2_optimizer.CausalH2CorpusBO(problem,BASE,17,
        model_settings("hybrid_onehot",profile_for("smoke")),
        "/tmp/unit-only-h2-never-created-coverage")
    def coverage_parent(*args):
        candidate=pool[0]
        engine.pending={"design":digest(candidate),"contract":problem.contract_id,
                        "records":[],"prefix_digest":digest([]),
                        "branch":engine._branch(candidate),"prior_branch_best":None}
        return candidate,{"selected_index":0,
                 "posterior_fit_skipped":True,
                 "coverage_reason":"scheduled_coverage"}
    with patch.object(CorpusDiscoveryBO,"propose",side_effect=coverage_parent):
        _,meta=engine.propose([],pool,0)
    assert meta["H2_prior_forecast_commitment"]["skip"]=="coverage"
    assert engine.h2.n_scored==0


def test_multiple_synthetic_task_streams_are_seed_deterministic_and_no_oracle():
    assert len(SEEDS)==8 and len(SCENARIOS)==4 and STEPS==96
    rows,a=run(SEEDS[0],SCENARIOS[0])
    same,b=run(SEEDS[0],SCENARIOS[0])
    assert rows==same and a==b
    assert len(rows)==96
    assert set(r["observed_kind"] for r in rows)=={"exact","interval"}
    assert rows[0]["controller_weight_before"]==.5
    assert all(.1<=r["controller_weight_after"]<=.9 for r in rows)
    scored=summarize(rows)
    assert len([r for r in scored if r["n"]])==8
    assert all(r["mean_nll"] is not None for r in scored if r["n"])
