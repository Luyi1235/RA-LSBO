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
    assert reported["H2_prior_forecast_commitment"]["onehot"]=={
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


def test_real_stable_tail_EP_mixed_SOC_heads_with_math_only_hierarchy():
    """Fit original 4x2 mixed-likelihood GPs on purely synthetic records."""
    from ra_lsbo.experiments.simplified_aircraft.h2_optimizer import H2RiskPosterior
    class Features:
        def __init__(self,latent):
            self.latent=latent
            self.identity="synthetic_latent" if latent else "synthetic_onehot"
        def transform(self,candidates):
            out=[]
            for d in candidates:
                m=d["material"]
                discrete=([float(m==0),float(m==1)]
                           if not self.latent else
                           [float(m),float(1-m)])
                out.append([d["x"],d["z"],1.,float(d["mask"]),
                            *discrete])
            return np.array(out,float)
    problem=SimpleNamespace(
        contract_id="mathematical_hierarchy_only",
        required_constraints=("soc",),
        analytical_constraints=(),
        space=SimpleNamespace(continuous=("x","z")),
        objective_reference_scale=1.)
    history=[]
    for i in range(9):
        d={"material":i%2,"mask":int(i%2==1),
           "x":(i+1)/11,"z":((i*3)%8)/10 if i%2 else 0.}
        soc=(OutputObservation("interval",f"s{i}",lower=.02)
             if i%4==0 else OutputObservation("exact",f"s{i}",value=-.2+.19*i/8))
        packet=RunObservation(
            design_id=digest(d),contract_id=problem.contract_id,
            evidence_id=f"synthetic{i}",status="completed",
            objective=OutputObservation("exact",f"o{i}",value=1.5+.03*i),
            constraints=(("soc",soc),))
        history.append(DesignRecord(d,packet,{"mathematical_only":True}))
    settings=SimpleNamespace(average_kernel=False,
        continuous_grid=(.05,.15,.4,1.),discrete_grid=(.5,1.5))
    one=Features(False)
    lat=Features(True)
    model=H2RiskPosterior(problem,history,17,settings,lat,one,.25,
        {"schema":"synthetic_prefix_only"})
    test_designs=[{"material":m,"mask":int(m==1),"x":.36,"z":.42 if m else 0.}
                  for m in (0,1)]
    pred=model.predict(test_designs)
    soc=pred["constraints"]["soc"]
    assert soc["components"]["weights"].tolist()==pytest.approx([.25,.75])
    assert soc["mean"].shape==(2,)
    assert all(np.isfinite(soc["variance"]))
    assert np.all(soc["variance"]>0)
    before=model.diagnostics()
    assert before["SOC_probability_rule"]=="componentwise_gaussian_mixture_CDF"
    assert len(before["heads"]["soc"]["candidates"])==8
    assert len(before["heads"]["soc:latent_guard"]["candidates"])==8
    assert before["heads"]["soc"]["selected"]["status"]=="ok"
    assert before["heads"]["soc:latent_guard"]["selected"]["status"]=="ok"
