"""Mechanism tests; not evidence of online optimization superiority."""
from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.stats import norm

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core import protocol as core
from ra_lsbo.experiments.discovery_stages import methods, optimizer, protocol, runner
from ra_lsbo.experiments.discovery_stages import evidence_transfer as et
from ra_lsbo.experiments.discovery_stages.reviewed_phase import prerequisite_receipt, verify_reviewed_phase
from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.soc_revision.acquisition import quantities
from test_discovery_stages import samples, fake_model


def pred(mean=0., variance=1., scale=1.):
    return dict(mean=mean, variance=variance, scale=scale)


@pytest.mark.parametrize('lo,hi', [(None,0.), (0.,None),(-.5,.5),(8.,9.),(-40.,-39.)])
def test_interval_mass_not_midpoint(lo,hi):
    obs=OutputObservation('interval','i',lower=lo,upper=hi)
    score=et.evidence_log_score(pred(),obs)
    assert np.isfinite(score) and score<=0
    if lo is None: expected=norm.logcdf(hi)
    elif hi is None: expected=norm.logsf(lo)
    elif lo>=0: expected=norm.logsf(lo)+np.log1p(-np.exp(norm.logsf(hi)-norm.logsf(lo)))
    else: expected=norm.logcdf(hi)+np.log1p(-np.exp(norm.logcdf(lo)-norm.logcdf(hi)))
    assert np.isclose(score,expected,atol=1e-10)


def test_exact_density_units_and_tail():
    obs=OutputObservation('exact','x',value=1.)
    assert np.isclose(et.evidence_log_score(pred(),obs),norm.logpdf(1.,scale=np.sqrt(1+1e-6)))
    assert np.isfinite(et.evidence_log_score(pred(100.,0.),obs))
    assert et.evidence_log_score(pred(),OutputObservation('missing','m',reason='unknown')) is None


@pytest.mark.parametrize('forecast',[pred(variance=-1),pred(mean=np.nan),pred(scale=0),pred(scale=np.inf)])
def test_bad_forecast_is_rejected(forecast):
    with pytest.raises(ValueError):et.evidence_log_score(forecast,OutputObservation('exact','x',value=0.))


def test_per_constraint_weights_use_real_intervals_and_omit_unknown():
    _,_,good,bad=samples()
    state={'region':[-math.log(2)]*2,'resource':[-math.log(2)]*2}
    forecasts={'region':[pred(2.),pred(-2.)],'resource':[pred(-.2),pred(20.)]}
    updated=et.advance(state,forecasts,bad,'prequential')
    assert et.weights(updated['region'])[0]>.5 and et.weights(updated['resource'])[0]>.5
    assert state=={'region':[-math.log(2)]*2,'resource':[-math.log(2)]*2}
    assert et.advance(state,forecasts,bad,'equal')==state
    unknown=replace(bad,observation=replace(bad.observation,status='unknown'))
    assert et.advance(state,forecasts,unknown,'prequential')==state
    missing=replace(bad,observation=replace(bad.observation,constraints=(
        ('region',OutputObservation('missing','u',reason='not_run')),
        ('resource',OutputObservation('exact','r',value=-.2)))))
    u=et.advance(state,forecasts,missing,'prequential')
    assert u['region']==state['region'] and u['resource']!=state['resource']


def test_mixture_keeps_components_and_between_expert_variance():
    class Expert:
        best=None
        def __init__(self,mean):self.mean=mean
        def predict(self,x):
            return {'completion':{'mean':np.zeros(len(x)),'variance':np.ones(len(x))},'objective':None,
                    'constraints':{'g':{'mean':np.full(len(x),self.mean),'variance':np.ones(len(x)),'scale':1.}}}
    mix=et.ConstraintMixture([Expert(-3.),Expert(1.)],{'g':[-math.log(2)]*2})
    p=mix.predict([{}]);g=p['constraints']['g']
    assert g['mean'][0]==-1 and g['variance'][0]==5
    result=quantities(p,None,1.,.5)
    assert np.isclose(result['probabilities']['g'][0],(norm.cdf(3)+norm.cdf(-1))/2)
    assert not np.isclose(result['probabilities']['g'][0],norm.cdf(1/np.sqrt(5)))


def fake_experts(monkeypatch):
    class Expert:
        calls=[]
        def __init__(self,problem,base,records,seed,settings):
            self.best=None;self.latent=settings.latent;self.calls.append(settings.latent)
        def predict(self,pool):
            n=len(pool)
            return {'completion':{'mean':np.zeros(n),'variance':np.ones(n)},'objective':None,
                    'constraints':{k:{'mean':np.full(n,1. if self.latent else -1.),
                        'variance':np.ones(n),'scale':1.} for k in ('region','resource')}}
        def diagnostics(self):return {'expert':self.latent}
    monkeypatch.setattr(et,'RiskPosterior',Expert)
    return Expert


def make(problem,mode='prequential'):
    return et.EvidenceTransferBO(problem,core.BASE,7,methods.settings_for('ra_lsbo',core.PROFILES['smoke']),mode=mode)


def test_forecasts_sealed_before_outcome_and_update_after_accept(monkeypatch):
    problem,pool,_,bad=samples();ex=fake_experts(monkeypatch)
    opt=make(problem)
    initial=deepcopy(opt.log_weights)
    c,meta=opt.propose([bad],pool,0)
    assert ex.calls==[True,False] and opt.log_weights==initial
    assert meta['selection_posterior_fit_skipped'] and not meta['posterior_fit_skipped']
    assert meta['evidence_transfer']['forecasts'] and meta['coverage_reason']=='scheduled_coverage'
    settled=replace(bad,candidate=c,observation=replace(bad.observation,design_id=digest(c)))
    proposal={'candidate':c,'iteration':0,'pool':pool,'metadata':meta,'prefix_digest':digest([bad.to_dict()])}
    opt.accept(settled)
    assert opt.log_weights!=initial
    replay=et.audit_transfer(problem,{'transfer_mode':'prequential'},[bad],[proposal],[settled])
    assert replay['final_log_weights']==opt.log_weights
    broken=deepcopy(proposal);broken['metadata']['evidence_transfer']['forecasts']['region'][0]['mean']=9.
    with pytest.raises(ValueError):et.audit_transfer(problem,{'transfer_mode':'prequential'},[bad],[broken],[settled])
    with pytest.raises(ValueError):opt.accept(settled)


def test_wrong_settlement_cannot_change_weights(monkeypatch):
    problem,pool,good,bad=samples();fake_experts(monkeypatch);opt=make(problem)
    opt.propose([bad],pool,0);old=deepcopy(opt.log_weights)
    with pytest.raises(ValueError):opt.accept(good)
    assert opt.log_weights==old


@pytest.mark.parametrize('mode',['equal','prequential'])
def test_warm_path_is_the_fixed_latent_phase_policy(monkeypatch,mode):
    problem,pool,good,_=samples();fake_model(monkeypatch,singleton=True)
    monkeypatch.setattr(et,'RiskPosterior',lambda *args:pytest.fail('No expert adaptation on warm prefix'))
    opt=make(problem,mode)
    base=optimizer.DiscoveryBO(problem,core.BASE,7,opt.settings,coverage_scope='cold')
    for obj in (opt,base):monkeypatch.setattr(obj,'_augment',lambda *args:(pool,{'local_reports':[]}))
    c,info=opt.propose([good],pool,0);b,bi=base.propose([good],pool,0)
    assert c==b and info['coverage_reason'] is None and not info['evidence_transfer']['active']
    assert info['evidence_transfer']['forecasts'] is None
    assert et.audit_transfer(problem,{'transfer_mode':mode},[good],
        [{'candidate':c,'metadata':info}], [replace(good,candidate=c,observation=replace(good.observation,design_id=digest(c)))])['cold_scored_calls']==0


def test_plans_are_new_budgeted_and_phase_pinned(tmp_path):
    p=protocol.build_plan('transfer','smoke');protocol.verify_plan(p)
    assert p['budget']['maximum_model_calls']==112 and len(p['arms'])==6
    q=protocol.build_plan('transfer','pilot')
    assert tuple(q['profile']['seeds'])==tuple(range(9801,9809))
    assert q['budget']['maximum_model_calls']==3108 and q['budget']['maximum_rcaide_mda_calls']==0
    assert set(q['tasks']['branin_sparse']['initial'])==set(map(str,range(9801,9809)))
    with pytest.raises(ValueError,match='receipt'):protocol.verify_plan(q)
    q['prerequisite']=prerequisite_receipt();protocol.verify_plan(q)
    q['profile']['seeds']=[9701]
    with pytest.raises(ValueError):protocol.verify_plan(q)
    with pytest.raises(ValueError,match='prerequisite'):runner.prepare(tmp_path/'x','transfer','pilot')
    assert not (tmp_path/'x').exists()


def test_unreviewed_phase_evidence_cannot_unlock_new_study(tmp_path):
    (tmp_path/'plan.json').write_text('{}')
    with pytest.raises(ValueError):verify_reviewed_phase(tmp_path)
