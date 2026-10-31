"""Actual VAE/custom GP tests with synthetic labels; no real evaluator calls.

Only the frozen schema/record loader is narrowed to those two genuine modules;
BoTorch's unrelated native baseline is not imported. No GP or VAE is mocked.
"""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_structure_corpus import MODULES, VARIABLES, BASE, raw_records
from ra_lsbo.budget import digest
from ra_lsbo import features as base_features
from ra_lsbo.partial_bo import models
from ra_lsbo.structured_encoder import hierarchy_space
from ra_lsbo.soc_revision.optimizer import RevisionSettings
from ra_lsbo.soc_revision.posterior import RiskPosterior
from ra_lsbo.experiments.simplified_aircraft import corpus_optimizer as c
from ra_lsbo.risk_core.observations import RunObservation, OutputObservation
from ra_lsbo.partial_bo.records import DesignRecord

class SchemaProblem:
    contract_id='test-contract'
    required_constraints=('soc',)
    analytical_constraints=()
    objective_reference_scale=1.
    def __init__(self):
        self.space=hierarchy_space(MODULES,VARIABLES)
        self.candidate_class=MODULES['space'].Candidate
    def canonical(self,raw):return self.space.canonical(self.candidate_class(**raw)).to_dict()
    def precheck(self,raw):return {}

@pytest.fixture
def problem(monkeypatch):
    torch.set_num_threads(1)
    loader=lambda base:(MODULES,{})
    monkeypatch.setattr(c,'load_base',loader)
    monkeypatch.setattr(base_features,'load_base',loader)
    monkeypatch.setattr(models,'load_base',loader)
    return SchemaProblem()

def settings(horizon=10):
    return RevisionSettings(horizon=horizon,encoder_epochs=30,latent=True,local=False,
        continuous_grid=(.4,),discrete_grid=(1.5,),average_kernel=False)

def candidates(p,i):
    out=[]
    for j,d in enumerate(p.space.configurations()):
        raw=p.space.default().to_dict();raw['discrete']=d
        raw['continuous']['area']=.05+(i*12+j)/300
        out.append(p.canonical(raw))
    return out

def observation(p,raw,i):
    return DesignRecord(raw,RunObservation(digest(raw),p.contract_id,f'online-{i}','completed',
        OutputObservation('exact',f'y-{i}',value=1.+raw['continuous']['area']),
        (('soc',OutputObservation('exact',f'g-{i}',value=-.2)),)),{})

@pytest.mark.parametrize('n',[0,-1,True,1.5])
def test_reject_bad_horizon(n):
    with pytest.raises(ValueError):c.training_iterations(n)

def test_schedule():assert c.training_iterations(30)==[0,8,16,24]

def test_fixed_encoder_settings_and_unsupported_flags(problem,tmp_path):
    with pytest.raises(ValueError):c.CorpusDiscoveryBO(problem,BASE,3,replace(settings(),latent=False),tmp_path)
    with pytest.raises(ValueError):c.CorpusDiscoveryBO(problem,BASE,3,replace(settings(),encoder_epochs=120),tmp_path)
    assert c.encoder_settings().initial_epochs==30 and c.encoder_settings().update_epochs==20

def test_joint_features_head_adapter_is_numerically_identical(problem):
    history=raw_records(problem.space,5)
    original=RiskPosterior(problem,BASE,history,4,settings())
    injected=c.SnapshotRiskPosterior(problem,history,4,settings(),original.features)
    for name in ['completion','objective']:
        a=getattr(original,name).predict(original.features.transform(candidates(problem,0)))
        b=getattr(injected,name).predict(injected.features.transform(candidates(problem,0)))
        np.testing.assert_array_equal(a['mean'],b['mean']);np.testing.assert_array_equal(a['variance'],b['variance'])
    assert original.diagnostics()==injected.diagnostics()

def test_full_schedule_state_chain_and_audit(problem,tmp_path):
    initial=raw_records(problem.space,5);history=list(initial);online=[];props=[]
    engine=c.CorpusDiscoveryBO(problem,BASE,7,settings(9),tmp_path)
    first_state=None
    for i in range(9):
        raw,meta=engine.propose(history,candidates(problem,i),i)
        prop={'iteration':i,'candidate':raw,'prefix_digest':digest([r.to_dict() for r in history]),'metadata':meta}
        props.append(prop);r=observation(problem,raw,i);online.append(r);engine.accept(r);history.append(r)
        if i==0:
            first_state=engine.session._sealed_state
            assert meta['corpus_training']['cumulative_gradient_updates']==480
        if 0<i<8:assert engine.session._sealed_state==first_state
    assert engine.session.total_updates==800 and len(engine.session.events)==2
    result=c.audit_training(problem,BASE,7,initial,props,online,tmp_path,settings(9))
    assert result['fit_count']==2 and result['gradient_updates']==800
    assert result['fit_seconds']==sum(engine.session.fit_seconds)
    assert [e['iteration'] for e in result['checkpoint_events']]==[0,8]
    assert engine.session.events[-1]['observed_rows']==13
    assert engine.session.events[-1]['parameter_state_before']==first_state
    # Feature history uses current observations even between encoder updates.
    assert props[7]['metadata']['corpus_training']['latest']['observed_rows']==5
    assert props[7]['metadata']['corpus_training']['current_gp_prefix']==props[7]['prefix_digest']
    bad=deepcopy(props);bad[3]['metadata']['corpus_training']['cumulative_gradient_updates']=900
    with pytest.raises(ValueError):c.audit_training(problem,BASE,7,initial,bad,online,tmp_path,settings(9))

def test_fit_before_cold_coverage_no_fake_physics(problem,tmp_path):
    history=raw_records(problem.space,5)
    history=[replace(r,observation=replace(r.observation,constraints=(('soc',OutputObservation('interval',str(i),lower=0.,lower_inclusive=False)),)))
             for i,r in enumerate(history)]
    engine=c.CorpusDiscoveryBO(problem,BASE,7,settings(),tmp_path)
    raw,meta=engine.propose(history,candidates(problem,0),0)
    assert meta['posterior_fit_skipped'] and meta['coverage_reason']=='scheduled_coverage'
    assert meta['corpus_training']['representation_fit_on_this_proposal']
    assert engine.session.total_updates==480
    assert (tmp_path/'representation/iteration_000/state.npz').exists()
    with pytest.raises(ValueError):engine.propose(history,candidates(problem,0),0)

def test_history_rewrite_and_unknown_refused_before_fit(problem,tmp_path):
    history=raw_records(problem.space,5)
    bad=replace(history[0],observation=replace(history[0].observation,contract_id='foreign'))
    engine=c.CorpusDiscoveryBO(problem,BASE,7,settings(),tmp_path)
    with pytest.raises(ValueError):engine.propose([bad]+history[1:],candidates(problem,0),0)
    assert not (tmp_path/'representation').exists()

def test_no_overwrite_or_implicit_retry(problem,tmp_path):
    history=raw_records(problem.space,5)
    folder=tmp_path/'representation/iteration_000';folder.mkdir(parents=True)
    engine=c.CorpusDiscoveryBO(problem,BASE,7,settings(),tmp_path)
    with pytest.raises(FileExistsError):engine.propose(history,candidates(problem,0),0)
    assert engine.failed
    with pytest.raises(ValueError):engine.propose(history,candidates(problem,0),0)

def test_checkpoint_mutation_refused(problem,tmp_path):
    initial=raw_records(problem.space,5)
    engine=c.CorpusDiscoveryBO(problem,BASE,7,settings(),tmp_path)
    raw,meta=engine.propose(initial,candidates(problem,0),0)
    prop={'iteration':0,'candidate':raw,'prefix_digest':digest([r.to_dict() for r in initial]),'metadata':meta}
    row=observation(problem,raw,0)
    path=tmp_path/'representation/iteration_000/diagnostics.json'
    value=json.loads(path.read_bytes());value['raw_mean_active_exact']=-1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):c.audit_training(problem,BASE,7,initial,[prop],[row],tmp_path,settings())

def test_checkpoint_files_not_created_on_foreign_step(problem,tmp_path):
    engine=c.CorpusDiscoveryBO(problem,BASE,7,settings(),tmp_path)
    with pytest.raises(ValueError):engine.propose(raw_records(problem.space,5),candidates(problem,0),1)
    assert not list(tmp_path.iterdir())
