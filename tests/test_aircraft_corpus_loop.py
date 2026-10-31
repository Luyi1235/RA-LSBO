"""Corpus loop integration and evidence tests; artificial evaluator returns only."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from ra_lsbo.budget import digest
from ra_lsbo.experiments.simplified_aircraft import corpus_pilot as p
from ra_lsbo.experiments.simplified_aircraft import corpus_optimizer as co
from ra_lsbo.experiments.simplified_aircraft import corpus_audit as ca
from ra_lsbo.experiments.discovery_stages.optimizer import DiscoveryBO
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.soc_revision.posterior import RiskPosterior
from tests.test_aircraft_nominal_pilot import row


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def test_frozen_budget_and_new_seeds():
    assert p.budget_for(p.PROFILES['smoke'])['maximum_model_calls'] == 42
    assert p.budget_for(p.PROFILES['pilot']) == {'shared_initial':192,'online':2400,
            'incumbent_step_verification':80,'maximum_model_calls':2672,'maximum_rcaide_mda_calls':0}
    assert p.PROFILES['pilot']['seeds']==list(range(9921,9929))
    assert p.ARMS==('random','ra_old30','ra_old480','ra_corpus','ra_onehot')
    assert co.scheduled_updates(30)==[0,8,16,24]
    assert co.scheduled_updates(2)==[0]


@pytest.mark.parametrize('h',[0,-1,True,1.1])
def test_bad_training_horizon(h):
    with pytest.raises(ValueError):co.scheduled_updates(h)


def test_single_intervention_specs():
    a,b=p.method_spec('ra_old30',p.PROFILES['pilot']),p.method_spec('ra_old480',p.PROFILES['pilot'])
    assert a['settings']['encoder_epochs']==30 and b['settings']['encoder_epochs']==480
    aa,bb=deepcopy(a),deepcopy(b)
    for x in (aa,bb):x.pop('representation');x['settings'].pop('encoder_epochs')
    assert aa==bb
    c=p.method_spec('ra_corpus',p.PROFILES['pilot'])
    assert c['corpus_training']['count']==2000 and c['corpus_training']['scheduled_even_when_GP_skipped']
    assert c['corpus_training']['initial_and_updates']['initial_epochs']==30
    assert c['corpus_training']['initial_and_updates']['update_epochs']==20
    assert p.method_spec('ra_onehot',p.PROFILES['pilot'])['settings']['latent'] is False


@pytest.mark.parametrize('method',['bad','ev_prequential','phase_sign'])
def test_unregistered_arm(method):
    with pytest.raises(ValueError):p.method_spec(method,p.PROFILES['pilot'])


def test_order_rotation_not_outcome_dependent():
    plan={'profile':p.PROFILES['pilot']}
    orders=[p.execution_order(plan,c,s) for c in p.CASES for s in plan['profile']['seeds']]
    assert all(set(o)==set(p.ARMS) for o in orders)
    assert orders[0]!=orders[1]
    for m in p.ARMS:
        counts=[sum(o[i]==m for o in orders) for i in range(5)]
        assert max(counts)-min(counts)<=1


def history(prob,n=6):
    candidates,_=p.candidate_pool(prob,16543,n)
    return [row(prob,c,objective=1000+100*i) for i,c in enumerate(candidates)]


@pytest.mark.parametrize('latent',[False,True])
@pytest.mark.parametrize('averaging',[False,True])
def test_supplied_posterior_equals_frozen_heads(latent,averaging):
    problem=p.problem_for(p.CASES[0]);records=history(problem)
    settings=replace(p.settings_for('ra_lsbo',p.PROFILES['smoke']),
                     latent=latent,average_kernel=averaging,encoder_epochs=4)
    f=PrefixFeatures(problem,p.BASE,records,41,latent,4)
    actual=co.SuppliedFeaturePosterior(problem,records,41,settings,f)
    expected=RiskPosterior(problem,p.BASE,records,41,settings)
    candidates,_=p.candidate_pool(problem,811,8)
    assert actual.diagnostics()==expected.diagnostics()
    assert digest(p.plain(actual.predict(candidates)))==digest(p.plain(expected.predict(candidates)))


@pytest.mark.parametrize('latent',[False,True])
def test_control_logging_does_not_change_actual_proposal(latent):
    problem=p.problem_for(p.CASES[0]);records=history(problem)
    settings=replace(p.settings_for('ra_lsbo',p.PROFILES['smoke']),latent=latent,encoder_epochs=4)
    a=co.LoggedControlBO(problem,p.BASE,36,settings,cold_global=True,protect=True,coverage_scope='cold')
    b=DiscoveryBO(problem,p.BASE,36,settings,cold_global=True,protect=True,coverage_scope='cold')
    pool,_=p.candidate_pool(problem,811,10)
    caa,ma=a.propose(records,pool,0);cbb,mb=b.propose(records,pool,0)
    assert caa==cbb
    info=ma.pop('representation_fit')
    assert digest(p.plain(ma))==digest(p.plain(mb))
    assert info['encoder_fit_calls']==int(latent)


def test_cold_controls_skip_model_without_fake_cost():
    prob=p.problem_for(p.CASES[0]);rows=[row(prob,c,margin=.1,partial=True) for c in p.candidate_pool(prob,13,6)[0]]
    settings=p.settings_for('ra_lsbo',p.PROFILES['smoke'])
    e=co.LoggedControlBO(prob,p.BASE,4,settings,cold_global=True,protect=True,coverage_scope='cold')
    _,m=e.propose(rows,p.candidate_pool(prob,15,8)[0],0)
    assert m['posterior_fit_skipped'] and m['representation_fit'] is None


def isolated_training_trial(monkeypatch,folder,horizon=10):
    # Only acquisition selection is stubbed. Real 2000-draw training, snapshots,
    # journal writes and acceptance are exercised; no GP or evaluator is run.
    prob=p.problem_for(p.CASES[0]);initial=history(prob)
    settings=replace(p.settings_for('ra_lsbo',p.PROFILES['smoke']),horizon=horizon)
    e=co.CorpusDiscoveryBO(prob,p.BASE,7,settings,folder)
    def choose(self,records,pool,i):
        c=pool[0]
        return self._finish(c,records,{'posterior_fit_skipped':True},digest([r.to_dict() for r in records]))
    monkeypatch.setattr(DiscoveryBO,'propose',choose)
    rows=list(initial);online=[];props=[];snap=None
    for i in range(horizon):
        pool,_=p.candidate_pool(prob,900+i,3,{r.observation.design_id for r in rows})
        c,m=e.propose(rows,pool,i)
        props.append({'iteration':i,'candidate':c,'metadata':m})
        r=row(prob,c,objective=1200+i);e.accept(r);rows.append(r);online.append(r)
        if i==0:snap=e.current_features
    return prob,e,initial,online,props,snap


def test_persistent_updates_and_immutable_snapshots(monkeypatch,tmp_path):
    prob,e,initial,online,props,snap=isolated_training_trial(monkeypatch,tmp_path/'trial')
    assert [event['observed_rows'] for event in e.session.events]==[6,14]
    assert e.session.total_updates==800 and e.session.corpus['draws']==2000
    assert props[7]['metadata']['persistent_representation']['last_fit_iteration']==0
    assert props[8]['metadata']['persistent_representation']['last_fit_iteration']==8
    assert snap.state_hash!=e.current_features.state_hash
    assert snap.transform([initial[0].candidate]).shape==(1,16)
    result=ca.audit_representation(prob,p.BASE,tmp_path/'trial',initial,online,props,7,'ra_corpus')
    assert result['fits']==2 and result['gradient_updates']==800
    cost=p.representation_accounting(props,'ra_corpus')
    assert cost['encoder_fit_calls']==2 and cost['encoder_fit_seconds']==sum(e.session.fit_seconds)
    with pytest.raises(ValueError):e.propose(initial,p.candidate_pool(prob,32,2)[0],0)


@pytest.mark.parametrize('target',['npz','corpus','history','event','feature','checkpoint_path','extra'])
def test_persistent_tampering_rejected(monkeypatch,tmp_path,target):
    prob,e,initial,online,props,_=isolated_training_trial(monkeypatch,tmp_path/'trial',horizon=1)
    root=tmp_path/'trial';cp=root/'representation/checkpoints/t000'
    if target=='npz':(cp/'state.npz').write_bytes(b'bad')
    elif target=='corpus':
        x=p.read(root/'representation/corpus.json');x['draw_indices'][0]=999
        (root/'representation/corpus.json').write_text(json.dumps(x))
    elif target=='history':initial[0]=row(prob,initial[0].candidate,objective=999.)
    elif target=='event':
        rows=p.lines(root/'representation/events.jsonl');rows[0]['iteration']=8
        (root/'representation/events.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    elif target=='feature':props[0]['metadata']['persistent_representation']['feature_identity']='bad'
    elif target=='checkpoint_path':props[0]['metadata']['persistent_representation']['checkpoint']='../other'
    else:(cp/'extra.json').write_text('{}')
    with pytest.raises((ValueError,RuntimeError,KeyError)):
        ca.audit_representation(prob,p.BASE,root,initial,online,props,7,'ra_corpus')


def test_failed_fit_is_retained_no_retry(monkeypatch,tmp_path):
    prob=p.problem_for(p.CASES[0]);initial=history(prob)
    e=co.CorpusDiscoveryBO(prob,p.BASE,3,p.settings_for('ra_lsbo',p.PROFILES['smoke']),tmp_path/'trial')
    monkeypatch.setattr(e.session,'fit',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('synthetic failed fit')))
    pool=p.candidate_pool(prob,9,5)[0]
    with pytest.raises(RuntimeError):e.propose(initial,pool,0)
    assert len(p.lines(tmp_path/'trial/representation/events.jsonl'))==1
    with pytest.raises(ValueError):e.propose(initial,pool,0)


class FakeEngine:
    stop_reason=None
    def propose(self,history,pool,iteration):return pool[0],{'method':'fake'}
    def accept(self,record):pass


def fake_runner(monkeypatch):
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase,'evaluate',lambda self,c:row(self,c))
    monkeypatch.setattr(p,'engine_for',lambda *a,**k:FakeEngine())
    monkeypatch.setattr(p,'validate_decision',lambda *a:None)
    monkeypatch.setattr(p,'audit_representation',lambda *a:None)
    monkeypatch.setattr(p,'representation_accounting',lambda *a:{'synthetic_runner_test':True})


def test_new_runner_roundtrip_no_extra_calls(monkeypatch,tmp_path):
    fake_runner(monkeypatch);out=tmp_path/'new'
    p.prepare(out);r=p.run(out,42);a=p.audit(out)
    assert r['charged_model_calls']==42 and a['checked_proposals']==20 and a['trial_count']==10
    assert p.audit(out,False)==a and p.report(out).exists()
    assert len(r['primary_verified_pairs'])==8
    assert all(x['counts']=={'tie':1} for x in r['primary_verified_pairs'])
    with pytest.raises(FileExistsError):p.run(out,42)
    with pytest.raises(FileExistsError):p.prepare(out)


@pytest.mark.parametrize('cap',[None,41,43,True,0])
def test_exact_cap_required_before_lock(monkeypatch,tmp_path,cap):
    fake_runner(monkeypatch);out=tmp_path/'new';p.prepare(out)
    with pytest.raises(ValueError):p.run(out,cap)
    assert not (out/'execution.lock').exists() and not (out/'calls.jsonl').exists()


@pytest.mark.parametrize('target',['source','result','trial','timing','ledger','plan'])
def test_runner_evidence_tamper_refused(monkeypatch,tmp_path,target):
    fake_runner(monkeypatch);out=tmp_path/'new';p.prepare(out);p.run(out,42);p.audit(out)
    if target=='source':f=next((out/'source_snapshot').iterdir())
    elif target=='result':f=out/'results.json'
    elif target=='trial':f=next((out/'optimization').rglob('summary.json'))
    elif target=='timing':f=next((out/'optimization').rglob('timing.json'))
    elif target=='ledger':f=out/'calls.jsonl'
    else:f=out/'plan.json'
    f.write_text('{}')
    with pytest.raises((ValueError,RuntimeError,KeyError,TypeError)):p.report(out)


def test_no_nominal_feasible_no_held_out_eval(monkeypatch,tmp_path):
    fake_runner(monkeypatch)
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase,'evaluate',lambda self,c:row(self,c,margin=.1,partial=True))
    out=tmp_path/'new';p.prepare(out);r=p.run(out,42)
    assert r['charged_model_calls']==32 and p.audit(out)['settled_calls']==32
    assert all(x['counts']=={'neither_verified':1} for x in r['primary_verified_pairs'])


def test_verification_not_feedback_or_reselection(monkeypatch,tmp_path):
    fake_runner(monkeypatch)
    from ra_lsbo.experiments.core.problems import AircraftCase
    seen=[]
    class Track(FakeEngine):
        def accept(self,r):seen.append(r.observation.objective.value)
    monkeypatch.setattr(p,'engine_for',lambda *a,**k:Track())
    monkeypatch.setattr(AircraftCase,'evaluate',lambda self,c:row(self,c,
        objective=1. if self.settings['integration_step_s']==5. else 100.,
        margin=.1 if self.settings['integration_step_s']==5. else -.1))
    out=tmp_path/'new';p.prepare(out);r=p.run(out,42);p.audit(out)
    assert seen==[100.]*20
    assert all(s['best_feasible']==100. and not s['step_verification']['verified_usable'] for s in r['summaries'])


def test_unknown_settles_and_stops_whole_campaign(monkeypatch,tmp_path):
    fake_runner(monkeypatch)
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase,'evaluate',lambda self,c:row(self,c,numerical=True))
    out=tmp_path/'new';p.prepare(out)
    with pytest.raises(RuntimeError):p.run(out,42)
    assert len(p.lines(out/'calls.jsonl'))==2 and not (out/'results.json').exists()


@pytest.mark.parametrize('kwargs',[{}, {'qualification':'missing'}, {'corpus_evidence':'missing'}])
def test_pilot_requires_both_reviewed_inputs(tmp_path,kwargs):
    with pytest.raises(ValueError):p.prepare(tmp_path/'new','pilot',**kwargs)
    assert not (tmp_path/'new').exists()


def test_pilot_prepare_only_reads_receipts(monkeypatch,tmp_path):
    called=[]
    monkeypatch.setattr(p,'verify_offline_archive',lambda f:called.append(f) or ca.offline_receipt())
    monkeypatch.setattr(p,'qualification_report',lambda f:{'receipt':p.q.receipt(),'new_model_calls':0})
    out=tmp_path/'new';plan=p.prepare(out,'pilot',tmp_path/'old',tmp_path/'archive.zip')
    assert len(called)==1 and p.load_plan(out)==plan
    assert plan['reviewed_offline_corpus']==ca.offline_receipt()
    assert not (out/'calls.jsonl').exists()


def test_offline_wrong_hash_fails(tmp_path):
    f=tmp_path/'bad.zip';f.write_bytes(b'bad')
    with pytest.raises(ValueError):ca.verify_offline_archive(f)


def test_npz_included_in_seal(tmp_path):
    (tmp_path/'state.npz').write_bytes(b'numeric checkpoint')
    assert 'state.npz' in p.evidence_hashes(tmp_path)


def test_cli_no_implicit_campaign():
    with pytest.raises(SystemExit):p.main([])


def primary_fixture():
    return [{'case':c,'seed':1,'method':m,'step_verification':{
        'verified_usable':True,'refined_objective_Wh':100.}}
        for c in p.CASES for m in p.ARMS]


def test_primary_requires_unique_complete_pairs():
    rows=primary_fixture()
    with pytest.raises(ValueError):p.primary_pairs(rows+[rows[0]])
    with pytest.raises(ValueError):p.primary_pairs(rows[1:])


def test_primary_never_imputes_infeasible_objectives():
    rows=primary_fixture()
    for r in rows:
        if r['method']=='ra_corpus':
            r['step_verification']={'verified_usable':False,'refined_objective_Wh':None}
    assert all(r['counts']=={'loss':1} and r['median_difference_Wh'] is None
               for r in p.primary_pairs(rows))


def test_control_invented_cost_rejected(tmp_path):
    prob=p.problem_for(p.CASES[0])
    prop={'metadata':{'posterior_fit_skipped':True,'representation_fit':{'encoder_fit_calls':1}}}
    with pytest.raises(ValueError):
        ca.audit_representation(prob,p.BASE,tmp_path,[],[],[prop],3,'ra_old30')
