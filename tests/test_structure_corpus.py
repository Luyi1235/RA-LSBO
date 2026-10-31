"""Synthetic structural tests: no aircraft, mathematical or MDA evaluation."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.risk_core.observations import OutputObservation, RunObservation
from ra_lsbo.structured_encoder import EncoderSettings, hierarchy_space, state_digest
from ra_lsbo.structure_corpus import (CorpusRepresentation, sampled_grammar,
    native_observations, grammar_diagnostics)

BASE = Path(__file__).resolve().parents[1]/'runs/compact_mda_20261021T081006Z_20eb608d/qualification_recovered_20261025/frozen_base'
# Only the actual frozen schema and dataclasses, not the GP or evaluator modules.
NAME = '_structure_corpus_test_base'
if NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(NAME, BASE/'lsbo/__init__.py', submodule_search_locations=[str(BASE/'lsbo')])
    package = importlib.util.module_from_spec(spec);sys.modules[NAME] = package;spec.loader.exec_module(package)
import importlib
MODULES = {n:importlib.import_module(NAME+'.'+n) for n in ('space','records')}
VARIABLES = [
    {'name':'prop_layout','kind':'nominal','default':'wingtip','values':['wingtip','distributed']},
    {'name':'propeller_count','kind':'ordinal','default':2,'values':[2,4,6], 'active_when':{'prop_layout':['distributed']}},
    {'name':'parallel_strings','kind':'ordinal','default':40,'values':[32,40,48]},
    {'name':'area','kind':'continuous','default':.5,'bounds':[0.,1.]},
    {'name':'spacing','kind':'continuous','default':.5,'bounds':[0.,1.], 'active_when':{'prop_layout':['distributed']}}]


@pytest.fixture
def space():
    torch.set_num_threads(1)
    return hierarchy_space(MODULES,VARIABLES)


def raw_records(space,n=5):
    result=[]
    for i in range(n):
        c=space.default().to_dict();c['discrete']=space.configurations()[i%12]
        c['continuous']['area']=(i+1)/(n+2)
        c=space.canonical(MODULES['space'].Candidate(**c)).to_dict()
        identity=digest(c)
        result.append(DesignRecord(c,RunObservation(identity,'test-contract','obs'+str(i),'completed',
            OutputObservation('exact','y'+str(i),value=float(i+1)),
            (('soc',OutputObservation('exact','g'+str(i),value=-.2)),)),{}))
    return result


def config():
    return EncoderSettings(initial_epochs=2,update_epochs=1,width=12,batch_size=128,joint_view='latent')


def session(space):
    return CorpusRepresentation(space,7,settings=config())


def test_exact_2000_and_duplicates_not_deduplicated(space):
    t,p=sampled_grammar(space,2000,12)
    assert t.shape==(2000,3) and p['draws']==2000
    assert p['available_configurations']==12 and p['distinct_configurations']==12
    assert sum(p['counts'])==2000 and len(p['draw_indices'])==2000
    assert p['model_evaluations']==0 and p['simulation_labels'] is False


@pytest.mark.parametrize('count',[0,-1,True,1.5])
def test_bad_count(space,count):
    with pytest.raises(ValueError):sampled_grammar(space,count,0)


@pytest.mark.parametrize('seed',[-1,True,1.5,'7'])
def test_bad_seed(space,seed):
    with pytest.raises(ValueError):CorpusRepresentation(space,seed)


def test_sampling_reproducible_and_global_rng_unchanged(space):
    nr=np.random.get_state();tr=torch.get_rng_state().clone()
    a,ap=sampled_grammar(space,2000,4);b,bp=sampled_grammar(space,2000,4)
    assert torch.equal(a,b) and ap==bp
    assert torch.equal(tr,torch.get_rng_state())
    assert np.array_equal(nr[1],np.random.get_state()[1])
    assert sampled_grammar(space,2000,5)[1]['identity']!=ap['identity']


def test_all_draws_canonical_inactive_count_fixed(space):
    t,p=sampled_grammar(space,2000,4)
    for i in p['draw_indices']:
        c=p['configurations'][i]
        if c['prop_layout']=='wingtip':assert c['propeller_count']==2
    assert len({digest(c) for c in p['configurations']})==12


def test_joint_rows_include_observed_and_step_count(space):
    s=session(space);r=raw_records(space,12);old=deepcopy([x.to_dict() for x in r])
    e=s.fit(native_observations(r,MODULES,space))
    assert e['structural_rows']==2012 and e['random_structural_rows']==2000
    assert e['batches_per_epoch']==16 and e['gradient_updates']==32
    assert e['trusted_exact_target_rows']==12
    assert [x.to_dict() for x in r]==old
    assert torch.equal(s.representation.grammar[:2000],s.sampled_indices)


@pytest.mark.parametrize('kind',['missing','interval','unknown','infrastructure_error','numerical'])
def test_unlabelled_never_supplies_performance_target(space,kind):
    records=raw_records(space,1);r=records[0]
    if kind=='missing':o=OutputObservation('missing','y',reason='not_observed');r=replace(r,observation=replace(r.observation,objective=o))
    if kind=='interval':o=OutputObservation('interval','y',lower=10.);r=replace(r,observation=replace(r.observation,objective=o))
    if kind in {'unknown','infrastructure_error'}:r=replace(r,observation=replace(r.observation,status=kind))
    if kind=='numerical':r=replace(r,diagnostics={'numerical_failure':True})
    obs=native_observations([r],MODULES,space)
    assert not obs[0].result.has_objective() and obs[0].result.objective is None
    s=session(space);e=s.fit(obs)
    assert e['trusted_exact_target_rows']==0 and e['structural_rows']==2001
    assert not s.representation.performance_fitted


def test_exact_infeasible_label_still_trains_without_faking_feasibility(space):
    r=raw_records(space,1)[0]
    r=replace(r,observation=replace(r.observation,constraints=(('soc',OutputObservation('exact','g',value=1.)),)))
    obs=native_observations([r],MODULES,space)
    assert obs[0].result.has_objective() and not obs[0].result.feasible(('soc',))


def test_incremental_retains_state_and_uses_low_rate(space):
    s=session(space);obs=native_observations(raw_records(space,8),MODULES,space)
    first=s.fit(obs[:4]);before=s._sealed_state
    second=s.fit(obs,incremental=True)
    assert second['parameter_state_before']==before
    assert second['learning_rate']==s.settings.update_learning_rate
    assert s.representation.version==2
    assert second['gradient_updates']==16 and second['cumulative_gradient_updates']==48
    assert first['parameter_state_after']!=s._sealed_state


@pytest.mark.parametrize('change',['alter','truncate','repeat','reset'])
def test_prefix_mutations_rejected_without_training(space,change):
    s=session(space);obs=native_observations(raw_records(space,8),MODULES,space);s.fit(obs[:4])
    before=s._sealed_state
    if change=='alter':x=deepcopy(obs);x[0].result.objective+=1
    elif change=='truncate':x=obs[:3]
    elif change=='repeat':x=obs[:4]
    else:x=obs
    with pytest.raises(ValueError):s.fit(x,incremental=change!='reset')
    assert state_digest(s.representation)==before


def test_checkpoint_restores_then_same_incremental_update(space,tmp_path):
    s=session(space);obs=native_observations(raw_records(space,8),MODULES,space);s.fit(obs[:4])
    s.save_checkpoint(tmp_path/'checkpoint')
    b=CorpusRepresentation.load_checkpoint(tmp_path/'checkpoint',space,obs[:4])
    assert b._sealed_state==s._sealed_state and b.corpus==s.corpus
    assert s.fit(obs,incremental=True)==b.fit(obs,incremental=True)
    assert s._sealed_state==b._sealed_state
    with pytest.raises(FileExistsError):s.save_checkpoint(tmp_path/'checkpoint')


def test_checkpoint_rejects_history_and_state_change(space,tmp_path):
    s=session(space);obs=native_observations(raw_records(space,5),MODULES,space);s.fit(obs);s.save_checkpoint(tmp_path/'c')
    with pytest.raises(ValueError):CorpusRepresentation.load_checkpoint(tmp_path/'c',space,obs[:-1])
    p=tmp_path/'c/state.npz';p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError):CorpusRepresentation.load_checkpoint(tmp_path/'c',space,obs)


def test_corpus_and_model_mutation_detected(space):
    s=session(space);obs=native_observations(raw_records(space,8),MODULES,space);s.fit(obs[:4])
    with torch.no_grad():s.representation.model.mu.bias.add_(.1)
    with pytest.raises(RuntimeError):s.fit(obs,incremental=True)
    s=session(space);s.corpus['draw_indices'][0]=0
    s.corpus['draw_indices'][1]=99
    with pytest.raises(ValueError):s.fit(obs)


def test_grammar_diagnostics_preserve_training_rng_and_state(space):
    s=session(space);s.fit(native_observations(raw_records(space),MODULES,space));h=s._sealed_state
    a=grammar_diagnostics(s.representation);b=grammar_diagnostics(s.representation)
    assert a==b and state_digest(s.representation)==h
    assert len(a['mu'])==12 and a['training_grammar_not_holdout']
    assert not a['automatic_BO_authorization']
    assert 0<=a['raw_sampled_active_exact']<=1


def test_offline_budgets_and_score_semantics():
    from tools.study_structure_corpus_2000 import limits,forecast_score
    assert limits('smoke')['encoder_fit_calls']==20
    assert limits('full')['encoder_fit_calls']==160
    assert limits('smoke')['GP_head_fit_calls']==40
    assert limits('full')['GP_head_fit_calls']==320
    exact=OutputObservation('exact','e',value=0.)
    interval=OutputObservation('interval','i',lower=0.,lower_inclusive=False)
    missing=OutputObservation('missing','m',reason='unknown')
    assert 'brier' not in forecast_score(0.,1.,0.,exact)
    assert forecast_score(0.,1.,0.,exact,constraint=True)['brier']==.25
    assert forecast_score(0.,1.,0.,interval)['nll']==pytest.approx(np.log(2.))
    assert forecast_score(0.,1.,0.,missing)['score_status']=='missing_not_imputed'
    assert 'nll' not in forecast_score(0.,1.,0.,missing)


def test_feature_snapshot_isolated_from_later_updates(space,monkeypatch):
    from types import SimpleNamespace
    from ra_lsbo import features as feature_module
    from ra_lsbo.structure_corpus import feature_snapshot
    monkeypatch.setattr(feature_module,'load_base',lambda base:(MODULES,{}))
    records=raw_records(space,8);obs=native_observations(records,MODULES,space)
    s=session(space);s.fit(obs[:4])
    problem=SimpleNamespace(space=space,contract_id='test-contract')
    first=feature_snapshot(problem,BASE,records[:4],s)
    matrix=first.transform([r.candidate for r in records])
    s.fit(obs,incremental=True)
    np.testing.assert_array_equal(matrix,first.transform([r.candidate for r in records]))
    second=feature_snapshot(problem,BASE,records,s)
    assert second.identity!=first.identity
    with torch.no_grad():first.encoder.model.mu.bias.add_(.1)
    with pytest.raises(RuntimeError):first.transform([records[0].candidate])


def test_failed_fit_cannot_silently_continue(space,monkeypatch):
    s=session(space);obs=native_observations(raw_records(space),MODULES,space)
    def fail(*a,**k):raise FloatingPointError('synthetic training failure')
    monkeypatch.setattr(s.representation,'fit',fail)
    with pytest.raises(FloatingPointError):s.fit(obs)
    with pytest.raises(RuntimeError,match='Failed fit'):s.fit(obs)


def test_offline_explicit_caps_before_io(tmp_path):
    from tools.study_structure_corpus_2000 import run
    with pytest.raises(ValueError,match='caps'):run(tmp_path/'missing.zip',tmp_path/'out','full',20,40)
    assert not (tmp_path/'out').exists()


def make_audit_fixture(root):
    from tools.study_structure_corpus_2000 import save,sha,ARMS,CASES,HEADS,CHECKPOINTS,limits
    root.mkdir();plan={'profile':'smoke','cases':CASES,'seeds':[9911]};save(root/'plan.json',plan)
    rows=[]
    for case in CASES:
        for t in CHECKPOINTS:
            for arm in ARMS:
                rows.append({'case':case,'seed':9911,'iteration':t,'arm':arm,'history_identity':'same'+str(t),
                    'predictions':[{'head':head,'scores':[{'candidate':{'x':1},'evidence_id':'test',
                                   'observation':{'kind':'exact','value':0.}}]} for head in HEADS],
                    'corpus':{'draws':2000,'draw_indices':[0]*2000} if arm.startswith('corpus_') else None})
    (root/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    counts={k:limits('smoke')[k] for k in ('encoder_fit_calls','onehot_transforms','GP_head_fit_calls')}
    attempts=[{'kind':k,'number':i} for k,n in counts.items() for i in range(1,n+1)]
    (root/'fit_attempts.jsonl').write_text('\n'.join(json.dumps(r) for r in attempts),encoding='utf-8')
    files={p.name:sha(p) for p in root.iterdir()};save(root/'audit.json',{'files':files,'counts':counts})
    return rows


def test_readonly_audit_does_not_fit_or_evaluate(tmp_path,monkeypatch):
    from tools import study_structure_corpus_2000 as study
    root=tmp_path/'out';make_audit_fixture(root)
    monkeypatch.setattr(study,'GroupedMixedHead',lambda *a,**k:pytest.fail('no fitting in audit'))
    assert study.audit_output(root)['rows']==20


@pytest.mark.parametrize('change',['hash','target','prefix','duplicate','missing','count'])
def test_readonly_audit_rejects_changed_evidence(tmp_path,change):
    from tools import study_structure_corpus_2000 as study
    root=tmp_path/'out';rows=make_audit_fixture(root)
    if change=='target':rows[1]['predictions'][0]['scores'][0]['observation']['value']=42
    if change=='prefix':rows[1]['history_identity']='future'
    if change=='duplicate':rows.append(rows[0])
    if change=='missing':rows.pop()
    if change=='count':rows[2]['corpus']={'draws':12,'draw_indices':[0]*12}
    (root/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n',encoding='utf-8')
    if change!='hash':
        a=json.loads((root/'audit.json').read_bytes());a['files']['rows.jsonl']=study.sha(root/'rows.jsonl')
        study.save(root/'audit.json',a)  # Test semantic checks beyond a re-sealed digest.
    with pytest.raises(ValueError):study.audit_output(root)
