"""Runner/ledger integration tests. Requires the unchanged full model dependencies.

No skip-on-missing-dependency: a collection error must stop local acceptance.
Mission outputs below are explicitly synthetic, never research ledger evidence.
"""
from copy import deepcopy
from pathlib import Path
import json
import pytest

from ra_lsbo.experiments.simplified_aircraft import corpus_pilot as p
from tests.test_aircraft_nominal_pilot import row

@pytest.fixture(autouse=True)
def no_real_missions(monkeypatch):
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase,'evaluate',lambda *a:pytest.fail('No real evaluator in unit tests'))
    # Unit storage fixtures are not executable research plans.
    monkeypatch.setattr(p,'require_clean_git',lambda:None)

@pytest.mark.parametrize('name,cap,ntrial',[('smoke',42,10),('pilot',2672,80)])
def test_frozen_budget_and_sourcing(name,cap,ntrial):
    plan=p.build_plan(name)
    assert plan['budget']['maximum_model_calls']==cap
    assert plan['budget']['incumbent_step_verification']==ntrial
    assert plan['tasks']['aircraft_ordinary']['initial']==plan['tasks']['aircraft_sparse']['initial']
    assert plan['tasks']['aircraft_sparse']['settings']['nominal_cruise_distance_m']==140000
    assert plan['reference']=='corpus_persistent'
    assert not plan['qualification_labels_available_to_optimizer']

def test_no_pilot_before_new_smoke_and_qualification(tmp_path):
    for qual,smoke in [(None,None),('old',None),(None,'smoke')]:
        with pytest.raises(ValueError):p.prepare(tmp_path/'new','pilot',qual,smoke)
    assert not (tmp_path/'new').exists()

class FakeEngine:
    stop_reason=None
    def propose(self,history,pool,iteration):return pool[0],{'method':'fake'}
    def accept(self,record):pass

def setup_fake(monkeypatch):
    from ra_lsbo.experiments.core.problems import AircraftCase
    accepts=[]
    monkeypatch.setattr(AircraftCase,'evaluate',lambda self,c:row(self,c))
    def factory(problem,method,seed,profile,output):
        obj=FakeEngine();obj.accept=lambda r:accepts.append(r.observation.contract_id);return obj
    monkeypatch.setattr(p,'engine_for',factory)
    monkeypatch.setattr(p,'validate_decision',lambda *a:None)
    # Only the runner is isolated here; genuine VAE/GP state is tested separately.
    monkeypatch.setattr(p,'audit_training',lambda *a:{'fit_count':1,'gradient_updates':480,'fit_seconds':0.,
        'checkpoint_events':[{'mean_roundtrip_target_met':True}], 'training_replayed':False})
    return accepts

def test_full_storage_roundtrip_and_no_verification_feedback(monkeypatch,tmp_path):
    accepts=setup_fake(monkeypatch);out=tmp_path/'smoke'
    p.prepare(out);result=p.run(out,42);seal=p.audit(out)
    assert result['charged_model_calls']==42 and len(accepts)==20
    assert len(result['summaries'])==10 and seal['checked_proposals']==20
    assert p.audit(out,False)==seal
    assert p.report(out).exists()
    assert p.smoke_receipt(out)['source_commit']==p.read(out/'plan.json')['source_commit']
    with pytest.raises(FileExistsError):p.run(out,42)
    with pytest.raises(FileExistsError):p.prepare(out)

@pytest.mark.parametrize('cap',[None,0,41,43,True])
def test_exact_cap_checked_before_lock(monkeypatch,tmp_path,cap):
    setup_fake(monkeypatch);out=tmp_path/'smoke';p.prepare(out)
    with pytest.raises(ValueError):p.run(out,cap)
    assert not (out/'execution.lock').exists()

def test_pilot_registration_requires_same_commit_runtime_and_unchanged_smoke(monkeypatch,tmp_path):
    setup_fake(monkeypatch);smoke=tmp_path/'smoke';p.prepare(smoke);p.run(smoke,42);p.audit(smoke)
    monkeypatch.setattr(p,'qualification_report',lambda f:{'receipt':p.q.receipt(),'new_model_calls':0})
    out=tmp_path/'pilot';plan=p.prepare(out,'pilot','old_qualification',smoke)
    assert not (out/'calls.jsonl').exists() and p.load_plan(out)==plan
    f=smoke/'results.json';value=p.read(f);value['charged_model_calls']=-1;f.write_text(json.dumps(value))
    with pytest.raises(ValueError):p.load_plan(out)

@pytest.mark.parametrize('target',['source','result','timing','observation','verification','proposal','npz'])
def test_tampering_detected(monkeypatch,tmp_path,target):
    setup_fake(monkeypatch);out=tmp_path/'smoke';p.prepare(out);p.run(out,42);p.audit(out)
    trial=next((out/'optimization').glob('*/*'))
    if target=='source':
        plan=p.read(out/'plan.json');f=out/'source_snapshot'/(next(iter(plan['sources'].values()))+'.source');f.write_text('changed')
    elif target=='npz':
        # The evidence seal must include numeric payloads, even an unregistered addition.
        (trial/'extra.npz').write_bytes(b'new unregistered payload')
    elif target in ('observation','proposal'):
        f=trial/('observations.jsonl' if target=='observation' else 'proposals.jsonl')
        f.write_text('\n'.join(f.read_text().splitlines()[:-1])+'\n')
    else:
        f=out/'results.json' if target=='result' else trial/('timing.json' if target=='timing' else 'verification.json')
        value=p.read(f);value['tampered']=True;f.write_text(json.dumps(value))
    with pytest.raises((ValueError,KeyError)):p.report(out)

def test_no_unregistered_early_stop():
    with pytest.raises(ValueError):p.validate_stop('old30',[None]*5,30,'native stop')
    with pytest.raises(ValueError):p.validate_stop('corpus_persistent',[None]*29,30,None)

@pytest.mark.parametrize('kind',['unknown','infrastructure_error','numerical'])
def test_untrusted_evidence_stops_before_optimizer_feedback(kind):
    from dataclasses import replace
    problem=p.problem_for('aircraft_sparse');r=row(problem,problem.anchors()[0],numerical=kind=='numerical')
    if kind!='numerical':r=replace(r,observation=replace(r.observation,status=kind))
    with pytest.raises(RuntimeError):p.ensure_trusted(r)

def test_final_verification_cannot_select_another_design():
    problem=p.problem_for('aircraft_sparse');a,b=problem.anchors()
    with pytest.raises(ValueError):p.verification_result(row(problem,a),row(problem,b))

def test_real_adapter_factories_only_registered_changes(tmp_path):
    problem=p.problem_for('aircraft_ordinary');profile=p.PROFILES['smoke']
    for arm in ('old30','old480','onehot','corpus_persistent'):
        engine=p.engine_for(problem,arm,9930,profile,tmp_path/arm)
        assert engine.method==arm and engine.stop_reason is None
        assert engine.engine.coverage_scope=='cold' and engine.engine.cold_global
        assert engine.engine.settings==p.model_settings(arm,profile)
        assert not (tmp_path/arm).exists()  # No model evaluation or fitting during construction.
