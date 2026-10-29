"""The development gate must catch warm degradation, not just initial improvement."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import pytest

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core import protocol as core
from ra_lsbo.experiments.discovery_stages import methods, optimizer, policies, protocol, runner
from ra_lsbo.experiments.discovery_stages.phase_metrics import phase_diagnostics, warm_pairs
from ra_lsbo.experiments.discovery_stages.reviewed_evidence import (
    _read, prerequisite_receipt, verify_reviewed_discovery)
from test_discovery_stages import samples, fake_model


@pytest.mark.parametrize('scope,cold,expected', [
    ('all',False,True), ('all',True,True), ('cold',False,False),
    ('cold',True,True), ('none',False,False), ('none',True,False)])
def test_explicit_coverage_scope(scope,cold,expected):
    assert policies.coverage_enabled(scope,cold) is expected


@pytest.mark.parametrize('scope,cold', [('warm',False), ('cold',1), ('cold',None)])
def test_bad_scope_is_not_silently_coerced(scope,cold):
    with pytest.raises(ValueError): policies.coverage_enabled(scope,cold)


def engine(problem, scope, cold_global=False):
    return optimizer.DiscoveryBO(problem,core.BASE,7,
        methods.settings_for('ra_lsbo',core.PROFILES['smoke']),
        coverage_scope=scope,cold_global=cold_global,protect=True)


@pytest.mark.parametrize('scope', ['cold','none'])
@pytest.mark.parametrize('step', [0,1,5])
def test_warm_model_development_not_replaced_even_at_singleton(monkeypatch,scope,step):
    problem,pool,good,_=samples()
    fake_model(monkeypatch,singleton=True)
    opt=engine(problem,scope); opt.settings=replace(opt.settings,horizon=10); opt.next_iteration=step
    calls=[]
    def local(*args):
        calls.append(True)
        return pool, {'local_reports':[]}
    monkeypatch.setattr(opt,'_augment',local)
    candidate,info=opt.propose([good],pool,step)
    assert calls and info['local_enabled']
    assert candidate==pool[0] and info['coverage_reason'] is None
    assert not info['coverage_active'] and not info['coverage_scheduled']
    assert not info['posterior_fit_skipped']
    assert info['selected_index']==info['model_selected_index']


def test_cold_scope_keeps_original_cold_coverage_then_switches(monkeypatch):
    problem,pool,_,bad=samples()
    opt=engine(problem,'cold',cold_global=True)
    monkeypatch.setattr(optimizer,'RiskPosterior',lambda *a:pytest.fail('Cold coverage fits no model'))
    candidate,info=opt.propose([bad],pool,0)
    assert info['coverage_reason']=='scheduled_coverage'
    good=problem.record(candidate,3.,{'region':-.1,'resource':-.2},{})
    opt.accept(good)
    fake_model(monkeypatch,singleton=True)
    novel=[c for c in pool if c!=candidate]
    monkeypatch.setattr(opt,'_augment',lambda *a:(novel,{'local_reports':[]}))
    _,info=opt.propose([bad,good],novel,1)
    assert not info['cold_start'] and not info['coverage_active'] and info['local_enabled']
    assert info['coverage_reason'] is None


def test_missing_objective_does_not_trigger_phase_transition(monkeypatch):
    problem,pool,_,bad=samples()
    fake_model(monkeypatch,singleton=True)
    opt=engine(problem,'cold',cold_global=True); opt.next_iteration=1
    _,info=opt.propose([bad],pool,1)
    assert info['cold_start'] and info['coverage_active']
    assert info['coverage_reason']=='eligible_at_most_one' and not info['local_enabled']


def test_near_guard_survives_coverage_switch_off():
    problem,_,good,_=samples()
    near=deepcopy(good.candidate); near['continuous']['x']+=1e-10
    for scope in ('cold','none'):
        with pytest.raises(RuntimeError,match='No novel'):
            engine(problem,scope).propose([good],[near],0)


def test_all_phase_default_retains_old_decision(monkeypatch):
    problem,pool,_,bad=samples()
    fake_model(monkeypatch,singleton=True)
    default=optimizer.DiscoveryBO(problem,core.BASE,7,methods.settings_for('ra_lsbo',core.PROFILES['smoke']))
    explicit=engine(problem,'all',cold_global=True)
    assert default.propose([bad],pool,0)==explicit.propose([bad],pool,0)


def test_phase_budget_and_source_settings_match():
    p=protocol.build_plan('phase','smoke');protocol.verify_plan(p)
    assert p['budget']['maximum_model_calls']==120 and len(p['arms'])==7
    p=protocol.build_plan('phase','pilot')
    assert p['budget']['maximum_model_calls']==8916
    assert tuple(p['profile']['seeds'])==tuple(range(9701,9721))
    assert p['budget']['maximum_rcaide_mda_calls']==0
    with pytest.raises(ValueError,match='receipt'):protocol.verify_plan(p)
    p['prerequisite']=prerequisite_receipt();protocol.verify_plan(p)
    p['prerequisite']['settled_calls']=0
    with pytest.raises(ValueError,match='receipt'):protocol.verify_plan(p)
    settings=[methods.specification(m,core.PROFILES['pilot'])['settings']
              for m in methods.STAGES['phase'] if m!='random']
    assert all(s==settings[0] for s in settings)


def test_phase_pilot_cannot_prepare_without_real_reviewed_evidence(tmp_path):
    with pytest.raises(ValueError,match='prerequisite'):
        runner.prepare(tmp_path/'new','phase','pilot')
    with pytest.raises(ValueError,match='Missing'):
        runner.prepare(tmp_path/'new','phase','pilot',tmp_path/'absent')
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('stage', ['representation','baselines'])
def test_failed_development_gate_stops_later_pilots(tmp_path,stage):
    with pytest.raises(ValueError,match='not scientifically approved'):
        runner.prepare(tmp_path/stage,stage,'pilot',tmp_path/'anything')
    assert not (tmp_path/stage).exists()


@pytest.mark.parametrize('path', ['../secret','/absolute','C:/work/plan','a\\b','./plan.json'])
def test_import_rejects_unsafe_paths(tmp_path,path):
    with pytest.raises(ValueError,match='Unsafe'):_read(tmp_path,path)


def test_import_rejects_unreviewed_header(tmp_path):
    (tmp_path/'plan.json').write_text('{}')
    with pytest.raises(ValueError,match='Not the reviewed'):
        verify_reviewed_discovery(tmp_path)


def test_receipt_is_not_a_mutable_global():
    r=prerequisite_receipt(); r['published_files_sha256']['plan.json']='x'
    assert prerequisite_receipt()['published_files_sha256']['plan.json']!='x'


def test_warm_initial_improvement_does_not_mask_pairwise_regression():
    rows=[]
    for seed,x,y in [(1,2.,1.),(2,.5,1.),(3,1.,1.)]:
        for m,v in [('a',x),('b',y)]:
            rows.append({'case':'toy','seed':seed,'method':m,'initial_kind':'warm',
                         'initial_best':10.,'best_feasible':v,'has_feasible':True})
    pair=warm_pairs(rows)[0]
    assert pair['n']==3 and (pair['better'],pair['ties'],pair['worse'])==(1,1,1)
    with pytest.raises(ValueError,match='Missing'):warm_pairs(rows[:-1])
    with pytest.raises(ValueError,match='Duplicate'):warm_pairs(rows+rows[:1])
    rows[0]['initial_best']=11.
    with pytest.raises(ValueError,match='Unmatched'):warm_pairs(rows)


def test_phase_metric_assigns_first_success_to_pre_observation_phase():
    _,_,good,bad=samples()
    props=[{'candidate':bad.candidate,'metadata':{'coverage_reason':'scheduled_coverage',
             'posterior_fit_skipped':True}},
           {'candidate':good.candidate,'metadata':{'coverage_reason':'eligible_at_most_one'}},
           {'candidate':good.candidate,'metadata':{'boundary_slot':True}}]
    d=phase_diagnostics([bad],props,[bad,good,good])
    assert d['cold_calls']==2 and d['warm_calls']==1
    assert d['cold_coverage']==2 and d['warm_coverage']==0
    assert d['first_feasible_channel']=='eligible_at_most_one' and d['warm_model_boundary']==1


def test_audit_rejects_warm_coverage_for_phase_policy(monkeypatch):
    problem,pool,good,_=samples();fake_model(monkeypatch,singleton=True)
    opt=engine(problem,'cold',cold_global=True)
    monkeypatch.setattr(opt,'_augment',lambda *a:(pool,{'local_reports':[]}))
    candidate,meta=opt.propose([good],pool,0);meta['method']='ra_phase_discovery'
    spec=methods.specification('ra_phase_discovery',core.PROFILES['smoke'])
    p={'iteration':0,'candidate':candidate,'pool':pool,'metadata':meta}
    plan={'methods':{'ra_phase_discovery':spec}}
    runner.validate_decision(problem,plan,'ra_phase_discovery',7,[good],p)
    for key,value in [('coverage_active',True),('coverage_scheduled',True),
                      ('coverage_reason','eligible_at_most_one')]:
        broken=deepcopy(p);broken['metadata'][key]=value
        with pytest.raises(ValueError):runner.validate_decision(problem,plan,'ra_phase_discovery',7,[good],broken)
