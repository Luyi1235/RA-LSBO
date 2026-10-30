"""Transition tests use synthetic labels; true reduced missions run only in CI smoke."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from ra_lsbo.budget import digest
from ra_lsbo.experiments.simplified_aircraft import transition as t
from ra_lsbo.risk_core.observations import OutputObservation


@pytest.fixture(autouse=True)
def forbid_missions(monkeypatch):
    monkeypatch.setattr(t.AircraftCase, 'evaluate',
        lambda *a: pytest.fail('Unit suite must not call a real aircraft mission'))


def synthetic(problem, raw, objective=10., *, margin=-.1, numerical=False, partial=False):
    value = OutputObservation('interval','unit',lower=3.) if partial else objective
    return problem.record(raw,value,{n: margin for n in problem.required_constraints},
        {'termination_reason':'unit_partial' if partial else 'complete', 'numerical_failure':numerical})


def fake_missions(monkeypatch):
    monkeypatch.setattr(t.AircraftCase,'evaluate',lambda self,raw:synthetic(self,raw))


@pytest.mark.parametrize('stage,size,cap', [('smoke','full',36),('smoke','ci',36),
                                          ('qualification','full',316),('qualification','ci',64)])
def test_registered_budgets_and_no_old_data_prerequisite(stage,size,cap):
    plan=t.build_plan(stage,size)
    assert plan['budget']['maximum_model_calls']==cap
    assert plan['budget']['maximum_rcaide_mda_calls']==0
    assert not plan['old_transfer_diagnostic_required']
    assert not plan['prerequisite_directory_required']
    assert not plan['automatic_pilot_authorization']
    assert all(len(x['configurations'])==12 for x in plan['tasks'].values())
    if stage=='qualification':
        assert len(plan['calls'])==cap
        assert plan['tasks']['aircraft_ordinary']['panel']==plan['tasks']['aircraft_sparse']['panel']


@pytest.mark.parametrize('stage', ['pilot','confirm','transfer','mda','resume'])
def test_no_unapproved_experiment_stage(stage):
    with pytest.raises(ValueError): t.build_plan(stage)


def test_stage_latent_has_no_experts_and_matches_existing_phase():
    spec=t.method_spec('ev_latent')
    assert spec['coverage_scope']=='cold' and spec['cold_global'] and spec['protect']
    assert spec['feedback']=='interval' and spec['transfer_mode'] is None
    assert spec['settings']['average_kernel'] is False
    problem=t.problem_for('aircraft_sparse')
    engine=t.engine_for(problem,'ev_latent',9901,None).engine
    assert type(engine) is t.DiscoveryBO


def test_exact_view_only_changes_feedback_not_raw_evidence():
    spec=t.method_spec('phase_exact')
    other=t.method_spec('ev_latent')
    assert spec=={**other,'feedback':'exact'}
    opt=t.engine_for(t.problem_for('aircraft_sparse'),'phase_exact',9901,None).engine
    assert opt.feedback=='exact' and opt.coverage_scope=='cold'


@pytest.mark.parametrize('method',['ev_equal','ev_prequential','cbob','unknown'])
def test_no_unregistered_or_adaptive_method(method):
    with pytest.raises(ValueError): t.method_spec(method)


@pytest.mark.parametrize('numerical,partial,margin,state',[
    (False,False,-.1,'usable_feasible'), (False,False,.1,'known_infeasible'),
    (True,False,.1,'unresolved'), (False,True,-.1,'unresolved')])
def test_physical_classification_does_not_fabricate_failure(numerical,partial,margin,state):
    p=t.problem_for('aircraft_ordinary')
    r=synthetic(p,p.anchors()[0],margin=margin,numerical=numerical,partial=partial)
    assert t.physical_state(r)==state


def test_unknowns_retained_in_denominators():
    p=t.problem_for('aircraft_sparse'); c=p.anchors()[0]
    rows=[synthetic(p,c),synthetic(p,c,margin=.2),synthetic(p,c,numerical=True)]
    out=t.summarize_panel(rows)
    assert out['n']==3 and out['counts']=={'usable_feasible':1,'known_infeasible':1,'unresolved':1}
    assert out['physical_feasibility_partial_identification']==[1/3,2/3]
    assert out['unknown_conservative_binomial_envelope_95'][1]>=out['usable_binomial_95'][1]
    assert not out['sparse_below_10pct_supported']


@pytest.mark.parametrize('values,expected', [([10,10.05,10.02],'passed_screen'),
                                           ([10,11,12],'needs_review')])
def test_integration_energy_screen(values,expected):
    p=t.problem_for('aircraft_sparse'); c=p.anchors()[0]
    result=t.convergence([synthetic(p,c,v) for v in values])
    assert result['status']==expected


def test_partial_results_cannot_pass_convergence():
    p=t.problem_for('aircraft_sparse'); c=p.anchors()[0]
    assert t.convergence([synthetic(p,c,partial=True)]*2)['status']=='inconclusive_partial_outputs'
    assert t.convergence([synthetic(p,c,numerical=True)]*2)['status']=='needs_review_numerical_failure'
    assert t.convergence([synthetic(p,c),synthetic(p,c,partial=True)])['status']=='needs_review_changed_termination'
    assert t.convergence([synthetic(p,c,margin=-.1),synthetic(p,c,margin=.1)])['status']=='needs_review'


def test_qualification_uses_no_optimizer_and_audit_no_evaluator(monkeypatch,tmp_path):
    fake_missions(monkeypatch)
    monkeypatch.setattr(t,'engine_for',lambda *a:pytest.fail('Qualification cannot optimize'))
    folder=tmp_path/'new'; plan=t.prepare(folder,'qualification','ci')
    result=t.run(folder,64)
    assert result['charged_model_calls']==64
    assert not result['automatic_pilot_authorization']
    assert not (folder/'optimization').exists()
    monkeypatch.setattr(t.AircraftCase,'evaluate',lambda *a:pytest.fail('Audit cannot evaluate'))
    receipt=t.audit(folder)
    assert receipt['settled_calls']==64 and not receipt['algorithm_replay']
    assert t.audit(folder,write=False)==receipt
    t.report(folder)
    assert t.audit(folder,write=False)==receipt
    with pytest.raises(FileExistsError): t.report(folder)
    with pytest.raises(FileExistsError): t.run(folder,64)
    with pytest.raises(FileExistsError): t.prepare(folder,'qualification','ci')


@pytest.mark.parametrize('cap',[None,0,63,True])
def test_full_cap_required_before_lock_or_calls(tmp_path,cap):
    folder=tmp_path/'new';t.prepare(folder,'qualification','ci')
    with pytest.raises(ValueError): t.run(folder,cap)
    assert not (folder/'calls.jsonl').exists() and not (folder/'execution.lock').exists()


@pytest.mark.parametrize('target',['plan','snapshot','record','summary','missing_call','extra_call'])
def test_tamper_is_rejected(monkeypatch,tmp_path,target):
    fake_missions(monkeypatch)
    folder=tmp_path/'new'; plan=t.prepare(folder,'qualification','ci'); t.run(folder,64); t.audit(folder)
    if target=='plan':
        path=folder/'plan.json'; data=t.read(path);data['budget']['maximum_model_calls']=1
        path.write_text(json.dumps(data))
    elif target=='snapshot':
        next((folder/'source_snapshot').iterdir()).write_text('changed')
    elif target=='summary':
        path=folder/'results.json';data=t.read(path);data['charged_model_calls']=1;path.write_text(json.dumps(data))
    else:
        path=folder/'calls.jsonl'; data=t.lines(path)
        if target=='record': data[1]['record']['observation']['objective']['value']=999.
        elif target=='missing_call': data=data[:-2]
        else: data+=data[-2:]
        path.write_text('\n'.join(json.dumps(x) for x in data)+'\n')
    with pytest.raises((ValueError,FileNotFoundError)): t.audit(folder,write=False)


def test_prepare_has_no_model_calls(tmp_path):
    folder=tmp_path/'new';t.prepare(folder,'qualification','full')
    assert not (folder/'calls.jsonl').exists()
    assert t.load_plan(folder)['budget']['maximum_model_calls']==316


def test_cli_does_not_execute_without_command(tmp_path):
    with pytest.raises(SystemExit): t.main([])
    with pytest.raises(SystemExit): t.main(['prepare','--output',str(tmp_path/'none')])
    assert not (tmp_path/'none').exists()
