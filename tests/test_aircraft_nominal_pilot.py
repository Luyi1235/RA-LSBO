"""Storage/control tests use synthetic labels; CI alone executes bounded real smoke."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from ra_lsbo.experiments.simplified_aircraft import nominal_pilot as p
from ra_lsbo.risk_core.observations import OutputObservation


@pytest.fixture(autouse=True)
def no_real_missions(monkeypatch):
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase,'evaluate',lambda *args:pytest.fail('No real missions in unit tests'))


def row(problem,candidate,objective=10.,margin=-.1,partial=False,numerical=False):
    value=OutputObservation('interval','unit-energy',lower=3.) if partial else objective
    return problem.record(candidate,value,{n:margin for n in problem.required_constraints},
        {'termination_reason':'unit_partial' if partial else 'complete','numerical_failure':numerical})


@pytest.mark.parametrize('profile,cap,ntrial',[('smoke',54,14),('pilot',3664,112)])
def test_complete_budget_and_frozen_equations(profile,cap,ntrial):
    plan=p.build_plan(profile)
    assert plan['budget']['maximum_model_calls']==cap
    assert plan['budget']['incumbent_step_verification']==ntrial
    assert plan['budget']['maximum_rcaide_mda_calls']==0
    assert plan['tasks']['aircraft_ordinary']['initial']==plan['tasks']['aircraft_sparse']['initial']
    assert not plan['qualification_labels_available_to_optimizer']
    assert not plan['verification']['fed_back_to_optimizer']
    assert plan['tasks']['aircraft_sparse']['settings']['nominal_cruise_distance_m']==140000


@pytest.mark.parametrize('name',['confirm','mda','resume','scenario','transfer'])
def test_no_unsupported_profile(name):
    with pytest.raises(ValueError):p.build_plan(name)


@pytest.mark.parametrize('method',['ev_equal','ev_prequential','cbob','unknown'])
def test_no_unregistered_model(method):
    with pytest.raises(ValueError):p.method_spec(method,p.PROFILES['pilot'])


@pytest.mark.parametrize('method,view',[('phase_exact','exact'),('phase_sign','sign')])
def test_information_views_change_only_declared_training_view(method,view):
    ref=p.method_spec('ev_latent',p.PROFILES['smoke'])
    assert p.method_spec(method,p.PROFILES['smoke'])=={**ref,'feedback':view}
    engine=p.engine_for(p.problem_for('aircraft_ordinary'),method,9910,p.PROFILES['smoke'],None).engine
    assert type(engine) is p.DiscoveryBO and engine.feedback==view and engine.coverage_scope=='cold'


def test_no_qualification_no_pilot(tmp_path):
    out=tmp_path/'run'
    with pytest.raises(ValueError):p.prepare(out,'pilot')
    assert not out.exists()


def test_prepare_imports_only_reviewed_receipt(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr(p,'qualification_report',lambda f:seen.append(f) or
                        {'receipt':p.q.receipt(),'cases':{},'new_model_calls':0})
    out=tmp_path/'new';plan=p.prepare(out,'pilot','old')
    assert seen==['old'] and plan['reviewed_qualification']==p.q.receipt()
    assert not (out/'calls.jsonl').exists() and not (out/'execution.lock').exists()
    assert p.load_plan(out)==plan


def test_incumbent_keeps_chronological_tie_and_initial(monkeypatch):
    prob=p.problem_for('aircraft_sparse');a,b=prob.anchors()
    rows=[row(prob,a,5),row(prob,b,5)]
    assert p.incumbent(rows) is rows[0]
    assert p.incumbent([row(prob,a,partial=True)]) is None


def test_held_out_changed_label_is_reported_not_used_to_replace():
    prob=p.problem_for('aircraft_sparse');c=prob.anchors()[0]
    nominal=row(prob,c,5);refined=row(prob,c,5,margin=.01)
    result=p.verification_result(nominal,refined)
    assert not result['verified_usable'] and result['screen']['status']=='needs_review'
    assert result['nominal_objective_Wh']==5 and result['record']==p.plain(refined.to_dict())
    assert p.DesignRecord.from_dict(result['record']).to_dict()==refined.to_dict()
    assert p.incumbent([nominal]) is nominal


def test_no_feasible_no_verification():
    assert p.verification_result(None,None)['status']=='no_nominal_feasible_incumbent'
    prob=p.problem_for('aircraft_sparse')
    with pytest.raises(ValueError):p.verification_result(None,row(prob,prob.anchors()[0]))


def test_verification_cannot_switch_candidate():
    prob=p.problem_for('aircraft_sparse');a,b=prob.anchors()
    with pytest.raises(ValueError):p.verification_result(row(prob,a),row(prob,b))


@pytest.mark.parametrize('state',['numerical','unknown','infrastructure_error'])
def test_untrusted_outcome_halts_before_training(state):
    prob=p.problem_for('aircraft_sparse');r=row(prob,prob.anchors()[0],numerical=state=='numerical')
    if state!='numerical':r=replace(r,observation=replace(r.observation,status=state))
    with pytest.raises(RuntimeError):p.ensure_trusted(r)


def test_native_stop_not_relaxed_or_retried():
    prob=p.problem_for('aircraft_sparse');r=row(prob,prob.anchors()[0],partial=True)
    p.validate_stop('original_lsbo',[r]*5,30,p.NATIVE_STOP)
    for method,online,reason in [('original_lsbo',[r]*4,p.NATIVE_STOP),('ev_latent',[r]*5,p.NATIVE_STOP),
                                  ('original_lsbo',[r]*6,p.NATIVE_STOP),('original_lsbo',[r]*30,None)]:
        with pytest.raises(ValueError):p.validate_stop(method,online,30,reason)


class FakeEngine:
    stop_reason=None
    def propose(self,history,pool,iteration):return pool[0],{'method':'fake'}
    def accept(self,record):pass


def setup_fake(monkeypatch):
    from ra_lsbo.experiments.core.problems import AircraftCase
    seen=[]
    monkeypatch.setattr(AircraftCase,'evaluate',lambda self,c:row(self,c))
    def factory(problem,method,seed,profile,output):
        obj=FakeEngine()
        obj.accept=lambda record:seen.append(record.observation.contract_id)
        return obj
    monkeypatch.setattr(p,'engine_for',factory)
    # These tests isolate runner/evidence. Real guarded-policy tests run separately.
    monkeypatch.setattr(p,'validate_decision',lambda *a:None)
    return seen


def test_roundtrip_verification_not_fed_back(monkeypatch,tmp_path):
    seen=setup_fake(monkeypatch);out=tmp_path/'run';p.prepare(out)
    result=p.run(out,54);audit=p.audit(out)
    assert result['charged_model_calls']==54 and len(seen)==28
    assert len(result['summaries'])==14 and audit['checked_proposals']==28
    assert all(x['step_verification']['verified_usable'] for x in result['summaries'])
    assert p.audit(out,False)==audit and p.report(out).exists()
    with pytest.raises(FileExistsError):p.run(out,54)
    with pytest.raises(FileExistsError):p.prepare(out)


@pytest.mark.parametrize('cap',[None,0,53,True])
def test_cap_before_lock(monkeypatch,tmp_path,cap):
    setup_fake(monkeypatch);out=tmp_path/'run';p.prepare(out)
    with pytest.raises(ValueError):p.run(out,cap)
    assert not (out/'execution.lock').exists() and not (out/'calls.jsonl').exists()


@pytest.mark.parametrize('target',['summary','validation','missing_online','timing','result','source'])
def test_tamper_refused(monkeypatch,tmp_path,target):
    setup_fake(monkeypatch);out=tmp_path/'run';p.prepare(out);p.run(out,54);p.audit(out)
    trial=next((out/'optimization').glob('*/*'))
    if target=='summary':
        f=trial/'summary.json';value=p.read(f);value['best_feasible']=-100
    elif target=='validation':
        f=trial/'verification.json';value=p.read(f);value['observation']['objective']['value']=-100
    elif target=='missing_online':
        f=trial/'proposals.jsonl';f.write_text('\n'.join(f.read_text().splitlines()[:-1])+'\n')
    elif target=='timing':
        f=trial/'timing.json';value=p.read(f);value['proposal_seconds']=[-1,-1]
    elif target=='result':
        f=out/'results.json';value=p.read(f);value['charged_model_calls']=1
    else:
        f=next((out/'source_snapshot').iterdir());f.write_bytes(b'tampered')
    if target not in {'missing_online','source'}:
        import json
        f.write_text(json.dumps(value))
    with pytest.raises((ValueError,RuntimeError)):p.report(out)


@pytest.mark.parametrize('name',['../secret','/absolute','C:/a','a\\b','./plan.json'])
def test_safe_prerequisite_paths(tmp_path,name):
    with pytest.raises(ValueError):p.q.safe_read(tmp_path,name)


def test_wrong_prerequisite_headers_refused(tmp_path):
    (tmp_path/'plan.json').write_text('{}')
    with pytest.raises(ValueError):p.q.verify(tmp_path)


def test_receipt_not_mutable_global():
    r=p.q.receipt();r['headers_sha256']['plan.json']='bad'
    assert p.q.receipt()['headers_sha256']['plan.json']==p.q.HEADERS['plan.json']


def test_cli_requires_explicit_command_and_output():
    with pytest.raises(SystemExit):p.main([])


def test_no_feasible_skips_all_terminal_calls(monkeypatch,tmp_path):
    setup_fake(monkeypatch)
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase,'evaluate',lambda self,c:row(self,c,margin=.1,partial=True))
    out=tmp_path/'run';p.prepare(out);r=p.run(out,54)
    assert r['charged_model_calls']==40 and p.audit(out)['settled_calls']==40
    assert all(x['step_verification']['status']=='no_nominal_feasible_incumbent' for x in r['summaries'])
    assert not list((out/'optimization').rglob('verification.json'))


def test_numeric_failure_is_settled_and_campaign_halts(monkeypatch,tmp_path):
    setup_fake(monkeypatch)
    from ra_lsbo.experiments.core.problems import AircraftCase
    monkeypatch.setattr(AircraftCase,'evaluate',lambda self,c:row(self,c,numerical=True))
    out=tmp_path/'run';p.prepare(out)
    with pytest.raises(RuntimeError):p.run(out,54)
    events=p.lines(out/'calls.jsonl')
    assert len(events)==2 and events[-1]['event']=='settled'
    assert (out/'execution.lock').exists() and not (out/'results.json').exists()


def test_output_must_not_be_inside_old_qualification(tmp_path):
    old=tmp_path/'old';old.mkdir()
    with pytest.raises(ValueError):p.prepare(old/'new','pilot',old)
    assert not (old/'new').exists()


def test_qualification_diagnostic_seal_checked(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'qualification_report',lambda f:{'receipt':p.q.receipt(),'new_model_calls':0})
    out=tmp_path/'new';p.prepare(out,'pilot',tmp_path/'old')
    (out/'qualification_diagnostics.json').write_text('{}')
    with pytest.raises(ValueError):p.load_plan(out)


def test_validated_incumbent_is_never_received_by_any_optimizer(monkeypatch,tmp_path):
    from ra_lsbo.experiments.core.problems import AircraftCase
    seen=setup_fake(monkeypatch)
    def evaluate(self,c):
        fine=self.settings['integration_step_s']==5.
        return row(self,c,objective=1. if fine else 10.,margin=.1 if fine else -.1)
    monkeypatch.setattr(AircraftCase,'evaluate',evaluate)
    out=tmp_path/'new';p.prepare(out);result=p.run(out,54);p.audit(out)
    assert len(seen)==28
    for r in result['summaries']:
        assert r['best_feasible']==10. and not r['step_verification']['verified_usable']
        assert r['step_verification']['refined_objective_Wh']==1.


def test_runtime_change_before_run_refused(monkeypatch,tmp_path):
    out=tmp_path/'new';p.prepare(out)
    old=p.runtime()
    monkeypatch.setattr(p,'runtime',lambda:{**old,'platform':'changed'})
    with pytest.raises(ValueError):p.run(out,54)
    assert not (out/'execution.lock').exists()


def test_observations_stay_raw_under_sign_ablation():
    from ra_lsbo.experiments.discovery_stages.policies import feedback_view
    prob=p.problem_for('aircraft_sparse');r=row(prob,prob.anchors()[0],partial=True)
    constraints=dict(r.observation.constraints)
    constraints['soc']=OutputObservation('interval','soc-unit',lower=.003)
    r=replace(r,observation=replace(r.observation,constraints=tuple(sorted(constraints.items()))))
    original=deepcopy(r.to_dict());view=feedback_view([r],'sign')[0]
    assert r.to_dict()==original
    assert view.observation.objective.kind=='missing'
    assert view.observation.constraint('soc').lower==0.
    assert not view.observation.constraint('soc').lower_inclusive
    assert view.observation.constraint('stall').kind=='exact'
