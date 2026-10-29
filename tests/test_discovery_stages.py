"""Staged policies, independent controls, budget gates and evidence corruption."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core import protocol as core_protocol
from ra_lsbo.experiments.core.problems import MathematicalCase
from ra_lsbo.experiments.discovery_stages import methods, optimizer as model_module, policies, protocol, runner
from ra_lsbo.partial_bo.problems import candidate_pool
from ra_lsbo.risk_core.observations import OutputObservation

BASE = core_protocol.BASE


def samples():
    problem = MathematicalCase(BASE)
    pool, _ = candidate_pool(problem, 197029, 16)
    raw = pool[0]
    good = problem.record(raw, 5., {'region': -.1, 'resource': -.2}, {})
    bad = problem.record(raw, OutputObservation('missing', 'f', reason='early_stop'),
        {'region': OutputObservation('interval', 'g', lower=0., lower_inclusive=False), 'resource': -.2}, {})
    return problem, pool[1:], good, bad


def test_distance_ignores_inactive_and_separates_branches():
    problem, _, _, _ = samples()
    left = problem.canonical({'continuous': {'x': .2, 'y': .3, 'trim': .1},
                              'discrete': {'architecture': 'basic', 'module': 1}})
    right = deepcopy(left)
    right['continuous']['trim'] = .9
    assert policies.active_distance(problem, left, right) == 0
    right['continuous']['x'] += 5e-9
    assert policies.active_distance(problem, left, right) == pytest.approx(5e-9)
    right['discrete']['architecture'] = 'augmented'
    assert policies.active_distance(problem, left, right) is None


def test_near_filter_does_not_change_identity_or_observations():
    problem, pool, good, _ = samples()
    almost = deepcopy(good.candidate)
    almost['continuous']['x'] += 1e-10
    before = good.to_dict()
    kept, indices, rejected = policies.filter_near(problem, [almost, pool[0], pool[0]], [good])
    assert kept == [pool[0]] and indices == [1]
    assert len(rejected) == 2
    assert digest(almost) != good.observation.design_id
    assert good.to_dict() == before


@pytest.mark.parametrize('tol', [-1, float('nan'), float('inf')])
def test_bad_distance_tolerance(tol):
    problem, pool, good, _ = samples()
    with pytest.raises(ValueError):
        policies.filter_near(problem, pool, [good], tol)


def test_coverage_is_deterministic_and_does_not_evaluate(monkeypatch):
    problem, pool, good, _ = samples()
    monkeypatch.setattr(problem, 'evaluate', lambda *a: pytest.fail('No model calls in decision utilities'))
    first = policies.coverage_pick(problem, pool, [good], 20, 5)
    assert first == policies.coverage_pick(problem, pool, [good], 20, 5)
    assert 0 <= first[0] < len(pool)
    with pytest.raises(RuntimeError):
        policies.coverage_pick(problem, [], [good], 20, 5)


@pytest.mark.parametrize('mode', ['exact', 'sign', 'interval'])
def test_feedback_retains_raw_and_never_fills_missing(mode):
    _, _, _, bad = samples()
    before = bad.to_dict()
    row = policies.feedback_view([bad], mode)[0]
    assert bad.to_dict() == before
    assert row.observation.objective.kind == 'missing'
    assert row.observation.constraint('resource').kind == 'exact'
    assert row.observation.constraint('region').kind == ('missing' if mode == 'exact' else 'interval')
    assert row.observation.design_id == bad.observation.design_id


@pytest.mark.parametrize('lower,upper,label', [(0., None, False), (-3., -1., True), (-1., 1., None)])
def test_interval_to_sign_preserves_resolved_semantics(lower, upper, label):
    _, _, _, bad = samples()
    interval = OutputObservation('interval', 'g', lower=lower, upper=upper, lower_inclusive=False)
    original = replace(bad, observation=replace(bad.observation, constraints=(('region', interval),)))
    output = policies.feedback_view([original], 'sign')[0].observation.constraint('region')
    assert output.constraint_satisfaction() is label
    if label is None:
        assert output.kind == 'missing'
    elif label:
        assert output.lower is None and output.upper == 0
    else:
        assert output.lower == 0 and not output.lower_inclusive


def test_objective_bounds_are_not_given_an_arbitrary_sign():
    _, _, good, _ = samples()
    row = replace(good, observation=replace(good.observation,
        objective=OutputObservation('interval', 'f', lower=3., upper=7.)))
    assert policies.feedback_view([row], 'sign')[0].observation.objective.kind == 'missing'
    assert policies.feedback_view([row], 'interval')[0].observation.objective.upper == 7.


def test_utb_sign_flip_zero_beta_and_expanding_uncertainty():
    mean = np.array([-.2, .1, 2.])
    std = np.array([.1, .2, .3])
    bound = policies.optimistic_bound(mean, std)
    assert np.allclose(bound, -((-mean)+1.96*std))
    assert np.allclose(policies.optimistic_bound(mean, std, 0), mean)
    assert np.all(policies.optimistic_bound(mean, 2*std) <= bound)
    assert np.all((bound <= 0) <= (policies.optimistic_bound(mean, 2*std) <= 0))


@pytest.mark.parametrize('std,beta', [(-1,1), (1,-1), (float('nan'),1)])
def test_invalid_utb_inputs(std, beta):
    with pytest.raises(ValueError):
        policies.optimistic_bound(1, std, beta)


def fake_model(monkeypatch, singleton=False):
    class Posterior:
        def __init__(self, problem, base, records, seed, settings):
            self.best = min((r.observation.objective.value for r in records if r.usable_feasible), default=None)
        def predict(self, candidates):
            return candidates
        def diagnostics(self):
            return {'test_posterior': True}
    def choose(candidates, best, iteration, settings, probability):
        n = len(candidates)
        return 0, {'mode': 'feasible_discovery', 'fallback': None, 'kappa': 1.,
                   'eligible': [True]+[not singleton]*(n-1), 'boundary_slot': iteration % 5 == 4}
    monkeypatch.setattr(model_module, 'RiskPosterior', Posterior)
    monkeypatch.setattr(model_module, 'choose_acquisition', choose)


def make_optimizer(problem, cold_global=True, protect=True):
    return model_module.DiscoveryBO(problem, BASE, 7,
        methods.settings_for('ra_lsbo', core_protocol.PROFILES['smoke']),
        cold_global=cold_global, protect=protect)


def test_scheduled_coverage_bypasses_all_model_gates(monkeypatch):
    problem, pool, _, bad = samples()
    monkeypatch.setattr(model_module, 'RiskPosterior', lambda *a: pytest.fail('Scheduled coverage must not fit/gate'))
    opt = make_optimizer(problem)
    candidate, metadata = opt.propose([bad], pool, 0)
    assert candidate in pool and metadata['posterior_fit_skipped']
    assert metadata['coverage_reason'] == 'scheduled_coverage'
    assert not metadata['selected_from_local_augmentation']
    opt.accept(problem.record(candidate, 3., {'region': -.1, 'resource': -.2}, {}))
    assert opt.pending is None and opt.next_iteration == 1


def test_near_exhaustion_stops_without_relaxation(monkeypatch):
    problem, _, good, _ = samples()
    almost = deepcopy(good.candidate)
    almost['continuous']['x'] += 1e-10
    opt = make_optimizer(problem)
    with pytest.raises(RuntimeError, match='No novel'):
        opt.propose([good], [almost], 0)
    assert opt.pending is None


@pytest.mark.parametrize('warm', [False, True])
def test_cold_local_disabled_and_warm_local_restored(monkeypatch, warm):
    problem, pool, good, bad = samples()
    fake_model(monkeypatch)
    opt = make_optimizer(problem, protect=False)
    called = []
    def augment(posterior, records, pool, metadata):
        called.append(True)
        return pool, {'local_reports': [], 'augmented_pool_size': len(pool)}
    monkeypatch.setattr(opt, '_augment', augment)
    _, info = opt.propose([good if warm else bad], pool, 0)
    assert bool(called) == warm and info['local_enabled'] == warm


def test_singleton_gate_fallback_is_ungated_global(monkeypatch):
    problem, pool, _, bad = samples()
    fake_model(monkeypatch, singleton=True)
    opt = make_optimizer(problem)
    opt.next_iteration = 1
    candidate, info = opt.propose([bad], pool, 1)
    assert info['mode'] == 'degenerate_gate_coverage'
    assert info['model_eligible_count'] == 1 and candidate in pool
    assert info['pool'][info['selected_index']] == candidate


def test_proposal_and_feedback_state_is_bound(monkeypatch):
    problem, pool, _, bad = samples()
    opt = make_optimizer(problem)
    candidate, _ = opt.propose([bad], pool, 0)
    with pytest.raises(ValueError):
        opt.propose([bad], pool, 1)
    with pytest.raises(ValueError):
        opt.accept(bad)
    opt.accept(problem.record(candidate, 4., {'region': -.1, 'resource': -.2}, {}))
    with pytest.raises(ValueError, match='prefix'):
        opt.propose([bad], pool, 1)


@pytest.mark.parametrize('stage,n,budget', [('discovery',5,6516), ('representation',8,10116), ('baselines',8,10116)])
def test_pilot_budgets_and_unchanged_seeds(stage, n, budget):
    plan = protocol.build_plan(stage, 'pilot')
    assert len(plan['arms']) == n and plan['budget']['maximum_model_calls'] == budget
    assert tuple(plan['profile']['seeds']) == tuple(range(9701,9721))
    assert plan['budget']['maximum_rcaide_mda_calls'] == 0
    protocol.verify_plan(plan)


def test_factorial_feedback_views_are_matched():
    profile = core_protocol.PROFILES['pilot']
    for feedback in ('exact', 'sign', 'interval'):
        left = methods.specification('latent_'+feedback, profile)
        right = methods.specification('onehot_'+feedback, profile)
        left['settings']['latent'] = False
        assert left == right
    assert methods.specification('ra_cold_global', profile)['settings'] == methods.specification('ra_coverage', profile)['settings']


def test_no_confirm_or_mda_registration():
    with pytest.raises(ValueError, match='no confirm or MDA'):
        protocol.build_plan('discovery', 'confirm')


def test_later_pilot_requires_reviewed_prerequisite(tmp_path):
    with pytest.raises(ValueError, match='prerequisite'):
        runner.prepare(tmp_path/'later', 'representation', 'pilot')
    assert not (tmp_path/'later').exists()


def test_modified_specification_rejected():
    plan = protocol.build_plan('discovery', 'smoke')
    plan['methods']['ra_discovery']['coverage_period'] = 1
    with pytest.raises(ValueError, match='specification'):
        protocol.verify_plan(plan)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    profile = {**core_protocol.PROFILES['smoke'], 'cases': ('branin_sparse',), 'seeds': (9601,),
               'initial': 4, 'panel': 2, 'pool': 8, 'horizon': 2}
    monkeypatch.setitem(core_protocol.PROFILES, 'smoke', profile)
    monkeypatch.setattr(MathematicalCase, 'evaluate', lambda p,c: p.record(c, 5., {'region':-.1,'resource':-.2}, {}))
    fake_model(monkeypatch)
    class Uniform:
        stop_reason = None
        def __init__(self, method): self.method = method
        def propose(self, history, pool, iteration): return pool[0], {'method': self.method}
        def accept(self, record): pass
    original = methods.optimizer_for
    def factory(problem, base, seed, method, profile, output):
        if method in ('random','ra_v2'):
            return Uniform(method)
        opt = original(problem, base, seed, method, profile, output)
        monkeypatch.setattr(opt.engine, '_augment', lambda posterior,records,pool,info: (pool, {'local_reports':[]}))
        return opt
    monkeypatch.setattr(runner, 'optimizer_for', factory)
    folder = tmp_path/'trial'
    plan = runner.prepare(folder, 'discovery', 'smoke')
    with pytest.raises(ValueError, match='cap'):
        runner.run(folder)
    assert not (folder/'execution.lock').exists()
    result = runner.run(folder, plan['budget']['maximum_model_calls'])
    assert result['charged_model_calls'] == 18
    return folder


def test_campaign_round_trip_and_no_resume(toy):
    first = runner.audit(toy)
    assert first['settled_calls'] == 18 and first['checked_proposals'] == 10
    assert runner.audit(toy) == first and not first['algorithm_replay']
    assert runner.report(toy).exists()
    with pytest.raises(FileExistsError):
        runner.run(toy, 18)


def rewrite(path, obj):
    path.write_text(json.dumps(obj, allow_nan=False), encoding='utf-8')


@pytest.mark.parametrize('damage', ['horizon','ledger','snapshot','phase','coverage','summary','results','pool'])
def test_corrupted_evidence_rejected(toy, damage):
    proposal_path = toy/'optimization/branin_sparse/9601_ra_discovery/proposals.jsonl'
    if damage in {'horizon','phase','coverage','pool'}:
        rows = runner.lines(proposal_path)
        if damage == 'horizon': rows.pop()
        if damage == 'phase': rows[0]['metadata']['cold_start'] = True
        if damage == 'coverage': rows[0]['metadata']['coverage_reason'] = None
        if damage == 'pool': rows[0]['pool'] = rows[0]['pool'][::-1]
        proposal_path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    elif damage == 'ledger':
        path = toy/'calls.jsonl'
        rows = runner.lines(path)
        rows[1]['call_id'] = 'changed'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    elif damage == 'snapshot':
        path = next((toy/'source_snapshot').glob('*.source'))
        path.write_bytes(path.read_bytes()+b'\n# changed')
    elif damage == 'summary':
        path = proposal_path.parent/'summary.json'
        obj = runner.read(path); obj['seed'] = 0; rewrite(path,obj)
    else:
        path = toy/'results.json'
        obj = runner.read(path); obj['charged_model_calls'] += 1; rewrite(path,obj)
    with pytest.raises(ValueError):
        runner.audit(toy)


def test_report_rejects_stale_audit(toy):
    runner.audit(toy)
    rewrite(toy/'new_evidence.json', {'unexpected': True})
    with pytest.raises(ValueError, match='unchanged'):
        runner.report(toy)
