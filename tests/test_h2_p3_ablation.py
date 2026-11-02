"""Zero-aircraft P3 contract, synthetic closed-loop and audit tests."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.simplified_aircraft import h2_p3_pilot as pilot, h2_p3_protocol as protocol
from ra_lsbo.experiments.simplified_aircraft.corpus_optimizer import CorpusDiscoveryBO
from ra_lsbo.experiments.simplified_aircraft.h2_optimizer import CausalH2CorpusBO
from ra_lsbo.experiments.simplified_aircraft.h2_p3_optimizer import FixedH2CorpusBO, FixedSOCController
from ra_lsbo.experiments.simplified_aircraft.h2_risk_controller import CausalSOCController, component_mixture
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.partial_bo.problems import candidate_pool
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.risk_core.observations import OutputObservation, RunObservation


@pytest.fixture(autouse=True)
def prohibit_aircraft_truth(monkeypatch):
    from ra_lsbo.experiments.core.problems import AircraftCase
    from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft
    def forbidden(*_args, **_kwargs):
        pytest.fail('A physical aircraft evaluator was called')
    monkeypatch.setattr(AircraftCase, 'evaluate', forbidden)
    monkeypatch.setattr(SimplifiedElectricAircraft, 'evaluate', forbidden)


def test_roster_budgets_and_identical_nonweight_settings():
    assert protocol.ARMS == ('h2_dynamic', 'h2_fixed50')
    smoke, full = protocol.profile_for('smoke'), protocol.profile_for('pilot')
    assert smoke['seeds'] == [12070] and full['seeds'] == list(range(12071, 12079))
    assert protocol.budget_for(smoke) == dict(shared_initial=12, online=8,
        incumbent_step_verification=4, maximum_model_calls=24, maximum_rcaide_mda_calls=0)
    assert protocol.budget_for(full) == dict(shared_initial=192, online=576,
        incumbent_step_verification=32, maximum_model_calls=800, maximum_rcaide_mda_calls=0)
    for profile in (smoke, full):
        a, b = (protocol.method_spec(m, profile) for m in protocol.ARMS)
        assert a.pop('weight_policy') == 'causal'
        assert b.pop('weight_policy') == 'fixed_0.5'
        assert a == b
        assert protocol.model_settings('h2_dynamic', profile) == protocol.model_settings('h2_fixed50', profile)


@pytest.mark.parametrize('method', protocol.ARMS)
def test_synthetic_optimizer_closed_loop_and_replay(method, tmp_path):
    problem = problem_for('aircraft_ordinary')
    candidates, _ = candidate_pool(problem, 812403, 6)
    settings = protocol.model_settings(method, protocol.profile_for('pilot'))
    cls = CausalH2CorpusBO if method == 'h2_dynamic' else FixedH2CorpusBO
    engine = cls(problem, BASE, 17, settings, tmp_path / 'never-created')
    history, proposals, online, weights = [], [], [], []

    def fake_parent(records, _pool, iteration):
        candidate = candidates[iteration]
        prefix = digest([r.to_dict() for r in records])
        engine.pending = {'design': digest(candidate), 'contract': problem.contract_id,
            'records': list(records), 'prefix_digest': prefix,
            'branch': engine._branch(candidate), 'prior_branch_best': None}
        mixed = component_mixture({'mean': np.array([-.3]), 'variance': np.array([.02])},
                                  {'mean': np.array([.4]), 'variance': np.array([.02])},
                                  engine.h2.weight, .1)
        return candidate, {'raw_history_digest': prefix, 'selected_index': 0,
            'posterior_fit_skipped': False, 'coverage_reason': None,
            'forecasts': {'constraints': {'soc': {'components': {
                k: v.tolist() for k, v in mixed['components'].items()}}}}}

    with patch.object(CorpusDiscoveryBO, 'propose', side_effect=fake_parent):
        for iteration in range(6):
            candidate, metadata = engine.propose(history, candidates, iteration)
            prefix = digest([r.to_dict() for r in history])
            weights.append(metadata['H2_controller_before']['onehot_weight'])
            proposals.append({'candidate': candidate, 'prefix_digest': prefix, 'metadata': metadata})
            assert engine.h2.pending and engine.h2.n_scored == iteration
            constraints = []
            for name in problem.required_constraints:
                value = (OutputObservation('interval', name, upper=-.02)
                         if name == 'soc' and iteration == 5 else
                         OutputObservation('exact', name, value=-.05))
                constraints.append((name, value))
            packet = RunObservation(design_id=digest(candidate), contract_id=problem.contract_id,
                evidence_id=f'synthetic-{iteration}', status='completed',
                objective=OutputObservation('exact', 'objective', value=1.), constraints=tuple(constraints))
            record = DesignRecord(candidate, packet, {'synthetic_only': True})
            engine.accept(record)
            history.append(record)
            online.append(record)

    assert weights[:4] == [.5] * 4
    assert (weights[4] > .5) if method == 'h2_dynamic' else (weights == [.5] * 6)
    replay = pilot.replay_h2_controller(proposals, online, method)
    assert replay['scored_events'] == 6 and replay['exact_scores'] == 5 and replay['interval_scores'] == 1
    assert replay['final_state'] == engine.h2.snapshot()
    scored = pilot.h2_prequential_soc(proposals, online)
    assert scored['exact'] == 5 and scored['interval'] == 1
    tampered = deepcopy(proposals)
    tampered[4]['metadata']['H2_controller_before']['onehot_weight'] = .99
    with pytest.raises(ValueError, match='controller'):
        pilot.replay_h2_controller(tampered, online, method)
    tampered = deepcopy(proposals)
    tampered[0]['prefix_digest'] = 'future-history'
    with pytest.raises(ValueError, match='prefix'):
        pilot.replay_h2_controller(tampered, online, method)


@pytest.mark.parametrize('controller', [CausalSOCController, FixedSOCController])
def test_fallback_does_not_invent_score(controller):
    c = controller()
    for reason in ('coverage', 'model_failure', 'no_forecast'):
        key = f'design-{reason}'
        c.register(key, 'prefix', skip=reason)
        event = c.settle(key, 'prefix', OutputObservation('exact', 'soc', value=-.1))
        assert event['skipped'] == reason and c.n_scored == 0 and c.weight == .5


def test_fixed_controller_records_opposite_scores_without_weight_change():
    c = FixedSOCController()
    for i in range(8):
        c.register(str(i), str(i), {'mean': -.5, 'variance': .02}, {'mean': .5, 'variance': .02})
        event = c.settle(str(i), str(i), OutputObservation('exact', 'soc', value=(-.5 if i < 4 else .5)))
        assert event['weight_before'] == event['weight_after'] == .5
    assert c.n_scored == 8 and c.events[0]['delta_raw'] > 0 and c.events[-1]['delta_raw'] < 0


def test_pairing_retains_infeasible_seeds_and_separates_tasks():
    rows = []
    for case in protocol.CASES:
        for method in protocol.ARMS:
            feasible = not (case == 'aircraft_sparse' and method == 'h2_fixed50')
            rows.append(dict(case=case, seed=12071, method=method, initial_kind='cold',
                has_feasible=feasible, best_feasible=10. if feasible else None))
    pairs = pilot.ablation_pairs(rows)
    assert len(pairs) == 2 and pairs[0]['dynamic_minus_fixed'] == 0.
    assert pairs[1]['dynamic_minus_fixed'] is None and not pairs[1]['fixed_feasible']


def test_zero_evaluator_registration_and_snapshot_tampering(tmp_path, monkeypatch):
    from ra_lsbo.experiments.core.runner import Ledger
    from ra_lsbo.experiments.simplified_aircraft import reviewed_qualification
    monkeypatch.setattr(pilot, 'require_clean_git', lambda: None)
    monkeypatch.setattr(pilot, 'qualification_report',
                        lambda _path: {'receipt': reviewed_qualification.receipt()})
    monkeypatch.setattr(Ledger, 'evaluate', lambda *_a, **_kw: pytest.fail('truth evaluator called'))
    qualification = tmp_path / 'qualification'
    qualification.mkdir()
    out = tmp_path / 'registered'
    registered = pilot.prepare(out, 'smoke', qualification=qualification)
    loaded = pilot.load_plan(out)
    assert registered == loaded and loaded['budget']['maximum_model_calls'] == 24
    assert all(not path.startswith(str(pilot.ROOT)) for path in loaded['sources'])
    assert not any((out / x).exists() for x in ('execution.lock', 'calls.jsonl', 'results.json'))
    snapshot = out / 'source_snapshot' / (next(iter(loaded['sources'].values())) + '.source')
    snapshot.write_bytes(snapshot.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='Snapshot changed'):
        pilot.load_plan(out)


def test_synthetic_math_sequential_closed_loop_without_gp_or_truth_evaluator():
    from scipy.special import ndtr

    def run(controller):
        candidates = [0., .25, .5, .75, 1.]
        picked, scores, weights = [], [], []
        for step in range(10):
            available = [x for x in candidates if x not in picked[-2:]]
            forecasts = {x: ({'mean': x - .6, 'variance': .04},
                             {'mean': x - .2, 'variance': .04}) for x in available}
            w = controller.weight
            def utility(x):
                one, lat = forecasts[x]
                probability = w * ndtr(-one['mean'] / .2) + (1-w) * ndtr(-lat['mean'] / .2)
                return float(probability - .12 * (x-.6)**2)
            candidate = max(available, key=utility)
            prefix = digest({'picked': picked, 'scores': scores})
            one, lat = forecasts[candidate]
            controller.register(str(step), prefix, one, lat)
            weights.append(controller.weight)
            # Explicit mathematical synthetic feedback, created after forecast commitment.
            outcome = OutputObservation('exact', str(step), value=candidate-.6)
            event = controller.settle(str(step), prefix, outcome)
            picked.append(candidate)
            scores.append(event['delta_capped'])
        return picked, scores, weights

    dynamic, fixed = run(CausalSOCController()), run(FixedSOCController())
    assert dynamic[2][:4] == [.5] * 4 and dynamic[2][4] != .5
    assert fixed[2] == [.5] * 10
    assert len(dynamic[0]) == len(fixed[0]) == 10


def test_math_only_prepare_run_audit_report_and_no_resume(tmp_path, monkeypatch):
    """Exercise the actual P3 ledger/runner with a synthetic mathematical problem."""
    from dataclasses import asdict
    from ra_lsbo.experiments.simplified_aircraft import reviewed_qualification
    from ra_lsbo.experiments.discovery_stages import policies

    class MathProblem:
        settings = {'kind': 'mathematics-only'}
        def __init__(self, case, step):
            self.contract_id = f'math:{case}:{step}'
        def canonical(self, candidate):
            return candidate
        def precheck(self, _candidate):
            return {}
        def evaluate(self, candidate):
            packet = RunObservation(design_id=digest(candidate), contract_id=self.contract_id,
                evidence_id='math-only', status='completed',
                objective=OutputObservation('exact', 'objective', value=1.),
                constraints=(('soc', OutputObservation('exact', 'soc', value=.1)),))
            return DesignRecord(candidate, packet, {'synthetic_only': True})

    def pool(_problem, seed, count, exclude=None):
        exclude = exclude or set()
        values = []
        i = 0
        while len(values) < count:
            candidate = {'x': seed * 1000 + i}
            if digest(candidate) not in exclude:
                values.append(candidate)
            i += 1
        return values, []

    class MathEngine:
        stop_reason = None
        def __init__(self, method, profile):
            self.method = method
            self.settings = asdict(protocol.model_settings(method, profile))
            self.controller = (CausalSOCController() if method == 'h2_dynamic'
                               else FixedSOCController())
        def propose(self, history, candidates, _iteration):
            candidate = candidates[0]
            prefix = digest([r.to_dict() for r in history])
            before = self.controller.snapshot()
            prior = self.controller.register(digest(candidate), prefix, skip='coverage')
            return candidate, {'method':self.method, 'raw_history_digest':prefix,
                'settings':self.settings, 'global_near_rejections':[], 'pool':candidates,
                'selected_index':0, 'coverage_reason':'scheduled_coverage',
                'posterior_fit_skipped':True, 'H2_controller_before':before,
                'H2_prior_forecast_commitment':prior}
        def accept(self, row):
            self.controller.settle(row.observation.design_id, self.controller.pending['prefix'],
                                   row.observation.constraint('soc'))

    monkeypatch.setattr(pilot, 'require_clean_git', lambda: None)
    monkeypatch.setattr(pilot, 'qualification_report',
                        lambda _: {'receipt': reviewed_qualification.receipt()})
    monkeypatch.setattr(pilot, 'problem_for', lambda case, step=20.: MathProblem(case, step))
    monkeypatch.setattr(pilot, 'candidate_pool', pool)
    monkeypatch.setattr(pilot, 'engine_for', lambda _p, method, _s, profile, _o: MathEngine(method, profile))
    monkeypatch.setattr(pilot, 'audit_training', lambda *_a: {'fit_seconds':0., 'fit_count':0,
                        'gradient_updates':0, 'checkpoint_events':[]})
    monkeypatch.setattr(pilot, 'decision_diagnostics', lambda *_a: {})
    monkeypatch.setattr(policies, 'filter_near', lambda _p, pool, _h: (pool, [], []))
    monkeypatch.setattr(policies, 'nearest_distance', lambda *_a: 1.)
    qualification = tmp_path / 'qualification'
    qualification.mkdir()
    output = tmp_path / 'math-campaign'
    pilot.prepare(output, 'smoke', qualification=qualification)
    with pytest.raises(ValueError, match='complete-call cap'):
        pilot.run(output, 23)
    assert not (output / 'execution.lock').exists()
    result = pilot.run(output, 24)
    assert result['charged_model_calls'] == 20  # no feasible incumbent, hence no 5s checks
    assert len(result['summaries']) == 4
    audit = pilot.audit(output)
    assert audit['passed'] and audit['checked_proposals'] == 8
    assert pilot.report(output).is_file()
    assert pilot.smoke_receipt(output)['smoke_no_feasible_trial_count'] == 4
    with pytest.raises((FileExistsError, RuntimeError)):
        pilot.run(output, 24)
    ledger = output / 'calls.jsonl'
    original = ledger.read_text(encoding='utf-8')
    ledger.write_text(original.replace('"role": "shared_initial"', '"role": "online"', 1), encoding='utf-8')
    with pytest.raises(ValueError, match='Ledger identity'):
        pilot.audit(output, False)
