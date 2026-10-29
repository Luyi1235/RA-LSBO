"""Raw-incumbent, native-stop and descriptive-evidence regression tests."""
from copy import deepcopy
from dataclasses import replace

import pytest

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core import protocol as core_protocol
from ra_lsbo.experiments.core.problems import MathematicalCase
from ra_lsbo.experiments.discovery_stages import methods, optimizer as model_module, runner
from ra_lsbo.partial_bo.problems import candidate_pool
from ra_lsbo.risk_core.observations import OutputObservation

BASE = core_protocol.BASE


def samples():
    problem = MathematicalCase(BASE)
    pool, _ = candidate_pool(problem, 197029, 16)
    good = problem.record(pool[0], 5., {'region': -.1, 'resource': -.2}, {})
    bad = problem.record(pool[0], OutputObservation('missing', 'f', reason='early_stop'),
        {'region': OutputObservation('interval', 'g', lower=0., lower_inclusive=False), 'resource': -.2}, {})
    return problem, pool[1:], good, bad


def fake_model(monkeypatch):
    class Posterior:
        def __init__(self, problem, base, records, seed, settings):
            self.best = min((r.observation.objective.value for r in records if r.usable_feasible), default=None)
        def predict(self, candidates):
            return candidates
        def diagnostics(self):
            return {'test_posterior': True}
    monkeypatch.setattr(model_module, 'RiskPosterior', Posterior)


def make_optimizer(problem, protect=False):
    return model_module.DiscoveryBO(problem, BASE, 7,
        methods.settings_for('ra_lsbo', core_protocol.PROFILES['smoke']), protect=protect)


def test_raw_feasible_incumbent_survives_exact_feedback_ablation(monkeypatch):
    problem, pool, good, _ = samples()
    good = replace(good, observation=replace(good.observation, constraints=(
        ('region', OutputObservation('interval', 'g', upper=0.)),
        ('resource', OutputObservation('exact', 'r', value=-.2)))))
    assert good.usable_feasible
    fake_model(monkeypatch)
    best_seen = []
    def choose(candidates, best, iteration, settings, probability):
        best_seen.append(best)
        return 0, {'mode': 'feasible_discovery', 'fallback': None, 'kappa':1.,
                   'eligible':[True]*len(candidates)}
    monkeypatch.setattr(model_module, 'choose_acquisition', choose)
    opt = make_optimizer(problem, protect=False)
    opt.feedback = 'exact'
    monkeypatch.setattr(opt, '_augment', lambda posterior,records,pool,info: (pool, {'local_reports':[]}))
    opt.propose([good], pool, 0)
    assert best_seen == [good.observation.objective.value]*2


def test_native_missing_objective_stop_consumes_only_settled_labels(monkeypatch, tmp_path):
    problem, _, _, bad = samples()
    monkeypatch.setattr(problem, 'evaluate', lambda *a: pytest.fail('Native adapter cannot call true evaluator'))
    profile = {**core_protocol.PROFILES['smoke'], 'horizon': 6}
    opt = methods.NativeLSBO(problem, BASE, 1701, profile, tmp_path)
    history = [bad]
    for step in range(5):
        candidate, metadata = opt.propose(history, [], step)
        row = replace(bad, candidate=candidate,
                      observation=replace(bad.observation, design_id=digest(candidate)))
        opt.accept(row)
        history.append(row)
        assert (opt.stop_reason is not None) == (step == 4)
    assert opt.stop_reason == runner.NATIVE_STOP and opt.pending is None
    with pytest.raises(ValueError):
        opt.propose(history, [], 5)


def test_descriptive_paired_rule_retains_no_solution_and_cold_denominators():
    from ra_lsbo.experiments.discovery_stages.metrics import lexicographic_pairs
    rows = []
    for seed, (a,b) in enumerate([(None,None),(None,3.),(2.,None),(1.,2.),(3.,1.),(2.,2.)]):
        for method,value in [('ref',a),('other',b)]:
            rows.append({'case':'toy','seed':seed,'initial_policy':'random','method':method,
                         'initial_kind':'cold','has_feasible':value is not None,'best_feasible':value})
    result = lexicographic_pairs(rows,'ref')[0]
    assert result['paired_n'] == result['cold_n'] == 6
    assert result['reference_wins'] == result['comparator_wins'] == result['ties'] == 2
    assert result['cold_reference_wins'] == result['cold_comparator_wins'] == result['cold_ties'] == 2
    with pytest.raises(ValueError):
        lexicographic_pairs(rows+rows[:1], 'ref')


def test_recorded_decision_diagnostics_count_near_repeats_without_evaluation(monkeypatch):
    from ra_lsbo.experiments.discovery_stages.metrics import decision_diagnostics
    problem, _, good, bad = samples()
    monkeypatch.setattr(problem, 'evaluate', lambda *a: pytest.fail('Diagnostics must not evaluate'))
    row = replace(bad, candidate=deepcopy(bad.candidate))
    proposals = [{'candidate':row.candidate, 'metadata':{
        'coverage_reason':'eligible_at_most_one', 'eligible':[True],
        'local_reports':[{'slsqp_clipping_warnings':2,'local_error':None}]}}]
    result = decision_diagnostics(problem,[bad],proposals,[row])
    assert result['near_repeat_calls'] == result['cold_near_repeat_calls'] == 1
    assert result['singleton_fallback_calls'] == result['model_singleton_steps'] == 1
    assert result['slsqp_clipping_warnings'] == 2
