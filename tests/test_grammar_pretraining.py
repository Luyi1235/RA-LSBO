"""Cheap isolated checks. No aircraft evaluator, GP fit, or optimizer is called."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest
import torch

from ra_lsbo import grammar_pretraining as g
from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.structured_encoder import EncoderSettings, StructuredRepresentation, state_digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'runs/compact_mda_20261021T081006Z_20eb608d/qualification_recovered_20261025/frozen_base/lsbo'


def load(name):
    package = '_grammar_unit'
    if package not in sys.modules:
        parent = ModuleType(package); parent.__path__ = [str(BASE)]
        sys.modules[package] = parent
    key = package+'.'+name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, BASE/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


@pytest.fixture
def sample():
    space_module, record_module = load('space'), load('records')
    space = space_module.DesignSpace.from_dict({'variables': [
        {'name': 'layout', 'kind': 'nominal', 'default': 'a', 'values': ['a', 'b']},
        {'name': 'count', 'kind': 'ordinal', 'default': 2, 'values': [2, 4, 6],
         'active_when': {'layout': ['b']}},
        {'name': 'battery', 'kind': 'ordinal', 'default': 32, 'values': [32, 40, 48]},
        {'name': 'x', 'kind': 'continuous', 'default': .5, 'bounds': [0., 1.]}]})
    f = PrefixFeatures.__new__(PrefixFeatures)
    f.space, f.latent = space, True
    f.modules = {'space': space_module, 'records': record_module}
    f.encoder = StructuredRepresentation(space, EncoderSettings(initial_epochs=2, joint_view='latent'), seed=7)
    f.encoder.fit([], epochs=2)
    f.state_hash = state_digest(f.encoder)
    rows = []
    for i, d in enumerate(space.configurations()[:4]):
        c = space.canonical(space_module.Candidate({'x': i/4}, d)).to_dict()
        packet = SimpleNamespace(status='completed', objective=OutputObservation('exact', f'e{i}', value=float(i)), constraints=())
        r = SimpleNamespace(candidate=c, observation=packet)
        r.to_dict = lambda c=c, i=i: {'candidate': c, 'objective': i}
        rows.append(r)
    g.reset_normalization(f, rows, 'legacy')
    f.identity = 'unit-template'
    return f, rows


def test_schema_enumeration_uses_all_known_unobserved_configurations(sample):
    f, rows = sample
    assert len(g.schema_candidates(f)) == 12 > len(rows)


def test_pretraining_cannot_use_objective_labels(sample, monkeypatch):
    f, rows = sample
    monkeypatch.setattr(g, 'PRETRAIN_UPDATES', 3)
    cp = g.pretrain_grammar(f, 7)
    assert not cp.performance_fitted
    assert cp.settings.performance_weight == 0
    assert all(r['losses']['performance'] == 0 for r in cp.training_trace)


def test_checkpoint_clone_is_immutable_and_repeatable(sample, monkeypatch):
    f, rows = sample
    monkeypatch.setattr(g, 'PRETRAIN_UPDATES', 3)
    monkeypatch.setattr(g, 'PREFIX_UPDATES', 3)
    cp = g.pretrain_grammar(f, 7); before = state_digest(cp)
    a = g.from_checkpoint(f, rows, cp, 7)
    b = g.from_checkpoint(f, rows, cp, 7)
    assert a.state_hash == b.state_hash and state_digest(cp) == before
    assert a.encoder is not cp and a.encoder is not b.encoder
    assert a.encoder.performance_fitted


def test_checkpoint_wrong_seed_rejected(sample, monkeypatch):
    f, rows = sample
    monkeypatch.setattr(g, 'PRETRAIN_UPDATES', 2)
    cp = g.pretrain_grammar(f, 7)
    with pytest.raises(ValueError): g.from_checkpoint(f, rows, cp, 8)


def test_mutated_checkpoint_rejected(sample, monkeypatch):
    f, rows = sample
    monkeypatch.setattr(g, 'PRETRAIN_UPDATES', 2)
    cp = g.pretrain_grammar(f, 7)
    with torch.no_grad(): next(cp.model.parameters()).add_(.1)
    with pytest.raises(ValueError): g.from_checkpoint(f, rows, cp, 7)


def test_metrics_do_not_change_training_random_state(sample):
    f, _ = sample
    before = state_digest(f.encoder)
    x = g.reconstruction(f.encoder)
    assert before == state_digest(f.encoder)
    assert 0 <= x['mean_code_exact_fraction'] <= 1
    assert len(x['codebook']) == 12


def test_scale_only_changes_last_four_columns(sample):
    f, rows = sample
    a = f.transform(g.schema_candidates(f))
    old = f.scale.copy()
    scaled = g.scaled_copy(f, rows)
    b = scaled.transform(g.schema_candidates(f))
    np.testing.assert_array_equal(a[:, :-4], b[:, :-4])
    np.testing.assert_array_equal(f.scale, old)
    assert np.all(b[:, -4:] >= 0) and np.all(b[:, -4:] <= 1 + 1e-12)
    assert scaled.state_hash == f.state_hash


def test_schema_scale_independent_of_observed_subset(sample):
    f, rows = sample
    a, b = g.scaled_copy(f, rows[:1]), g.scaled_copy(f, rows)
    np.testing.assert_array_equal(a.scale, b.scale)
    np.testing.assert_array_equal(a.offset, b.offset)


def test_scale_rejects_unknown_policy(sample):
    f, rows = sample
    with pytest.raises(ValueError): g.reset_normalization(f, rows, 'anything')


def test_constant_code_is_not_amplified(sample):
    f, rows = sample
    with torch.no_grad():
        f.encoder.model.mu.weight.zero_(); f.encoder.model.mu.bias.fill_(1.)
    f.state_hash = state_digest(f.encoder)
    result = g.scaled_copy(f, rows)
    assert result.inactive_latent_coordinates == [0, 1, 2, 3]
    np.testing.assert_array_equal(result.scale[-4:], [1]*4)
    assert np.isfinite(result.transform(g.schema_candidates(f))).all()


def test_unknown_status_not_used_by_performance(sample):
    f, rows = sample
    rows[0].observation.status = 'unknown'
    obs = g._observations(f, rows)
    assert not obs[0].result.has_objective()
    assert obs[1].result.has_objective()


def test_interval_and_exact_scores_separate():
    obs = [OutputObservation('exact', 'e', value=0.),
           OutputObservation('interval', 'i', lower=0., lower_inclusive=False),
           OutputObservation('missing', 'm', reason='not_executed')]
    s = g.score_observations(obs, [0, 0, 0], [1, 1, 1], constraint=True)
    assert s['n_exact'] == s['n_interval'] == s['n_missing'] == 1
    assert s['n_sign'] == 2 and s['exact_rmse'] == 0
    assert s['interval_nlpm'] == pytest.approx(np.log(2))
    assert s['sign_brier'] == pytest.approx(.25)


def test_interval_spanning_zero_has_no_sign():
    s = g.score_observations([OutputObservation('interval', 'i', lower=-1., upper=1.)], [0], [1], constraint=True)
    assert s['n_sign'] == 0 and s['sign_brier'] is None
    assert s['n_exact'] == 0 and s['exact_rmse'] is None


@pytest.mark.parametrize('mean,var', [([np.nan],[1]), ([0],[-1]), ([0],[np.inf]), ([0,1],[1])])
def test_invalid_forecasts_rejected(mean,var):
    with pytest.raises(ValueError): g.score_observations([OutputObservation('exact','e',value=0)], mean,var)


def test_all_missing_retains_denominator():
    s = g.score_observations([OutputObservation('missing','m',reason='unknown')], [0],[1])
    assert s['n_total'] == s['n_missing'] == 1
    assert s['exact_nll'] is None and s['interval_nlpm'] is None


def test_objective_does_not_get_constraint_sign():
    s = g.score_observations([OutputObservation('exact','e',value=2.)], [2],[1])
    assert s['n_sign'] == 0


def test_exact_noise_only_applies_to_density():
    obs=[OutputObservation('exact','e',value=0.), OutputObservation('interval','i',upper=0.)]
    a=g.score_observations(obs,[0,0],[1,1]); b=g.score_observations(obs,[0,0],[1,1],exact_noise=1.)
    assert b['exact_nll'] > a['exact_nll'] and b['interval_nlpm'] == a['interval_nlpm']
