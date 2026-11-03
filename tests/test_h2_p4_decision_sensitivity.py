import copy
import json

import pytest

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.selector import PartialBOSettings
from ra_lsbo.soc_revision.acquisition import choose_acquisition
from tools import h2_p4_decision_sensitivity as p4


def fixture_meta():
    pool = [{'x':0}, {'x':1}]
    soc = p4.component_mixture({'mean':[-1.,1.], 'variance':[.1,.1]},
                               {'mean':[1.,-1.], 'variance':[.1,.1]}, .5, 1.)
    forecasts = {'completion':{'mean':[-1.,-1.], 'variance':[.1,.1]},
                 'objective':None, 'constraints':{'soc':soc}}
    settings = {'boundary_fraction':.2, 'probability_exploitation':True}
    idx, detail = choose_acquisition(forecasts, None, 0, PartialBOSettings(horizon=18), True)
    return {'pool':pool, 'forecasts':forecasts, 'settings':settings, 'best_observed':None,
            'SOC_view_weight_onehot':.5, 'model_selected_index':idx, 'scores':detail['scores']}


def test_exact_mixture_and_fixed_pool_ranking_without_evaluator(monkeypatch):
    meta = fixture_meta()
    ids = [digest(x) for x in meta['pool']]
    result = p4.counterfactual(meta, 0, [.9,.1], ids)
    assert result['identified'] and result['model_rank_reversal']
    assert [x['candidate_id'] for x in result['reconstructed']] == ids
    assert result['reconstructed'][0]['actual_candidate_ranks'][ids[0]]['rank'] == 1
    assert result['local_search_path_reconstructed'] is False


def test_counterfactual_refuses_unreproduced_scores_or_coverage():
    meta = fixture_meta()
    meta['scores'][0] += 1
    assert p4.counterfactual(meta, 0, [.9,.1])['reason'] == 'saved_acquisition_not_reproduced'
    meta['coverage_reason'] = 'eligible_at_most_one'
    assert p4.counterfactual(meta, 0, [.9,.1])['reason'] == 'coverage_overrides_acquisition'


def test_common_forecast_comparison_uses_identical_designs_only():
    a = fixture_meta()
    b = copy.deepcopy(a)
    assert p4.common_forecast_check(a,b)['all_absolute_differences_at_most_1e-10']
    b['forecasts']['constraints']['soc']['components']['mean'][0][0] += .1
    assert not p4.common_forecast_check(a,b)['all_absolute_differences_at_most_1e-10']


def test_exact_interval_scoring_and_clipping():
    forecast = {'mean':0., 'variance':1.}
    exact = p4.proper_nll(forecast, p4.OutputObservation('exact','x',value=0.))
    interval = p4.proper_nll(forecast, p4.OutputObservation('interval','y',lower=0.))
    assert exact['type'] == 'exact_density'
    assert interval['type'] == 'interval_mass'
    assert exact['nll'] == pytest.approx(.9189385332046727)
    assert interval['nll'] == pytest.approx(.6931471805599453)


def test_archive_tampering_rejected(tmp_path):
    archive = tmp_path/'bad.zip'
    archive.write_bytes(b'not the frozen archive')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        p4.validate_archive(archive)


def test_zero_evaluator_and_no_model_fit_calls():
    import ast
    tree = ast.parse(p4.Path(p4.__file__).read_text(encoding='utf-8'))
    forbidden = {'fit', 'propose', 'evaluate', 'run_model', 'train'}
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not forbidden.intersection(calls)


def test_archived_outputs_when_supplied():
    import os
    path = os.environ.get('H2_P4_OUTPUT')
    if not path:
        return
    d = json.loads((p4.Path(path)/'diagnosis.json').read_text(encoding='utf-8'))
    s = json.loads((p4.Path(path)/'scores_by_type.json').read_text(encoding='utf-8'))
    assert len(d['cases']) == 16 and d['new_aircraft_calls'] == d['new_gp_fits'] == d['new_vae_fits'] == 0
    assert sum(x['subsequent_different_decisions_descriptive_only'] for x in d['cases']) == 210
    assert all(x['common_history_verified'] for x in d['cases'])
    assert all(x['matched_prefix_steps'][-1]['novel_global_pool_equal_ordered'] for x in d['cases'])
    assert all(not x['matched_prefix_steps'][-1]['pool_equal_ordered'] for x in d['cases'])
    assert sum(x['matched_prefix_steps'][-1]['fixed_pool_counterfactual'].get('model_rank_reversal',False)
               for x in d['cases']) == 8
    assert s['totals']['h2_dynamic']['exact_density'] == 254
    assert s['totals']['h2_fixed50']['interval_mass'] == 34
