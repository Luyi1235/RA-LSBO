"""No evaluator/model fitting: test sealed-score and decision-effect diagnostics."""
from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.special import ndtr

from ra_lsbo.experiments.discovery_stages.decision_effect import (
    cold_model_choice, compare_saved_choices, log_interval_mass, normalized_weights,
    observed_log_score, reweight_constraints, update_saved_weights)


def observation(kind='exact', **kw):
    return dict(kind=kind, value=kw.get('value'), lower=kw.get('lower'), upper=kw.get('upper'),
                lower_inclusive=kw.get('lower_inclusive', True), upper_inclusive=True)


def prediction():
    return {'completion': {'mean': [-3., -3.], 'variance': [.1, .1]}, 'objective': None,
        'constraints': {'g': {'mean': [0., 0.], 'variance': [26., 26.], 'scale': 1.,
          'components': {'weights': [.5,.5], 'mean': [[5., -5.], [-5., 5.]],
                         'variance': [[1.,1.],[1.,1.]]}}}}


@pytest.mark.parametrize('bad', [[0.], [0.,float('nan')], [0.,float('inf')]])
def test_reject_invalid_weights(bad):
    with pytest.raises(ValueError): normalized_weights(bad)


def test_weights_shift_invariant():
    assert np.allclose(normalized_weights([-10000.,-10001.]),normalized_weights([0.,-1.]))


@pytest.mark.parametrize('bounds', [(-np.inf,0.), (0.,np.inf), (-1.,2.), (8.,9.), (30.,31.)])
def test_interval_mass_finite_and_reflection(bounds):
    a,b=bounds
    x=log_interval_mass(a,b)
    assert np.isfinite(x) and x <= 0
    assert np.isclose(x,log_interval_mass(-b,-a))


@pytest.mark.parametrize('bounds', [(0.,0.), (2.,1.), (np.nan,1.)])
def test_bad_interval(bounds):
    with pytest.raises(ValueError): log_interval_mass(*bounds)


def test_half_line_does_not_impute_midpoint():
    p={'mean':1.,'variance':4.,'scale':1.}
    assert np.isclose(observed_log_score(p,observation('interval',lower=0.)),np.log(ndtr(.5)))
    assert observed_log_score({},observation('missing')) is None


def test_exact_density_units_change_but_relative_scores_do_not():
    p=[{'mean':.2,'variance':.8,'scale':1.},{'mean':1.,'variance':.3,'scale':1.}]
    a=[observed_log_score(x,observation(value=.4)) for x in p]
    k=1000.
    b=[observed_log_score({'mean':x['mean']*k,'variance':x['variance']*k*k,'scale':k},
                         observation(value=.4*k)) for x in p]
    assert np.allclose(np.array(b)-a,-math.log(k))
    assert np.isclose(b[0]-b[1],a[0]-a[1])


@pytest.mark.parametrize('status', ['unknown','infrastructure_error'])
def test_untrusted_outputs_never_update(status):
    w={'g':[-math.log(2)]*2}
    row={'observation':{'status':status,'constraints':[['g',observation(value=99.)]]}}
    result,scores=update_saved_weights(w,{'g':[{},{}]},row,'prequential')
    assert result==w and scores['g'] is None


def test_equal_and_adaptive_scoring_share_likelihood():
    w={'g':[-math.log(2)]*2};p={'g':[{'mean':0.,'variance':1.,'scale':1.},{'mean':4.,'variance':1.,'scale':1.}]}
    row={'observation':{'status':'completed','constraints':[['g',observation(value=0.)]]}}
    a,s=update_saved_weights(w,p,row,'equal');b,t=update_saved_weights(w,p,row,'prequential')
    assert a==w and s==t and normalized_weights(b['g'])[0]>.99
    assert w['g']==[-math.log(2)]*2


def test_missing_does_not_update():
    w={'g':[0.,-2.]}
    row={'observation':{'status':'completed','constraints':[['g',observation('missing')]]}}
    assert update_saved_weights(w,{'g':[{},{}]},row,'prequential')[0]==w


def test_mixture_includes_between_expert_variance_and_does_not_mutate():
    p=prediction();before=deepcopy(p)
    q=reweight_constraints(p,{'g':[.9,.1]},{'g':[1.,1.]})
    assert p==before
    assert np.allclose(q['constraints']['g']['mean'],[4.,-4.])
    assert np.allclose(q['constraints']['g']['variance'],[10.,10.])


@pytest.mark.parametrize('w', [[-.1,1.1],[.5,.6],[np.nan,.5]])
def test_invalid_mixture(w):
    with pytest.raises(ValueError): reweight_constraints(prediction(),{'g':w},{'g':[1.,1.]})


def test_cold_selector_can_change_with_expert_weights():
    p=prediction()
    a=cold_model_choice(reweight_constraints(p,{'g':[.99,.01]},{'g':[1.,1.]}),1)
    b=cold_model_choice(reweight_constraints(p,{'g':[.01,.99]},{'g':[1.,1.]}),1)
    assert a['model_index']==1 and b['model_index']==0
    assert a['eligible_count']==b['eligible_count']==1


def test_boundary_schedule_and_zero_variance_feasibility():
    p={'completion':{'mean':[-3.,-3.],'variance':[0.,0.]},'objective':None,
       'constraints':{'g':{'mean':[0.,1.],'variance':[0.,0.],'scale':1.}}}
    a=cold_model_choice(p,4)
    assert a['boundary_slot'] and a['eligible']==[True,False] and a['model_index']==0


def pair():
    p={'iteration':1,'prefix_digest':'same','candidate':{'x':2},'metadata':{
       'pool':[{'x':0},{'x':1},{'x':2}],'model_selected_index':0,
       'eligible':[True,False,False],'coverage_reason':'eligible_at_most_one'}}
    q=deepcopy(p);q['metadata'].update(model_selected_index=1,eligible=[False,True,False])
    return p,q


def test_internal_choice_change_can_be_erased_by_fallback():
    a,b=pair();r=compare_saved_choices(a,b)
    assert r['model_candidate_changed'] and r['gate_changed']
    assert r['model_change_overridden_by_coverage'] and not r['actual_candidate_changed']


def test_actual_choice_change_is_not_an_outcome_claim():
    a,b=pair();b['candidate']={'x':1}
    r=compare_saved_choices(a,b)
    assert r['actual_candidate_changed'] and not r['model_change_overridden_by_coverage']
    assert not any('objective' in key for key in r)


@pytest.mark.parametrize('field', ['prefix_digest','iteration','pool'])
def test_comparing_different_histories_or_pools_is_rejected(field):
    a,b=pair()
    if field=='pool': b['metadata']['pool']=[]
    else: b[field]='changed'
    with pytest.raises(ValueError): compare_saved_choices(a,b)


def test_scheduled_coverage_has_no_model_comparison():
    a,b=pair()
    for p in (a,b):
        p['metadata'].pop('model_selected_index');p['metadata'].pop('eligible')
        p['metadata']['coverage_reason']='scheduled_coverage'
    r=compare_saved_choices(a,b)
    assert not r['model_comparable'] and not r['model_candidate_changed']


def test_reader_keeps_strict_positive_interval_infeasible_and_unknown_missing():
    from tools.review_transfer_pilot import satisfaction
    assert satisfaction(observation('interval',lower=0.,lower_inclusive=False)) is False
    assert satisfaction(observation('interval',lower=0.)) is None
    assert satisfaction(observation('interval',upper=0.)) is True
    assert satisfaction(observation('missing')) is None


def test_pinned_review_rejects_absent_headers_without_creating_output(tmp_path):
    from tools.review_transfer_pilot import review
    root=tmp_path/'input';root.mkdir();output=tmp_path/'output'
    with pytest.raises(ValueError): review(root,output)
    assert not output.exists()


def test_review_rejects_in_source_output(tmp_path):
    from tools.review_transfer_pilot import review
    with pytest.raises(ValueError): review(tmp_path,tmp_path/'derived')
