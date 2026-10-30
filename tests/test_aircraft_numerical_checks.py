"""Pure stored-data checks; no GP fitting or aircraft evaluation."""
from copy import deepcopy

import pytest

from ra_lsbo.experiments.simplified_aircraft.numerical_checks import sensitivity
from ra_lsbo.experiments.simplified_aircraft import reviewed_qualification as q


def value(kind='exact', number=0.):
    return {'kind':kind,'value':number if kind=='exact' else None,
            'lower':number if kind=='interval' else None,'upper':None,
            'lower_inclusive':True,'upper_inclusive':True}


def record(t=100.,soc=.19,partial=True):
    return {'candidate':{'x':1},'observation':{'status':'completed',
        'objective':value('interval' if partial else 'exact',10.),
        'constraints':[['soc',value('interval' if partial else 'exact',.2-soc)],
                       ['stall',value('exact',-.1)]]},
        'diagnostics':{'termination_reason':'physical_soc_reserve_abort' if partial else 'complete',
                      'trace':[{'stage':'climb','elapsed_s':50.,'soc':.8,'energy_prefix_Wh':4.},
                               {'stage':'cruise','elapsed_s':t,'soc':soc,'energy_prefix_Wh':10.}]}}


def test_partial_label_agreement_is_not_full_convergence():
    rows=[record(100,.19),record(95,.199)];before=deepcopy(rows)
    r=sensitivity(rows,[20,5])
    assert rows==before and r['constraint_labels_agree'] and r['termination_agrees']
    assert r['terminal_time_range_s']==5
    assert r['status']=='partial_outputs_not_certified'
    assert not r['full_response_convergence_proven']
    assert r['interval_bounds']['soc']['lower']==pytest.approx([.01,.001])
    assert set(r['completed_stage_endpoints'])=={'climb'}
    assert r['completed_stage_endpoints']['climb']['soc_range']==0.


def test_complete_and_partial_disagreement_retained():
    r=sensitivity([record(),record(soc=.21,partial=False)],[20,5])
    assert not r['termination_agrees'] and not r['constraint_labels_agree']
    assert r['physical_states']==['known_infeasible','usable_feasible']
    assert r['status']=='partial_outputs_not_certified'


def test_no_trace_does_not_invent_terminal_time():
    rows=[record(),record()]
    for row in rows:row['diagnostics']['trace']=[]
    r=sensitivity(rows,[20,5])
    assert r['terminal_time_range_s'] is None and r['completed_stage_endpoints']=={}


def test_compare_completed_stages_only_at_same_time():
    rows=[record(),record()];rows[1]['diagnostics']['trace'][0]['elapsed_s']=51.
    r=sensitivity(rows,[20,5])
    assert r['completed_stage_endpoints']['climb']['soc_range'] is None
    assert not r['completed_stage_endpoints']['climb']['same_time']


def test_complete_is_descriptive_not_certificate():
    r=sensitivity([record(soc=.3,partial=False),record(soc=.301,partial=False)],[20,5])
    assert r['status']=='complete_outputs_descriptive_check'
    assert not r['physical_validation'] and not r['full_response_convergence_proven']
    assert r['absolute_exact_constraint_ranges']['soc']==pytest.approx(.001)


@pytest.mark.parametrize('steps',[[20],[20,20],[20,0],[20,-1],[20,float('nan')]])
def test_invalid_steps_refused(steps):
    with pytest.raises(ValueError):sensitivity([record(),record()],steps)


def test_different_design_refused():
    rows=[record(),record()];rows[1]['candidate']['x']=2
    with pytest.raises(ValueError):sensitivity(rows,[20,5])


def test_missing_constraints_never_made_safe():
    rows=[record(),record()]
    for r in rows:r['observation']['constraints'][1][1]=value('missing')
    answer=sensitivity(rows,[20,5])
    assert answer['constraint_labels']['stall']==[None,None]


def test_unknown_status_retained():
    rows=[record(),record()];rows[1]['observation']['status']='unknown'
    assert sensitivity(rows,[20,5])['physical_states'][1]=='unresolved'


def test_physical_classification_open_zero_is_violation():
    v=value('interval',0.);v['lower_inclusive']=False
    assert q.satisfaction(v) is False
    v['lower_inclusive']=True
    assert q.satisfaction(v) is None
