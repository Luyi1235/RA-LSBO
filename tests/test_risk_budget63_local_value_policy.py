"""PR63 pure selected-candidate credibility/value tests; never fit VAE or GP."""
from dataclasses import dataclass
from pathlib import Path

import pytest

from ra_lsbo.experiments.risk_budget63_local_value_policy import (
    SCHEMA, VARIANTS, _normalized_gains, local_credibility, choose_local_value,
)


@dataclass(frozen=True)
class Var:
    name: str
    bounds: tuple


class Space:
    continuous=(Var("x",(0.,1.)),Var("y",(0.,1.)),Var("trim",(0.,1.)))

    def active(self,var,discrete):
        return var.name!="trim" or discrete["architecture"]=="augmented"


class Problem:
    space=Space()


class Observation:
    status="completed"


class Record:
    def __init__(self,candidate,feasible):
        self.candidate=candidate
        self.observation=Observation()
        self.physical_feasible=feasible
        self.usable_feasible=feasible


def candidate(x,y=None,arch="baseline",trim=0.):
    return {"continuous":{"x":x,"y":x if y is None else y,"trim":trim},
            "discrete":{"architecture":arch,"module":2}}


def meta(risks,mode="probability_weighted_feasible_improvement"):
    return {
        "mode":mode,
        "eligible":[True]*len(risks),
        "feasibility_proxy_no_status":[1.-v for v in risks],
        "risk_budget":{
            "selected_predicted_failure":risks[0],
            "selected_predicted_feasible_proxy":1.-risks[0],
            "candidate_filter_applied":True,
            "selection_rule":"predicted_allowance_filter",
        },
        "scores":[1.]*len(risks),
        "fallback":None,
    }


def select(candidates,history,risks,variant,*,ei=None,mode="feasible_discovery",
           iteration=0):
    return choose_local_value(
        Problem(),meta(risks,mode),candidates,history,ei,[0.]*len(candidates),
        variant=variant,iteration=iteration)


def test_same_discrete_local_active_distance_prevents_beta_branch_pooling():
    hist=[Record(candidate(.1,trim=.1),False)]*6+[
        Record(candidate(.9,trim=.9),True)]*6
    near_failure=local_credibility(Problem(),hist,candidate(.1,trim=.8),.1)
    near_success=local_credibility(Problem(),hist,candidate(.9,trim=.1),.1)
    assert near_failure["same_discrete_branch_observations"]==12
    assert near_success["same_discrete_branch_observations"]==12
    assert near_failure["candidate_blended_failure_proxy"] > .5
    assert near_success["candidate_blended_failure_proxy"] < .15
    assert near_failure["nearest_same_branch_distance"]==pytest.approx(0.)
    # An inactive trim coordinate must not silently perturb local risk.
    same=local_credibility(Problem(),hist,candidate(.1,trim=.1),.1)
    assert same["candidate_blended_failure_proxy"]==pytest.approx(
        near_failure["candidate_blended_failure_proxy"])
    assert near_failure["independent_bernoulli_assumption_made"] is False


def test_soft_gp_vs_local_blend_change_actual_candidate_ranking_not_hard_gate():
    designs=[candidate(.1),candidate(.9)]
    history=[Record(candidate(.1),False)]*6+[Record(candidate(.9),True)]*6
    pure,raw=select(designs,history,[.1,.2],"soft_gp",ei=[1.,1.],
                    mode="probability_weighted_feasible_improvement")
    fused,blend=select(designs,history,[.1,.2],"local_blend",ei=[1.,1.],
                      mode="probability_weighted_feasible_improvement")
    assert pure==0 and fused==1
    assert raw["risk_budget"]["no_hard_risk_gate"] is True
    assert blend["risk_budget"]["candidate_filter_applied"] is False
    assert blend["credibility_decision"]["selected_local_evidence"][
        "weighted_local_support_mass"]>5
    assert blend["credibility_decision"]["selected_predicted_failure"]<.2
    assert blend["credibility_decision"]["schema"]==SCHEMA
    assert blend["scores"][fused]==pytest.approx(
        blend["candidate_net_utility"][fused])


def test_real_expected_gain_can_justify_evaluation_above_old_failure_allowance():
    history=[Record(candidate(.5),True)]*12
    designs=[candidate(.3),candidate(.7)]
    i,d=select(designs,history,[.1,.8],"soft_gp",
               ei=[.01,100.],mode="probability_weighted_feasible_improvement")
    assert i==1
    assert d["credibility_decision"]["selected_predicted_failure"]==pytest.approx(.8)
    assert d["credibility_decision"]["candidate_filter_applied"] is False
    assert d["credibility_decision"]["would_defer_in_unconstrained_utility_model"] is False
    assert d["credibility_decision"]["actual_action"]=="evaluate_under_frozen_six_call_protocol"
    assert d["credibility_decision"]["probability_guarantee"] is False


def test_no_acceptable_candidate_is_a_logged_outside_option_not_false_safety():
    history=[Record(candidate(.5),False)]*12
    i,d=select([candidate(.2),candidate(.8)],history,[.9,.99],
               "soft_gp",mode="feasible_discovery")
    assert i==0
    assert d["credibility_decision"]["would_defer_in_unconstrained_utility_model"] is True
    assert d["credibility_decision"]["selected_net_utility"]<0
    assert d["credibility_decision"]["candidate_risk_within_allowance"] is None
    assert d["credibility_decision"]["outside_option_utility"]==0
    assert d["credibility_decision"]["abstention_not_performed_in_equal_budget_study"] is True
    assert d["fallback"] is None


def test_value_of_information_is_bounded_and_gated_by_objective_promise():
    history=[Record(candidate(.5),True)]*12
    designs=[candidate(.1),candidate(.9)]
    a,soft=select(designs,history,[.2,.9],"local_blend",
                  ei=[1.,0.],mode="probability_weighted_feasible_improvement")
    b,with_info=select(designs,history,[.2,.9],"local_value",
                       ei=[1.,0.],mode="probability_weighted_feasible_improvement")
    assert a==b
    assert with_info["credibility_decision"]["selected_information_bonus"]>=0
    assert (with_info["candidate_net_utility"][0]-
            soft["candidate_net_utility"][0])>=0
    # Zero EI gets zero information reward; high-risk does not lure exploration.
    assert with_info["candidate_net_utility"][1]==pytest.approx(
        soft["candidate_net_utility"][1])
    gains=_normalized_gains("budgeted_boundary_information",[0.,.5,1.],None,
                            [True,True,True])
    assert gains==pytest.approx([0.,1.,0.])


def test_history_is_strictly_before_proposal_and_unknown_labels_are_rejected():
    history=[Record(candidate(.5),True)]*12
    designs=[candidate(.3),candidate(.7)]
    with pytest.raises(ValueError,match="prefix contract"):
        select(designs,history+[Record(candidate(.3),False)],
               [.2,.2],"local_value")
    unknown=Record(candidate(.4),True)
    unknown.physical_feasible=None
    with pytest.raises(ValueError,match="untrusted"):
        local_credibility(Problem(),history+[unknown],candidate(.3),.1)
    c=candidate(.2,arch="augmented",trim=.1)
    unobserved=local_credibility(Problem(),history,c,.3)
    assert unobserved["weighted_local_support_mass"]==0
    assert unobserved["candidate_blended_failure_proxy"]==pytest.approx(.3)
    assert unobserved["same_discrete_branch_observations"]==0


def test_invalid_probability_support_or_mode_fails_closed():
    hist=[Record(candidate(.5),True)]*12
    cs=[candidate(.3),candidate(.7)]
    with pytest.raises(ValueError,match="probability"):
        select(cs,hist,[float("nan"),.2],"soft_gp")
    with pytest.raises(ValueError,match="Unregistered"):
        select(cs,hist,[.1,.2],"local_bad")
    broken=meta([.1,.2])
    broken["eligible"]=[False,False]
    with pytest.raises(ValueError,match="support"):
        choose_local_value(Problem(),broken,cs,hist,None,[0.,0.],
                           variant="local_blend",iteration=0)
    with pytest.raises(ValueError,match="phase"):
        select(cs,hist,[.1,.2],"soft_gp",mode="future_mode")
    assert tuple(VARIANTS)==("soft_gp","local_blend","local_value")


def test_new_engine_source_binds_exact_model_pool_and_clears_coverage_receipts():
    p=Path(__file__).resolve().parents[1]/"ra_lsbo"/"experiments"
    s=(p/"risk_budget63_local_value_engine.py").read_text(encoding="utf8")
    assert "class LocalCredibilityValueBO(BudgetedDualViewMathBO)" in s
    assert "self.owner._pr63_last_predicted_candidates=list(candidates)" in s
    assert "selected_candidate_sha256" in s
    assert 'meta.pop("credibility_decision")' in s
    assert "model_selected_utility_not_attributed_to_coverage" in s
    assert "super()._choose_acquisition(" in s
