"""Pure regression tests for opt-in PR54 risk decisions; zero model evaluations."""
import ast
import math
from pathlib import Path

import pytest

from ra_lsbo.experiments.risk_budget54_policy import (
    SCHEMA, choose_semantic_or_budget, observed_failures,
)


def fake(mode="probability_weighted_feasible_improvement",
         probs=(.95,.30,.80), eligible=(0.,0.,0.)):
    n=len(probs)
    return {
        "mode":mode,"fallback":None,
        "probability_guarantee":False,
        "per_constraint_probability":{
            "region":list(probs),
            "resource":[1.]*n,
        },
        "optimistic_violation":list(eligible),
        "scores":[float(n-i) for i in range(n)],
        "completion_probability":[1.,.01,1.][:n],
    }


def choose(meta, policy="semantic_only", t=0, failures=0, ei=(.1,10.,1.)):
    return choose_semantic_or_budget(
        meta,None if ei is None else list(ei),
        [0.]*len(meta["optimistic_violation"]),
        policy=policy,iteration=t,horizon=12,
        failure_cap=6,failures_seen=failures)


def test_semantics_ignores_one_class_status_probability_and_ranks_by_ei_times_p():
    meta=fake()
    # Scores based on p*EI -> 3 > .8 > .095, select index 1.
    k,decision=choose(meta)
    assert k==1
    assert decision["risk_budget"]["schema"]==SCHEMA
    assert decision["risk_semantics_rule"]=="constraint_only_not_record_completion"
    assert decision["risk_budget"]["candidate_filter_applied"] is False
    assert decision["risk_budget"]["selected_predicted_feasible_proxy"]==pytest.approx(.3)
    assert decision["probability_guarantee"] is False
    revised=fake()
    revised["completion_probability"]=[0.,1.,.01]
    j,_=choose(revised)
    assert j==k  # a one-class pseudo-completion head cannot change ranking


def test_budget_changes_the_actually_selected_candidate():
    meta=fake()
    semantic,_=choose(meta,"semantic_only")
    budgeted,decision=choose(meta,"budgeted")
    assert semantic==1 and budgeted==2
    b=decision["risk_budget"]
    assert b["predicted_per_call_failure_allowance"]==pytest.approx(.5)
    assert b["selected_predicted_failure"]==pytest.approx(.2)
    assert b["candidate_risk_within_allowance"] is True
    assert b["selection_rule"]=="predicted_allowance_filter"
    assert decision["original_selector_index_before_semantic_fix"]==0


def test_fallback_when_no_candidate_passes_allowance_cannot_claim_safety():
    meta=fake(probs=(.6,.4,.1))
    chosen,decision=choose(meta,"budgeted",t=8,failures=5)
    assert chosen==0  # smallest *predicted* risk, still exceeds allowance
    b=decision["risk_budget"]
    assert b["predicted_per_call_failure_allowance"]==pytest.approx(.25)
    assert b["candidate_risk_within_allowance"] is False
    assert b["selection_rule"]=="no_candidate_below_allowance_min_predicted_failure"
    assert b["probability_guarantee"] is False


def test_boundary_entropy_separate_from_improvement_and_model_failure_support():
    meta=fake(mode="budgeted_boundary_information",
              probs=(.99,.50,.20))
    idx,decision=choose(meta,"semantic_only",ei=None)
    assert idx==1
    # Budget filter excludes p=.5 when allowance=(6/12)=.5? risk=.5
    # equal allowance is accepted, so boundary pick still 1.
    idx,_=choose(meta,"budgeted",ei=None)
    assert idx==1
    difficult=fake(mode="feasible_discovery",
                   eligible=(1.0,.1,0.4))
    idx,decision=choose(difficult,"semantic_only",ei=None)
    assert idx==1
    assert decision["fallback"]=="no_constraint_optimistic_support"


def test_invalid_or_exhausted_budget_refuses_to_score_not_fabricate_predictions():
    with pytest.raises(ValueError,match="cap"):
        choose(fake(),"budgeted",t=11,failures=6)
    meta=fake()
    meta["per_constraint_probability"]["region"][2]=float("nan")
    with pytest.raises(ValueError,match="probability"):
        choose(meta)
    meta=fake()
    meta["optimistic_violation"][0]=float("nan")
    with pytest.raises(ValueError,match="optimistic"):
        choose(meta)


class _Packet:
    def __init__(self,status="completed"):
        self.status=status


class _R:
    def __init__(self,usable=True,physical=True,status="completed"):
        self.observation=_Packet(status)
        self.physical_feasible=physical
        self.usable_feasible=usable


def test_only_settled_online_nonfeasible_spends_observed_cap():
    rows=[_R(usable=False,physical=False) for _ in range(12)]
    rows += [_R(True,True), _R(False,False), _R(False,False)]
    assert observed_failures(rows,12,3)==2
    with pytest.raises(ValueError,match="prefix"):
        observed_failures(rows,12,2)
    rows[-1]=_R(False,None)
    with pytest.raises(ValueError,match="Unknown"):
        observed_failures(rows,12,3)


def test_default_discovery_bo_selector_still_delegates_to_frozen_acquisition():
    path=Path(__file__).resolve().parents[1]/"ra_lsbo"/"experiments"/"discovery_stages"/"optimizer.py"
    tree=ast.parse(path.read_text(encoding="utf8"))
    obj=next(item for item in tree.body if isinstance(item,ast.ClassDef) and item.name=="DiscoveryBO")
    selector=next(item for item in obj.body if isinstance(item,ast.FunctionDef)
                  and item.name=="_choose_acquisition")
    assert len(selector.body)==2 or len(selector.body)==1  # docstring + direct return
    ret=next(item for item in selector.body if isinstance(item,ast.Return))
    assert isinstance(ret.value,ast.Call) and isinstance(ret.value.func,ast.Name)
    assert ret.value.func.id=="choose_acquisition"
    propose=next(item for item in obj.body if isinstance(item,ast.FunctionDef) and item.name=="propose")
    hooks=[node for node in ast.walk(propose)
           if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
           and node.func.attr=="_choose_acquisition"]
    assert len(hooks)==2
