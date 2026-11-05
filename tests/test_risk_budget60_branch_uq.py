"""PR60: branch posterior and acquisition pure-Python regressions (no GP fits)."""
import ast
import math
from pathlib import Path

import pytest

from ra_lsbo.experiments.risk_budget60_branch_uq_policy import (
    SCHEMA, VARIANTS, branch_posterior, choose_branch_evidence,
)


class Packet:
    status="completed"


class Record:
    def __init__(self,branch,feasible):
        self.candidate={"discrete":{"branch":branch},"continuous":{"x":.3}}
        self.observation=Packet()
        self.physical_feasible=feasible
        self.usable_feasible=feasible


def candidate(branch):
    return {"discrete":{"branch":branch},"continuous":{"x":.2}}


def meta(probs, mode="probability_weighted_feasible_improvement"):
    return {
        "mode":mode,
        "eligible":[True]*len(probs),
        "feasibility_proxy_no_status":[1.-p for p in probs],
        "scores":[1.]*len(probs),
        "risk_budget":{"selected_predicted_failure":probs[0],
                       "selected_predicted_feasible_proxy":1.-probs[0]},
        "fallback":None, "probability_guarantee":False,
    }


def choose(m,c,history,variant,ei=None):
    return choose_branch_evidence(m,c,history,ei,
                                   [0.]*len(c),variant=variant,
                                   iteration=0,horizon=12,failure_cap=6)


def test_beta_posterior_mean_and_dispersion_depend_only_on_past_true_records():
    cs=[candidate("A"),candidate("B")]
    history=[Record("A",True)]*12
    post=branch_posterior(history,cs)
    a,b=post
    assert a["n_settled_in_branch"]==12
    assert a["observed_successes"]==12
    assert a["observed_failures"]==0
    assert a["posterior_failure_mean"]==pytest.approx(1/14)
    assert b["n_settled_in_branch"]==0
    assert b["posterior_failure_mean"]==pytest.approx(.5)
    assert b["posterior_failure_sd"]==pytest.approx(math.sqrt(1/12))
    assert b["posterior_failure_mean_plus_sd"]>a["posterior_failure_mean_plus_sd"]


def test_mean_only_vs_branch_dispersion_select_different_actual_candidates():
    cs=[candidate("A"),candidate("B")]
    history=[Record("A",True)]*12
    m=meta([.2,.1])
    a,ma=choose(m,cs,history,"branch_mean",[.4,10.])
    b,mb=choose(m,cs,history,"branch_dispersion",[.4,10.])
    assert a==1  # B: attractive objective, mean empirical risk 0.5 passes.
    assert b==0  # B: no observed support, dispersion surcharge rejects.
    assert ma["risk_budget"]["candidate_risk_within_allowance"] is True
    assert mb["risk_budget"]["candidate_risk_within_allowance"] is True
    assert ma["risk_budget"]["schema"]==SCHEMA
    assert mb["risk_budget"]["selected_branch_posterior"]["n_settled_in_branch"]==12
    assert mb["probability_guarantee"] is False


def test_information_fallback_changes_decision_and_never_calls_it_safe():
    cs=[candidate("A"),candidate("B")]
    history=[Record("A",True)]*12
    m=meta([.6,.9])
    a,ma=choose(m,cs,history,"branch_dispersion",[1.,1.])
    b,mb=choose(m,cs,history,"branch_explore",[1.,1.])
    assert a==0
    assert b==1
    assert mb["risk_budget"]["candidate_risk_within_allowance"] is False
    assert mb["risk_budget"]["selection_rule"]=="no_candidate_below_allowance_max_branch_information"
    assert mb["risk_budget"]["fallback_not_a_safety_guarantee"] is True
    assert ma["risk_budget"]["selection_rule"]=="no_candidate_below_allowance_min_corrected_proxy"
    assert mb["risk_budget"]["selected_predicted_failure"]>.5


def test_legacy_status_probability_is_not_used_as_mission_risk():
    cs=[candidate("A"),candidate("B")]
    history=[Record("A",True)]*12
    m=meta([.2,.1])
    m["completion_probability"]=[1.,0.]
    a,_=choose(m,cs,history,"branch_mean",[.4,10.])
    m["completion_probability"]=[0.,1.]
    b,_=choose(m,cs,history,"branch_mean",[.4,10.])
    assert a==b


def test_no_outcome_leakage_and_reject_invalid_history_or_arrays():
    cs=[candidate("A"),candidate("B")]
    hist=[Record("A",True)]*12
    m=meta([.1,.1])
    orig,meta_before=choose(m,cs,hist,"branch_dispersion",[1.,1.])
    assert choose(m,cs,hist,"branch_dispersion",[1.,1.])[0]==orig
    # A future result would alter posterior: that record is NOT part of the
    # current history. The algorithm has no access to it at selection time.
    future=hist+[Record("B",False)]
    posterior_after=branch_posterior(future,cs)
    assert posterior_after[1]!=branch_posterior(hist,cs)[1]
    bad=Record("A",True)
    bad.physical_feasible=None
    with pytest.raises(ValueError,match="settled"):
        branch_posterior(hist+[bad],cs)
    broken=meta([.1,.1])
    broken["feasibility_proxy_no_status"][1]=float("nan")
    with pytest.raises(ValueError,match="probability"):
        choose(broken,cs,hist,"branch_mean",[1.,1.])
    with pytest.raises(ValueError,match="Unregistered"):
        choose(m,cs,hist,"not_an_arm",[1.,1.])


def test_boundary_mode_without_ei_and_support_preservation():
    cs=[candidate("A"),candidate("B")]
    hist=[Record("A",True)]*12
    m=meta([.2,.9],mode="budgeted_boundary_information")
    m["eligible"]=[True,False]
    idx,dec=choose(m,cs,hist,"branch_explore",None)
    assert idx==0
    assert dec["eligible"]==[True,False]
    assert dec["scores"][1] is None
    assert dec["risk_budget"]["selected_predicted_failure"]<1.


def test_new_engine_isolated_from_exact_frozen_pr54_source():
    root=Path(__file__).resolve().parents[1]/"ra_lsbo"/"experiments"
    source=(root/"risk_budget60_branch_uq_engine.py").read_text(encoding="utf8")
    tree=ast.parse(source)
    assert "class BranchUQDualViewMathBO" in source
    assert "BudgetedDualViewMathBO" in source
    assert "_TrackedPosterior" in source
    assert "super()._choose_acquisition(" in source
    assert "choose_branch_evidence(" in source
    assert "original_index,original" in source
    assert "def predict(self,candidates)" in source
    assert not any(isinstance(t,ast.ImportFrom) and t.module=="risk_budget59_diagnostics"
                   for t in ast.walk(tree))
