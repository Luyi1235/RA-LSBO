"""PR64 pure protocol tests; no GP/VAE or evaluator calls."""
import copy
from pathlib import Path

import pytest
from ra_lsbo.experiments import risk_budget64_protocol as p


def trial(case,seed,arm,*,cap=2,abstain=False):
    initial=10.
    n=cap
    decisions=[]
    for i in range(n):
        if abstain and i==n-1:
            decisions.append({
                "iteration":i,"candidate_sha256":"a"*64,
                "evaluation_performed":False,"action":"abstain_stop_before_evaluator",
                "selected_net_utility":-.1,"predicted_nonfeasible":.9,
                "realized_nonfeasible":None,"no_evaluator_call":True,
            })
        else:
            decisions.append({
                "iteration":i,"candidate_sha256":(str(i)*64)[:64],
                "evaluation_performed":True,"action":"evaluate",
                "realized_nonfeasible":int(i==0),"predicted_nonfeasible":.2,
            })
    actual=sum(x["evaluation_performed"] for x in decisions)
    failures=sum(x.get("realized_nonfeasible")==1 for x in decisions)
    curve=[initial]*(actual+1)
    return {
        "case":case,"seed":seed,"arm":arm,"max_online_calls":cap,
        "online_calls":actual,"observed_nonfeasible":failures,
        "initial_best":initial,"final_best":initial,
        "best_feasible_at_each_actual_call":curve,
        "initial_prefix_sha256":"b"*64,"decisions":decisions,
        "stop_reason":"soft_utility_outside_option" if abstain else "max_budget_exhausted",
    }


def test_registered_protocol_uses_equal_maximum_not_equal_actual_budget():
    full=p.plan("full","a"*40)
    smoke=p.plan("smoke","a"*40)
    assert p.ARMS==("semantic_only","budgeted","soft_gp_continue","soft_gp_abstain")
    assert p.SEEDS==tuple(range(998101,998113))
    assert full["max_new_analytic_evaluation_budget"]==4*12*(12+4*6)
    assert smoke["max_new_analytic_evaluation_budget"]==4*(12+4*2)
    assert full["same_maximum_online_evaluation_budget"] is True
    assert full["actual_calls_may_be_lower_only_for_abstention_arm"] is True
    assert full["abstention_evaluator_call_count"]==0
    assert full["no_unexecuted_evaluation_is_counted_as_success"] is True
    with pytest.raises(ValueError):
        p.plan("invalid","a"*40)


def test_abstention_has_no_evaluation_and_no_fake_realized_label():
    r=trial(p.CASES[0],p.SEEDS[0],"soft_gp_abstain",abstain=True)
    assert p.check_trial(r,profile="smoke") is r
    bad=copy.deepcopy(r)
    bad["decisions"][-1]["realized_nonfeasible"]=0
    with pytest.raises(ValueError,match="abstention"):
        p.check_trial(bad,profile="smoke")
    bad=copy.deepcopy(r)
    bad["decisions"][-1]["evaluation_performed"]=True
    with pytest.raises(ValueError,match="evaluator count"):
        p.check_trial(bad,profile="smoke")
    bad=copy.deepcopy(r)
    bad["decisions"][-1]["selected_net_utility"]=.1
    with pytest.raises(ValueError,match="nonpositive"):
        p.check_trial(bad,profile="smoke")


def test_continue_arm_cannot_abstain_and_cannot_exceed_budget():
    r=trial(p.CASES[0],p.SEEDS[0],"soft_gp_continue")
    assert p.check_trial(r,profile="smoke") is r
    bad=copy.deepcopy(r)
    bad["decisions"].append({"evaluation_performed":False,
        "action":"abstain_stop_before_evaluator","selected_net_utility":-.1,
        "realized_nonfeasible":None})
    with pytest.raises(ValueError,match="only registered"):
        p.check_trial(bad,profile="smoke")
    bad=copy.deepcopy(r)
    bad["online_calls"]=3
    with pytest.raises(ValueError,match="maximum-call"):
        p.check_trial(bad,profile="smoke")


def test_paired_summary_reports_cost_saved_and_does_not_count_abstain_as_failure():
    rows=[]
    for c in p.CASES:
        for arm in p.ARMS:
            rows.append(trial(c,p.SEEDS[0],arm,abstain=arm=="soft_gp_abstain"))
    by,pairs=p.summarize(rows,profile="smoke")
    a=by["branin_ordinary/soft_gp_abstain"]
    assert a["actual_online_calls"]==4
    assert a["max_online_call_budget"]==8
    assert a["budget_saved"]==4
    assert a["observed_nonfeasible"]==4
    assert a["stopped_by_abstention"]==1
    assert pairs["branin_ordinary/soft_gp_abstain_vs_soft_gp_continue"]["candidate_abstentions"]==1
    assert pairs["branin_ordinary/soft_gp_abstain_vs_soft_gp_continue"]["candidate_actual_calls"]==1


def test_duplicate_or_missing_trial_rejected_and_runner_never_evaluates_none():
    rows=[trial(c,p.SEEDS[0],a,abstain=a=="soft_gp_abstain")
          for c in p.CASES for a in p.ARMS]
    with pytest.raises(ValueError,match="Duplicate"):
        p.summarize(rows+[rows[0]],profile="smoke")
    with pytest.raises(ValueError,match="roster"):
        p.summarize(rows[:-1],profile="smoke")
    src=(Path(__file__).resolve().parents[1]/"ra_lsbo"/"experiments"/
         "risk_budget64_runner.py").read_text(encoding="utf8")
    assert "if candidate is None:" in src
    assert "abstain_stop_before_evaluator" in src
    assert "rec=problem.evaluate(candidate)" in src
    assert "charged>spec[\"max_new_analytic_evaluation_budget\"]" in src
    assert "new_aircraft_calls" in src and "new_rcaide_calls" in src
