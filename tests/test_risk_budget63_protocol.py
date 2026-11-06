"""PR63 frozen 6-arm/4-task/12-seed equal-real-effort protocol; pure Python."""
import copy
from pathlib import Path

import pytest
from ra_lsbo.experiments import risk_budget63_protocol as p


SOFT=("soft_gp","local_blend","local_value")


def trial(case,seed,arm,*,n=6,cold=False,found=False,best=8.):
    initial=None if cold else 10.
    first=(2 if found else None) if cold else (1 if n>=2 else 0)
    fails=(set(range(n)) if cold and not found else
           set(range(n))-{first} if cold else
           {0} if first>0 else set())
    curve=[initial]*(n+1)
    for i in range(first+1 if first is not None else n+1,n+1):
        curve[i]=best if initial is None else min(initial,best)
    receipts=[]
    for i in range(n):
        have=arm!="legacy"
        soft=arm in SOFT
        gp=.2
        local=.4
        r=(gp if arm=="soft_gp" else local) if soft else .3
        receipts.append({
            "iteration":i,"realized_nonfeasible":int(i in fails),
            "forecast_available_for_selected":have,
            "predicted_nonfeasible":r if have else None,
            "budget_rule":("no_hard_gate_max_expected_net_value" if soft else
                           "predicted_allowance_filter" if arm=="budgeted" else
                           "constraint_only_ranking" if have else None),
            "selected_local_evidence":{
                "same_discrete_branch_observations":3,
                "weighted_local_support_mass":2.,
                "local_trust_weight":.5,
                "candidate_local_ambiguity":.4,
                "candidate_blended_failure_proxy":local,
            } if soft else None,
            "original_gp_predicted_failure":gp if soft else None,
            "selected_net_utility":-.1 if arm=="local_value" and i==0
                                   else .25 if soft else None,
            "no_hard_risk_gate":True if soft else None,
            "actual_action":"evaluate_under_frozen_six_call_protocol" if soft else None,
            "would_defer_in_unconstrained_utility_model":
                True if arm=="local_value" and i==0 else False if soft else None,
        })
    return {
        "case":case,"seed":seed,"arm":arm,
        "initial_best":initial,"final_best":curve[-1],
        "online_calls":n,"observed_nonfeasible":len(fails),
        "first_online_feasible_iteration":first,
        "best_feasible_at_each_actual_call":curve,
        "risk_receipts":receipts,
        "initial_prefix_sha256":"a"*64,
        "selected_path_sha256":(str(seed)+arm).ljust(64,"f"),
        "six_call_horizon":6,"forecast_internal_horizon":12,
        "evidence_before_result_was_future_blind":True,
    }


def test_registered_seeds_and_equal_new_evaluation_cost_are_frozen():
    full=p.plan("full","a"*40)
    smoke=p.plan("smoke","a"*40)
    assert p.SCHEMA=="ra_lsbo_pr63_candidate_credibility_soft_value_onpolicy_v1"
    assert len(p.ARMS)==6 and len(p.SEEDS)==12 and len(p.CASES)==4
    assert p.SEEDS==tuple(range(997201,997213))
    assert full["fresh_nonoverlap_with_pr54_pr55_pr58_pr60"] is True
    assert full["new_analytic_evaluation_ceiling"]==4*12*(12+6*6)==2304
    assert smoke["new_analytic_evaluation_ceiling"]==4*(12+6*2)==96
    assert full["engine_internal_horizon_unmodified"]==12
    assert full["actual_online_calls_per_arm"]==6
    assert full["primary_contrast"]=="local_value_vs_budgeted_per_task"
    assert full["soft_choice_always_evaluates_in_six_call_experiment"] is True
    assert full["new_arms_never_use_a_hard_risk_probability_gate"] is True
    assert full["no_probability_safety_guarantee"] is True
    assert p.PAIRS[0]==("local_value","budgeted")
    with pytest.raises(ValueError):
        p.plan("future","a"*40)


def test_actual_six_call_and_frozen_before_result_local_receipt():
    row=trial("branin_sparse",p.SEEDS[0],"local_value",cold=True,found=True)
    assert p.check_trial(row,profile="full") is row
    broken=copy.deepcopy(row)
    broken["risk_receipts"][2]["selected_local_evidence"]=None
    with pytest.raises(ValueError,match="local trust/utility"):
        p.check_trial(broken,profile="full")
    broken=copy.deepcopy(row)
    broken["risk_receipts"][2]["no_hard_risk_gate"]=False
    with pytest.raises(ValueError,match="hard gate"):
        p.check_trial(broken,profile="full")
    broken=copy.deepcopy(row)
    broken["online_calls"]=5
    with pytest.raises(ValueError,match="fixed-call"):
        p.check_trial(broken,profile="full")
    broken=copy.deepcopy(row)
    broken["first_online_feasible_iteration"]=1
    with pytest.raises(ValueError,match="first feasible"):
        p.check_trial(broken,profile="full")
    broken=copy.deepcopy(row)
    broken["risk_receipts"][0]["realized_nonfeasible"]=0
    with pytest.raises(ValueError,match="nonfeasible"):
        p.check_trial(broken,profile="full")


def test_all_seed_coverage_soft_outside_option_and_winning_is_recorded():
    rows=[]
    for case in p.CASES:
        for i,seed in enumerate(p.SEEDS):
            cold=case.endswith("_sparse") and i%3==0
            for arm in p.ARMS:
                found=arm=="local_value" and i%2==0
                best={
                    "legacy":9.,"semantic_only":8.,"budgeted":7.,
                    "soft_gp":6.,"local_blend":5.,"local_value":4.,
                }[arm]
                rows.append(trial(case,seed,arm,cold=cold,
                                  found=(found if cold else False),best=best))
    by,paired=p.summarize(rows,profile="full")
    assert len(rows)==4*12*6
    assert len(by)==4*6 and len(paired)==4*7
    group=by["branin_sparse/local_value"]
    assert group["risk_gate_fallbacks"]==0
    assert group["soft_outside_option_preferred_but_evaluated"]==12
    assert group["actual_failure_after_soft_outside_option_preferred"]==12
    assert group["local_evidence_mass_mean_on_soft_model_calls"]==2.
    assert group["cold_start_feasible_discovered"]==2
    contrast=paired["branin_sparse/local_value_vs_budgeted"]
    assert contrast["pair_count"]==12
    assert contrast["only_candidate_feasible"]==2
    assert contrast["only_control_feasible"]==0
    assert contrast["primary_all_seed_feasible_first_wins"]>0
    assert contrast["n_real_online_calls_per_arm"]==6


def test_missing_duplicate_and_mismatched_initial_prefix_are_rejected():
    smoke=[trial(case,p.SEEDS[0],arm,n=2)
           for case in p.CASES for arm in p.ARMS]
    with pytest.raises(ValueError,match="Duplicate"):
        p.summarize(smoke+[smoke[0]],profile="smoke")
    with pytest.raises(ValueError,match="roster"):
        p.summarize(smoke[:-1],profile="smoke")
    changed=copy.deepcopy(smoke)
    changed[0]["initial_prefix_sha256"]="b"*64
    with pytest.raises(ValueError,match="initial observations"):
        p.summarize(changed,profile="smoke")
    assert len(p.summarize(smoke,profile="smoke")[0])==24


def test_real_runner_uses_same_old_gp_vae_and_actual_mathematical_evaluator():
    src=(Path(__file__).resolve().parents[1]/"ra_lsbo"/"experiments"/
         "risk_budget63_runner.py").read_text(encoding="utf8")
    assert "from .risk_budget63_protocol import" in src
    assert "from .risk_budget63_local_value_engine import LocalCredibilityValueBO" in src
    assert "from .risk_budget54_runner import _engine as _legacy_engine" in src
    assert "problem.evaluate(candidate)" in src
    assert "patch.object(AircraftCase" in src
    assert "patch.object(SimplifiedElectricAircraft" in src
    assert "risk_budget63_protocol.py" in src
    assert "PR54_POLICY_SHA256" in src and "PR54_ENGINE_SHA256" in src
