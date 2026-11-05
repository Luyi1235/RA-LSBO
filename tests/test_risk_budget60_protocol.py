"""PR60 frozen 6-arm/4-task/12-seed same-call protocol (pure Python)."""
import copy
from pathlib import Path

import pytest
from ra_lsbo.experiments import risk_budget60_protocol as p


def trial(case,seed,arm,*,n=6,cold=False,found=False,
          best=8.,model=True,fallback=False):
    if cold:
        initial=None
        first=2 if found else None
        failures=set(range(n)) if not found else set(range(n))-{first}
        curve=[None]*(n+1)
        if found:
            for i in range(first+1,n+1):curve[i]=best
    else:
        initial=10.
        first=1 if n>1 else 0
        failures={0} if first>0 else set()
        curve=[initial]*(n+1)
        for i in range(first+1,n+1):curve[i]=min(initial,best)
    receipts=[]
    for i in range(n):
        available=model and arm!="legacy"
        branch=arm.startswith("branch_")
        rule=("no_candidate_below_allowance_max_branch_information"
              if fallback and i==1 and arm=="branch_explore" else
              "no_candidate_below_allowance_min_corrected_proxy"
              if fallback and i==1 and branch else
              "predicted_allowance_filter" if available else None)
        receipts.append({
            "iteration":i, "realized_nonfeasible":int(i in failures),
            "forecast_available_for_selected":available,
            "predicted_nonfeasible":.8 if fallback and i==1 else
                                    .25 if available else None,
            "budget_rule":rule,
            "selected_branch_posterior":{
                "n_settled_in_branch":12,
                "posterior_failure_mean":.4,
                "posterior_failure_sd":.1,
            } if branch and available else None,
            "original_gp_predicted_failure":.1 if branch and available else None,
        })
    return {
        "case":case,"seed":seed,"arm":arm,
        "initial_best":initial,"final_best":curve[-1],
        "online_calls":n,"observed_nonfeasible":len(failures),
        "first_online_feasible_iteration":first,
        "best_feasible_at_each_actual_call":curve,
        "risk_receipts":receipts,
        "initial_prefix_sha256":"a"*64,
        "selected_path_sha256":(str(seed)+arm).ljust(64,"f"),
        "six_call_horizon":6,"forecast_internal_horizon":12,
        "evidence_before_result_was_future_blind":True,
    }


def test_frozen_six_arm_seed_registration_and_actual_evaluator_budget():
    full=p.plan("full","d"*40)
    smoke=p.plan("smoke","d"*40)
    assert full["schema"]==p.SCHEMA
    assert len(p.ARMS)==6 and len(p.SEEDS)==12 and len(p.CASES)==4
    assert p.SEEDS==(997101,997102,997103,997104,997105,997106,
                     997107,997108,997109,997110,997111,997112)
    assert full["fresh_nonoverlap_with_pr54_pr55_pr58"] is True
    assert full["new_analytic_evaluation_ceiling"]==4*12*(12+6*6)==2304
    assert smoke["new_analytic_evaluation_ceiling"]==4*(12+6*2)==96
    assert full["engine_internal_horizon_unmodified"]==12
    assert full["actual_online_calls_per_arm"]==6
    assert len(p.PAIRS)==7
    assert full["branch_beta_prior"]==[1.,1.]
    assert full["branch_evidence_only_from_this_arm_settled_prefix"] is True
    assert full["no_probability_safety_guarantee"] is True
    with pytest.raises(ValueError):
        p.plan("bad","d"*40)


def test_six_charged_calls_source_evidence_and_branch_receipts():
    t=trial("branin_sparse",p.SEEDS[0],"branch_dispersion",
            cold=True,found=True)
    assert p.check_trial(t,profile="full") is t
    invalid=copy.deepcopy(t)
    invalid["first_online_feasible_iteration"]=1
    with pytest.raises(ValueError,match="first feasible"):
        p.check_trial(invalid,profile="full")
    invalid=copy.deepcopy(t)
    invalid["risk_receipts"][0]["original_gp_predicted_failure"]=None
    with pytest.raises(ValueError,match="branch posterior"):
        p.check_trial(invalid,profile="full")
    invalid=copy.deepcopy(t)
    invalid["online_calls"]=5
    with pytest.raises(ValueError,match="fixed-call"):
        p.check_trial(invalid,profile="full")
    invalid=copy.deepcopy(t)
    invalid["risk_receipts"][0]["realized_nonfeasible"]=0
    with pytest.raises(ValueError,match="nonfeasible"):
        p.check_trial(invalid,profile="full")
    invalid=copy.deepcopy(t)
    invalid["best_feasible_at_each_actual_call"][-1]=float("inf")
    with pytest.raises(ValueError,match="finite"):
        p.check_trial(invalid,profile="full")


def test_all_seeds_included_and_explore_fallback_audited():
    rows=[]
    for case in p.CASES:
        for i,seed in enumerate(p.SEEDS):
            cold=case.endswith("_sparse") and i%3==0
            for arm in p.ARMS:
                found=(arm=="branch_explore" and i%2==0)
                best={"legacy":9.,"semantic_only":8.,"budgeted":7.,
                      "branch_mean":6.,"branch_dispersion":5.,
                      "branch_explore":4.}[arm]
                rows.append(trial(case,seed,arm,cold=cold,
                                  found=(found if cold else False),
                                  best=best,fallback=arm=="branch_explore"))
    by_arm,paired=p.summarize(rows,profile="full")
    assert len(rows)==4*12*6
    assert len(by_arm)==4*6
    assert len(paired)==4*7
    assert by_arm["branin_sparse/branch_explore"]["risk_gate_fallbacks"]==12
    assert by_arm["branin_sparse/branch_dispersion"]["risk_gate_fallbacks"]==0
    assert by_arm["branin_sparse/branch_explore"]["cold_start_feasible_discovered"]==2
    assert by_arm["branin_sparse/branch_explore"]["risk_forecast_receipts"]==12*6
    x=paired["branin_sparse/branch_explore_vs_budgeted"]
    assert x["pair_count"]==12
    assert x["only_candidate_feasible"]==2
    assert x["only_control_feasible"]==0
    assert x["same_real_online_call_count"] is True
    assert x["n_real_online_calls_per_arm"]==6
    assert x["primary_all_seed_feasible_first_wins"]>0


def test_duplicate_missing_or_unmatched_shared_initial_is_rejected():
    rows=[trial(case,p.SEEDS[0],arm,n=2)
          for case in p.CASES for arm in p.ARMS]
    with pytest.raises(ValueError,match="Duplicate"):
        p.summarize(rows+[rows[0]],profile="smoke")
    with pytest.raises(ValueError,match="roster"):
        p.summarize(rows[:-1],profile="smoke")
    broken=copy.deepcopy(rows)
    broken[0]["initial_prefix_sha256"]="b"*64
    with pytest.raises(ValueError,match="initial observations"):
        p.summarize(broken,profile="smoke")
    assert len(p.summarize(rows,profile="smoke")[0])==24


def test_actual_runner_uses_registered_optin_engine_and_zero_aircraft():
    code=(Path(__file__).resolve().parents[1]/"ra_lsbo"/"experiments"/
          "risk_budget60_runner.py").read_text(encoding="utf8")
    assert "from .risk_budget60_protocol import" in code
    assert "from .risk_budget60_branch_uq_engine import BranchUQDualViewMathBO" in code
    assert "from .risk_budget54_runner import _engine as _legacy_engine" in code
    assert "problem.evaluate(candidate)" in code
    assert "patch.object(AircraftCase" in code
    assert "patch.object(SimplifiedElectricAircraft" in code
    assert "PR54_POLICY_SHA256" in code
    assert "PR54_ENGINE_SHA256" in code
