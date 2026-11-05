"""PR58 pure protocol tests: actual GP/VAE/analytical runs are workstation-only."""
import copy
from pathlib import Path

import pytest

from ra_lsbo.experiments import risk_budget58_protocol as pr


def mock_trial(case, seed, arm, *, n=6, initial=10., final=8.,
               fail=2, discovery=None, fallback=False):
    if initial is None:
        curve=[None] * (n+1)
        if discovery is not None:
            for index in range(discovery+1,n+1):
                curve[index]=final
    else:
        curve=[initial] * n + [final]
    receipts=[]
    for i in range(n):
        forecast=arm!="legacy"
        is_fallback=forecast and arm=="budgeted" and fallback and i==1
        receipts.append({
            "iteration":i,
            "forecast_available_for_selected":forecast,
            "predicted_nonfeasible":.8 if is_fallback else .25 if forecast else None,
            "budget_rule":"no_candidate_below_allowance_min_predicted_failure"
                if is_fallback else "predicted_allowance_filter" if arm=="budgeted"
                else "constraint_only_ranking" if forecast else None,
            "realized_nonfeasible":int(i<fail),
        })
    return {
        "case":case,"seed":seed,"arm":arm,
        "initial_best":initial,
        "final_best":curve[-1],
        "online_calls":n,
        "observed_nonfeasible":fail,
        "first_online_feasible_iteration":discovery if initial is None else
            (fail if fail<n else None),
        "initial_prefix_sha256":"a"*64,
        "best_feasible_at_each_actual_call":curve,
        "risk_receipts":receipts,
        "selected_path_sha256":(arm[0]+str(seed)).ljust(64,"f"),
        "six_call_horizon":6,
        "forecast_internal_horizon":12,
        "evidence_before_result_was_future_blind":True,
    }


def test_registration_prespecified_fresh_seeds_unchanged_horizons():
    full=pr.plan("full","0"*40)
    smoke=pr.plan("smoke","0"*40)
    assert full["schema"]==pr.SCHEMA
    assert len(pr.CASES)==4 and len(pr.SEEDS)==24
    assert full["seeds"]==list(range(997001,997025))
    assert full["fresh_nonoverlap_with_pr54_pr55"] is True
    assert full["new_analytic_evaluation_ceiling"]==4*24*(12+3*6)==2880
    assert smoke["new_analytic_evaluation_ceiling"]==4*(12+3*2)==72
    assert full["actual_online_calls_per_arm"]==6
    assert full["engine_internal_horizon_unmodified"]==12
    assert full["observed_failure_cap"]==6
    assert full["predeclared_primary_metric"].startswith("all-seed")
    assert full["immutable_policy_sha256"].startswith("d2c32b77")
    with pytest.raises(ValueError):
        pr.plan("unknown","0"*40)


def test_actual_six_charged_calls_and_feasible_discovery_are_required():
    v=mock_trial("branin_sparse",pr.SEEDS[0],"budgeted",initial=None,
                 final=2.,discovery=2,fail=2)
    assert pr.check_trial(v,profile="full")==v
    wrong=copy.deepcopy(v)
    wrong["first_online_feasible_iteration"]=2
    with pytest.raises(ValueError,match="first feasible"):
        pr.check_trial(wrong,profile="full")
    wrong=copy.deepcopy(v)
    wrong["online_calls"]=5
    with pytest.raises(ValueError,match="fixed-call"):
        pr.check_trial(wrong,profile="full")
    wrong=copy.deepcopy(v)
    wrong["risk_receipts"][0]["realized_nonfeasible"]=0
    with pytest.raises(ValueError,match="nonfeasible count"):
        pr.check_trial(wrong,profile="full")
    wrong=copy.deepcopy(v)
    wrong["best_feasible_at_each_actual_call"][4]=5.
    with pytest.raises(ValueError,match="increased"):
        pr.check_trial(wrong,profile="full")


def test_all_seed_feasible_first_rank_never_fills_unobserved_objective():
    assert pr._compare({"final_best":None},{"final_best":None})==0
    assert pr._compare({"final_best":2.},{"final_best":None})==1
    assert pr._compare({"final_best":None},{"final_best":2.})==-1
    assert pr._compare({"final_best":1.},{"final_best":2.})==1
    assert pr._compare({"final_best":3.},{"final_best":2.})==-1
    assert pr._compare({"final_best":2.},{"final_best":2.})==0


def test_full_24_seed_4_case_matched_results_and_fallback_coverage():
    rows=[]
    for case in pr.CASES:
        for i,seed in enumerate(pr.SEEDS):
            cold=case.endswith("sparse") and i%2==0
            for arm in pr.ARMS:
                if cold:
                    # All three share initially infeasible, but only budget
                    # on one subgroup discovers a feasible design.
                    found=(i%4==0 and arm=="budgeted")
                    initial=None
                    final=2. if found else None
                    discover=2 if found else None
                else:
                    initial=10.
                    final={"legacy":9.,"semantic_only":7.,
                           "budgeted":6.}[arm]
                    discover=None
                rows.append(mock_trial(case,seed,arm,initial=initial,
                                      final=final,discovery=discover,
                                      fail=2 if not cold or found else 6,
                                      fallback=arm=="budgeted" and cold))
    group,pairs=pr.summarize(rows,profile="full")
    assert len(rows)==4*24*3
    assert len(group)==4*3
    assert len(pairs)==4*3
    fb=group["branin_sparse/budgeted"]
    assert fb["risk_forecast_receipts"]==24*6
    assert fb["risk_gate_fallbacks"]==12
    assert fb["fallback_observed_nonfeasible"]==12
    assert group["branin_sparse/budgeted"]["cold_start_feasible_discovered"]==6
    assert group["branin_sparse/semantic_only"]["cold_start_feasible_discovered"]==0
    contrast=pairs["branin_sparse/budgeted_vs_semantic_only"]
    assert contrast["pair_count"]==24
    assert contrast["primary_all_seed_feasible_first_wins"]==18
    assert contrast["primary_all_seed_feasible_first_losses"]==0
    assert contrast["primary_all_seed_feasible_first_ties"]==6
    assert contrast["only_candidate_feasible"]==6
    assert contrast["both_infeasible"]==6
    assert contrast["same_real_online_call_count"] is True
    assert contrast["n_real_online_calls_per_arm"]==6
    assert contrast["mean_observed_nonfeasible_delta"]==0


def test_missing_duplicate_or_different_initial_evidence_rejected():
    rows=[mock_trial(case,pr.SEEDS[0],arm)
          for case in pr.CASES for arm in pr.ARMS]
    with pytest.raises(ValueError,match="Duplicate"):
        pr.summarize(rows+[rows[0]],profile="smoke")
    with pytest.raises(ValueError,match="roster"):
        pr.summarize(rows[:-1],profile="smoke")
    altered=copy.deepcopy(rows)
    altered[0]["initial_prefix_sha256"]="b"*64
    with pytest.raises(ValueError,match="initial observations"):
        pr.summarize(altered,profile="smoke")


def test_smoke_observed_two_calls_and_no_synthetic_extra_truth():
    rows=[mock_trial(case,pr.SEEDS[0],arm,n=2,fail=1)
          for case in pr.CASES for arm in pr.ARMS]
    agg,pairs=pr.summarize(rows,profile="smoke")
    assert len(agg)==12 and len(pairs)==12
    assert pairs["rosenbrock_ordinary/budgeted_vs_semantic_only"]["n_real_online_calls_per_arm"]==2
    with pytest.raises(ValueError):
        pr.check_trial(rows[0],profile="full")


def test_runner_reuses_pinned_original_policy_engine_and_forbids_aircraft():
    # AST/string audit ensures PR58 test CI does not import the heavy runner.
    source=(Path(__file__).resolve().parents[1]/"ra_lsbo"/"experiments"/
            "risk_budget58_runner.py").read_text(encoding="utf8")
    assert "from .risk_budget54_runner import _engine" in source
    assert "PR54_POLICY_SHA256" in source
    assert "PR54_ENGINE_SHA256" in source
    assert "patch.object(AircraftCase" in source
    assert "patch.object(SimplifiedElectricAircraft" in source
    assert "problem.evaluate(candidate)" in source
