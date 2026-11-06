"""PR64 pure protocol tests; no GP/VAE fitting or objective calls."""
import copy
from pathlib import Path

import pytest
from ra_lsbo.experiments import risk_budget64_protocol as p


SOFT=("soft_gp_fixed128","soft_gp_fixed512","soft_gp_abstain512")


def _id(text):
    import hashlib
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def trial(case,seed,arm,*,n=2,stop=False,feasible_at=None,failures=None):
    maximum=2
    if failures is None:
        failures=[1]*n
    initial=None
    curve=[initial]
    first=None
    decisions=[]
    for i in range(n):
        y=int(failures[i])
        if y==0 and first is None:first=i
        if y==0 and curve[-1] is None:curve.append(5.)
        else:curve.append(curve[-1])
        soft=arm in SOFT
        forecast=soft or arm in ("semantic_only","budgeted")
        cid=_id(f"{case}/{seed}/{arm}/{i}")
        if arm in ("soft_gp_fixed512","soft_gp_abstain512"):
            cid=_id(f"{case}/{seed}/shared512/{i}")
        decisions.append({
            "proposal_step":i,"candidate_sha256":cid,"candidate_pool_size":512 if arm.endswith("512") else 128,
            "mode":"feasible_discovery","evaluated":True,"actual_call_index":i,
            "actual_action":"evaluate_under_maximum_budget",
            "forecast_available_for_selected":forecast,
            "predicted_nonfeasible":.8 if forecast else None,
            "selected_net_utility":.2 if soft else None,
            "would_defer_in_unconstrained_utility_model":False if soft else None,
            "no_hard_risk_gate":True if soft else None,
            "candidate_filter_applied":False if soft else None,
            "observed_online_nonfeasible_before":sum(failures[:i]),
            "realized_nonfeasible":y,
            "prediction_is_not_a_guarantee":True,
            "outcome_not_available_when_selected":True,
        })
    if stop:
        assert arm=="soft_gp_abstain512"
        i=n
        cid=_id(f"{case}/{seed}/shared512/{i}")
        decisions.append({
            "proposal_step":i,"candidate_sha256":cid,"candidate_pool_size":512,
            "mode":"feasible_discovery","evaluated":False,"actual_call_index":n,
            "actual_action":"abstain_before_evaluation",
            "forecast_available_for_selected":True,
            "predicted_nonfeasible":.95,"selected_net_utility":-.2,
            "would_defer_in_unconstrained_utility_model":True,
            "no_hard_risk_gate":True,"candidate_filter_applied":False,
            "observed_online_nonfeasible_before":sum(failures),
            "realized_nonfeasible":None,"prediction_is_not_a_guarantee":True,
            "outcome_not_available_when_selected":True,
        })
    return {
        "case":case,"seed":seed,"arm":arm,"online_calls":n,
        "maximum_online_calls":maximum,"observed_nonfeasible":sum(failures),
        "initial_best":initial,"final_best":curve[-1],
        "initial_prefix_sha256":"a"*64,"best_feasible_at_each_actual_call":curve,
        "first_online_feasible_iteration":first,"decision_receipts":decisions,
        "evaluated_path_sha256":_id(f"evaluated/{case}/{seed}/{arm}"),
        "decision_path_sha256":_id(f"decision/{case}/{seed}/{arm}"),
        "six_call_maximum_horizon":6,"forecast_internal_horizon":12,
        "stop_reason":"outside_option_nonpositive" if stop else "maximum_online_call_budget",
        "evidence_before_result_was_future_blind":True,
    }


def test_registered_seed_block_and_maximum_real_call_ceiling():
    full=p.plan("full","d"*40)
    smoke=p.plan("smoke","d"*40)
    assert len(p.ARMS)==6 and len(p.SEEDS)==24 and len(p.CASES)==4
    assert p.SEEDS==tuple(range(997301,997325))
    assert full["new_analytic_evaluation_ceiling"]==4*24*(12+6*6)==4608
    assert smoke["new_analytic_evaluation_ceiling"]==4*(12+6*2)==96
    assert full["maximum_online_calls_per_arm"]==6
    assert full["actual_calls_may_differ_only_for_soft_gp_abstain512"] is True
    assert full["abstention_is_real_no_evaluator_call"] is True
    assert full["no_probability_safety_guarantee"] is True
    assert full["primary_contrast"]=="soft_gp_abstain512_vs_soft_gp_fixed512 per task, 24 matched seeds"
    assert full["fresh_nonoverlap_with_pr54_pr55_pr58_pr60_pr63"] is True
    with pytest.raises(ValueError):p.plan("invalid","d"*40)


def test_real_abstention_has_no_outcome_and_does_not_charge_an_evaluation():
    row=trial("branin_sparse",p.SEEDS[0],"soft_gp_abstain512",
              n=1,stop=True,failures=[1])
    assert p.check_trial(row,profile="smoke") is row
    broken=copy.deepcopy(row)
    broken["decision_receipts"][-1]["realized_nonfeasible"]=1
    with pytest.raises(ValueError,match="UNKNOWN outcome"):
        p.check_trial(broken,profile="smoke")
    broken=copy.deepcopy(row)
    broken["decision_receipts"][-1]["actual_action"]="evaluate_under_maximum_budget"
    with pytest.raises(ValueError,match="abstention"):
        p.check_trial(broken,profile="smoke")
    broken=copy.deepcopy(row)
    broken["online_calls"]=2
    with pytest.raises(ValueError,match="invalid abstention"):
        p.check_trial(broken,profile="smoke")


def test_fixed_soft_arms_must_spend_the_maximum_budget_even_for_negative_utility():
    row=trial("branin_sparse",p.SEEDS[0],"soft_gp_fixed512",
              n=2,failures=[1,1])
    row["decision_receipts"][0]["selected_net_utility"]=-.2
    row["decision_receipts"][0]["would_defer_in_unconstrained_utility_model"]=True
    assert p.check_trial(row,profile="smoke") is row
    broken=copy.deepcopy(row)
    broken["online_calls"]=1
    broken["decision_receipts"]=broken["decision_receipts"][:1]
    broken["best_feasible_at_each_actual_call"]=[None,None]
    broken["final_best"]=None
    broken["observed_nonfeasible"]=1
    with pytest.raises(ValueError,match="maximum budget"):
        p.check_trial(broken,profile="smoke")


def test_summary_matches_abstained_candidate_to_fixed512_shadow_only_after_run():
    rows=[]
    for case in p.CASES:
        for arm in p.ARMS:
            if arm=="soft_gp_abstain512":
                rows.append(trial(case,p.SEEDS[0],arm,n=1,stop=True,failures=[1]))
            elif arm=="soft_gp_fixed512":
                rows.append(trial(case,p.SEEDS[0],arm,n=2,failures=[1,1]))
            else:
                rows.append(trial(case,p.SEEDS[0],arm,n=2,failures=[1,0]))
    by,paired,shadow=p.summarize(rows,profile="smoke")
    assert len(by)==24 and len(paired)==4*len(p.PAIRS)
    assert shadow["n_abstentions"]==4
    assert shadow["same_candidate_shadow_matches"]==4
    assert shadow["shadow_failures"]==4
    assert shadow["post_run_counterfactual_only"] is True
    assert all(x["shadow_outcome_is_not_available_to_abstention_algorithm"]
               for x in shadow["rows"])
    group=by["branin_sparse/soft_gp_abstain512"]
    assert group["actual_online_calls_total"]==1
    assert group["abstentions"]==1
    assert group["actual_nonfeasible_total"]==1
    assert paired["branin_sparse/soft_gp_abstain512_vs_soft_gp_fixed512"][
        "same_maximum_call_budget"] is True


def test_missing_duplicate_and_unmatched_initial_evidence_are_rejected():
    rows=[trial(c,p.SEEDS[0],a,n=2,failures=[1,0])
          for c in p.CASES for a in p.ARMS]
    with pytest.raises(ValueError,match="Duplicate"):
        p.summarize(rows+[rows[0]],profile="smoke")
    with pytest.raises(ValueError,match="roster"):
        p.summarize(rows[:-1],profile="smoke")
    changed=copy.deepcopy(rows)
    changed[0]["initial_prefix_sha256"]="b"*64
    with pytest.raises(ValueError,match="same initial"):
        p.summarize(changed,profile="smoke")


def test_runner_performs_abstention_before_any_real_problem_evaluate():
    src=(Path(__file__).resolve().parents[1]/"ra_lsbo"/"experiments"/
         "risk_budget64_runner.py").read_text(encoding="utf8")
    assert "LocalCredibilityValueBO" in src
    assert "POOL_WIDE+ENGINE_PLANNING_HORIZON,prior" in src
    assert 'arm=="soft_gp_abstain512"' in src
    assert '"abstain_before_evaluation"' in src
    assert src.index("if abstain:") < src.index("rec=problem.evaluate(candidate)")
    assert "patch.object(AircraftCase" in src
    assert "patch.object(SimplifiedElectricAircraft" in src
    assert "frozen PR63 GP-only soft utility source" in src
