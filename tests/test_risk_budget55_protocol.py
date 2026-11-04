"""PR55 pure-Python registration and paired scoring tests; NO model/aircraft imports."""
import copy
import math

import pytest

from ra_lsbo.experiments.risk_budget55_protocol import (
    SCHEMA, CASES, ARMS, SEEDS, INITIAL, ONLINE, OBSERVED_FAILURE_CAP,
    registration, summarize, validate_trial,
)


SHA = "a" * 40


def trial(case, seed, arm, *, cold=False, succeeds=True, steps=12,
          failures=2, first=1, initial_value=20., final_value=15.):
    initial = None if cold else initial_value
    final = final_value if succeeds else initial
    if cold:
        curve = [None] * (steps+1)
        if succeeds:
            for i in range(first+1,steps+1):
                curve[i]=final
        first_feasible = first if succeeds else None
    else:
        curve = [initial] * steps + [final]
        first_feasible = 0 if steps else None
    return {
        "case":case, "seed":seed, "arm":arm,
        "initial_best":initial, "final_best":final,
        "online_calls":steps, "online_nonfeasible":failures,
        "first_online_feasible_iteration":first_feasible,
        "best_feasible_after_each_actual_online_call":curve,
        "selected_path_digest":f"{case}/{seed}/{arm}",
        "pre_evaluation_risk_receipts":[
            {"iteration":i,"prediction_available_for_selected":arm!="legacy",
             "budget_gate_fallback":"predicted_allowance_filter",
             "realized_nonfeasible":int(i<failures),
             "selected_predicted_nonfeasible":0.2 if arm!="legacy" else None,
             "candidate_risk_within_allowance":True if arm!="legacy" else None,
             "probability_guarantee":False}
            for i in range(steps)],
    }


def test_frozen_roster_and_calculations():
    smoke=registration("smoke",SHA)
    full=registration("full",SHA)
    assert full["schema"]==SCHEMA
    assert len(CASES)==4 and len(SEEDS)==16
    assert all(not 995401<=seed<=995408 for seed in SEEDS)
    assert full["seeds"]==list(range(996101,996117))
    assert full["new_analytical_evaluation_cap"]==4*16*(12+3*12)==3072
    assert smoke["new_analytical_evaluation_cap"]==4*(12+3*2)==72
    assert full["online_steps_to_execute"]==ONLINE==12
    assert full["shared_initial"]==INITIAL==12
    assert full["observed_online_nonfeasible_stop"]==OBSERVED_FAILURE_CAP==6
    assert full["arms"]==list(ARMS)
    assert full["analytical_cases"]==list(CASES)
    assert full["selection_rule_unchanged_from_pr54"] is True
    assert full["new_aircraft_evaluations"]==full["new_rcaide_evaluations"]==0
    with pytest.raises(ValueError):
        registration("unregistered",SHA)
    with pytest.raises(ValueError):
        registration("full","not_a_git_sha")


def test_observed_failure_stop_and_no_missing_objective_imputation():
    early=trial("branin_sparse",SEEDS[0],"budgeted",cold=True,
                steps=6,failures=6,succeeds=False)
    validate_trial(early)
    assert early["final_best"] is None
    assert len(early["best_feasible_after_each_actual_online_call"])==7
    invalid=copy.deepcopy(early)
    invalid["online_nonfeasible"]=5
    with pytest.raises(ValueError,match="Trial stopped"):
        validate_trial(invalid)
    invalid=copy.deepcopy(early)
    invalid["online_nonfeasible"]=7
    with pytest.raises(ValueError,match="observed-budget"):
        validate_trial(invalid)
    invalid=copy.deepcopy(early)
    invalid["best_feasible_after_each_actual_online_call"][-1]=3.2
    with pytest.raises(ValueError,match="terminal"):
        validate_trial(invalid)


def test_discovery_timing_and_objective_path_nonregression():
    found=trial("rosenbrock_sparse",SEEDS[0],"semantic_only",
                cold=True,succeeds=True,first=3,steps=12,failures=5)
    assert validate_trial(found)["final_best"]==15.
    wrong=copy.deepcopy(found)
    wrong["first_online_feasible_iteration"]=2
    with pytest.raises(ValueError,match="discovery"):
        validate_trial(wrong)
    wrong=copy.deepcopy(found)
    wrong["best_feasible_after_each_actual_online_call"][-2]=10.
    with pytest.raises(ValueError,match="increased"):
        validate_trial(wrong)
    wrong=copy.deepcopy(found)
    wrong["final_best"]=math.nan
    with pytest.raises(ValueError,match="Nonfinite"):
        validate_trial(wrong)


def test_pairing_counts_missing_feasibility_and_conditional_objective():
    rows=[]
    for case in CASES:
        for seed in SEEDS[:1]:
            cold=case.endswith("_sparse")
            for arm in ARMS:
                succeeds=not cold or arm!="legacy"
                final=15. if arm=="semantic_only" else 13. if arm=="budgeted" else 16.
                rows.append(trial(case,seed,arm,cold=cold,
                                  succeeds=succeeds,first=1,
                                  steps=2,failures=1,final_value=final))
    agg,pairs=summarize(rows,"smoke")
    assert len(agg)==4*3
    assert len(pairs)==4*3
    sparse=pairs["branin_sparse/budgeted_vs_legacy"]
    assert sparse["both_final_feasible"]==0
    assert sparse["candidate_only_feasible"]==1
    assert sparse["mean_objective_delta_if_both_feasible"] is None
    comparison=pairs["branin_ordinary/budgeted_vs_semantic_only"]
    assert comparison["both_final_feasible"]==1
    assert comparison["mean_objective_delta_if_both_feasible"]==-2
    assert comparison["full_selected_paths_differ"]==1
    assert agg["branin_sparse/budgeted"]["initial_feasible"]==0
    assert agg["branin_sparse/budgeted"]["new_feasible_discoveries_in_initially_infeasible_groups"]==1
    assert agg["branin_sparse/legacy"]["final_feasible"]==0
    assert agg["rosenbrock_sparse/legacy"]["initially_infeasible_groups"]==1


def test_duplicate_missing_or_mismatched_initial_trials_refused():
    rows=[trial(case,seed,arm,steps=2,failures=0)
          for case in CASES for seed in SEEDS[:1] for arm in ARMS]
    with pytest.raises(ValueError,match="Duplicate"):
        summarize(rows+[rows[0]],"smoke")
    with pytest.raises(ValueError,match="Missing"):
        summarize(rows[:-1],"smoke")
    different=copy.deepcopy(rows)
    different[0]["initial_best"]=19.
    different[0]["best_feasible_after_each_actual_online_call"][0]=19.
    different[0]["best_feasible_after_each_actual_online_call"][1]=19.
    different[0]["best_feasible_after_each_actual_online_call"][-1]=15.
    with pytest.raises(ValueError,match="same shared initial"):
        summarize(different,"smoke")


def test_full_predeclared_16_seed_4_task_pairing():
    rows=[]
    for case in CASES:
        for i,seed in enumerate(SEEDS):
            cold=case.endswith("_sparse") and i%2==1
            for arm in ARMS:
                failure_cap_hit=(i%4==0)
                calls=6 if failure_cap_hit else ONLINE
                failures=6 if failure_cap_hit else 3
                successful=(not cold or i%3!=0 or arm=="budgeted")
                final=16. if arm=="legacy" else 15. if arm=="semantic_only" else 14.
                rows.append(trial(case,seed,arm,cold=cold,
                                  succeeds=successful,
                                  steps=calls,failures=failures,first=1,
                                  final_value=final))
    agg,pairs=summarize(rows,"full")
    assert len(rows)==4*16*3
    assert len(agg)==4*3
    assert len(pairs)==4*3
    assert pairs["rosenbrock_ordinary/budgeted_vs_semantic_only"]["independent_seed_groups"]==16
    assert agg["branin_ordinary/legacy"]["failure_cap_reached"]==4
    assert agg["branin_ordinary/legacy"]["mean_online_calls"]==10.5
    assert agg["rosenbrock_sparse/budgeted"]["n_seeds"]==16


def test_runner_schema_risk_receipts_are_counted_without_alias_guessing():
    rows=[]
    for case in CASES:
        for seed in SEEDS[:1]:
            for arm in ARMS:
                row=trial(case,seed,arm,steps=2,failures=1)
                if arm=="budgeted":
                    row["pre_evaluation_risk_receipts"][0].update({
                        "prediction_available_for_selected":True,
                        "budget_gate_fallback":"no_candidate_below_allowance_min_predicted_failure",
                        "candidate_risk_within_allowance":False,
                        "selected_predicted_nonfeasible":.8,
                        "realized_nonfeasible":1,
                    })
                    row["pre_evaluation_risk_receipts"][1].update({
                        "prediction_available_for_selected":True,
                        "budget_gate_fallback":"predicted_allowance_filter",
                        "candidate_risk_within_allowance":True,
                        "selected_predicted_nonfeasible":.1,
                        "realized_nonfeasible":0,
                    })
                rows.append(row)
    by_arm,_=summarize(rows,"smoke")
    for case in CASES:
        b=by_arm[f"{case}/budgeted"]
        assert b["predicted_risk_receipt_coverage"]==2
        assert b["predicted_allowance_gate_fallbacks"]==1
        assert b["observed_nonfeasible_in_predicted_gate_fallbacks"]==1
        assert b["predicted_receipt_observed_nonfeasible"]==1
        assert b["predicted_receipt_actual_nonfeasible_rate"]==.5
        assert b["mean_selected_predicted_nonfeasible_proxy"]==pytest.approx(.45)
        assert b["budget_gate_passed_count"]==1
        assert b["budget_gate_failed_count"]==1
