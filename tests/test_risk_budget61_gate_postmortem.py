"""PR61 read-only scientific audit regression: synthetic 288-trial source."""
import copy
import json
import math

import pytest
from ra_lsbo.experiments import risk_budget61_gate_postmortem as p


# Construct a COMPLETE synthetic source with the exact forecast/fallback/floor
# anchor totals, not a partial metadata stub. No user data or GP fit required.
COUNTS={
    "legacy":(0,0,0,0),
    "semantic_only":(266,0,0,0),
    "budgeted":(266,22,19,0),
    "branch_mean":(265,109,92,213),
    "branch_dispersion":(264,175,131,231),
    "branch_explore":(264,231,213,214),
}


def fixture_report():
    trials=[]
    for arm in p.ARMS:
        scored,fb_total,fb_failed,floor_count=COUNTS[arm]
        forecast_index=fallback_index=floor_index=0
        for case in p.CASES:
            for seed in p.SEEDS:
                receipts=[]
                spent=0
                for i in range(6):
                    have=forecast_index < scored
                    fallback=have and fallback_index < fb_total
                    floor=arm in p.BRANCH_ARMS and have and floor_index < floor_count
                    failure=int(fallback and fallback_index < fb_failed)
                    if arm in p.BRANCH_ARMS and have:
                        a,b=1.,1.
                        m=a/(a+b)
                        sd=math.sqrt(a*b/((a+b)**2*(a+b+1)))
                        original_gp=.1 if floor else .9
                        threshold=m if arm=="branch_mean" else m+sd
                        risk=max(original_gp,threshold)
                        branch={
                            "branch":"x"*64,"n_settled_in_branch":0,
                            "observed_failures":0,"observed_successes":0,
                            "posterior_failure_mean":m,"posterior_failure_sd":sd,
                            "posterior_failure_mean_plus_sd":m+sd,
                        }
                        rule=("no_candidate_below_allowance_max_branch_information"
                              if arm=="branch_explore" and fallback else
                              "no_candidate_below_allowance_min_corrected_proxy"
                              if fallback else
                              "empirical_branch_risk_under_predicted_allowance")
                    elif arm=="budgeted" and have:
                        original_gp=branch=None
                        risk=.75 if fallback else .2
                        rule=("no_candidate_below_allowance_min_predicted_failure"
                              if fallback else "predicted_allowance_filter")
                    elif arm=="semantic_only" and have:
                        original_gp=branch=None
                        risk=.3
                        rule="constraint_only_ranking"
                    else:
                        original_gp=branch=risk=rule=None
                    receipts.append({
                        "iteration":i,
                        "mode":"probability_weighted_feasible_improvement",
                        "outcome_not_available_when_selected":True,
                        "forecast_available_for_selected":have,
                        "realized_nonfeasible":failure,
                        "observed_online_nonfeasible_before":spent,
                        "predicted_nonfeasible":risk,
                        "budget_rule":rule,
                        "predicted_gate_passed":not fallback if have else None,
                        "original_gp_predicted_failure":original_gp,
                        "selected_branch_posterior":branch,
                    })
                    spent+=failure
                    if have:forecast_index+=1
                    if fallback:fallback_index+=1
                    if arm in p.BRANCH_ARMS and have:floor_index+=1
                trials.append({
                    "case":case,"seed":seed,"arm":arm,
                    "online_calls":6,"observed_nonfeasible":spent,
                    "initial_prefix_sha256":("i"+str(seed)).ljust(64,"0"),
                    "risk_receipts":receipts,
                })
        assert forecast_index==scored
        assert fallback_index==fb_total
        if arm in p.BRANCH_ARMS:assert floor_index==scored
    return {
        "schema":p.SOURCE_SCHEMA,
        "status":"COMPLETE_FRESH_FIXED_SIX_ON_POLICY",
        "code_sha":p.ORIGINAL_EXPERIMENT_SOURCE,
        "n_real_analytical_calls":2304,
        "n_new_on_policy_trials":288,
        "new_aircraft_calls":0,"new_rcaide_calls":0,
        "plan":{
            "cases":list(p.CASES),"arms":list(p.ARMS),
            "seeds":list(p.SEEDS),"source_sha":p.ORIGINAL_EXPERIMENT_SOURCE,
            "profile":"full","initial":12,
            "actual_online_calls_per_arm":6,
            "engine_internal_horizon_unmodified":12,
        },
        "trial_details":trials,
    }


def prepare(tmp_path,monkeypatch):
    source=tmp_path/"frozen_inputs"
    source.mkdir()
    file=source/"original.json"
    file.write_text(json.dumps(fixture_report()),encoding="utf8")
    monkeypatch.setattr(p,"ORIGINAL_REPORT_GIT_BLOB",p.git_blob_sha1(file))
    monkeypatch.setattr(p,"ORIGINAL_REPORT_SHA256",p.sha256(file))
    return file


def test_full_frozen_model_coverage_and_beta_floor_consistency(tmp_path,monkeypatch):
    file=prepare(tmp_path,monkeypatch)
    rows,identity=p.parse_original(file)
    assert len(rows)==1728
    assert identity["trial_count"]==288
    summary=p.aggregate(rows)
    assert summary["branch_mean/all"]["forecasted_calls"]==265
    assert summary["branch_mean/all"]["fallback_calls"]==109
    assert summary["branch_mean/all"]["fallback_actual_nonfeasible"]==92
    assert summary["branch_mean/all"]["n_branch_posterior_floor_bind"]==213
    assert summary["branch_dispersion/all"]["forecasted_calls"]==264
    assert summary["branch_dispersion/all"]["fallback_calls"]==175
    assert summary["branch_dispersion/all"]["n_branch_posterior_floor_bind"]==231
    assert summary["branch_explore/all"]["fallback_calls"]==231
    assert summary["branch_explore/all"]["fallback_actual_nonfeasible"]==213
    assert summary["branch_explore/all"]["n_branch_posterior_floor_bind"]==214
    assert summary["budgeted/all"]["fallback_calls"]==22
    assert summary["legacy/all"]["forecasted_calls"]==0
    assert summary["branch_mean/all"]["mean_branch_history_n"]==0
    assert summary["branch_mean/all"]["selected_branch_n_le_3"]==265
    assert all(r["risk_is_heuristic_not_a_safety_bound"] for r in rows)


def test_end_to_end_write_once_and_unchanged_source_bytes(tmp_path,monkeypatch):
    file=prepare(tmp_path,monkeypatch)
    original=file.read_bytes()
    out=tmp_path/"new_results"
    report=p.run(file,out)
    assert report["status"]=="COMPLETE_RETROSPECTIVE_BRANCH_BETA_GATE_COLLAPSE_AUDIT"
    assert report["n_original_online_receipts"]==1728
    assert (out/"branch_floor_gate_audit.json").is_file()
    assert (out/"branch_floor_selected_receipts.csv").is_file()
    assert report["n_new_analytical_calls"]==report["n_new_gp_vae_fits"]==0
    assert file.read_bytes()==original
    with pytest.raises(ValueError,match="fresh"):
        p.run(file,out)


def test_tampered_exact_source_rejected_before_any_output(tmp_path,monkeypatch):
    file=prepare(tmp_path,monkeypatch)
    with file.open("a",encoding="utf8") as f:f.write(" ")
    with pytest.raises(ValueError,match="Git blob SHA1 changed"):
        p.run(file,tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_rejects_future_beta_count_and_incorrect_posterior(tmp_path,monkeypatch):
    file=prepare(tmp_path,monkeypatch)
    raw=json.loads(file.read_text(encoding="utf8"))
    t=next(t for t in raw["trial_details"] if t["arm"]=="branch_mean")
    t["risk_receipts"][0]["selected_branch_posterior"]["n_settled_in_branch"]=19
    file.write_text(json.dumps(raw),encoding="utf8")
    monkeypatch.setattr(p,"ORIGINAL_REPORT_GIT_BLOB",p.git_blob_sha1(file))
    monkeypatch.setattr(p,"ORIGINAL_REPORT_SHA256",p.sha256(file))
    with pytest.raises(ValueError,match="Branch history counts"):
        p.parse_original(file)


def test_rejects_fabricated_coverage_forecast_and_ledger_drift(tmp_path,monkeypatch):
    file=prepare(tmp_path,monkeypatch)
    original=json.loads(file.read_text(encoding="utf8"))
    broken=copy.deepcopy(original)
    t=next(t for t in broken["trial_details"] if t["arm"]=="legacy")
    t["risk_receipts"][0]["predicted_nonfeasible"]=.6
    file.write_text(json.dumps(broken),encoding="utf8")
    monkeypatch.setattr(p,"ORIGINAL_REPORT_GIT_BLOB",p.git_blob_sha1(file))
    monkeypatch.setattr(p,"ORIGINAL_REPORT_SHA256",p.sha256(file))
    with pytest.raises(ValueError,match="invented GP probability"):
        p.parse_original(file)
    broken=copy.deepcopy(original)
    t=next(t for t in broken["trial_details"] if t["arm"]=="branch_explore")
    t["risk_receipts"][1]["observed_online_nonfeasible_before"]=999
    file.write_text(json.dumps(broken),encoding="utf8")
    monkeypatch.setattr(p,"ORIGINAL_REPORT_GIT_BLOB",p.git_blob_sha1(file))
    monkeypatch.setattr(p,"ORIGINAL_REPORT_SHA256",p.sha256(file))
    with pytest.raises(ValueError,match="time order"):
        p.parse_original(file)
