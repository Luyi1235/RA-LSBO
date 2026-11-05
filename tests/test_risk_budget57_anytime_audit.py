"""PR57 fixed-call/censoring tests: pure Python, no physics, GP or VAE."""
import copy
import json
from pathlib import Path

import pytest

from ra_lsbo.experiments import risk_budget57_anytime_audit as a


def record(case="branin_sparse", seed=995401, arm="budgeted", *,
           n=12, initial=10., final=8., fail=2, six=None):
    if n < 12:
        assert fail == 6
        final = initial
    curve = [initial] * (n + 1)
    if six is not None:
        for i in range(6, n+1):
            curve[i] = six
    if n == 12:
        curve[-1] = final
    if n < 12:
        curve[-1] = initial
    assert curve[-1] == (final if n==12 else initial)
    return {
        "case":case, "seed":seed, "arm":arm,
        "online_calls":n, "online_nonfeasible":fail,
        "initial_best":initial, "final_best":curve[-1],
        "first_online_feasible_iteration":0,
        "best_feasible_after_each_actual_online_call":curve,
        "pre_evaluation_risk_receipts":[
            {"iteration":i,"realized_nonfeasible":int(i<fail)}
            for i in range(n)],
    }


def test_strict_online_receipts_stop_and_nonincreasing_objective():
    original=record()
    assert a._validate_trial(original) is original
    broken=copy.deepcopy(original)
    broken["online_nonfeasible"]=3
    with pytest.raises(ValueError, match="Realized failures disagree"):
        a._validate_trial(broken)
    broken=copy.deepcopy(original)
    broken["best_feasible_after_each_actual_online_call"][3]=11.
    with pytest.raises(ValueError, match="increased"):
        a._validate_trial(broken)
    broken=copy.deepcopy(original)
    broken["pre_evaluation_risk_receipts"][0]["iteration"]=4
    with pytest.raises(ValueError, match="Misordered"):
        a._validate_trial(broken)
    stopped=record(n=6,fail=6)
    assert a._validate_trial(stopped) is stopped
    stopped["online_nonfeasible"]=5
    with pytest.raises(ValueError,match="Realized failures disagree"):
        a._validate_trial(stopped)


def test_all_seeds_are_eligible_for_six_but_later_horizons_are_censored():
    cases="branin_sparse"
    seeds=[995401,995402]
    found={}
    for seed in seeds:
        stopped=seed==995401
        for arm in a.ARMS:
            n=6 if stopped and arm=="semantic_only" else 12
            f=6 if n==6 else 2
            found[(cases,seed,arm)]=record(cases,seed,arm,n=n,fail=f,
                       final=8. if arm=="budgeted" else 9.)
    six=a.paired_at_horizon(found,"budgeted","semantic_only",cases,seeds,horizon=6)
    assert six["both_arms_observed_through_horizon"]==2
    assert six["both_feasible"]==2
    twelve=a.paired_at_horizon(found,"budgeted","semantic_only",cases,seeds,horizon=12)
    assert twelve["both_arms_observed_through_horizon"]==1
    assert twelve["pairs_not_observed_at_this_horizon_due_to_stop"]==1
    assert twelve["coverage_selection_bias_for_k_gt_6"] is True
    rows=a._per_seed("pr54",found,cases,seeds,"budgeted","semantic_only")
    pair=[x for x in rows if x["seed"]==995401][0]
    assert pair["common_online_calls"]==6
    assert pair["candidate_extra_online_calls_beyond_common"]==6
    assert pair["control_extra_online_calls_beyond_common"]==0
    assert pair["delta_at_six_if_both_feasible"]==0
    assert pair["delta_final_if_both_feasible"]==-2.


def test_algebraic_continuation_identity_is_not_causal_attribution():
    seed=995401
    case="branin_sparse"
    budget=record(case,seed,"budgeted",six=8.,final=6.)
    control=record(case,seed,"semantic_only",n=6,fail=6,initial=9.)
    leg=record(case,seed,"legacy",n=6,fail=6,initial=9.)
    # Both arms must have the same initial objective.
    budget["initial_best"]=9.
    budget["best_feasible_after_each_actual_online_call"][0]=9.
    for i in range(1,6):
        budget["best_feasible_after_each_actual_online_call"][i]=9.
    found={(case,seed,r["arm"]):r for r in (budget,control,leg)}
    pair=a._per_seed("pr54",found,case,[seed],"budgeted","semantic_only")[0]
    assert pair["common_online_calls"]==6
    assert pair["delta_at_common_if_both_feasible"]==-1.
    assert pair["candidate_extra_improvement_beyond_common"]==-2.
    assert pair["control_extra_improvement_beyond_common"]==0
    assert pair["delta_final_if_both_feasible"]==-3.


def _frozen_report(study):
    cfg=a.SOURCES[study]
    trials=[]
    for case in cfg["cases"]:
        for seed in cfg["seeds"]:
            for arm in a.ARMS:
                final={"legacy":9.5,"semantic_only":9.,"budgeted":8.}[arm]
                trials.append(record(case,seed,arm,final=final))
    registration={
        "profile":"full", "seeds":list(cfg["seeds"]),"arms":list(a.ARMS),
        "online_horizon":12,
        "initial_shared":12,"shared_initial":12,
        "observed_online_failure_cap":6,
        "observed_online_nonfeasible_stop":6,
        "cases":list(cfg["cases"]),
        "analytical_cases":list(cfg["cases"]),
    }
    payload={
        "status":cfg["status"],
        "source_sha":cfg["source_commit"],
        "new_rcaide_calls":0,
        "new_aircraft_evaluations":0,
        "new_aircraft_calls":0,
        "registration":registration,"plan":registration,
        "per_arm_results":trials,"trial_details":trials,
        "actual_new_analytical_evaluations":12*len(cfg["cases"])*len(cfg["seeds"])+len(trials)*12,
        "real_new_analytical_calls":12*len(cfg["cases"])*len(cfg["seeds"])+len(trials)*12,
    }
    if study=="pr55_corrected_pr56":
        payload["report_aggregation_audit"]={
            "trial_details_count":192,
            "new_analytical_evaluations":0,
            "aggregation_only":True,
            "original_report_sha256":
              "583d21106615cf05d8ec6c0a96a20d5a55f01de341efe7af76ee441139aa23f9",
        }
    return payload


def _prepare(tmp_path,monkeypatch):
    files={}
    for study in a.SOURCES:
        source=tmp_path / (study+".json")
        source.write_text(json.dumps(_frozen_report(study)),encoding="utf8")
        monkeypatch.setitem(a.SOURCES[study],"blob_sha1",a.git_blob_sha1(source))
        expected=12*len(a.SOURCES[study]["cases"])*len(a.SOURCES[study]["seeds"])+12*a.SOURCES[study]["n_trials"]
        monkeypatch.setitem(a.SOURCES[study],"n_analytical",expected)
        files[study]=source
    return files


def test_complete_240_paired_seed_rows_only_with_actual_outcomes(tmp_path, monkeypatch):
    source_dir=tmp_path/"inputs"
    source_dir.mkdir()
    files=_prepare(source_dir,monkeypatch)
    out=tmp_path/"new_audit"
    report=a.run(files["pr54"],files["pr55_corrected_pr56"],out)
    assert report["status"]=="COMPLETE_RETROSPECTIVE_FIXED_EFFORT_ANYTIME_AUDIT"
    assert report["n_original_trial_trajectories"]==240
    assert report["n_paired_method_seed_rows"]==240  # 8*2*3 + 16*4*3
    assert (out/"fixed_effort_paired_seeds.csv").exists()
    entry=report["paired_studies"]["pr55_corrected_pr56/branin_sparse/budgeted_vs_semantic_only"]
    assert entry["call_synchronized_fixed_horizons"]["6"]["both_arms_observed_through_horizon"]==16
    assert entry["original_full_stop_censored_endpoints"]["paired_mean_delta"]==-1.
    assert report["new_gp_vae_fits"]==report["new_analytical_evaluations"]==0
    with pytest.raises(ValueError,match="fresh"):
        a.run(files["pr54"],files["pr55_corrected_pr56"],out)


def test_byte_identity_failure_rejects_tampering_before_output(tmp_path,monkeypatch):
    source_dir=tmp_path/"inputs"
    source_dir.mkdir()
    files=_prepare(source_dir,monkeypatch)
    files["pr54"].write_text(files["pr54"].read_text(encoding="utf8")+" ",encoding="utf8")
    with pytest.raises(ValueError,match="Git blob SHA1"):
        a.run(files["pr54"],files["pr55_corrected_pr56"],tmp_path/"out")
    assert not (tmp_path/"out").exists()
