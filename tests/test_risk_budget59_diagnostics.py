"""PR59 synthetic no-model regression of genuine selected-point receipt semantics."""
import copy
import json

import pytest

from ra_lsbo.experiments import risk_budget59_diagnostics as d


def fake_rows():
    rs=[]
    for case in d.TASKS:
        for seed in d.SEEDS:
            for arm in d.ARMS:
                risk=[]
                for i in range(6):
                    available=(arm!="legacy" and i!=1)
                    predicted=.1 if i<3 else .8
                    failed=int(i in (0,2,4))
                    risk.append({
                        "iteration":i,
                        "realized_nonfeasible":failed,
                        "forecast_available_for_selected":available,
                        "predicted_nonfeasible":predicted if available else None,
                        "budget_rule": ("no_candidate_below_allowance_min_predicted_failure"
                                        if arm=="budgeted" and available and i==2 else
                                        "predicted_allowance_filter" if arm=="budgeted" and available else
                                        "constraint_only_ranking" if available else None),
                        "mode":"budgeted_boundary_information" if i==4 else
                               "scheduled_coverage" if i==1 else
                               "feasible_discovery",
                        "predicted_gate_passed":False if arm=="budgeted" and i==2 else True,
                        "observed_online_nonfeasible_before":sum(
                            int(k in (0,2,4)) for k in range(i)),
                        "outcome_not_available_when_selected":True,
                    })
                rs.append({"case":case,"seed":seed,"arm":arm,"online_calls":6,
                           "observed_nonfeasible":3,
                           "initial_prefix_sha256":"f"*64,
                           "risk_receipts":risk})
    return rs


def fake_report():
    rows=fake_rows()
    return {
        "schema":d.INPUT_SCHEMA,
        "status":"COMPLETE_FRESH_FIXED_SIX_ON_POLICY",
        "code_sha":d.SOURCE_CODE_SHA,
        "n_real_analytical_calls":2880,
        "n_original_on_policy_trials":288,
        "new_aircraft_calls":0,
        "new_rcaide_calls":0,
        "plan":{"cases":list(d.TASKS),"arms":list(d.ARMS),
                "seeds":list(d.SEEDS),
                "actual_online_calls_per_arm":6,"initial":12,
                "engine_internal_horizon_unmodified":12,
                "source_sha":d.SOURCE_CODE_SHA},
        "trial_details":rows,
    }


def _fixture(tmp_path,monkeypatch):
    folder=tmp_path/"source"
    folder.mkdir()
    path=folder/"pr58_report.json"
    path.write_text(json.dumps(fake_report()),encoding="utf8")
    monkeypatch.setattr(d,"INPUT_SHA1",d.git_blob_sha1(path))
    return path


def test_frozen_all_288_trial_receipts_and_1728_genuine_online_calls(tmp_path,monkeypatch):
    path=_fixture(tmp_path,monkeypatch)
    rows,ident=d._input(path)
    assert len(rows)==4*24*3*6==1728
    assert len(ident["report_sha256"])==64
    assert all(r["forecast_error"] is None for r in rows if not r["has_prediction"])
    assert sum(r["high_confidence_safe_failure"] for r in rows)>0
    assert len({r["seed"] for r in rows})==24


def test_phase_probability_bias_and_high_confidence_failure_rate(tmp_path,monkeypatch):
    path=_fixture(tmp_path,monkeypatch)
    rows,_=d._input(path)
    scores=d.analyze(rows)
    scored=scores["branin_sparse/budgeted/all"]
    assert scored["all_actual_selected_calls"]==144
    assert scored["forecasted_calls"]==120
    assert scored["no_forecast_calls"]==24
    assert scored["budget_gate_fallback_count"]==24
    assert scored["realized_failure_count_in_fallback"]==24
    assert scored["predicted_low_failure_calls"]==48
    assert scored["observed_failures_among_predicted_low_failure_calls"]==48
    assert scored["descriptive_cluster_bias"]["n_seed_clusters"]==24
    assert scored["descriptive_cluster_bias"]["bootstrap_draws"]==1000
    assert scores["branin_sparse/legacy/all"]["forecasted_calls"]==0
    assert scores["branin_sparse/budgeted/boundary"]["all_actual_selected_calls"]==24
    assert scores["branin_sparse/budgeted/coverage"]["no_forecast_calls"]==24
    assert scores["branin_sparse/budgeted/iteration_2"]["forecasted_calls"]==24


def test_cli_output_is_write_once_and_original_unchanged(tmp_path,monkeypatch):
    path=_fixture(tmp_path,monkeypatch)
    saved=path.read_bytes()
    out=tmp_path/"output"
    report=d.run(path,out)
    assert report["total_original_six_call_receipts"]==1728
    assert report["status"]=="COMPLETE_RETROSPECTIVE_PR58_SELECTED_RISK_DIAGNOSIS"
    assert report["new_gp_vae_fits"]==report["new_aircraft_calls"]==0
    assert (out/"risk_overconfidence_report.json").is_file()
    assert (out/"selected_risk_receipts.csv").is_file()
    assert path.read_bytes()==saved
    with pytest.raises(ValueError,match="must be NEW"):
        d.run(path,out)


def test_bytes_identity_and_synthetic_leak_rejected_before_output(tmp_path,monkeypatch):
    path=_fixture(tmp_path,monkeypatch)
    with path.open("a",encoding="utf8") as f:
        f.write("  ")
    with pytest.raises(ValueError,match="Git blob"):
        d.run(path,tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_legacy_status_is_never_pretended_as_real_risk_forecast(tmp_path,monkeypatch):
    path=_fixture(tmp_path,monkeypatch)
    raw=json.loads(path.read_text(encoding="utf8"))
    legacy=next(r for r in raw["trial_details"] if r["arm"]=="legacy")
    legacy["risk_receipts"][0]["forecast_available_for_selected"]=True
    legacy["risk_receipts"][0]["predicted_nonfeasible"]=.2
    path.write_text(json.dumps(raw),encoding="utf8")
    monkeypatch.setattr(d,"INPUT_SHA1",d.git_blob_sha1(path))
    with pytest.raises(ValueError,match="Legacy arm"):
        d._input(path)


def test_wrong_budget_history_prefix_is_rejected(tmp_path,monkeypatch):
    path=_fixture(tmp_path,monkeypatch)
    raw=json.loads(path.read_text(encoding="utf8"))
    row=next(r for r in raw["trial_details"] if r["arm"]=="budgeted")
    row["risk_receipts"][2]["observed_online_nonfeasible_before"]=0
    path.write_text(json.dumps(raw),encoding="utf8")
    monkeypatch.setattr(d,"INPUT_SHA1",d.git_blob_sha1(path))
    with pytest.raises(ValueError,match="historical prefix"):
        d._input(path)
