import json
from pathlib import Path

import pytest

from ra_lsbo.experiments.risk_budget56_report_audit import audit
from ra_lsbo.experiments.risk_budget55_protocol import SCHEMA, CASES, ARMS, SEEDS


def make_trial(case,arm):
    receipts=[
        {"iteration":0,"prediction_available_for_selected":arm!="legacy",
         "selected_predicted_nonfeasible":.8 if arm!="legacy" else None,
         "candidate_risk_within_allowance":False if arm=="budgeted" else None,
         "budget_gate_fallback":"no_candidate_below_allowance_min_predicted_failure"
            if arm=="budgeted" else None,
         "realized_nonfeasible":1},
        {"iteration":1,"prediction_available_for_selected":arm!="legacy",
         "selected_predicted_nonfeasible":.1 if arm!="legacy" else None,
         "candidate_risk_within_allowance":True if arm=="budgeted" else None,
         "budget_gate_fallback":"predicted_allowance_filter" if arm=="budgeted" else None,
         "realized_nonfeasible":0},
    ]
    return {
        "case":case,"seed":SEEDS[0],"arm":arm,
        "initial_best":10.,"final_best":9.,
        "online_calls":2,"online_nonfeasible":1,
        "first_online_feasible_iteration":0,
        "best_feasible_after_each_actual_online_call":[10.,10.,9.],
        "selected_path_digest":case+arm,
        "pre_evaluation_risk_receipts":receipts,
    }


def test_audit_corrects_field_mismatch_from_trial_receipts_and_preserves_original(tmp_path):
    trials=[make_trial(c,a) for c in CASES for a in ARMS]
    report={
        "schema":SCHEMA,"source_sha":"a"*40,
        "plan":{"profile":"smoke"},
        "trial_details":trials,
        "method_summary_by_case":{
            f"{c}/{a}":{"predicted_risk_receipt_coverage":0}
            for c in CASES for a in ARMS},
    }
    source=tmp_path/"original.json"
    dest=tmp_path/"corrected.json"
    source.write_text(json.dumps(report),encoding="utf8")
    original_bytes=source.read_bytes()
    result=audit(source,dest)
    fixed=json.loads(dest.read_text(encoding="utf8"))
    assert source.read_bytes()==original_bytes
    assert result["trial_count"]==len(CASES)*len(ARMS)
    for case in CASES:
        assert fixed["method_summary_by_case"][f"{case}/legacy"]["predicted_risk_receipt_coverage"]==0
        assert fixed["method_summary_by_case"][f"{case}/semantic_only"]["predicted_risk_receipt_coverage"]==2
        budget=fixed["method_summary_by_case"][f"{case}/budgeted"]
        assert budget["predicted_risk_receipt_coverage"]==2
        assert budget["predicted_allowance_gate_fallbacks"]==1
        assert budget["observed_nonfeasible_in_predicted_gate_fallbacks"]==1
        assert budget["predicted_receipt_actual_nonfeasible_rate"]==.5
        assert budget["mean_selected_predicted_nonfeasible_proxy"]==pytest.approx(.45)
    assert fixed["report_aggregation_audit"]["new_analytical_evaluations"]==0
    assert fixed["report_aggregation_audit"]["original_report_modified"] is False
    with pytest.raises(FileExistsError):
        audit(source,dest)


def test_audit_rejects_non_pr55_and_missing_trial_details(tmp_path):
    source=tmp_path/"bad.json"
    source.write_text(json.dumps({"schema":"wrong"}),encoding="utf8")
    with pytest.raises(ValueError,match="Not a PR55"):
        audit(source,tmp_path/"out.json")
    source.write_text(json.dumps({"schema":SCHEMA,"plan":{"profile":"full"}}),encoding="utf8")
    with pytest.raises(ValueError,match="lacks full"):
        audit(source,tmp_path/"out2.json")
