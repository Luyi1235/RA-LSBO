"""PR53 real semantics and no-leakage regressions, with ZERO aircraft/GP calls."""
import csv
import json

import pytest

from confirm12 import phase_prefix_risk as p


def fake(case="aircraft_sparse",arm="ra_lsbo",split="test",seed=600705,
         iteration=0, raw=.7, success=0, mode="budgeted_boundary_information"):
    return {
        "case":case,"arm":arm,"split":split,"seed":seed,
        "iteration":iteration,"mode":mode,"phase":p.phase_of(mode),
        "prediction_available":raw is not None,
        "prediction_skip_reason":"" if raw is not None else "posterior_fit_skipped",
        "feasible_true":success,"mission_true":success,
        "predicted_constraint_product":raw,
        "predicted_soc":raw,
        "predicted_record_completion": .98 if raw is not None else None,
        "predicted_legacy_completion_product": .98*raw if raw is not None else None,
    }


def test_prior_only_calibration_no_current_outcome_leakage():
    rows = [fake(iteration=i, raw=.65,
                 success=int(i%2==0),mode="probability_weighted_feasible_improvement")
            for i in range(10)]
    base = p.prefix_predictions(rows,"feasible")
    key = ("aircraft_sparse","ra_lsbo","test",600705,6)
    assert base[key]["prefix_prior_scored"] == 6
    assert base[key]["prefix_fit_active"] is True
    assert base[("aircraft_sparse","ra_lsbo","test",600705,5)]["prefix_fit_active"] is False
    changed = [dict(r) for r in rows]
    changed[6]["feasible_true"] = 1 - changed[6]["feasible_true"]
    new = p.prefix_predictions(changed,"feasible")
    assert new[key] == base[key]
    # Only subsequent steps may use the different revealed outcome.
    later = ("aircraft_sparse","ra_lsbo","test",600705,7)
    assert new[later]["probability"] != base[later]["probability"]


def test_phase_calibration_shrinkage_and_fallback():
    rows = []
    for i in range(96):
        for case,base in (("aircraft_ordinary",500641),
                          ("aircraft_sparse",600641)):
            for arm in p.ARMS:
                mode = "budgeted_boundary_information" if i<24 else "probability_weighted_feasible_improvement"
                row = fake(case=case,arm=arm,split="calibration",seed=base+i//18,
                           iteration=i%18,mode=mode,raw=.7,
                           success=(int(i%3==0) if i<24 else int(i%3!=0)))
                rows.append(row)
    fitted,coverage = p.cross_seed_calibrators(rows,full=False)
    model = fitted[("aircraft_sparse","ra_lsbo","feasible")]
    assert coverage["aircraft_sparse/ra_lsbo/feasible/boundary"]["phase_fit_supported"] is False
    assert model["phase_shifts"]["boundary"] == model["pooled_shift"]
    assert coverage["aircraft_sparse/ra_lsbo/feasible/other"]["phase_fit_supported"] is False
    # Explicit centered logistic intercept only uses the supplied calibration labels.
    own = p.fit_centered([.7]*64,[0]*48+[1]*16,0.)
    assert own<0
    assert p.phase_of("budgeted_boundary_information") == "boundary"
    assert p.phase_of("scheduled_coverage") == "other"


def test_scoring_and_risk_budget_diagnostic_do_not_hide_skipped_steps():
    rows=[]
    for i in range(18):
        for event in ("feasible","mission"):
            modeled=(i!=5)
            q = {
                "case":"aircraft_ordinary","arm":"ra_lsbo","seed":500705,
                "iteration":i,"mode":"budgeted_boundary_information" if i%6==0
                else "probability_weighted_feasible_improvement",
                "phase":"boundary" if i%6==0 else "other",
                "target_event":event,"prediction_available":modeled,
                "true_event":int(i%4!=0),
                "raw_probability": .7 if modeled else None,
                "pooled_cross_seed_probability":.65 if modeled else None,
                "phase_cross_seed_probability":.66 if modeled else None,
                "prefix_only_probability":.63 if modeled else None,
                "prefix_fit_active":False,
            }
            rows.append(q)
    scores = p.scoring(rows)
    all_f = scores["aircraft_ordinary/ra_lsbo/feasible/all"]
    assert all_f["total_original_selected_calls"]==18
    assert all_f["model_probabilities_available"]==17
    assert all_f["model_free_or_no_probability"]==1
    assert all_f["strategies"]["raw"]["thresholds"]["0.8"]["accepted"]==0
    assert all_f["paired_delta"]["phase_minus_pooled"]["clusters"]==1
    risk=p.budget_diagnostic(rows)["aircraft_ordinary/ra_lsbo"]
    assert risk["n_original_calls"]==18
    assert risk["mode_free_worst_case_debits"]==1
    assert risk["mean_realized_nonfeasible"]==5
    assert risk["budget_thresholds"]["4.0"]["actual_exceed"]==1
    assert risk["not_a_counterfactual_stop_policy"] is True


def _source_fixture(root,monkeypatch):
    """Complete 9,216-row synthetic PR52-shaped file, SHA pinned to fixture."""
    report = {
        "schema":p.SOURCE_SCHEMA,
        "status":"COMPLETE_NATIVE_PREQUENTIAL_RISK_AUDIT",
        "source_sha":p.SOURCE_PR45_SHA,
        "original_sqlite_sha256_unchanged":p.SOURCE_PR45_SQLITE_SHA,
        "per_block":64,
        "total_original_online_selected_calls_scanned":9216,
        "new_aircraft_calls":0,"new_gp_fits":0,
        "split_seed_indices":{"calibration":[640,703],"test":[704,767]},
        "calibration_logit_shifts":{},
    }
    (root/"prequential_risk_report.json").write_text(json.dumps(report),encoding="utf8")
    fields = (
        "case","arm","split","seed","iteration","mode","cold_start",
        "proposal_sha256","prediction_available","prediction_skip_reason",
        "mission_true","feasible_true","soc_true","stall_true","root_bending_true",
        "historical_feasible_prevalence","historical_mission_prevalence",
        "predicted_record_completion","predicted_constraint_product",
        "predicted_legacy_completion_product","predicted_soc",
        "predicted_stall","predicted_root_bending",
        "online_calibrated_constraint_product","online_calibrated_legacy_completion_product",
        "online_calibrated_soc_mission","online_calibrated_record_completion_as_mission",
    )
    with (root/"prequential_risk_predictions.csv").open(
            "w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for case,base in (("aircraft_ordinary",500001),("aircraft_sparse",600001)):
            for arm in p.ARMS:
                for split,start in (("calibration",640),("test",704)):
                    for k in range(64):
                        for i in range(18):
                            good = int((k+i)%3 != 0)
                            # fixed every sixth step, but no probability only
                            # for a declared coverage mode, never for fake fits.
                            available = (i!=5)
                            prob = .75 if (k+i)%2 else .55
                            record = dict.fromkeys(fields,"")
                            record.update({
                                "case":case,"arm":arm,"split":split,
                                "seed":base+start+k,"iteration":i,
                                "mode":"scheduled_coverage" if not available else
                                       "budgeted_boundary_information" if i%6==0 else
                                       "probability_weighted_feasible_improvement",
                                "prediction_available":str(available),
                                "prediction_skip_reason":"" if available else "posterior_fit_skipped",
                                "mission_true":good,"soc_true":good,"feasible_true":good,
                            })
                            if available:
                                record.update(predicted_record_completion=.95,
                                              predicted_constraint_product=prob,
                                              predicted_legacy_completion_product=.95*prob,
                                              predicted_soc=prob)
                            writer.writerow(record)
    monkeypatch.setattr(p,"SOURCE_REPORT_GIT_BLOB_SHA",
                        p.git_blob_sha(root/"prequential_risk_report.json"))
    monkeypatch.setattr(p,"SOURCE_CSV_GIT_BLOB_SHA",
                        p.git_blob_sha(root/"prequential_risk_predictions.csv"))
    return root


def test_pinned_source_and_end_to_end_smoke(tmp_path, monkeypatch):
    root = tmp_path/"source"
    root.mkdir()
    _source_fixture(root,monkeypatch)
    rows,_,identity = p._read_source(root)
    assert len(rows)==9216 and len(identity["source_csv_sha256"])==64
    out=tmp_path/"output"
    result=p.run(root,out,per_block=1)
    assert result["status"]=="PARTIAL_SMOKE_ONLY"
    assert result["n_test_event_rows"]==144
    assert result["n_test_original_selected_calls"]==72
    assert len(result["cross_seed_calibration"])==2*2*2*2
    assert len(result["scoring_on_heldout_selected_calls"])==2*2*2*3
    assert (out/"phase_prefix_predictions.csv").is_file()
    assert result["new_optimizer_proposals"]==0
    with pytest.raises(ValueError,match="new output"):
        p.run(root,out,per_block=1)


def test_tampered_pr52_data_rejected_before_output_creation(tmp_path,monkeypatch):
    root=tmp_path/"source"
    root.mkdir()
    _source_fixture(root,monkeypatch)
    path=root/"prequential_risk_predictions.csv"
    with path.open("a",encoding="utf8") as stream:
        stream.write("forged_row\n")
    with pytest.raises(ValueError,match="blob identity"):
        p.run(root,tmp_path/"out",per_block=1)
    assert not (tmp_path/"out").exists()
