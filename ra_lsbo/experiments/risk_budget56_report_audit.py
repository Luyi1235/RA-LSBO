"""Rebuild PR55 risk aggregation from immutable trial_details; never rerun an evaluator.

Input PR55 report and its evidence remain read-only. Output must be a NEW path.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .risk_budget55_protocol import summarize, SCHEMA


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def audit(source, destination):
    source=Path(source).resolve()
    destination=Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source==destination or destination.exists():
        raise FileExistsError("Choose a distinct, not-yet-existing corrected report path")
    raw=source.read_bytes()
    original=json.loads(raw.decode("utf8"))
    if original.get("schema")!=SCHEMA:
        raise ValueError("Not a PR55 frozen report")
    plan=original.get("plan") or {}
    profile=plan.get("profile")
    if profile not in ("smoke","full"):
        raise ValueError("Missing frozen smoke/full registration")
    trials=original.get("trial_details")
    if not isinstance(trials,list) or not trials:
        raise ValueError("Report lacks full immutable per-trial details")
    by_arm,paired=summarize(trials,profile)
    corrected=dict(original)
    old=original.get("method_summary_by_case")
    corrected["method_summary_by_case"]=by_arm
    corrected["paired_seed_comparisons"]=paired
    corrected["report_aggregation_audit"]={
        "schema":"ra_lsbo_pr56_recomputed_aggregation_from_pr55_immutable_trial_details_v1",
        "original_report_sha256":hashlib.sha256(raw).hexdigest(),
        "trial_details_count":len(trials),
        "source_experiment_sha":original.get("source_sha"),
        "aggregation_only":True,
        "new_analytical_evaluations":0,
        "new_aircraft_evaluations":0,
        "new_rcaide_evaluations":0,
        "original_report_modified":False,
        "source_proposals_or_observations_modified":False,
        "summary_changed":old!=by_arm,
        "root_cause":"PR55 runner writes prediction_available_for_selected; original summary incorrectly tested forecast_available",
        "risk_metrics_recomputed_from_each_trial_pre_evaluation_risk_receipts":True,
    }
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x",encoding="utf8") as f:
        json.dump(corrected,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    return {
        "source":str(source),"source_sha256":sha256(source),
        "corrected":str(destination),"corrected_sha256":sha256(destination),
        "trial_count":len(trials),"summary_changed":old!=by_arm,
        "risk_coverage":{k:v["predicted_risk_receipt_coverage"]
                         for k,v in by_arm.items()},
        "risk_fallbacks":{k:v["predicted_allowance_gate_fallbacks"]
                          for k,v in by_arm.items()},
    }


def main():
    parser=argparse.ArgumentParser(
        description="PR56 correct only PR55 report aggregation from saved trial receipts; no evaluations.")
    parser.add_argument("--source",type=Path,required=True,
                        help="Existing PR55 risk_budget55_report.json")
    parser.add_argument("--output",type=Path,required=True,
                        help="New corrected report path; never overwrite original")
    args=parser.parse_args()
    print(json.dumps(audit(args.source,args.output),indent=2),flush=True)


if __name__=="__main__":
    main()
