"""PyCharm entry for read-only PR57 fixed-call comparative analysis.

Reads two byte-identical GitHub result files, not PR45/PR54/55 SQLite/trial
archives. Zero calls to GP/VAE, aircraft/RCAIDE, or analytical evaluators.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import traceback

from ra_lsbo.experiments.risk_budget57_anytime_audit import run

PR54_DEFAULT=Path(r"D:\LSBO\RA-LSBO-risk-budget-math54-20261104-full\risk_budget_math_report.json")
PR56_DEFAULT=Path(r"D:\LSBO\RA-LSBO-pr56-corrected-20261104\risk_budget55_report_corrected_pr56.json")


def main():
    parser=argparse.ArgumentParser(
        description="Fixed six online evaluations vs post-stop-censored outcomes, true PR54/PR55 trajectories only.")
    parser.add_argument("--pr54",type=Path,default=PR54_DEFAULT,
                        help="Exact PR54 full risk_budget_math_report.json")
    parser.add_argument("--pr56",type=Path,default=PR56_DEFAULT,
                        help="Exact PR56 corrected risk_budget55_report_corrected_pr56.json")
    parser.add_argument("--output",type=Path,
                        help="New output folder outside both original input folders.")
    args=parser.parse_args()
    output=args.output or Path(r"D:\LSBO")/(
        "RA-LSBO-pr57-fixed-effort-no-evaluation-"+
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    try:
        sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        sha="GIT_SHA_UNAVAILABLE"
    print("PR57 analysis_code_sha:",sha,flush=True)
    print("PR54 original:",args.pr54,flush=True)
    print("PR56 corrected:",args.pr56,flush=True)
    print("PR57 fresh output:",output,flush=True)
    try:
        result=run(args.pr54,args.pr56,output)
    except Exception:
        traceback.print_exc()
        raise
    print("status:",result["status"],flush=True)
    print("observed original on-policy trial paths:",result["n_original_trial_trajectories"],flush=True)
    print("paired method-seed records:",result["n_paired_method_seed_rows"],flush=True)
    print("new_analytical/aircraft/RCAIDE evaluations: 0/0/0",flush=True)


if __name__=="__main__":
    main()
