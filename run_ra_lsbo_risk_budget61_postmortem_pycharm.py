"""PR61 PyCharm entry: pure read-only PR60 report mechanism postmortem.

No numpy, torch, scipy, GP/VAE, mathematical evaluator or aircraft/RCAIDE.
The source is checked against the exact original uploaded Git blob and SHA256.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import traceback

from ra_lsbo.experiments.risk_budget61_gate_postmortem import run

DEFAULT=Path(
    r"D:\LSBO\RA-LSBO-pr60-branch-evidence-UQ-20261105-full\fixed_six_onpolicy_report.json")


def main():
    parser=argparse.ArgumentParser(
        description="PR61 forensic Beta-floor and budget-gate audit of PR60's real selected decisions.")
    parser.add_argument("--source",type=Path,default=DEFAULT,
                        help="Exact original complete PR60 fixed_six_onpolicy_report.json")
    parser.add_argument("--output",type=Path,
                        help="NEW output directory outside original PR60 result directory")
    args=parser.parse_args()
    output=args.output or Path(r"D:\LSBO")/(
        "RA-LSBO-pr61-branch-gate-audit-no-model-"+
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    try:
        sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        sha="CODE_SHA_UNAVAILABLE"
    print("PR61 analysis HEAD:",sha,flush=True)
    print("original PR60 source:",args.source,flush=True)
    print("new PR61 output:",output,flush=True)
    try:
        report=run(args.source,output)
    except Exception:
        traceback.print_exc()
        raise
    print("status:",report["status"],flush=True)
    print("original historical selected-call receipts:",
          report["n_original_online_receipts"],flush=True)
    for arm in ("budgeted","branch_mean","branch_dispersion","branch_explore"):
        row=report["summary_by_arm_case_and_mode"][arm+"/all"]
        print(arm,
              "model_scored",row["forecasted_calls"],
              "beta_floor_binds",row["n_branch_posterior_floor_bind"],
              "budget_fallbacks",row["fallback_calls"],
              "fallback_failures",row["fallback_actual_nonfeasible"],flush=True)
    print("No new optimizer, GP/VAE, analytical, aircraft or RCAIDE calls.",flush=True)


if __name__=="__main__":
    main()
