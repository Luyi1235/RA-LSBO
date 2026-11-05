"""PR59 Windows PyCharm entry: PR58 selected-risk retrospective audit ONLY.

No Python torch/numpy/scipy, no GP/VAE fits, no evaluator calls. The source
must be byte identical to the frozen PR58 result commit. Never overwrite it.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import traceback
from ra_lsbo.experiments.risk_budget59_diagnostics import run

DEFAULT_REPORT=Path(r"D:\LSBO\RA-LSBO-pr58-fixed-six-full-20261105-v2\fixed_six_onpolicy_report.json")


def main():
    parser=argparse.ArgumentParser(description="Frozen selected-candidate GP risk overconfidence and fallback diagnostics, no new experiments.")
    parser.add_argument("--source",type=Path,default=DEFAULT_REPORT)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    output=args.output or Path(r"D:\LSBO")/(
        "RA-LSBO-pr59-risk-diagnostics-no-model-"+
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    try:
        head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        head="GIT_SHA_UNAVAILABLE"
    print("PR59 source SHA:",head,flush=True)
    print("Original PR58 report:",args.source,flush=True)
    print("New PR59 output:",output,flush=True)
    try:
        result=run(args.source,output)
    except Exception:
        traceback.print_exc()
        raise
    print("status:",result["status"],flush=True)
    print("historical original on-policy receipts:",result["total_original_six_call_receipts"],flush=True)
    for k in ("branin_sparse/budgeted/all","rosenbrock_sparse/budgeted/all"):
        summary=result["group_scores"][k]
        print(k,"forecasted",summary["forecasted_calls"],
              "bias_predicted_minus_realized",summary["signed_probability_bias_predicted_minus_realized"],
              "fallback",summary["budget_gate_fallback_count"],
              "fallback_fail",summary["realized_failure_count_in_fallback"],flush=True)
    print("new GP/VAE fits 0, new analytical/aircraft/RCAIDE evaluations 0",flush=True)

if __name__=="__main__":
    main()
