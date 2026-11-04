"""PR54 Windows/PyCharm: genuine on-policy *analytical* risk-budget trial.

IMPORTANT: Runs new objective evaluations of analytical hierarchical Branin,
NOT original PR45 aircraft and NOT RCAIDE. No PR45 database is read/changed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import traceback

from ra_lsbo.experiments.risk_budget54_runner import run


def main():
    parser=argparse.ArgumentParser(
        description="Opt-in dual-view GP/VAE with new budget-aware candidate choices on analytical problems.")
    parser.add_argument("--profile",choices=("smoke","full"),default="full")
    parser.add_argument("--output",type=Path,
                        help="Fresh root outside original PR45/46/48-53 results.")
    args=parser.parse_args()
    out=args.output or Path(r"D:\LSBO") / (
        "RA-LSBO-risk-budget-math54-" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") +
        "-" + args.profile)
    try:
        sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        sha="SHA_UNAVAILABLE"
    print("code_sha:",sha,"profile:",args.profile,"output:",out,flush=True)
    try:
        report=run(out,profile=args.profile)
    except Exception as exc:
        traceback.print_exc()
        if out.exists():
            (out/"failure.json").write_text(
                json.dumps({"source_sha":sha,"error_type":type(exc).__name__,
                            "message":str(exc)},ensure_ascii=False,indent=2),
                encoding="utf8")
        raise
    print("status:",report["status"],flush=True)
    print("actual_new_analytical_evaluations:",
          report["actual_new_analytical_evaluations"],flush=True)
    print("new_aircraft/RCAIDE:",report["new_aircraft_evaluations"],
          report["new_rcaide_calls"],flush=True)
    print("All three arms used new on-policy analytical outcomes, NOT PR45 replay.",
          flush=True)


if __name__=="__main__":
    main()
