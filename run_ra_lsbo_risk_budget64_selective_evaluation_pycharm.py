"""PR64 Windows PyCharm: selective real evaluation with a true outside option.

Uses the frozen PR63 GP-only soft utility. Compares fixed evaluation against
pre-evaluation abstention with a 512-candidate screen under the same maximum
six-call budget. Mathematical tasks only; no aircraft/RCAIDE calls.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import traceback

from ra_lsbo.experiments.risk_budget64_runner import run


def main():
    parser=argparse.ArgumentParser(
        description="PR64 real abstention vs fixed-budget soft-GP candidate evaluation.")
    parser.add_argument("--profile",choices=("smoke","full"),default="full")
    parser.add_argument("--output",type=Path,
                        help="Brand-new output directory; never overwrite previous runs.")
    args=parser.parse_args()
    out=args.output or Path(r"D:\LSBO") / (
        "RA-LSBO-pr64-selective-evaluation-"+
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+args.profile)
    try:
        sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        sha="GIT_SHA_UNAVAILABLE"
    print("PR64 fixed SHA:",sha,flush=True)
    print("profile:",args.profile,"new output:",out,flush=True)
    try:
        result=run(out,profile=args.profile)
    except Exception as exc:
        traceback.print_exc()
        if out.exists():
            (out/"failure.json").write_text(json.dumps({
                "code_sha":sha,"error_type":type(exc).__name__,"message":str(exc)},
                ensure_ascii=False,indent=2),encoding="utf8")
        raise
    print("status:",result["status"],flush=True)
    print("actual REAL analytical evaluations:",result["n_real_analytical_calls"],flush=True)
    print("maximum evaluation ceiling:",result["maximum_analytic_evaluation_ceiling"],flush=True)
    print("actual online evaluations:",result["n_actual_online_evaluations"],flush=True)
    print("maximum online evaluation ceiling:",result["n_maximum_online_evaluations"],flush=True)
    print("new aircraft evaluations:",result["new_aircraft_calls"],flush=True)
    print("new RCAIDE calls:",result["new_rcaide_calls"],flush=True)
    print("Abstentions are real: no problem.evaluate call and no outcome label.",flush=True)


if __name__=="__main__":
    main()
