"""Windows PyCharm PR55: fresh on-policy mathematical, no aircraft/RCAIDE.

Run smoke first in one fresh output. On success run full in a *different*
fresh output. Never resume or overwrite partial evidence after an error.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import traceback

from ra_lsbo.experiments.risk_budget55_runner import run


def main():
    parser=argparse.ArgumentParser(
        description="PR55 independent seeds/families, true analytical GP/VAE on-policy only.")
    parser.add_argument("--profile",choices=("smoke","full"),default="full")
    parser.add_argument("--output",type=Path,help="New output dir outside frozen PR45–54 results.")
    args=parser.parse_args()
    out=args.output or Path(r"D:\LSBO")/(
        "RA-LSBO-pr55-independent-budget-math-"+
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+args.profile)
    try:
        sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        sha="UNAVAILABLE_GIT_SHA"
    print("PR55 code SHA:",sha,flush=True)
    print("analytical experiment:",args.profile,"output:",out,flush=True)
    try:
        result=run(out,profile=args.profile)
    except Exception as exc:
        traceback.print_exc()
        if out.exists():
            (out/"failure.json").write_text(
                json.dumps({"source_sha":sha,"error_type":type(exc).__name__,
                            "message":str(exc)},ensure_ascii=False,indent=2),
                encoding="utf8")
        raise
    print("status:",result["status"],flush=True)
    print("new_analytical_calls:",result["real_new_analytical_calls"],flush=True)
    print("new_aircraft_calls:",result["new_aircraft_calls"],flush=True)
    print("new_rcaide_calls:",result["new_rcaide_calls"],flush=True)
    print("Validation is ON-POLICY math on four tasks, never historic PR45 physics.",
          flush=True)


if __name__=="__main__":
    main()
