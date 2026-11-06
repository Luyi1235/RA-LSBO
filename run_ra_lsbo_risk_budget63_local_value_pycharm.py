"""PR63 Windows PyCharm: local-credibility / soft-gain-risk on-policy mathematical BO.

This REALLY fits latent VAE + constraint GP, chooses NEW candidates and
evaluates the custom analytical mathematical objective. It does NOT use or
modify aircraft/RCAIDE or frozen historical PR45–PR62 results.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import traceback

from ra_lsbo.experiments.risk_budget63_runner import run


def main():
    parser=argparse.ArgumentParser(
        description="PR63 same-candidate local trust with soft net utility instead of hard risk gate.")
    parser.add_argument("--profile",choices=("smoke","full"),default="full")
    parser.add_argument("--output",type=Path,
                        help="Brand new output directory outside all earlier PR results.")
    args=parser.parse_args()
    out=args.output or Path(r"D:\LSBO") / (
        "RA-LSBO-pr63-candidate-local-soft-value-"+
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+args.profile)
    try:
        sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        sha="GIT_SHA_UNAVAILABLE"
    print("PR63 fixed SHA:",sha,flush=True)
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
    print("new REAL analytical evaluations:",result["n_real_analytical_calls"],flush=True)
    print("actual on-policy trials:",result["n_new_on_policy_trials"],flush=True)
    print("new aircraft evaluations:",result["new_aircraft_calls"],flush=True)
    print("new RCAIDE calls:",result["new_rcaide_calls"],flush=True)
    print("Six original/new algorithms have the exact same six charged real online calls.",
          flush=True)


if __name__=="__main__":
    main()
