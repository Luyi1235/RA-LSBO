"""PR67 Windows PyCharm entry: matched eligible-only vs all-support soft-GP."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import traceback

from ra_lsbo.experiments.information_value_stage2.runner import run


def main():
    parser = argparse.ArgumentParser(description="PR67 matched no-support-gate soft-GP control")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(r"D:\LSBO") / (
        "RA-LSBO-pr67-all-support-soft-gp-" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.profile)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "GIT_SHA_UNAVAILABLE"
    print("PR67 source SHA:", sha, flush=True)
    print("profile:", args.profile, "output:", output, flush=True)
    try:
        result = run(output, profile=args.profile)
    except Exception as exc:
        traceback.print_exc()
        if output.exists():
            (output / "failure.json").write_text(json.dumps({
                "code_sha": sha, "error_type": type(exc).__name__, "message": str(exc)
            }, ensure_ascii=False, indent=2), encoding="utf8")
        raise
    print("status:", result["status"], flush=True)
    print("actual mathematical calls:", result["actual_mathematical_calls"], flush=True)
    print("aircraft/RCAIDE:", result["new_aircraft_calls"], result["new_rcaide_calls"], flush=True)


if __name__ == "__main__":
    main()
