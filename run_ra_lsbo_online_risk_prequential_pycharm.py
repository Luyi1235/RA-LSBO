"""PR52 Windows PyCharm: original online risk-proposal replay, NO aircraft/GP fits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import traceback

from confirm12.online_prequential_risk import run

DEFAULT_SOURCE = Path(r"D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z")


def main():
    parser = argparse.ArgumentParser(description="Only frozen PR45 original proposal logs/SQLite; no physics or GP/VAE fitting.")
    parser.add_argument("--root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-block", type=int, default=64, help="Each task: N independent calibration plus N test seed groups (1..64).")
    args = parser.parse_args()
    out = args.output or Path(r"D:\LSBO") / (
        "RA-LSBO-native-risk-prequential-no-aircraft-" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "SOURCE_SHA_UNAVAILABLE"
    print("analysis SHA:", sha, "original:", args.root, "output:", out, flush=True)
    print("seed_groups_per_block:", args.per_block, flush=True)
    try:
        result = run(args.root, out, per_block=args.per_block)
    except Exception as exc:
        traceback.print_exc()
        if out.exists():
            (out/"failure.json").write_text(json.dumps(
                {"source_sha": sha, "error_type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False, indent=2), encoding="utf8")
        raise
    print("status:", result["status"], flush=True)
    print("online selected calls:", result["total_original_online_selected_calls_scanned"],
          "test model-scored calls:", result["test_scored_calls"], flush=True)
    print("No new aircraft, RCAIDE, GP or VAE calls; original evidence unchanged", flush=True)


if __name__ == "__main__":
    main()
