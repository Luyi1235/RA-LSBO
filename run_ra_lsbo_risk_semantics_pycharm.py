"""Windows PyCharm entrypoint: read-only PR45 evaluation-risk event semantics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import traceback

from confirm12.risk_event_semantics import run

DEFAULT_SOURCE = Path(r"D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z")


def main():
    parser = argparse.ArgumentParser(description="No aircraft/RCAIDE calls. Original PR45 SQLite read-only.")
    parser.add_argument("--root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-records", type=int, help="Smoke only: first N original calls, NOT a full study")
    args = parser.parse_args()
    out = args.output or Path(r"D:\LSBO") / (
        "RA-LSBO-risk-semantics-no-aircraft-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "GIT_SHA_UNAVAILABLE"
    print("analysis_code_sha:", sha, flush=True)
    print("source_read_only:", args.root, flush=True)
    print("output:", out, flush=True)
    try:
        result = run(args.root, out, max_records=args.max_records)
    except Exception as exc:
        traceback.print_exc()
        if out.exists():
            (out / "failure.json").write_text(json.dumps(
                {"source_code_sha": sha, "error_type": type(exc).__name__,
                 "message": str(exc)}, indent=2), encoding="utf8")
        raise
    print("status:", result["status"], "rows:", result["calls_scanned"], flush=True)
    print("source unchanged; no new aircraft/RCAIDE calls", flush=True)


if __name__ == "__main__":
    main()
