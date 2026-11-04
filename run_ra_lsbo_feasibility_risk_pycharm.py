"""Windows PyCharm direct entry for PR51; zero original aircraft/RCAIDE calls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import traceback

from confirm12.feasibility_risk_calibration import run

DEFAULT_SOURCE = Path(r"D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z")


def main():
    parser = argparse.ArgumentParser(description="Disjoint-seed risk probability calibration; original PR45 SQLite read-only.")
    parser.add_argument("--root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--folds", type=int, default=2, help="Disjoint seed blocks per task (1..8).")
    parser.add_argument("--sizes", type=str, default="12,24,48,96,192")
    args = parser.parse_args()
    sizes = tuple(int(x.strip()) for x in args.sizes.split(","))
    output = args.output or Path(r"D:\LSBO") / (
        "RA-LSBO-feasibility-risk-no-aircraft-" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "GIT_SHA_UNAVAILABLE"
    print("analysis_code_sha:", sha, flush=True)
    print("original_source_read_only:", args.root, flush=True)
    print("output:", output, flush=True)
    print("folds:", args.folds, "sizes:", sizes, flush=True)
    try:
        result = run(args.root, output, folds=args.folds, sizes=sizes)
    except Exception as exc:
        traceback.print_exc()
        if output.exists():
            (output / "failure.json").write_text(
                json.dumps({"source_code_sha": sha,
                            "error_type": type(exc).__name__, "message": str(exc)},
                           indent=2, ensure_ascii=False), encoding="utf8")
        raise
    print("status:", result["status"], "model_configs:",
          result["n_model_configs_complete"], "prediction_rows:",
          result["n_test_prediction_rows"], flush=True)
    if result["fit_errors"]:
        print("FIT_ERRORS: preserved in report; do not modify code on workstation", flush=True)
        for error in result["fit_errors"]:
            print(error["case"], "fold", error["fold"], "n", error["n"],
                  error["error_type"], error["message"], flush=True)
    print("zero aircraft/RCAIDE calls, original database unchanged", flush=True)


if __name__ == "__main__":
    main()
