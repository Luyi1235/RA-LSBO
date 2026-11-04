"""PR53 PyCharm entry: read only the frozen PR52 prediction CSV/JSON."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import traceback

from confirm12.phase_prefix_risk import run

DEFAULT_SOURCE = Path(
    r"D:\LSBO\RA-LSBO-native-risk-prequential-no-aircraft-20261104-full")


def main():
    parser = argparse.ArgumentParser(
        description="Native online risk phase/prefix calibration: NO physics or GP/VAE fits.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="The complete PR52 output folder, containing CSV and JSON.")
    parser.add_argument("--output", type=Path,
                        help="Must be fresh, outside PR52 source folder.")
    parser.add_argument("--per-block", type=int, default=64,
                        help="1 for smoke, 64 for full frozen 64+64 seed study.")
    args = parser.parse_args()
    out = args.output or Path(r"D:\LSBO") / (
        "RA-LSBO-phase-prefix-risk-no-aircraft-" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    try:
        sha = subprocess.check_output(["git","rev-parse","HEAD"],
                                      text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        sha = "SOURCE_SHA_UNAVAILABLE"
    print("PR53 analysis code SHA:", sha, flush=True)
    print("PR52 frozen source:", args.source, flush=True)
    print("new output:", out, "per block:", args.per_block, flush=True)
    try:
        result = run(args.source, out, per_block=args.per_block)
    except Exception as exc:
        traceback.print_exc()
        if out.exists():
            (out/"failure.json").write_text(
                json.dumps({"code_sha":sha, "error_type":type(exc).__name__,
                            "message":str(exc)},indent=2,ensure_ascii=False),
                encoding="utf8")
        raise
    print("status:",result["status"],flush=True)
    print("original heldout calls:",result["n_test_original_selected_calls"],
          "event prediction rows:",result["n_test_event_rows"],flush=True)
    print("No new aircraft/RCAIDE calls; no GP/VAE fits; no optimization proposals",
          flush=True)


if __name__ == "__main__":
    main()
