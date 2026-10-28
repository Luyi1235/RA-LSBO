"""Prepare or execute one explicitly approved, budgeted physical worker check."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid

from ra_lsbo.x57_live import execute_smoke, prepare_smoke


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", type=Path, metavar="REQUEST")
    parser.add_argument("--base-project", type=Path, default=root.parent / "LSBO" / "X57-LSBO-refactored")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--active-run", type=Path)
    parser.add_argument("--output", type=Path, default=root / "runs")
    args = parser.parse_args()
    if args.prepare:
        if args.config is None:
            parser.error("--prepare requires --config")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = args.output.resolve() / ("x57_smoke_request_" + stamp + "_" + uuid.uuid4().hex[:8])
        print("Unapproved request:", prepare_smoke(args.base_project, args.config, folder, args.active_run))
        print("No MDA executed.")
    else:
        result = execute_smoke(args.execute)
        print("Result:", result["status"])
