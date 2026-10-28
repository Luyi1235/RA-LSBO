"""Print live boundary campaign progress; read-only, with no dispatch or retries."""

import argparse
import json
from pathlib import Path
import sys
import time
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.budget import RiskLedger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    previous = None
    while True:
        try:
            report = load_json((args.folder / "result.json").read_bytes())
            ledger = RiskLedger(args.folder / "budget.jsonl").summary()
        except (ValueError, OSError):
            # Writers may be between a truncate and the final flushed bytes.
            time.sleep(15)
            continue
        current = (report["status"], len(report["results"]), ledger["charged_calls"])
        if current != previous:
            print(json.dumps({
                "status": report["status"], "returned": len(report["results"]),
                "charged": ledger["charged_calls"],
                "states": dict(Counter(item["risk"]["state"] for item in report["results"]
                                       if item["risk"] is not None)),
                "local_forecasts_saved": (args.folder / "test_forecasts.json").exists(),
                "reason": report.get("reason"),
            }), flush=True)
            previous = current
        if report["status"] != "running":
            break
        time.sleep(15)
