"""Prepare, run, audit, and report external simplified-aircraft comparisons."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ra_lsbo.experiments.comparison.runner import audit, prepare, report, run


def run_default():
    output = (Path(__file__).resolve().parents[3] / "runs" /
              f"aircraft_comparison_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}")
    print(f"Preparing simplified aircraft pilot comparison: {output}", flush=True)
    plan = prepare(output, "pilot", "simplified_aircraft")
    print(f"Methods: {', '.join(plan['arms'])}", flush=True)
    print(f"Maximum model calls: {plan['budget']['maximum_model_calls']}", flush=True)
    print("Running comparison...", flush=True)
    result = run(output)
    print(f"Completed model calls: {result['charged_model_calls']}", flush=True)
    print("Auditing results...", flush=True)
    audit(output)
    print(f"Report: {report(output)}", flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return run_default()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "audit", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot", "confirm"), default="pilot")
    parser.add_argument("--approved-call-limit", type=int, default=None)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.command == "prepare":
        plan = prepare(output, args.profile, "simplified_aircraft")
        print({"suite": plan["suite"], "arms": plan["arms"],
               "budget": plan["budget"]}, flush=True)
    elif args.command == "run":
        result = run(output, args.approved_call_limit)
        print({key: result[key] for key in
               ("charged_model_calls", "maximum_model_calls", "new_rcaide_mda_calls")})
    elif args.command == "audit":
        print(audit(output), flush=True)
    else:
        print(report(output), flush=True)


if __name__ == "__main__":
    main()
