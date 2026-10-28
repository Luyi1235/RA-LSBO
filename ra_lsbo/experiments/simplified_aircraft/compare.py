"""Prepare, explicitly run, audit, and report external aircraft comparisons."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ra_lsbo.experiments.comparison.runner import audit, prepare, report, run


def run_default():
    """No arguments prepare a plan only; never implicitly authorize execution."""
    output = (Path(__file__).resolve().parents[3] / "runs" /
              f"aircraft_comparison_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}")
    plan = prepare(output, "pilot", "simplified_aircraft")
    maximum = plan["budget"]["maximum_model_calls"]
    print(f"Prepared only: {output}\nMethods: {', '.join(plan['arms'])}\nMaximum model calls: {maximum}", flush=True)
    print(f'After reviewing the full budget: python -m ra_lsbo.experiments.simplified_aircraft.compare '
          f'run --output "{output}" --approved-call-limit {maximum}', flush=True)
    return plan


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return run_default()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "audit", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="pilot")
    parser.add_argument("--approved-call-limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "run" and args.approved_call_limit is None:
        parser.error("run requires an explicit --approved-call-limit for the complete campaign")
    output = args.output.resolve()
    if args.command == "prepare":
        plan = prepare(output, args.profile, "simplified_aircraft")
        print({"suite": plan["suite"], "arms": plan["arms"], "budget": plan["budget"]}, flush=True)
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
