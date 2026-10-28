"""Prepare, explicitly run, audit, and report external mathematical comparisons."""

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
              f"mathematical_comparison_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}")
    print(f"Preparing mathematical pilot comparison: {output}", flush=True)
    plan = prepare(output, "pilot", "mathematical")
    print(f"Methods: {', '.join(plan['arms'])}", flush=True)
    maximum = plan["budget"]["maximum_model_calls"]
    print(f"Maximum model calls: {maximum}", flush=True)
    print("Prepared only. After reviewing the full budget, execute:", flush=True)
    print(f'python -m ra_lsbo.experiments.mathematical_benchmarks.compare run --output "{output}" '
          f'--approved-call-limit {maximum}', flush=True)
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
        plan = prepare(output, args.profile, "mathematical")
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
