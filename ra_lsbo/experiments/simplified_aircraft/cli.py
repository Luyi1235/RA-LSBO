"""Prepare, execute, audit, and report simplified-aircraft campaigns."""

import argparse
from pathlib import Path

from ..core.runner import audit, prepare, report, run
from . import SUITE


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "audit", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot", "confirm"), default="smoke")
    parser.add_argument("--approved-call-limit", type=int, default=500)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.command == "prepare":
        value = prepare(output, args.profile, SUITE)
        print(value["budget"], flush=True)
    elif args.command == "run":
        value = run(output, args.approved_call_limit)
        print({key: value[key] for key in (
            "charged_model_calls", "maximum_model_calls", "new_rcaide_mda_calls")})
    elif args.command == "audit":
        print(audit(output), flush=True)
    else:
        print(report(output), flush=True)


if __name__ == "__main__":
    main()
