"""PR68 workstation entry. No experiment is dispatched by imports or --help."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="PR68 real GP censored-fantasy two-step pilot")
    parser.add_argument("--task", choices=("run", "verify"), default="run")
    parser.add_argument("--profile", choices=("ci", "smoke", "full"), default="smoke")
    parser.add_argument("--expected-sha")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    from ra_lsbo.experiments.information_value_stage3.runner import run, verify
    if args.task == "verify":
        if args.input is None:
            parser.error("--input is required for verify")
        print(json.dumps(verify(args.input), ensure_ascii=False, indent=2), flush=True)
    else:
        if args.output is None or args.expected_sha is None:
            parser.error("--output and --expected-sha are required for a fresh run")
        result = run(args.output, args.profile, args.expected_sha)
        print(json.dumps({k: result[k] for k in ("status", "code_sha", "actual_calls_started", "actual_calls_settled", "aircraft_calls", "rcaide_calls")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
