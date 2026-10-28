"""Package-local API and launcher for the mathematical comparison experiment."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ra_lsbo.experiments.comparison import ARMS
from ra_lsbo.experiments.comparison.methods import METHODS, optimizer_for as shared_optimizer_for
from ra_lsbo.experiments.comparison.runner import audit as shared_audit
from ra_lsbo.experiments.comparison.runner import prepare as shared_prepare
from ra_lsbo.experiments.comparison.runner import report as shared_report
from ra_lsbo.experiments.comparison.runner import run as shared_run
from ra_lsbo.experiments.core.protocol import BASE, PROFILES
from ra_lsbo.experiments.mathematical_benchmarks import CASES, SUITE, make_problem


def problem_for(case):
    if case not in CASES:
        raise ValueError(f"Unknown {SUITE} case: {case}")
    return make_problem(BASE, case)


def optimizer_for(case, seed, method, profile):
    problem_for(case)
    if method not in ARMS:
        raise ValueError(f"Unknown comparison method: {method}")
    return shared_optimizer_for(problem_for(case), BASE, seed, method, profile)


def prepare(folder, profile="pilot"):
    return shared_prepare(folder, profile, SUITE)


def run(folder, approved_call_limit=None):
    return shared_run(folder, approved_call_limit)


def audit(folder):
    return shared_audit(folder)


def report(folder):
    return shared_report(folder)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "audit", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="pilot")
    parser.add_argument("--approved-call-limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        plan = prepare(args.output.resolve(), args.profile)
        print({"suite": SUITE, "cases": plan["profile"]["cases"],
               "arms": ARMS, "budget": plan["budget"]}, flush=True)
    elif args.command == "run":
        result = run(args.output.resolve(), args.approved_call_limit)
        print({key: result[key] for key in
               ("charged_model_calls", "maximum_model_calls", "new_rcaide_mda_calls")})
    elif args.command == "audit":
        print(audit(args.output.resolve()), flush=True)
    else:
        print(report(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
