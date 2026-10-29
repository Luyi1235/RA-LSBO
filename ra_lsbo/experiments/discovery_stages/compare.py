"""Explicit prepare/run/audit/report for staged mathematical research (no MDA)."""
import argparse
from pathlib import Path

from .methods import STAGES
from .runner import audit, prepare, report, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'audit', 'report'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=tuple(STAGES), default='discovery')
    parser.add_argument('--profile', choices=('smoke', 'pilot'), default='pilot')
    parser.add_argument('--prerequisite', type=Path)
    parser.add_argument('--approved-call-limit', type=int)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        plan = prepare(args.output.resolve(), args.stage, args.profile, args.prerequisite)
        print({'stage': plan['stage'], 'arms': plan['arms'], 'budget': plan['budget']}, flush=True)
    elif args.command == 'run':
        if args.approved_call_limit is None:
            parser.error('run requires --approved-call-limit; a plan is not execution authorization')
        result = run(args.output.resolve(), args.approved_call_limit)
        print({k: result[k] for k in ('charged_model_calls', 'maximum_model_calls', 'new_rcaide_mda_calls')})
    elif args.command == 'audit':
        result = audit(args.output.resolve())
        print({k:v for k,v in result.items() if k != 'evidence_hashes'})
    else:
        print(report(args.output.resolve()))


if __name__ == '__main__':
    main()
