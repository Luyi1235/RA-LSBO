"""Phase A only: --task plan/run/verify. No historical experiments are rerun."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('plan','run','verify'), default='plan')
    parser.add_argument('--profile', choices=('ci','smoke','full'), default='smoke')
    parser.add_argument('--expected-sha')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    from risk_learning.core import plan
    if args.task=='plan':
        result=plan(args.profile)
    elif args.task=='verify':
        if args.input is None:
            parser.error('--input required')
        from risk_learning.study import verify
        result=verify(args.input,args.expected_sha)
    else:
        if args.output is None or args.expected_sha is None:
            parser.error('--output and --expected-sha required')
        from risk_learning.io import source_receipt
        from risk_learning.study import run
        root=Path(__file__).resolve().parent
        output=args.output.resolve()
        if output==root or root in output.parents:
            parser.error('Output must be outside the fixed source worktree')
        source=source_receipt(root,args.expected_sha)
        result=run(output,args.profile,source)
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
