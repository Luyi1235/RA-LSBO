"""Phase B1: default plan is free; run requires explicit task and fixed SHA."""
import argparse
import json
from pathlib import Path
from b1_core import PARENT,plan


def main():
    parser=argparse.ArgumentParser(description='Matched objective/risk fixed-decision experiment')
    parser.add_argument('--task',choices=('plan','run','verify'),default='plan')
    parser.add_argument('--profile',choices=('ci','smoke','full'),default='smoke')
    parser.add_argument('--expected-sha')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--input',type=Path)
    args=parser.parse_args()
    if args.task=='plan':
        result=plan(args.profile)
    elif args.task=='verify':
        if args.input is None: parser.error('--input required')
        from b1_verify import verify
        result=verify(args.input,args.expected_sha)
    else:
        if args.output is None or args.expected_sha is None: parser.error('--output and --expected-sha required')
        from risk_learning.io import source_receipt,git
        repo=Path(__file__).resolve().parent
        source=source_receipt(repo,args.expected_sha)
        frozen=('risk_learning','decision_value_core.py','run_frozen_risk_decisions.py')
        if git(repo,'diff','--name-only',PARENT,args.expected_sha,'--',*frozen):
            raise ValueError('Frozen Phase A or B0 code changed')
        source.update(phase='B1',phase_parent_sha=PARENT,historical_risk_learning_unchanged=True)
        from b1_study import run
        result=run(args.output,args.profile,source)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
