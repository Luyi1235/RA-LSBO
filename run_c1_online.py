"""Default plan is free. Run requires explicit fixed SHA and fresh external output."""
import argparse
import json
from pathlib import Path
from c1_protocol import PARENT,plan


def main():
    p=argparse.ArgumentParser(description='C1 equal-call risk x posterior-update pilot')
    p.add_argument('--task',choices=('plan','run','verify'),default='plan')
    p.add_argument('--profile',choices=('ci','smoke','full'),default='smoke')
    p.add_argument('--expected-sha');p.add_argument('--output',type=Path);p.add_argument('--input',type=Path)
    a=p.parse_args()
    if a.task=='plan':
        r=plan(a.profile)
    elif a.task=='verify':
        if a.input is None:p.error('--input is required')
        from c1_verify import verify
        r=verify(a.input,a.expected_sha)
    else:
        if a.output is None or a.expected_sha is None:p.error('--output and --expected-sha required')
        from risk_learning.io import source_receipt,git
        repo=Path(__file__).resolve().parent
        if repo==a.output.resolve() or repo in a.output.resolve().parents:
            raise ValueError('Evidence must be outside source worktree')
        source=source_receipt(repo,a.expected_sha)
        frozen=('ra_lsbo','risk_learning','b1_core.py','b1_native.py','b1_study.py','b1_verify.py')
        if git(repo,'diff','--name-only',PARENT,a.expected_sha,'--',*frozen):
            raise ValueError('Historical model/evaluator/Phase A/B1 source changed')
        source.update(phase='C1',c1_parent_sha=PARENT,historical_sources_unchanged=True)
        from c1_study import run
        r=run(a.output,a.profile,source)
    print(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
