"""C1R safe entrypoint: defaults to plan, never silently reruns old experiments."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='Frozen-method, fresh-seed C1 replication')
    p.add_argument('--task',choices=('plan','run','verify','analyze','tamper','review-pilot'),default='plan')
    p.add_argument('--profile',choices=('ci','smoke','full'),default='full')
    p.add_argument('--expected-sha')
    p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--evidence-root',type=Path)
    a=p.parse_args()
    from online_confirm import plan,run,verify,analysis,tamper
    if a.task=='plan':
        result=plan(a.profile)
    else:
        if not a.expected_sha or len(a.expected_sha)!=40:
            p.error('Exact fixed --expected-sha required')
        if a.task=='run':
            if a.output is None:p.error('--output required')
            result=run(a.output,a.profile,a.expected_sha)
        elif a.task=='review-pilot':
            if a.evidence_root is None or a.output is None:p.error('--evidence-root and --output required')
            result=analysis(None,a.output,a.expected_sha,pilot=True,evidence_root=a.evidence_root)
        else:
            if a.input is None:p.error('--input required')
            if a.task=='verify':result=verify(a.input,a.expected_sha)
            elif a.task=='tamper':result=tamper(a.input,a.expected_sha)
            else:
                if a.output is None:p.error('--output required')
                result=analysis(a.input,a.output,a.expected_sha)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
