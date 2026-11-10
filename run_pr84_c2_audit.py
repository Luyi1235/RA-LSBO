"""Fixed-source PR84 audit CLI. There is no optimization/evaluation command."""
import argparse
import json
from pathlib import Path
from pr84_c2_audit import execute, RESULT_HEAD, CODE_SHA, REL, ZERO


def main():
    p=argparse.ArgumentParser(description='PR84 read-only C2 archive audit (PR #85)')
    p.add_argument('--task',choices=('plan','run','verify'),default='plan')
    p.add_argument('--profile',choices=('smoke','full'),default='full')
    p.add_argument('--expected-sha')
    p.add_argument('--evidence-root',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.task=='plan':
        result=dict(result_head=RESULT_HEAD,archived_code_sha=CODE_SHA,archive_relative_path=REL,
            profile=a.profile,units=4 if a.profile=='smoke' else 64,
            trials=24 if a.profile=='smoke' else 384,
            historical_online_observations=72 if a.profile=='smoke' else 9216,
            new_bill=ZERO,read_only=True,output_write_once=True)
    else:
        if not a.expected_sha or a.evidence_root is None or a.output is None:
            p.error('--expected-sha, --evidence-root and --output are required')
        result=execute(Path(__file__).resolve().parent,a.expected_sha,a.evidence_root,
                       a.profile,a.output,verify=a.task=='verify')
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
