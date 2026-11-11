"""PR86-C command-line entrypoint. Write-once run, independent verify and analysis."""
import argparse
import json
from pathlib import Path
from pr86c_protocol import plan


def main():
    parser=argparse.ArgumentParser(description="PR86-C prospective objective-aware BO")
    parser.add_argument("--task",choices=("plan","run","verify","analyze"),default="plan")
    parser.add_argument("--profile",choices=("ci","smoke","full"),default="ci")
    parser.add_argument("--expected-sha")
    parser.add_argument("--input",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.task=="plan":
        result=plan(args.profile)
    elif args.task=="run":
        if not args.output or not args.expected_sha:
            parser.error("run requires --output and --expected-sha")
        from risk_learning.io import source_receipt
        from pr86c_study import run
        source=source_receipt(Path(__file__).resolve().parent,args.expected_sha)
        source.update(phase="PR86-C",protocol="pr86c_objective_aware_prospective_v1")
        result=run(args.output,args.profile,source)
    elif args.task=="verify":
        if not args.input or not args.expected_sha:
            parser.error("verify requires --input and --expected-sha")
        from pr86c_verify import verify
        result=verify(args.input,args.expected_sha)
    else:
        if not args.input or not args.output or not args.expected_sha:
            parser.error("analyze requires --input, --output and --expected-sha")
        from pr86c_analysis import analyze
        result=analyze(args.input,args.output,args.expected_sha)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
