"""PR86-B CLI: sequential risk-information acquisition."""
import argparse, json
from pathlib import Path
from pr86b_protocol import plan

def main():
    p=argparse.ArgumentParser(description="PR86-B sequential risk-information acquisition")
    p.add_argument("--task",choices=("plan","run","verify","analyze"),default="plan")
    p.add_argument("--profile",choices=("ci","smoke","full"),default="smoke")
    p.add_argument("--expected-sha")
    p.add_argument("--output",type=Path)
    p.add_argument("--input",type=Path)
    a=p.parse_args()
    if a.task=="plan":
        result=plan(a.profile)
    elif a.task=="run":
        if not a.output or not a.expected_sha:
            p.error("run requires --output and --expected-sha")
        from risk_learning.io import source_receipt
        from pr86b_study import run
        root=Path(__file__).resolve().parent
        source=source_receipt(root,a.expected_sha)
        source.update(phase="PR86-B",protocol="pr86b_sequential_risk_information_v1")
        result=run(a.output,a.profile,source)
    elif a.task=="verify":
        if not a.input or not a.expected_sha:
            p.error("verify requires --input and --expected-sha")
        from pr86b_verify import verify
        result=verify(a.input,a.expected_sha)
    else:
        if not a.input or not a.output or not a.expected_sha:
            p.error("analyze requires --input, --output and --expected-sha")
        from pr86b_analysis import analyze
        result=analyze(a.input,a.output,a.expected_sha)
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))

if __name__=="__main__":
    main()
