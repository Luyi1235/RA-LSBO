"""C2 runner CLI. Default plan is read-only; run requires fixed SHA and fresh output."""
import argparse,json
from pathlib import Path
from c2_protocol import plan
PARENT="0a532a16e59c0f94e5263ad77a8a7e6f145196d1"

def main():
 p=argparse.ArgumentParser(description="C2 risk-head x objective-head factorial")
 p.add_argument("--task",choices=("plan","run","verify","analyze","tamper"),default="plan")
 p.add_argument("--profile",choices=("ci","smoke","full"),default="smoke")
 p.add_argument("--expected-sha");p.add_argument("--output",type=Path);p.add_argument("--input",type=Path)
 a=p.parse_args()
 if a.task=="plan":r=plan(a.profile)
 elif a.task=="run":
  if a.output is None or not a.expected_sha:p.error("--output and --expected-sha required")
  from risk_learning.io import source_receipt,git
  root=Path(__file__).resolve().parent
  source=source_receipt(root,a.expected_sha)
  frozen=("ra_lsbo","risk_learning","c1_native.py","c1_protocol.py","c1_study.py","c1_verify.py",
          "c2_protocol.py","c2_native.py","c2_study.py","c2_stats.py")
  if git(root,"diff","--name-only",PARENT,a.expected_sha,"--",*frozen):
   raise ValueError("Frozen C1 evaluator/model/protocol changed")
  source.update(phase="C2",c2_parent_sha=PARENT,head_factorial=True)
  from c2_study import run
  r=run(a.output,a.profile,source)
 elif a.task=="verify":
  if a.input is None:p.error("--input required")
  from c2_verify import verify
  r=verify(a.input,a.expected_sha)
 elif a.task=="analyze":
  if a.input is None or a.output is None:p.error("--input and --output required")
  from c2_analysis import analyze
  r=analyze(a.input,a.output,a.expected_sha)
 else:
  if a.input is None:p.error("--input required")
  from c2_verify import tamper
  r=tamper(a.input,a.expected_sha)
 print(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
if __name__=="__main__":main()
