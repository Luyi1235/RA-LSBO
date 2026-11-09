"""C2 CLI. No run without explicit fixed SHA and fresh output directory."""
import argparse,json
from pathlib import Path
PARENT="0a532a16e59c0f94e5263ad77a8a7e6f145196d1"
PROTECTED=("c1_protocol.py","c1_native.py","c1_study.py","c1_verify.py","run_c1_online.py",
 "online_confirm.py","online_confirm_stats.py","ra_lsbo","risk_learning")
def main():
 p=argparse.ArgumentParser(description="C2 risk/objective posterior-head factorial")
 p.add_argument("--task",choices=("plan","run","verify","analyze"),default="plan")
 p.add_argument("--analysis-output",type=Path)
 p.add_argument("--profile",choices=("ci","smoke","full"),default="smoke")
 p.add_argument("--expected-sha");p.add_argument("--output",type=Path);p.add_argument("--input",type=Path)
 a=p.parse_args()
 from c2_protocol import plan
 if a.task=="plan":result=plan(a.profile)
 elif a.task=="verify":
  if a.input is None:p.error("--input required")
  from c2_verify import verify
  result=verify(a.input,a.expected_sha)
 elif a.task=="analyze":
  if a.input is None or a.analysis_output is None or a.expected_sha is None:p.error("--input, --analysis-output and --expected-sha required")
  from c2_analysis import analyze
  result=analyze(a.input,a.analysis_output,a.expected_sha)
 else:
  if a.output is None or a.expected_sha is None:p.error("--output and --expected-sha required")
  from risk_learning.io import source_receipt,git
  root=Path(__file__).resolve().parent
  if a.output.resolve()==root or root in a.output.resolve().parents:raise ValueError("Output must be outside source worktree")
  receipt=source_receipt(root,a.expected_sha)
  if git(root,"diff","--name-only",PARENT,a.expected_sha,"--",*PROTECTED):
   raise ValueError("Frozen C1 model/evaluator sources changed")
  receipt.update(phase="C2",parent_sha=PARENT,design_issue=82)
  from c2_study import run
  result=run(a.output,a.profile,receipt)
 print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
if __name__=="__main__":main()
