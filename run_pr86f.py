"""PR86-F exact-source CLI. No runtime tuning parameters or resume option."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from pr86f_protocol import BASE_SHA, plan


def source_receipt(expected):
    root = Path(__file__).resolve().parent
    def git(*args):
        return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()
    if not re.fullmatch(r"[a-f0-9]{40}",expected or ""):
        raise ValueError("Full 40-character code SHA required")
    if git("rev-parse","HEAD")!=expected or git("status","--porcelain"):
        raise ValueError("Wrong SHA or dirty source worktree")
    frozen = ("risk_learning","ra_lsbo","pr86c_acquisition.py","pr86c_models.py",
              "pr86c_evaluator.py","pr86c_protocol.py")
    if git("diff","--name-only",BASE_SHA,expected,"--",*frozen):
        raise ValueError("Historical model, evaluator or baseline source changed")
    return dict(code_sha=expected,parent_sha=BASE_SHA,frozen_models_unchanged=True)


def main():
    parser=argparse.ArgumentParser(description="PR86-F rank vs magnitude acquisition ablation")
    parser.add_argument("--task",choices=("plan","run","verify","analyze"),default="plan")
    parser.add_argument("--profile",choices=("ci","full"),default="ci")
    parser.add_argument("--expected-sha")
    parser.add_argument("--input",type=Path)
    parser.add_argument("--output",type=Path)
    a=parser.parse_args()
    if a.task=="plan":
        answer=plan(a.profile)
    else:
        from pr86f_experiment import run, verify, analyze
        if not re.fullmatch(r"[a-f0-9]{40}",a.expected_sha or ""):
            parser.error("exact --expected-sha is required")
        if a.task=="run":
            if a.output is None:
                parser.error("run requires --output")
            answer=run(a.output,a.profile,source_receipt(a.expected_sha))
        else:
            if a.input is None or (a.task=="analyze" and a.output is None):
                parser.error("verify/analyze requires --input; analyze also requires --output")
            answer=verify(a.input,a.expected_sha) if a.task=="verify" else analyze(a.input,a.output,a.expected_sha)
    print(json.dumps(answer,sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
