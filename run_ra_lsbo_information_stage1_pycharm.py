"""Workstation entry: stage-1 opportunity audit and finite mechanism checks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from ra_lsbo.experiments.information_value_stage1.io import (
    manifest, pack, source_receipt, verify, write_new,
)
from ra_lsbo.experiments.information_value_stage1.mechanism import run_mechanisms
from ra_lsbo.experiments.information_value_stage1.runner import NativeAdapter, execute, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("opportunity", "mechanism", "verify", "pack"),
                        default="opportunity")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.task in ("verify", "pack"):
        if args.input is None or (args.task == "pack" and args.output is None):
            parser.error("verify requires --input; pack requires --input and fresh --output")
        result = verify(args.input) if args.task == "verify" else pack(args.input, args.output)
        print(json.dumps(result, indent=2), flush=True)
        return
    root = Path(__file__).resolve().parent
    out = (args.output or Path(r"D:\LSBO") / (
        "RA-LSBO-information-stage1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + args.task + "-" + args.profile)).resolve()
    if root == out or root in out.parents:
        parser.error("Keep evidence and logs OUTSIDE the clean source worktree")
    if out.exists():
        raise FileExistsError("A fresh output directory is mandatory: %s" % out)
    receipt = source_receipt(root)
    print("fixed code SHA:", receipt["head_sha"], "output:", out, flush=True)
    if args.task == "mechanism":
        out.mkdir(parents=True, exist_ok=False)
        write_new(out/"source_receipt.json", receipt)
        write_new(out/"mechanism_report.json", run_mechanisms())
        manifest(out)
    else:
        # Import/model/base failures happen before any evaluator call.
        api = NativeAdapter()
        result = execute(out, plan(args.profile), api, receipt)
        print(result["status"], "charged mathematical calls:",
              result["actual_total_mathematical_calls"], flush=True)
    print("Evidence verification:", verify(out), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
