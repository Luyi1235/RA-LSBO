"""PyCharm: READ-ONLY PR45 completion/feasibility/paired-data postrun review.

Right-click Run using Rcaide_vsp. No aircraft calls and no changes to the
original 151006-call ledger, 20GB raw evidence or original PR45 results.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path

from confirm12.postrun_pr45 import analyze

DEFAULT_ORIGINAL=Path(
    r"D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z"
)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(
        os.environ.get("PR45_REAL_RESULT_DIR",str(DEFAULT_ORIGINAL))))
    parser.add_argument("--archive",type=Path,default=None,
                        help="Optional archived trial-results-and-verifications.zip, else read local 20GB source result.json")
    parser.add_argument("--output",type=Path,default=None)
    parser.add_argument("--skip-event-chain",action="store_true",
                        help="Only for quick exploratory diagnostics, NOT an independent full audit")
    args=parser.parse_args(argv)
    if args.output is None:
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output=Path(r"D:\LSBO")/("RA-LSBO-pr45-independent-postrun-"+stamp)
    print("PR45 independent READ-ONLY review",flush=True)
    print("Original source:",args.root,flush=True)
    print("New report output:",args.output,flush=True)
    result=analyze(args.root,args.output,args.archive,
                   check_event_chain=not args.skip_event_chain)
    print(json.dumps({
        "independent_status":result["independent_postrun_status"],
        "original_ledger_status":result["original_ledger_status_unchanged"],
        "calls":result["accounting"]["calls"],
        "unused_optional_verifications":result["unused_optional_verifications"],
        "shared_initial_qA_feasibility":result["shared_initial_qA_feasibility"],
        "primary_paired_exploratory":result["primary_paired_exploratory"],
        "output_dir":str(args.output),
    },indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
