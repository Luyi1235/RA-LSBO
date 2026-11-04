"""PyCharm read-only RA-LSBO latent versus onehot GP holdout experiment.

This computes new surrogate fits on existing PR45 observations only; ZERO new
aircraft, RCAIDE or mission simulation calls. Does not alter any source data.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from confirm12.matched_gp_mechanism import run

DEFAULT_SOURCE=Path(r"D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z")


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    ap.add_argument("--output",type=Path,default=None)
    ap.add_argument("--pairs-per-task",type=int,default=8,
                    help="Prespecified independent seed pairs per task (1..128), not outcome-selected")
    args=ap.parse_args(argv)
    output=args.output or (Path(r"D:\LSBO")/(
        "RA-LSBO-matched-GP-no-aircraft-"+
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    print("MATCHED GP STUDY; physical aircraft calls ZERO",flush=True)
    print("SOURCE",args.source,flush=True)
    print("OUTPUT",output,flush=True)
    report=run(args.source,output,per_case=args.pairs_per_task)
    print(json.dumps({
        "status":"COMPLETE_MATCHED_SURROGATE_DIAGNOSTIC",
        "case_seed_pairs":report["training_seed_pairs"],
        "prediction_rows":sum(v["records"] for v in report["model_heads"].values()),
        "output_dir":str(output),
        "new_aircraft_calls":report["new_aircraft_calls"],
        "new_RCAIDE_calls":report["new_rcaide_calls"],
    },ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
