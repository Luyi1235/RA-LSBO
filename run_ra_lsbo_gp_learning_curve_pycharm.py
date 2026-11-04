"""PyCharm RA-LSBO GP sample-size learning curves from PR45 real records.

Read-only. Never calls any aircraft model; never changes the source experiment.
An offline cross-seed training data experiment, NOT a new 12+18 BO campaign.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from confirm12.sample_size_gp_learning import run,SIZES

DEFAULT_SOURCE=Path(r"D:\LSBO\RA-LSBO-pr45-real-20261104_01\results\true_aircraft_12workers_20261103T173750Z")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    p.add_argument("--output",type=Path,default=None)
    p.add_argument("--folds-per-task",type=int,default=2)
    p.add_argument("--sizes",type=str,default=",".join(str(n) for n in SIZES),
                   help="Ascending subset of 12,24,48,96,192, e.g. 12,24 for first smoke")
    args=p.parse_args(argv)
    try:
        sizes=tuple(int(a) for a in args.sizes.split(","))
    except ValueError as exc:
        p.error("--sizes requires comma-separated integers")
    output=args.output or (Path(r"D:\LSBO")/(
        "RA-LSBO-GP-learning-curve-no-aircraft-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    print("RA-LSBO offline size effect: 0 aircraft calls",flush=True)
    print("PR45 source:",args.source,flush=True)
    print("New results:",output,flush=True)
    report=run(args.source,output,folds=args.folds_per_task,sizes=sizes)
    print(json.dumps({
        "status":report["status"],
        "completed_fit_pairs":report["successful_fit_pairs"],
        "fit_errors":report["fit_errors"],
        "training_sizes":report["training_sizes"],
        "folds_per_task":report["folds_per_task"],
        "output_dir":str(output),
        "new_aircraft_calls":report["new_aircraft_calls"],
        "new_RCAIDE_calls":report["new_RCAIDE_calls"],
    },ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
