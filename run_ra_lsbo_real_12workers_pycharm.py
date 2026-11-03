"""PyCharm one-click TRUE full 12-worker aircraft experiment.

RUNNING THIS FILE EVALUATES THE REAL REDUCED-AIRCRAFT MODEL, up to
151040 times. This is NOT the zero-truth preflight or offline seven-arm demo.
Scientific qualification remains PROVISIONAL until separately assessed.

Select the Rcaide_vsp interpreter and right-click Run only when you intend
to commit the machine to the whole 1024/256 nested experimental workload.
"""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
os.environ["MPLBACKEND"]="Agg"

from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path

from confirm12.real_aircraft_campaign import run

ROOT=Path(__file__).resolve().parent


if __name__=="__main__":
    multiprocessing.freeze_support()
    output=ROOT/"results"/(
        "true_aircraft_12workers_"+
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    print("REAL AIRCRAFT CAMPAIGN: 7 actual method candidates, "
          "12 Windows spawn workers, <=151040 reduced-aircraft evaluations",
          flush=True)
    print("Results:",output,flush=True)
    print("Scientific interpretation: PROVISIONAL (unqualified task/baselines)",
          flush=True)
    result=run(output,workers=12)
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
