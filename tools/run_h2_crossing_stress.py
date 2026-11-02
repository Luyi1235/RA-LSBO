"""Second, separately registered H2 mathematical stress test: crossing SOC views.

NOT a replacement for the first fixed analytic benchmark. This deliberately
creates both onehot-optimistic and latent-optimistic designs and measures
how an adaptive weight differs from min-feasibility under the same shared
24-step pool stream. No GP/VAE/aircraft or RCAIDE evaluation.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import math
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.special import ndtr

from tools.run_h2_synthetic_decision import (
    pool_for,objective_proxy,stable_seed,ARMS,STEPS,POOL)
from tools.run_h2_hierarchical_synthetic import observe
from ra_lsbo.experiments.simplified_aircraft.h2_risk_controller import (
    CausalSOCController,H2Settings)

SEEDS=tuple(range(18001,18009))
SCENARIOS=("crossing_structural","crossing_conditional")
SD=.2
SCHEMA="ra_h2_crossing_synthetic_stress_v1"


def forecasts(d):
    common=-.12+.5*(d["x"]-.5)+.1*(d["ordinal"]-1.5)
    signed=1. if d["material"] in (0,2) else -1.
    mask=d["active_z"]
    p=common+.23*signed+.12*mask*(d["z"]-.5)
    q=common-.23*signed+.03*mask*(d["z"]-.5)
    return {"mean":p,"variance":SD**2},{"mean":q,"variance":SD**2}


def truth_mean(scenario,d):
    a,b=forecasts(d)
    if scenario=="crossing_structural":
        return a["mean"] if d["ordinal"]>=2 else b["mean"]
    if scenario=="crossing_conditional":
        return a["mean"] if d["active_z"] else b["mean"]
    raise ValueError("Unregistered crossing scenario")


def probability(pred):
    return float(ndtr(-pred["mean"]/math.sqrt(pred["variance"])))


def choose(pool,arm,weight):
    scores=[]
    for d in pool:
        a,b=forecasts(d)
        pa,pb=probability(a),probability(b)
        p=(pa if arm=="onehot" else
           pb if arm=="latent" else
           min(pa,pb) if arm=="min_guard" else
           weight*pa+(1-weight)*pb)
        scores.append(-objective_proxy(d)+.12*math.log(max(p,1e-12)))
    return int(np.argmax(scores))


def run(scenario,seed,arm):
    controller=CausalSOCController(H2Settings()) if arm=="h2" else None
    best=None
    usable=0
    censored=0
    first_different_from_onehot=0
    for t in range(STEPS):
        pool=pool_for(scenario,seed,t)
        w=controller.weight if controller is not None else None
        idx=choose(pool,arm,w)
        d=pool[idx]
        a,b=forecasts(d)
        prefix=f"{scenario}/{seed}/{arm}/{t}"
        if controller is not None:
            controller.register(f"{seed}/{arm}/{t}",prefix,a,b)
        # Deterministic counterfactual outcomes are created only after
        # the candidate is committed; same design/seed/t yields same truth.
        rng=np.random.default_rng(stable_seed([scenario,seed,t,d,"truth"]))
        true=truth_mean(scenario,d)+.16*float(rng.normal())
        energy=objective_proxy(d)+.02+.05*float(rng.uniform())
        observation=observe(true,t)
        censored+=int(observation.kind=="interval")
        feasible=true<=0
        usable+=int(feasible)
        if feasible:
            best=energy if best is None else min(best,energy)
        if controller is not None:
            controller.settle(f"{seed}/{arm}/{t}",prefix,observation)
        first_different_from_onehot+=int(idx!=choose(pool,"onehot",None))
    return {"scenario":scenario,"seed":seed,"arm":arm,
            "completed":True,"synthetic_calls":STEPS,
            "usable":usable,"censored":censored,"has_feasible":best is not None,
            "final_feasible_proxy_energy":best,
            "different_choices_vs_onehot":first_different_from_onehot,
            "final_onehot_weight":controller.weight if controller else None}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError("Crossing stress output must be new")
    # Check the synthetic setup actually forces both probability orderings
    # before we report any comparison.
    crossing=[]
    for scenario in SCENARIOS:
        pool=pool_for(scenario,SEEDS[0],0)
        one_below_lat=sum(probability(forecasts(d)[0])<probability(forecasts(d)[1]) for d in pool)
        lat_below_one=sum(probability(forecasts(d)[1])<probability(forecasts(d)[0]) for d in pool)
        if min(one_below_lat,lat_below_one)<POOL//4:
            raise ValueError("Stress scenario did not create sufficient view crossing")
        crossing.append({"scenario":scenario,
                         "onehot_more_pessimistic":one_below_lat,
                         "latent_more_pessimistic":lat_below_one})
    rows=[run(scenario,seed,arm)
          for scenario in SCENARIOS for seed in SEEDS for arm in ARMS]
    assert len(rows)==len(SCENARIOS)*len(SEEDS)*len(ARMS)
    summaries=[]
    for scenario in SCENARIOS:
        for arm in ARMS:
            group=[x for x in rows if x["scenario"]==scenario and x["arm"]==arm]
            feasible=[x["final_feasible_proxy_energy"] for x in group if x["has_feasible"]]
            summaries.append({"scenario":scenario,"arm":arm,"seed_n":len(group),
                "feasible_runs":len(feasible),
                "mean_feasible_proxy_energy":sum(feasible)/len(feasible) if feasible else None,
                "usable_total":sum(x["usable"] for x in group),
                "censored_total":sum(x["censored"] for x in group),
                "different_choices_vs_onehot":sum(x["different_choices_vs_onehot"] for x in group),
                "scope":"analytic on-policy design stress; not calibrated GP or aircraft"})
    args.output.mkdir(parents=True)
    for name,value in [
        ("registration.json",{"schema":SCHEMA,"seeds":SEEDS,
            "scenarios":SCENARIOS,"arms":ARMS,"steps":STEPS,
            "max_synthetic_truth_calls":len(rows)*STEPS,
            "controller":asdict(H2Settings()),"new_aircraft_calls":0,
            "new_GP_fits":0,"new_VAE_fits":0,"new_RCAIDE_calls":0}),
        ("view_crossing_precheck.json",crossing),
        ("trial_rows.json",rows),("arm_summaries.json",summaries)]:
        with (args.output/name).open("x",encoding="utf8") as f:
            json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
            f.write("\n")
    print(json.dumps({"crossing_verified":crossing,
          "total_synthetic_truth_calls":len(rows)*STEPS,
          "new_aircraft_calls":0,"comparisons":summaries},allow_nan=False))


if __name__=="__main__":
    main()
