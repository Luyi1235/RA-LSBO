"""Synthetic hierarchical and censored SOC forecast-stream benchmark for H2.

This is NOT aircraft evaluation or an end-to-end BO result. The two
deliberately imperfect analytic forecasters depend only on the design.
H2, fixed-onehot, fixed-latent and fixed-conservative rules share the
same deterministic online candidate and truth sequence. No forecaster
uses future realized noise/labels or the future censoring type.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import json
import math
import sys

# Direct python tools/... invocation otherwise hides the repository root.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.special import ndtr,logsumexp

from ra_lsbo.experiments.simplified_aircraft.h2_risk_controller import (
    CausalSOCController,H2Settings,proper_nll)
from ra_lsbo.risk_core.observations import OutputObservation

SEEDS=tuple(range(16001,16009))
SCENARIOS=("onehot_valid","latent_valid","temporal_shift",
           "conditional_hierarchy")
STEPS=96
FIXED_WEIGHT={"onehot":1.,"latent":0.}
REGRESSION_SETTINGS=H2Settings()
MODEL_STD=.18


def design_from_rng(rng):
    """Mixed categorical/ordinal/conditional continuous design."""
    material=int(rng.integers(4))
    ordinal=int(rng.integers(4))
    layer=int(rng.integers(2))
    x=float(rng.uniform())
    # z is active only in layer 1 and materials 1,2,3.
    mask=int(layer==1 and material>0)
    z=float(rng.uniform()) if mask else 0.
    return {"material":material,"ordinal":ordinal,"layer":layer,
            "x":x,"z":z,"active_z":mask}


def analytic_forecasts(d):
    x=d["x"]
    common=-.13+.52*(x-.5)+.13*(d["ordinal"]-1.5)
    one=common+.18*(d["material"] in (2,3))+.16*d["active_z"]*(d["z"]-.5)
    latent=common+.08*(d["material"]-1.5)+.06*d["active_z"]*(d["z"]-.5)
    return {"mean":float(one),"variance":MODEL_STD**2}, \
           {"mean":float(latent),"variance":MODEL_STD**2}


def truth_location(scenario,d,t):
    one,latent=analytic_forecasts(d)
    if scenario=="onehot_valid":
        signal=one["mean"]
    elif scenario=="latent_valid":
        signal=latent["mean"]
    elif scenario=="temporal_shift":
        signal=one["mean"] if t<STEPS//2 else latent["mean"]
    elif scenario=="conditional_hierarchy":
        signal=one["mean"] if d["active_z"] else latent["mean"]
    else:
        raise ValueError("Unregistered synthetic scenario")
    return signal+.025*math.sin(2*math.pi*d["x"])


def observe(g,i):
    """One-sided failure censoring, no invented exact SOC label."""
    if g>.08:
        return OutputObservation("interval",f"synthetic:{i}",lower=.08)
    if g<-.48:
        return OutputObservation("interval",f"synthetic:{i}",upper=-.48)
    return OutputObservation("exact",f"synthetic:{i}",value=float(g))


def mix_log_score(observation,one,latent,w):
    a,b=proper_nll(one,observation),proper_nll(latent,observation)
    if a is None:return None
    # This is the TRUE marginal log mixture, not the log density of
    # moment-matched Gaussian or a weighted average of component NLLs.
    vals=[]
    if w>0:vals.append(math.log(w)-a["nll"])
    if w<1:vals.append(math.log1p(-w)-b["nll"])
    return {"type":a["type"],"nll":float(-logsumexp(vals))}


def event_probability(one,latent,w):
    po=float(ndtr(-one["mean"]/math.sqrt(one["variance"])))
    pl=float(ndtr(-latent["mean"]/math.sqrt(latent["variance"])))
    return w*po+(1-w)*pl


def run(seed,scenario):
    rng=np.random.default_rng(seed)
    c=CausalSOCController(REGRESSION_SETTINGS)
    rows=[]
    for t in range(STEPS):
        design=design_from_rng(rng)
        one,lat=analytic_forecasts(design)
        prefix=f"{scenario}:{seed}:{t}"
        proposal=c.register(f"design:{seed}:{t}",prefix,one,lat)
        weight=proposal["weight_at_proposal"]
        # In the experiment timeline this true outcome/noise is drawn
        # only AFTER both GP-view analog forecasts were registered.
        g=truth_location(scenario,design,t)+float(rng.normal(0,.16))
        observation=observe(g,t)
        po=event_probability(one,lat,1.)
        pl=event_probability(one,lat,0.)
        pessimistic=1. if po<=pl else 0.
        views={"h2":weight,"onehot":1.,"latent":0.,"min_guard":pessimistic}
        independent={}
        for arm,w in views.items():
            scored=mix_log_score(observation,one,lat,w)
            p=event_probability(one,lat,w)
            truth=bool(g<=0)  # synthetic true label, NOT used in H2 update
            # Resolved labels follow only observed evidence. The synthetic
            # oracle truth is recorded separately for the benchmark audit.
            resolved=observation.constraint_satisfaction()
            independent[arm]={"weight":w,"nll":scored["nll"],
                               "kind":scored["type"],
                               "brier":(p-float(resolved))**2
                               if resolved is not None else None,
                               "synthetic_truth":truth}
        settled=c.settle(f"design:{seed}:{t}",prefix,observation)
        rows.append({"scenario":scenario,"seed":seed,"t":t,
                     "material":design["material"],"ordinal":design["ordinal"],
                     "active_z":design["active_z"],
                     "observed_kind":observation.kind,
                     "observed_event":observation.constraint_satisfaction(),
                     "controller_weight_before":weight,
                     "controller_weight_after":settled["weight_after"],
                     "controller_delta_capped":settled["delta_capped"],
                     "forecasts":independent})
    assert len(c.events)==STEPS and c.n_scored==STEPS and c.pending is None
    return rows,c.snapshot()


def summarize(rows):
    summary=[]
    for scenario in SCENARIOS:
        for kind in ("exact_density","interval_mass"):
            for arm in ("h2","onehot","latent","min_guard"):
                subset=[r["forecasts"][arm] for r in rows
                        if r["scenario"]==scenario and
                           r["forecasts"][arm]["kind"]==kind]
                n=len(subset)
                summary.append({"scenario":scenario,"score_type":kind,
                    "arm":arm,"n":n,
                    "mean_nll":sum(x["nll"] for x in subset)/n if n else None,
                    "mean_brier_resolved":sum(
                        x["brier"] for x in subset if x["brier"] is not None)/
                        sum(x["brier"] is not None for x in subset)
                        if any(x["brier"] is not None for x in subset) else None,
                    "benchmark_scope":"Analytic synthetic forecasts, shared sequence; not BO/GP model training"})
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("Write-once synthetic output path required")
    rows=[]
    final=[]
    for scenario in SCENARIOS:
        for seed in SEEDS:
            values,last=run(seed,scenario)
            rows.extend(values)
            final.append({"scenario":scenario,"seed":seed,**last})
    assert len(rows)==len(SCENARIOS)*len(SEEDS)*STEPS
    summary=summarize(rows)
    args.output.mkdir(parents=True)
    for path,data in [
        ("registered_design.json",{
            "schema":"ra_h2_hierarchical_censored_synthetic_v1",
            "scenarios":SCENARIOS,"seeds":SEEDS,"steps":STEPS,
            "controller":asdict(REGRESSION_SETTINGS),
            "model_std":MODEL_STD,"prediction_inputs":"design_only",
            "truth_generation":"after_prediction_commitment",
            "new_aircraft_calls":0,"new_RCAIDE_calls":0,"new_GP_fits":0,
            "meaning":"forecast-stream stress test, not full BO or fitted GP proof"}),
        ("summary.json",summary),("final_controller_states.json",final),
        ("trajectory.json",rows)]:
        with (args.output/path).open("x",encoding="utf8") as f:
            json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
    print(json.dumps({"total_synthetic_observations":len(rows),
          "controller_updates":len(rows),
          "exact":sum(r["observed_kind"]=="exact" for r in rows),
          "interval":sum(r["observed_kind"]=="interval" for r in rows),
          "by_scenario":summary},allow_nan=False))


if __name__=="__main__":
    main()
