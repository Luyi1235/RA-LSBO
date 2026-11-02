"""Deterministic toy decision loop for causal H2 risk weighting.

This is an analytic-forecaster mathematical OPTIMIZATION TEST, not an
aircraft BO run and not an EP/GP benchmarking claim. Four declared arms
share each seed/time/candidate pool, with selected outcomes observed
strictly after proposal commitment. No simulator/RCAIDE or GP fit.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

# Direct execution from tools/ must still import the repository package.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.run_h2_hierarchical_synthetic import (
    SCENARIOS,design_from_rng,analytic_forecasts,truth_location,observe,
    event_probability)
from ra_lsbo.experiments.simplified_aircraft.h2_risk_controller import (
    CausalSOCController,H2Settings)

ARMS=("onehot","latent","min_guard","h2")
SEEDS=tuple(range(17001,17009))
STEPS=24
POOL=64
SCOPE="mathematical_analytic_surrogate_not_GP"


def stable_seed(parts):
    data=json.dumps(parts,sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8],"big")


def pool_for(scenario,seed,step):
    rng=np.random.default_rng(stable_seed([scenario,seed,step,"pool"]))
    return [design_from_rng(rng) for _ in range(POOL)]


def objective_proxy(design):
    m=design["material"]
    x=design["x"]
    # Lower Wh surrogate corresponds to an energy-efficient, potentially
    # riskier design. Same function for all four arms.
    return (x-.85)**2+.07*(m==1)+.02*(design["ordinal"]-2)**2 + \
           .08*design["active_z"]*(design["z"]-.6)**2


def selected_truth(scenario,seed,t,design):
    # Deterministic design-specific shock; same (design,step,seed) yields
    # the same answer regardless of which arm selects that candidate.
    rng=np.random.default_rng(stable_seed([scenario,seed,t,design,"outcome"]))
    g=truth_location(scenario,design,t)+.16*float(rng.normal())
    e=objective_proxy(design)+.05*float(rng.uniform())+.02
    return g,e


def choose(arm,weight,candidates):
    scores=[]
    for design in candidates:
        one,lat=analytic_forecasts(design)
        p1=event_probability(one,lat,1.)
        p2=event_probability(one,lat,0.)
        w=weight if arm=="h2" else (
            1. if arm=="onehot" else
            0. if arm=="latent" else
            (1. if p1<p2 else 0.))
        p=event_probability(one,lat,w)
        # Fixed probability-weighted mathematical acquisition proxy,
        # not the original RA-LSBO acquisition.
        score=-objective_proxy(design) + .12*math.log(max(1e-12,p))
        scores.append(score)
    idx=int(np.argmax(scores))
    return idx


def run_trial(scenario,seed,arm):
    c=CausalSOCController(H2Settings()) if arm=="h2" else None
    traj=[]
    incumbent=None
    feasible=0
    for t in range(STEPS):
        pool=pool_for(scenario,seed,t)
        weight=c.weight if c is not None else None
        idx=choose(arm,weight,pool)
        design=pool[idx]
        one,lat=analytic_forecasts(design)
        prefix=f"{scenario}:{seed}:{arm}:{t}"
        if c is not None:
            p=c.register(f"selected:{seed}:{arm}:{t}",prefix,one,lat)
            assert p["weight_at_proposal"]==weight
        # No true objective/SOC or censoring kind was accessed by choose.
        g,energy=selected_truth(scenario,seed,t,design)
        outcome=observe(g,t)
        is_feasible=g<=0
        feasible+=int(is_feasible)
        if is_feasible:
            incumbent=energy if incumbent is None else min(incumbent,energy)
        event=(c.settle(f"selected:{seed}:{arm}:{t}",prefix,outcome)
               if c is not None else None)
        traj.append({
            "scenario":scenario,"seed":seed,"arm":arm,"online_t":t,
            "candidate_id":hashlib.sha256(
                json.dumps(design,sort_keys=True).encode()).hexdigest(),
            "candidate":design,
            "true_feasible_synthetic_only":is_feasible,
            "observation_kind":outcome.kind,
            "best_feasible_proxy_energy":incumbent,
            "H2_weight_before":weight,
            "H2_weight_after":c.weight if c else None,
            "H2_delta_clipped":event["delta_capped"] if event else None,
        })
    return {"scenario":scenario,"seed":seed,"arm":arm,
            "feasible_count":feasible,"has_feasible":incumbent is not None,
            "final_best_proxy_energy":incumbent,
            "H2_final_weight":c.weight if c else None,
            "synthetic_calls":STEPS,
            "censored_count":sum(x["observation_kind"]=="interval" for x in traj)},traj


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("Use a new synthetic evidence directory")
    results=[]
    records=[]
    for scenario in SCENARIOS:
        for seed in SEEDS:
            for arm in ARMS:
                row,trajectory=run_trial(scenario,seed,arm)
                results.append(row)
                records.extend(trajectory)
    expected=len(SCENARIOS)*len(SEEDS)*len(ARMS)*STEPS
    assert len(records)==expected
    summaries=[]
    for scenario in SCENARIOS:
        for arm in ARMS:
            group=[r for r in results if r["scenario"]==scenario and r["arm"]==arm]
            energy=[r["final_best_proxy_energy"] for r in group
                    if r["final_best_proxy_energy"] is not None]
            summaries.append({"scenario":scenario,"arm":arm,
                "seeds":len(group),"feasible_trials":len(energy),
                "mean_feasible_proxy_energy":sum(energy)/len(energy) if energy else None,
                "total_feasible_designs":sum(r["feasible_count"] for r in group),
                "total_censored_results":sum(r["censored_count"] for r in group),
                "inferential_scope":"toy on-policy proxy objective, not GP or aircraft claims"})
    args.output.mkdir(parents=True)
    for name,value in (
        ("registration.json",{
            "schema":"ra_h2_synthetic_online_decision_v1",
            "scenarios":SCENARIOS,"seeds":SEEDS,"arms":ARMS,
            "steps":STEPS,"pool_size":POOL,
            "total_synthetic_truth_calls":expected,
            "controller":asdict(H2Settings()),
            "fixed_scoring":"-energy_proxy + .12 log(p_feasible)",
            "new_aircraft_calls":0,"new_GP_fits":0,
            "new_RCAIDE_calls":0,"scope":SCOPE}),
        ("trial_summaries.json",results),("arm_comparisons.json",summaries),
        ("online_trajectories.json",records)):
        with (args.output/name).open("x",encoding="utf-8") as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
    print(json.dumps({"new_aircraft_calls":0,"new_RCAIDE_calls":0,
          "synthetic_truth_calls":expected,"summaries":summaries},allow_nan=False))


if __name__=="__main__":
    main()
