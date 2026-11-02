"""Fixed common-stream ablation of the TWO opt-in RA-LSBO-H2 controllers.

This is a deterministic analytic-forecast, synthetic-outcome comparison,
not an actual trained GP, BO, airplane or independent physical study.

All alternatives face exactly the same candidate AND observed SOC label at
each step. Feedback for H2-B hard-head Brier and H2-M Gaussian mixture is
applied only AFTER both forecasts/choices have been committed.

The two registered controllers have DIFFERENT fixed prior/warmup/learning
rates, so differences cannot be attributed solely to the scoring rule.
"""
from __future__ import annotations
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.special import logsumexp,ndtr

from ra_lsbo.experiments.simplified_aircraft.h2_controller import (
    CausalRiskController,Gaussian,H2Settings as H2BSettings)
from ra_lsbo.experiments.simplified_aircraft.h2_risk_controller import (
    CausalSOCController,H2Settings as H2MSettings,proper_nll)
from tools.run_h2_hierarchical_synthetic import (
    SCENARIOS,SEEDS,STEPS,design_from_rng,analytic_forecasts,
    truth_location,observe)

ARMS=("onehot","latent","min_guard","H2B_hard_Brier","H2M_log_mixture")
SCHEMA="ra_lsbo_h2_common_forecast_stream_v1"


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _p(model):
    return float(ndtr(-float(model["mean"])/math.sqrt(float(model["variance"]))))


def _score(observation,one,lat,prior_B,weight_M):
    a,b=proper_nll(one,observation),proper_nll(lat,observation)
    if a["type"]!=b["type"]:
        raise ValueError("Paired expert scores cannot have different measures")
    lower="onehot" if _p(one)<=_p(lat) else "latent"
    selected=one if prior_B=="onehot" else lat
    h2b=proper_nll(selected,observation)
    score_M=float(-logsumexp((math.log(weight_M)-a["nll"],
                              math.log1p(-weight_M)-b["nll"])))
    feasible=observation.constraint_satisfaction()
    probabilities={
        "onehot":_p(one),"latent":_p(lat),
        "min_guard":min(_p(one),_p(lat)),
        "H2B_hard_Brier":_p(selected),
        "H2M_log_mixture":weight_M*_p(one)+(1-weight_M)*_p(lat),
    }
    outputs={
        "onehot":a["nll"],
        "latent":b["nll"],
        "min_guard":(a if lower=="onehot" else b)["nll"],
        "H2B_hard_Brier":h2b["nll"],
        "H2M_log_mixture":score_M,
    }
    return {
        "score_type":a["type"],
        "NLL":outputs,
        "feasibility_probabilities":probabilities,
        "Brier":{arm:(
            (p-float(feasible))**2 if feasible is not None else None)
            for arm,p in probabilities.items()},
        "H2B_selected_view":prior_B,
        "H2M_prior_onehot_weight":weight_M,
        "min_guard_selected_view":lower,
    }


def run(seed,scenario):
    generator=np.random.default_rng(seed)
    brier=CausalRiskController(H2BSettings(max_steps=STEPS))
    mixture=CausalSOCController(H2MSettings())
    rows=[]
    for t in range(STEPS):
        d=design_from_rng(generator)
        one,lat=analytic_forecasts(d)
        candidate_id=digest(d)
        prefix_id=digest({"seed":seed,"scenario":scenario,"t":t,
                          "past_event_ids":[r["id"] for r in rows]})
        priorB=brier.begin(
            t,candidate_id,prefix_id,
            {"onehot":Gaussian(**one),"latent":Gaussian(**lat)})
        priorM=mixture.register(candidate_id,prefix_id,one,lat)
        # After BOTH prior predictions are irrevocably committed, reveal
        # the same synthetic oracle label for ALL five scoring strategies.
        true_g=truth_location(scenario,d,t)+float(generator.normal(0,.16))
        obs=observe(true_g,t)
        result=_score(obs,one,lat,priorB["selected_view"],priorM["weight_at_proposal"])
        bsettle=brier.settle(t,candidate_id,obs)
        msettle=mixture.settle(candidate_id,prefix_id,obs)
        rows.append({
            "schema":SCHEMA,"id":candidate_id,"scenario":scenario,"seed":seed,
            "step":t,"design":d,"SOC_kind":obs.kind,
            "SOC_event_resolved":obs.constraint_satisfaction(),
            "H2B_commit":priorB["commitment_sha256"],
            "H2M_commit":priorM["commitment"],
            "H2B_update_scored":bsettle["feedback_pair_scored"],
            "H2M_delta_capped":msettle["delta_capped"],
            "H2B_weight_after":brier.state()["onehot_weight"],
            "H2M_weight_after":mixture.weight,
            **result,
        })
    if len(brier.journal()["events"])!=STEPS or len(mixture.events)!=STEPS:
        raise ValueError("Synthetic future label/settlement chronology was lost")
    return rows


def benchmark():
    records=[]
    for scenario in SCENARIOS:
        for seed in SEEDS:
            records.extend(run(seed,scenario))
    expected=len(SCENARIOS)*len(SEEDS)*STEPS
    if len(records)!=expected:
        raise ValueError("Expected exactly one shared-outcome observation per seed/time")
    summary=[]
    for scenario in SCENARIOS:
        for kind in ("exact_density","interval_mass"):
            part=[x for x in records if x["scenario"]==scenario and x["score_type"]==kind]
            for arm in ARMS:
                brier=[x["Brier"][arm] for x in part if x["Brier"][arm] is not None]
                summary.append({
                    "scenario":scenario,"score_type":kind,"arm":arm,
                    "n":len(part),
                    "mean_NLL":sum(x["NLL"][arm] for x in part)/len(part) if part else None,
                    "median_NLL":float(np.median([x["NLL"][arm] for x in part])) if part else None,
                    "mean_resolved_Brier":sum(brier)/len(brier) if brier else None,
                    "trial_seed_clusters":len(SEEDS),
                })
    return {
        "schema":SCHEMA,
        "plan":{"seeds":list(SEEDS),"cases":list(SCENARIOS),
                "steps_per_case_seed":STEPS,"strategy_roster":list(ARMS),
                "H2B":asdict(H2BSettings(max_steps=STEPS)),
                "H2M":asdict(H2MSettings()),
                "synthetic_oracle_calls":expected,
                "new_aircraft_calls":0,"new_RCAIDE_calls":0,
                "new_GP_fits":0,"new_VAE_fits":0},
        "summaries":summary,
        "rows":records,
        "scientific_limit":"Synthetic analytic forecast emulator; common stream, not BO/GP training, not safety or algorithm superiority",
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("New write-once synthetic comparison output required")
    result=benchmark()
    args.output.mkdir(parents=True)
    for name,value in (("registration.json",result["plan"]),
                       ("summaries.json",result["summaries"]),
                       ("all_3072_shared_outcomes.json",result["rows"]),
                       ("limitations.json",{"scope":result["scientific_limit"]})):
        with (args.output/name).open("x",encoding="utf8") as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
    print(json.dumps({"scenario_count":len(SCENARIOS),
        "seed_count":len(SEEDS),"matched_synthetic_truth_calls":len(result["rows"]),
        "arm_count":len(ARMS),"summary":result["summaries"]},allow_nan=False))


if __name__=="__main__":
    main()
