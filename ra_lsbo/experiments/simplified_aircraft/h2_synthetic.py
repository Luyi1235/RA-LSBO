"""Deterministic synthetic hierarchy/censoring benchmark for the H2 controller.

This is a *controller mechanism test*, not a surrogate-model accuracy
benchmark and not an RA-LSBO aircraft study. Its two forecasts are explicit
fixed-form expert emulators; no VAE or GP fitting occurs. The oracle
response is revealed only after the selected candidate is committed.

Synthetic regimes:
- smooth_transfer: latent structural smoothing is more accurate
- discrete_switch: explicit family/ordinal effects dominate
- delayed_shift: after a fixed step, family-specific SOC effects change

Every arm gets the same seed/iteration candidate pool and fixed evaluator.
All label censoring, missing evidence and response-head outages are logged.
"""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
import argparse
import json
import math
from pathlib import Path

import numpy as np

from ...risk_core.observations import OutputObservation
from .h2_controller import CausalRiskController, Gaussian, H2Settings, digest

CASES=("smooth_transfer","discrete_switch","delayed_shift")
ARMS=("onehot","latent","min_feasibility","h2_causal")
SEEDS=tuple(range(7301,7309))
HORIZON=24
POOL=64
FAMILIES=("aluminum","carbon","titanium","polymer")
LEVELS=(0,1,2)


def _candidate_pool(seed,step):
    rng=np.random.default_rng(seed*10000+step+7150000)
    out=[]
    for i in range(POOL):
        family=FAMILIES[int(rng.integers(len(FAMILIES)))]
        level=int(rng.integers(3))
        mode="sport" if rng.integers(2) else "eco"
        x=float(rng.uniform(.03,.97))
        z=float(rng.uniform(.05,.95)) if mode=="sport" else None
        out.append({"family":family,"level":level,"mode":mode,
                    "x":x,"conditional_z":z})
    return out


def _true_margin(candidate,case,step):
    """Synthetic oracle, never imported or called by risk prediction."""
    x,f,level,z=(candidate["x"],FAMILIES.index(candidate["family"]),
                 candidate["level"],candidate["conditional_z"])
    active_term=.17*(z-.5) if z is not None else 0.
    if case=="smooth_transfer" or case=="delayed_shift" and step<10:
        family_term=.04*(f-1.5)
    elif case in ("discrete_switch","delayed_shift"):
        family_term=(.10,-.16,.24,-.08)[f]
    else:
        raise ValueError("Unknown benchmark case")
    # no unmeasured noise in ground-truth margin; same candidate -> same
    # truth given case/time. Stochasticity enters through predefined pools.
    return .48-x+.11*(level-1)+family_term+active_term


def _objective(candidate):
    f=FAMILIES.index(candidate["family"])
    return (float(.7+.9*candidate["x"]+.05*candidate["level"]
             +(.02,-.03,.09,.01)[f]
             +(.07 if candidate["mode"]=="sport" else 0.)))


def _expert_predictions(candidate):
    """Fixed-form risk experts, independent of the latent true oracle.

    The latent view assumes smooth family transfer. The onehot view uses
    explicit family offsets, with a small fixed approximation residual.
    The scenario truth can shift without any change to these forecasts.
    """
    x,f,level,z=(candidate["x"],FAMILIES.index(candidate["family"]),
                 candidate["level"],candidate["conditional_z"])
    condition=.17*(z-.5) if z is not None else 0.
    onehot=.48-x+.11*(level-1)+(.11,-.15,.23,-.07)[f]+condition+.01
    latent=.48-x+.11*(level-1)+.04*(f-1.5)+condition-.01
    return {"onehot":Gaussian(float(onehot),.07**2),
            "latent":Gaussian(float(latent),.07**2)}


def _observe(margin,case,seed,step,arm):
    # Unknown -> no invented label. Censor positive SOC violations to a
    # one-sided interval so a correctly resolved sign remains available.
    if step%11==10:
        return OutputObservation("missing",f"{case}/{seed}/{arm}/{step}",
                                 reason="synthetic_missing_evidence")
    if margin>.02:
        return OutputObservation("interval",f"{case}/{seed}/{arm}/{step}",
                                 lower=0.,lower_inclusive=False)
    return OutputObservation("exact",f"{case}/{seed}/{arm}/{step}",value=margin)


def _active(forecasts,step):
    # Predeclared numerical outage injection, independent of labels.
    if step==5:
        forecasts["onehot"]=None
    if step==14:
        forecasts["onehot"]=None
        forecasts["latent"]=None
    return forecasts


def run_one(case,seed,method):
    controller=CausalRiskController(H2Settings(max_steps=HORIZON)) if method=="h2_causal" else None
    seen=set()
    best=None
    events=[]
    for t in range(HORIZON):
        pool=_candidate_pool(seed,t)
        # The static experts are queried for available designs only. No
        # oracle truth, future censor type or observed outcome is passed here.
        forecast=[_active(_expert_predictions(x),t) for x in pool]
        scores=[]
        for candidate,f in zip(pool,forecast):
            p={v:(None if f[v] is None else f[v].probability_feasible()) for v in ("onehot","latent")}
            if method=="onehot":
                risk=p["onehot"] if p["onehot"] is not None else p["latent"]
            elif method=="latent":
                risk=p["latent"] if p["latent"] is not None else p["onehot"]
            elif method=="min_feasibility":
                available=[v for v in p.values() if v is not None]
                risk=min(available) if available else None
            elif method=="h2_causal":
                # The controller's currently preferred head is based ONLY
                # on t-1 and earlier; no update from these pool candidates.
                head=controller.state()["currently_preferred"]
                risk=p[head] if p[head] is not None else p["latent" if head=="onehot" else "onehot"]
            else:
                raise ValueError("Unknown synthetic policy")
            # Risk-restoration score is deterministic; if both heads fail,
            # ignore risk and select a candidate by stable energy tie-break.
            score=_objective(candidate)+2.0*(1-risk) if risk is not None else _objective(candidate)
            scores.append(score)
        index=min(range(len(pool)),key=lambda i:(scores[i],digest(pool[i])))
        candidate=pool[index]
        identity=digest(candidate)
        if identity in seen:
            raise ValueError("Synthetic duplicate candidate")
        seen.add(identity)
        prior_state=controller.state() if controller else None
        commitment=(controller.begin(t,identity,digest(events),forecast[index])
                    if controller is not None else None)
        # The synthetic oracle is invoked ONLY after selection and commitment.
        true_margin=_true_margin(candidate,case,t)
        observation=_observe(true_margin,case,seed,t,method)
        event=controller.settle(t,identity,observation) if controller else None
        if observation.constraint_satisfaction() is True:
            value=_objective(candidate)
            best=value if best is None else min(best,value)
        events.append({
            "step":t,"candidate":candidate,"id":identity,
            "true_margin_synthetic":true_margin,
            "observable_SOC_kind":observation.kind,
            "feasible_evidence":observation.constraint_satisfaction(),
            "objective":_objective(candidate),
            "best_feasible":best,
            "model_outage":("both" if all(x is None for x in forecast[index].values()) else
                "onehot" if forecast[index]["onehot"] is None else None),
            "expert_means":{v:None if forecast[index][v] is None
                            else forecast[index][v].mean for v in ("onehot","latent")},
            "controller_prior":prior_state,
            "commitment_sha256":commitment["commitment_sha256"] if commitment else None,
            "decision_view":commitment["selected_view"] if commitment else method,
            "paired_update":event["feedback_pair_scored"] if event else None,
        })
    summary={
        "case":case,"seed":seed,"method":method,
        "synthetic_calls":HORIZON,
        "final_best_feasible":best,
        "feasible_observed":sum(e["feasible_evidence"] is True for e in events),
        "censored":sum(e["observable_SOC_kind"]=="interval" for e in events),
        "missing":sum(e["observable_SOC_kind"]=="missing" for e in events),
        "head_outages":sum(e["model_outage"] is not None for e in events),
        "h2_latent_chosen":sum(e["decision_view"]=="latent" for e in events) if controller else None,
        "h2_online_brier_updates":controller.state()["resolved_pairs"] if controller else None,
        "terminal_H2_state":controller.state() if controller else None,
        "no_gp_fits":True,"no_vae_fits":True,"no_aircraft":True,"no_RCAIDE":True,
    }
    return summary,events


def run_benchmark():
    trials=[]
    full=[]
    for case in CASES:
        for seed in SEEDS:
            for method in ARMS:
                summary,trace=run_one(case,seed,method)
                trials.append(summary)
                full.append({"case":case,"seed":seed,"method":method,"events":trace})
    return {
        "schema":"ra_lsbo_h2_synthetic_controller_mechanism_v1",
        "limitations":"Fixed-form forecast emulators, synthetic truth; not genuine learned GP or aircraft/BO validation",
        "trials":trials,"traces":full,
        "synthetic_oracle_calls":len(CASES)*len(SEEDS)*len(ARMS)*HORIZON,
        "GP_fits":0,"VAE_fits":0,"aircraft_calls":0,"RCAIDE_calls":0,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("Write-once benchmark output directory must be new")
    payload=run_benchmark()
    args.output.mkdir(parents=True)
    (args.output/"benchmark.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
    digest_sha=sha256((args.output/"benchmark.json").read_bytes()).hexdigest()
    groups=[]
    for case in CASES:
        for arm in ARMS:
            s=[r for r in payload["trials"] if r["case"]==case and r["method"]==arm]
            energies=[r["final_best_feasible"] for r in s if r["final_best_feasible"] is not None]
            groups.append({
                "case":case,"method":arm,
                "seeds":len(s),"feasible_runs":len(energies),
                "mean_final_feasible_objective":sum(energies)/len(energies) if energies else None,
                "mean_feasible_observed":sum(r["feasible_observed"] for r in s)/len(s),
                "mean_H2_latent_chosen":(sum(r["h2_latent_chosen"] for r in s)/len(s)
                                          if arm=="h2_causal" else None)})
    (args.output/"summary.json").write_text(
        json.dumps({"schema":payload["schema"],"groups":groups,
                    "benchmark_sha256":digest_sha,
                    "synthetic_truth_calls":payload["synthetic_oracle_calls"],
                    "new_GP_VAE_aircraft_calls":0},
                   indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps({"synthetic_oracle_calls":payload["synthetic_oracle_calls"],
                      "cases":list(CASES),"trials":len(payload["trials"]),
                      "trace_sha256":digest_sha,"groups":groups},sort_keys=True))


if __name__=="__main__":
    main()
