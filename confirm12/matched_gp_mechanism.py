"""Matched GP view diagnosis from PR45 shared initial real evaluations, NO aircraft calls.

For each case, choose training seeds by sorted frozen seed identity (outcome
blind). Train with exactly their 12 shared initial physical records. Evaluate
both response representations on the 12 shared initial physical records of a
different, disjoint held-out seed. The holdout observations are not used by the
VAE, GP, kernel evidence selection, calibration or training. No method-specific
online trajectories are used. The full four-arm 2x2 allocation reuses the same
two GP posteriors: differences are exclusively input representation allocation.

This is a retrospective fixed-input **prediction** experiment; NOT proof that
the original 151006-call optimizer would choose better designs on this panel.
"""
from collections import defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import time

import numpy as np
from scipy.special import ndtr

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.experiments.simplified_aircraft.dualview_pilot import BASE, model_settings, profile_for
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import HeterogeneousRiskPosterior
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.structure_corpus import CorpusRepresentation, feature_snapshot, native_observations
from ra_lsbo.structured_encoder import hierarchy_space
from ra_lsbo.experiments.simplified_aircraft.corpus_optimizer import encoder_settings
from ra_lsbo.experiments.core.runner import plain
from ra_lsbo.features import load_base

from .postrun_pr45 import (
    CASES, _read, _sqlite, FROZEN_SOURCE_SHA,
    FROZEN_PLAN_SHA256, FROZEN_REAL_SQLITE_SHA256,
    verify_roster,
)

SCHEMA="ra_lsbo_two_by_two_independent_holdout_gp_v1"
ARMS=("ra_lsbo","full_latent","full_onehot","reverse_representation")
VIEWS=("latent","onehot")
EPS=1e-12


def fixed_pairs(plan, per_case=8):
    """Use disjoint consecutive frozen seeds, never select by their labels."""
    if type(per_case) is not int or not 1 <= per_case <= 128:
        raise ValueError("Use 1..128 prespecified pairs per task")
    verify_roster(plan)
    result=[]
    for case in CASES:
        seeds=sorted(plan["cases"][case]["seeds"])
        for i in range(per_case):
            result.append((case,seeds[i],seeds[i+128]))
    return result


def _read_initial(db, case, seed, contract):
    rows=db.execute(
        "SELECT idx,record FROM calls WHERE case_name=? AND seed=? "
        "AND role='initial' AND state='settled' ORDER BY idx",
        (case,seed)).fetchall()
    if [row[0] for row in rows]!=list(range(12)):
        raise ValueError(f"Missing complete 12 actual initial calls: {case}/{seed}")
    records=[DesignRecord.from_dict(json.loads(blob)) for _,blob in rows]
    if any(r.observation.contract_id!=contract for r in records):
        raise ValueError("Foreign physical contract in shared initial records")
    if any(r.observation.status in ("unknown","infrastructure_error") or
           r.diagnostics.get("numerical_failure") for r in records):
        raise ValueError("Untrusted shared initial record")
    return records


def _train_views(problem, seed, training):
    """Exactly reproduce PR45's i=0 corpus training and onehot feature build."""
    from dataclasses import asdict as dc
    modules,_=load_base(BASE)
    space=hierarchy_space(modules,[dc(v) for v in problem.space.variables])
    session=CorpusRepresentation(space,seed,count=2000,settings=encoder_settings())
    session.fit(native_observations(training,modules,space),incremental=False)
    latent=feature_snapshot(problem,BASE,training,session)
    explicit=PrefixFeatures(problem,BASE,training,seed,False,30)
    settings=model_settings("hybrid_onehot",profile_for("pilot"))
    gp_latent=HeterogeneousRiskPosterior(
        problem,training,seed,settings,latent,latent,
        view_policy="matched_holdout_all_latent")
    gp_onehot=HeterogeneousRiskPosterior(
        problem,training,seed,settings,explicit,explicit,
        view_policy="matched_holdout_all_onehot")
    return {"latent":gp_latent,"onehot":gp_onehot}, {
        "corpus_identity":session.corpus["identity"],
        "representation_fit_count":len(session.events),
        "representation_initial_epochs":session.events[0]["epochs"],
        "latent_feature_identity":latent.identity,
        "onehot_feature_identity":explicit.identity,
        "fit_on_train_only":True,
    }


def _gaussian_probability(mean, variance):
    mu=np.asarray(mean,float);var=np.asarray(variance,float)
    if (mu.shape!=var.shape or not np.isfinite(mu).all() or
            not np.isfinite(var).all() or (var<0).any()):
        raise ValueError("Invalid Gaussian posterior")
    return ndtr(-mu/np.sqrt(np.maximum(var,EPS)))


def _completion_probability(pred):
    mu=np.asarray(pred["mean"],float)
    var=np.asarray(pred["variance"],float)
    if (mu.shape!=var.shape or not np.isfinite(mu).all() or
            not np.isfinite(var).all() or (var<0).any()):
        raise ValueError("Invalid completion GP")
    # PR45 completion observation is a probit threshold with unit noise.
    return ndtr(-mu/np.sqrt(1.+var))


def _score_row(case,train_seed,holdout_seed,idx,design,head,view,
               observation,prediction,scale):
    valid=observation.constraint_satisfaction() if head!="objective" else None
    truth=observation.value if observation.kind=="exact" else None
    mu=float(prediction["mean"][idx])
    variance=float(prediction["variance"][idx])
    if not math.isfinite(mu) or not math.isfinite(variance) or variance<0:
        raise ValueError("Nonfinite posterior prediction")
    p=float(_gaussian_probability([mu],[variance])[0]) if head!="objective" else None
    return {
        "case":case,"train_seed":train_seed,"holdout_seed":holdout_seed,
        "heldout_index":idx,"candidate_id":digest(design),
        "head":head,"view":view,
        "observed_kind":observation.kind,"exact_target":truth,
        "known_feasible":valid,
        "mean":mu,"variance":variance,"p_feasible":p,
        "brier":None if valid is None else (p-int(valid))**2,
        "false_feasible":valid is False and p>=.5 if valid is not None else None,
        "mae":abs(mu-truth) if truth is not None else None,
        "squared_error":(mu-truth)**2 if truth is not None else None,
        "near_boundary":(abs(truth)<=.2*scale if truth is not None and head!="objective" else None),
        "probability_label":"P(g<=0)" if head!="objective" else None,
    }


def score_pair(case, seed, holdout_seed, training, targets):
    """Two model fits, four prospective 2x2 combinations, same heldout labels."""
    problem=problem_for(case)
    if len(training)!=12 or len(targets)!=12:
        raise ValueError("Exactly 12 train and 12 holdout records required")
    training_ids={digest(r.candidate) for r in training}
    if any(digest(row.candidate) in training_ids for row in targets):
        raise ValueError("Heldout design was seen in training; refuse leakage")
    t0=time.perf_counter()
    fitted,identities=_train_views(problem,seed,training)
    training_hash=digest([r.to_dict() for r in training])
    holdout_hash=digest([r.to_dict() for r in targets])
    predictions={view:fitted[view].predict([r.candidate for r in targets]) for view in VIEWS}
    rows=[]
    for view,packet in predictions.items():
        objective=packet["objective"]
        if objective is not None:
            for i,row in enumerate(targets):
                rows.append(_score_row(
                    case,seed,holdout_seed,i,row.candidate,"objective",view,
                    row.observation.objective,objective,
                    fitted[view].objective.scale))
        for name,pred in packet["constraints"].items():
            for i,row in enumerate(targets):
                rows.append(_score_row(
                    case,seed,holdout_seed,i,row.candidate,name,view,
                    row.observation.constraint(name),pred,pred["scale"]))
        completion=_completion_probability(packet["completion"])
        for i,row in enumerate(targets):
            label=row.observation.completion_label()
            rows.append({
                "case":case,"train_seed":seed,"holdout_seed":holdout_seed,
                "heldout_index":i,"candidate_id":digest(row.candidate),
                "head":"completion","view":view,"observed_kind":"binary_or_missing",
                "exact_target":label,"known_feasible":bool(label) if label is not None else None,
                "mean":float(packet["completion"]["mean"][i]),
                "variance":float(packet["completion"]["variance"][i]),
                "p_feasible":float(completion[i]),
                "brier":None if label is None else float((completion[i]-label)**2),
                "false_feasible":bool(label==0 and completion[i]>=.5) if label is not None else None,
                "mae":None,"squared_error":None,"near_boundary":None,
                "probability_label":"P(completed)",
            })
    # The 2x2 method combinations are exactly recombinations of independent
    # latent/onehot objective/risk heads; no unnecessary third/fourth GP fit.
    arms={
        "ra_lsbo":{"objective":"latent","risk":"onehot"},
        "full_latent":{"objective":"latent","risk":"latent"},
        "full_onehot":{"objective":"onehot","risk":"onehot"},
        "reverse_representation":{"objective":"onehot","risk":"latent"},
    }
    for arm,views in arms.items():
        if not (views["objective"] in fitted and views["risk"] in fitted):
            raise AssertionError("Unknown 2x2 composition")
    return rows,{
        "case":case,"train_seed":seed,"holdout_seed":holdout_seed,
        "train_history_digest":training_hash,"holdout_history_digest":holdout_hash,
        "same_heldout_designs_for_both_models":True,
        "same_12_training_evaluations_for_both_models":True,
        "no_online_arm_outcomes_in_training":True,
        "actual_representation_allocation":arms,
        "fitted_models":2,
        "elapsed_seconds":time.perf_counter()-t0,
        **identities,
    }


def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row["case"],row["head"],row["view"])].append(row)
    summary={}
    for (case,head,view),records in sorted(groups.items()):
        by_seed=defaultdict(list)
        for r in records:by_seed[r["train_seed"]].append(r)
        name=f"{case}/{head}/{view}"
        observed=[r for r in records if r["brier"] is not None]
        exact=[r for r in records if r["squared_error"] is not None]
        boundary=[r for r in records if r["near_boundary"]]
        summary[name]={
            "records":len(records),
            "resolved_binary":len(observed),
            "exact_numeric":len(exact),
            "near_boundary_exact":len(boundary),
            "mean_brier":float(np.mean([r["brier"] for r in observed])) if observed else None,
            "rmse":float(np.sqrt(np.mean([r["squared_error"] for r in exact]))) if exact else None,
            "near_boundary_brier":float(np.mean([r["brier"] for r in boundary
                                                   if r["brier"] is not None])) if boundary else None,
            "false_feasible_count_at_0_5":sum(r["false_feasible"] is True for r in observed),
            "clustered_by_training_seed":len(by_seed),
            "uncertainty":"descriptive only; resample independent seed-pairs for CIs",
        }
    return summary


def paired_view_differences(rows):
    """Compute within-target deltas, with independent training seeds as clusters."""
    keyed=defaultdict(dict)
    for r in rows:
        key=(r["case"],r["head"],r["train_seed"],r["holdout_seed"],
             r["heldout_index"],r["candidate_id"])
        if r["view"] in keyed[key]:
            raise ValueError("Duplicate same-view prediction")
        keyed[key][r["view"]]=r
    groups=defaultdict(lambda:defaultdict(list))
    for (case,head,train_seed,_,_,_),views in keyed.items():
        if set(views)!=set(VIEWS):
            raise ValueError("Unpaired GP representation target")
        a,b=views["latent"],views["onehot"]
        for metric in ("brier","squared_error"):
            va,vb=a[metric],b[metric]
            if (va is None)!=(vb is None):
                raise ValueError("Paired views see inconsistent heldout evidence")
            if va is not None:
                groups[(case,head,metric)][train_seed].append(float(va-vb))
    report={}
    for (case,head,metric),clusters in sorted(groups.items()):
        per_seed=[statistics.mean(v) for _,v in sorted(clusters.items())]
        if not per_seed:continue
        k=len(per_seed)
        mean=float(statistics.mean(per_seed))
        sem=(statistics.stdev(per_seed)/math.sqrt(k) if k>1 else None)
        report[f"{case}/{head}/{metric}"]={
            "metric":"latent_minus_onehot",
            "mean_of_per_seed_mean_differences":mean,
            "cluster_count":k,
            "independent_unit":"training_seed_and_disjoint_holdout_seed_pair",
            "normal_approx_cluster_95_interval":(
                [mean-1.96*sem,mean+1.96*sem] if sem is not None else None),
            "direction":"negative favors latent representation",
            "posthoc_repeated_targets_not_iid":True,
        }
    return report


def run(original,output,*,per_case=8):
    """Only reads the existing immutable PR45 ledger; zero expensive calls."""
    # Explicit no-physical-call guard even if future refactors import an evaluator.
    from ra_lsbo.experiments.core.problems import AircraftCase
    from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft
    def prohibited(*args,**kwargs):
        raise RuntimeError("MECHANISM ANALYSIS FORBIDS AIRCRAFT EVALUATIONS")
    AircraftCase.evaluate=prohibited
    SimplifiedElectricAircraft.evaluate=prohibited
    original=Path(original).resolve()
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("Output must be a new directory")
    plan=_read(original/"plan.json")
    if plan.get("source_commit")!=FROZEN_SOURCE_SHA:
        raise ValueError("Wrong frozen PR45 campaign source")
    if hashlib.sha256((original/"plan.json").read_bytes()).hexdigest()!=FROZEN_PLAN_SHA256:
        raise ValueError("Wrong original plan bytes")
    expected=verify_roster(plan)
    pairs=fixed_pairs(plan,per_case)
    if any((case,seed) not in expected or (case,target) not in expected
           for case,seed,target in pairs):
        raise AssertionError("Unregistered pair")
    # We deliberately require the full database already independently audited
    # by PR46, and use its read-only bytes/SQLITE identity here.
    if hashlib.sha256((original/"real_calls.sqlite").read_bytes()).hexdigest()!=FROZEN_REAL_SQLITE_SHA256:
        raise ValueError("Original PR45 SQLite differs from Release original")
    results=[];receipts=[]
    output.mkdir(parents=True,exist_ok=False)
    with _sqlite(original/"real_calls.sqlite") as db:
        for case,seed,target_seed in pairs:
            contract=plan["cases"][case]["contracts"]["20s"]
            train=_read_initial(db,case,seed,contract)
            heldout=_read_initial(db,case,target_seed,contract)
            try:
                rows,receipt=score_pair(case,seed,target_seed,train,heldout)
            except Exception as exc:
                with (output/"failure.json").open("x",encoding="utf8") as f:
                    json.dump({"case":case,"seed":seed,"holdout_seed":target_seed,
                               "error_type":type(exc).__name__,
                               "message":str(exc)},f,indent=2)
                raise
            results.extend(rows);receipts.append(receipt)
            print("matched_gp:",case,seed,"=>",target_seed,flush=True)
    columns=list(results[0])
    with (output/"matched_gp_predictions.csv").open("x",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader();writer.writerows(results)
    report={
        "schema":SCHEMA,"source_sha":plan["source_commit"],
        "original_source_read_only":True,
        "new_aircraft_calls":0,"new_rcaide_calls":0,
        "training_seed_pairs":len(pairs),"target_points":12*len(pairs),
        "training_only_12_original_shared_initials":True,
        "heldout_only_disjoint_other_seed_initials":True,
        "2x2_allocation_from_two_independent_GP_view_models":True,
        "methods":list(ARMS),"model_heads":aggregate(results),
        "paired_view_differences":paired_view_differences(results),
        "receipts":receipts,
        "notes":[
            "Retrospective independent-seed initial holdout, not original online trajectories.",
            "Numerical GP errors are preserved, not converted to feasible predictions.",
            "No significance claims or selective task/seed filtering.",
            "Comparison of latent/onehot predictions on identical target designs; not causal online optimization effect.",
            "Shared initial samples have not established 140km sparse eligibility.",
        ],
    }
    with (output/"matched_gp_report.json").open("x",encoding="utf8") as f:
        json.dump(plain(report),f,ensure_ascii=False,indent=2,allow_nan=False)
    return report
