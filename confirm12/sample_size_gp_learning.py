"""Read-only fixed-qA sample-size learning curves for the RA-LSBO two GP views.

Goal: Determine whether poor n=12 risk/objective predictions improve after
more *real* response labels, without buying a single additional model call.

For each task and fold, train from an outcome-blind disjoint block of 16
PR45 IID shared-initial seed groups (12 records each); use a *different*
disjoint IID seed's 12 records as the common heldout set. Nested training
prefix sizes are exactly 12,24,48,96,192, with the same heldout design IDs
and targets at all levels. Models are freshly fitted from scratch per level
with the same 2000 unlabelled structural corpus and fixed GP/EP settings.

This is offline cross-seed pooling for diagnostic learning curves, NOT PR45
online algorithm evaluation; its larger training sets have different data
availability and non-comparable costs with the original 12+18 campaign.
"""
from collections import defaultdict
from contextlib import ExitStack
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.problems import AircraftCase
from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.experiments.core.runner import plain
from .matched_gp_mechanism import (
    VIEWS, _read_initial, _train_views, _score_row, _completion_probability,
    aggregate, paired_view_differences,
)
from .postrun_pr45 import (
    CASES, _read, _sqlite, verify_roster,
    FROZEN_PLAN_SHA256, FROZEN_SOURCE_SHA, FROZEN_REAL_SQLITE_SHA256,
)

SCHEMA="ra_lsbo_cross_seed_real_initial_learning_curve_v1"
SIZES=(12,24,48,96,192)
TRAIN_SEEDS_PER_FOLD=16
TEST_START_OFFSET=256
FOLDS_LIMIT=8


def fixed_study_plan(plan, *, folds=2, sizes=SIZES):
    """Plan before looking at any outcome; no feasibility-based seed selection."""
    verify_roster(plan)
    if type(folds) is not int or not 1<=folds<=FOLDS_LIMIT:
        raise ValueError("folds must be in 1..8")
    if (not isinstance(sizes,(tuple,list)) or not sizes or
            tuple(sizes)!=tuple(sorted(set(sizes))) or
            any(type(n) is not int or n not in SIZES for n in sizes)):
        raise ValueError("Require a nonempty, ascending subset of 12,24,48,96,192")
    tasks=[]
    for case in CASES:
        registered=sorted(plan["cases"][case]["seeds"])
        for fold in range(folds):
            block=registered[fold*TRAIN_SEEDS_PER_FOLD:(fold+1)*TRAIN_SEEDS_PER_FOLD]
            target=registered[TEST_START_OFFSET+fold]
            if len(block)!=TRAIN_SEEDS_PER_FOLD or target in block:
                raise AssertionError("Frozen training/target seed collision")
            tasks.append({
                "case":case,"fold":fold,"train_seeds":block,"target_seed":target,
                "sizes":list(sizes),"train_counts":[n//12 for n in sizes],
            })
    all_train={(t["case"],s) for t in tasks for s in t["train_seeds"]}
    all_test={(t["case"],t["target_seed"]) for t in tasks}
    if all_train & all_test:
        raise ValueError("Train/test seed overlap")
    return tasks


def _label_overview(records):
    """Coverage counts, including missing and one-class heads."""
    if not records:
        raise ValueError("Cannot describe an empty physical training prefix")
    packets=[r.observation for r in records]
    heads={"completion":[p.completion_label() for p in packets],
           "objective":[p.objective.value if p.objective.kind=="exact" else None
                        for p in packets]}
    for name,_ in packets[0].constraints:
        heads[name]=[p.constraint(name).constraint_satisfaction() for p in packets]
    report={}
    for name,observed in heads.items():
        count=sum(x is not None for x in observed)
        row={"total":len(records),"observed":count,"missing":len(records)-count}
        if name!="objective":
            row.update(positive=sum(x is True or x==1 for x in observed if x is not None),
                       negative=sum(x is False or x==0 for x in observed if x is not None))
            row["both_classes_observed"]=row["positive"]>0 and row["negative"]>0
        report[name]=row
    return report


def _baseline(prior,head):
    """Fixed training-only constant-response comparator; no test data allowed."""
    packets=[r.observation for r in prior]
    if head=="objective":
        values=[p.objective.value for p in packets if p.objective.kind=="exact"]
        return (float(statistics.mean(values)) if values else None),len(values)
    if head=="completion":
        values=[p.completion_label() for p in packets]
    else:
        values=[p.constraint(head).constraint_satisfaction() for p in packets]
    labels=[int(v) for v in values if v is not None]
    # Laplace smoothing only makes a descriptive simple prior; no safety claim.
    return (sum(labels)+1.)/(len(labels)+2.),len(labels)


def _fixed_boundary(targets):
    """Truth-defined 25% lowest |exact g| per head; same subset for both views.

    This is an assessment *stratum*, selected after observing holdout truth,
    never a BO candidate selection or GP training rule.
    """
    names=[name for name,_ in targets[0].observation.constraints]
    chosen={}
    for head in names:
        exact=[
            (i,abs(rec.observation.constraint(head).value))
            for i,rec in enumerate(targets)
            if rec.observation.constraint(head).kind=="exact"
        ]
        n=int(math.ceil(len(exact)*.25)) if len(exact)>=4 else 0
        chosen[head]=set(i for i,_ in sorted(exact,key=lambda t:(t[1],t[0]))[:n])
    return chosen


def forecast_level(case,fold,seed,holdout_seed,training,targets):
    """Fit both views on exactly the same n real labels, score same test IDs."""
    n=len(training)
    if n not in SIZES or len(targets)!=12:
        raise ValueError("Learning-curve sample size or target count differs")
    if any(a.observation.contract_id!=training[0].observation.contract_id
           for a in training+targets):
        raise ValueError("Mixed physical contracts")
    seen={digest(r.candidate) for r in training}
    if any(digest(r.candidate) in seen for r in targets):
        raise ValueError("Heldout design overlaps training")
    problem=problem_for(case)
    if problem.contract_id!=training[0].observation.contract_id:
        raise ValueError("Physical task contract drift")
    start=time.perf_counter()
    fitted,meta=_train_views(problem,seed,training)
    guesses={v:fitted[v].predict([r.candidate for r in targets]) for v in VIEWS}
    labels=_label_overview(training)
    base={head:_baseline(training,head)[0] for head in labels}
    boundary=_fixed_boundary(targets)
    scored=[]
    for view,predicted in guesses.items():
        for head,packet in [
            ("objective",predicted["objective"]),
            *[(name,predicted["constraints"][name])
              for name in sorted(predicted["constraints"])]
        ]:
            if packet is None:continue
            for i,rec in enumerate(targets):
                observation=(rec.observation.objective if head=="objective"
                             else rec.observation.constraint(head))
                item=_score_row(case,seed,holdout_seed,i,rec.candidate,
                                head,view,observation,packet,
                                fitted[view].objective.scale if head=="objective"
                                else packet["scale"])
                item["near_boundary"]=(i in boundary[head]) if head!="objective" else None
                item["baseline_brier"]=(
                    (base[head]-int(item["known_feasible"]))**2
                    if item["known_feasible"] is not None and base[head] is not None else None)
                item["baseline_squared_error"]=(
                    (base[head]-item["exact_target"])**2
                    if head=="objective" and item["exact_target"] is not None
                    and base[head] is not None else None)
                scored.append(item)
        comp=predicted["completion"]
        if comp is not None:
            p=_completion_probability(comp)
            for i,rec in enumerate(targets):
                label=rec.observation.completion_label()
                item={
                    "case":case,"train_seed":seed,"holdout_seed":holdout_seed,
                    "heldout_index":i,"candidate_id":digest(rec.candidate),
                    "head":"completion","view":view,
                    "observed_kind":"binary_or_missing","exact_target":label,
                    "known_feasible":None if label is None else bool(label),
                    "mean":float(comp["mean"][i]),
                    "variance":float(comp["variance"][i]),
                    "p_feasible":float(p[i]),
                    "brier":None if label is None else float((p[i]-label)**2),
                    "false_feasible":None if label is None else bool(label==0 and p[i]>=.5),
                    "mae":None,"squared_error":None,"near_boundary":None,
                    "probability_label":"P(completed)",
                    "baseline_brier":(
                        (base["completion"]-label)**2
                        if label is not None and base["completion"] is not None else None),
                    "baseline_squared_error":None,
                }
                scored.append(item)
    for row in scored:
        row["training_size"]=n
        row["fold"]=fold
        row["fixed_boundary_rule"]="lowest_absolute_quartile_of_heldout_exact_margins"
    return scored,{
        "case":case,"fold":fold,"n":n,
        "train_seed_anchor":seed,"test_seed":holdout_seed,
        "training_coverage":labels,
        "test_coverage":_label_overview(targets),
        "training_ids_sha256":digest([r.observation.design_id for r in training]),
        "target_ids_sha256":digest([r.observation.design_id for r in targets]),
        "fit_seconds":time.perf_counter()-start,
        "both_views_from_same_training_labels":True,
        "no_heldout_labels_in_training":True,
        **meta,
    }


def _summary(rows):
    buckets=defaultdict(list)
    for rec in rows:
        buckets[(rec["case"],rec["training_size"],rec["head"],rec["view"])].append(rec)
    result={}
    for (case,n,head,view),group in sorted(buckets.items()):
        resolved=[r for r in group if r["brier"] is not None]
        exact=[r for r in group if r["squared_error"] is not None]
        is_boundary=[r for r in resolved if r["near_boundary"] is True]
        entry={
            "case":case,"training_size":n,"head":head,"view":view,
            "predictions":len(group),"resolved_signs":len(resolved),
            "positive_signs":sum(r["known_feasible"] is True for r in resolved),
            "negative_signs":sum(r["known_feasible"] is False for r in resolved),
            "exact_targets":len(exact),
            "completion_has_both_classes":(
                (any(r["known_feasible"] is True for r in resolved) and
                 any(r["known_feasible"] is False for r in resolved))
                if head=="completion" else None),
            "rmse":math.sqrt(statistics.mean(r["squared_error"] for r in exact))
                   if exact else None,
            "constant_training_mean_rmse":(
                math.sqrt(statistics.mean(r["baseline_squared_error"] for r in exact
                                          if r["baseline_squared_error"] is not None))
                if any(r["baseline_squared_error"] is not None for r in exact)
                else None),
            "brier":statistics.mean(r["brier"] for r in resolved) if resolved else None,
            "constant_training_prevalence_brier":statistics.mean(
                r["baseline_brier"] for r in resolved
                if r["baseline_brier"] is not None
            ) if any(r["baseline_brier"] is not None for r in resolved) else None,
            "false_feasible_count":sum(r["false_feasible"] is True for r in resolved),
            "boundary_exact_quartile_resolved":len(is_boundary),
            "boundary_brier":statistics.mean(r["brier"] for r in is_boundary)
                            if is_boundary else None,
            "folds":len({r["fold"] for r in group}),
        }
        result[f"{case}/{n}/{head}/{view}"]=entry
    return result


def _paired_learning(rows):
    output={}
    sizes=sorted({r["training_size"] for r in rows})
    for n in sizes:
        r=[x for x in rows if x["training_size"]==n]
        for name,x in paired_view_differences(r).items():
            output[f"{n}/{name}"]=x
    return output


def _forbid(*args,**kwargs):
    raise RuntimeError("ZERO-TRUTH LEARNING CURVES: aircraft evaluation prohibited")


def run(original,output,*,folds=2,sizes=SIZES):
    """Retrospective learning curve; originals opened read-only; new folder only."""
    original=Path(original).resolve()
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("Never overwrite an old learning curve")
    plan=_read(original/"plan.json")
    if plan.get("source_commit")!=FROZEN_SOURCE_SHA:
        raise ValueError("Wrong frozen PR45 model source")
    if hashlib.sha256((original/"plan.json").read_bytes()).hexdigest()!=FROZEN_PLAN_SHA256:
        raise ValueError("PR45 registered plan byte identity differs")
    if hashlib.sha256((original/"real_calls.sqlite").read_bytes()).hexdigest()!=FROZEN_REAL_SQLITE_SHA256:
        raise ValueError("PR45 real-call SQLite byte identity differs")
    tasks=fixed_study_plan(plan,folds=folds,sizes=sizes)
    output.mkdir(parents=True,exist_ok=False)
    rows=[];receipts=[];fit_errors=[]
    # Source backend is explicitly disabled for this entire retrospective job.
    with ExitStack() as stack:
        stack.enter_context(patch.object(AircraftCase,"evaluate",_forbid))
        stack.enter_context(patch.object(SimplifiedElectricAircraft,"evaluate",_forbid))
        with _sqlite(original/"real_calls.sqlite") as db:
            for task in tasks:
                case=task["case"]
                contract=plan["cases"][case]["contracts"]["20s"]
                groups=[_read_initial(db,case,seed,contract)
                        for seed in task["train_seeds"]]
                target=_read_initial(db,case,task["target_seed"],contract)
                entire=[rec for batch in groups for rec in batch]
                if len(entire)!=192:
                    raise AssertionError("16 x 12 trained calls required")
                train_ids={rec.observation.design_id for rec in entire}
                if any(r.observation.design_id in train_ids for r in target):
                    raise ValueError("Heldout candidate appears in available training pool")
                for n in task["sizes"]:
                    training=entire[:n]
                    try:
                        scored,detail=forecast_level(
                            case,task["fold"],task["train_seeds"][0],
                            task["target_seed"],training,target)
                        rows.extend(scored);receipts.append(detail)
                        print("LEARNING_CURVE",case,"fold",task["fold"],
                              "train",n,"target",len(target),"rows",len(scored),
                              flush=True)
                    except Exception as error:
                        fit_errors.append({
                            "case":case,"fold":task["fold"],"n":n,
                            "error_type":type(error).__name__,"message":str(error),
                        })
                        print("LEARNING_CURVE_FIT_ERROR",case,task["fold"],n,
                              type(error).__name__,str(error),flush=True)
    with (output/"learning_curve_predictions.csv").open("x",newline="",encoding="utf8") as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        else:
            f.write("no_successful_predictions\n")
    out={
        "schema":SCHEMA,
        "status":("COMPLETE" if not fit_errors else "COMPLETE_WITH_RECORDED_MODEL_FIT_ERRORS"),
        "physical_source_sha":FROZEN_SOURCE_SHA,
        "source_plan_sha256":FROZEN_PLAN_SHA256,
        "source_db_sha256":FROZEN_REAL_SQLITE_SHA256,
        "new_aircraft_calls":0,"new_RCAIDE_calls":0,
        "original_db_read_only":True,
        "folds_per_task":folds,"training_sizes":list(sizes),
        "study_registration":tasks,
        "full_model_fits_requested":len(tasks)*len(sizes)*2,
        "successful_fit_pairs":len(receipts),
        "fit_errors":fit_errors,
        "receipts":receipts,
        "model_head_summary":_summary(rows),
        "paired_view_differences_by_training_size":_paired_learning(rows),
        "limitations":[
            "Offline cross-seed shared-initial qA training, NOT the original 12+18 sequential optimizer.",
            "Training sizes >12 pool REAL earlier PR45 calls across distinct seeds; this is extra supervised data, NOT free in a new online design.",
            "Test designs come from a disjoint frozen PR45 seed block and are held fixed as n changes; their outcomes do not fit any VAE, GP or kernel.",
            "The 2000 structural corpus has no objective/constraint labels.",
            "The heldout exact-near-boundary quartile is a posthoc assessment stratum, not candidate filtering or a constraint tolerance.",
            "Completion-class prevalence must be reported; all-positive tests cannot measure failure discrimination.",
            "Small folds, correlated 12-target clusters, non-calibrated EP probabilities: no safety guarantee or novelty claim.",
            "140km sparse qA threshold not scientifically qualified.",
            "Model fit errors preserved as outcomes; do not select successful folds after viewing errors.",
        ],
    }
    with (output/"learning_curve_report.json").open("x",encoding="utf8") as f:
        json.dump(plain(out),f,indent=2,ensure_ascii=False,allow_nan=False)
    return out
