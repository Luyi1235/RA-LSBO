"""PR86-D nested 12-vs-24 initial-sample allocation experiment."""
from copy import deepcopy
from pathlib import Path
import platform
import time
import traceback

from risk_learning.core import candidates, observed, digest
from risk_learning.io import append, seal, write
from risk_learning.study import new_bill
from risk_learning.native import predict as risk_predict
from pr86b_acquisition import metrics, brier_skill_score, anchor_prevalence
from pr86d_protocol import plan, METHODS, ANCHOR_SIZES
from pr86c_evaluator import truth_for_evaluation
from pr86c_models import predict_objective, outcome_state
from pr86c_acquisition import choose


def _evaluate(task,candidate,call_id,role,unit_name,root,bill):
    append(root/"evaluations.jsonl",dict(event="started",call_id=call_id,
        task=task,unit=unit_name,role=role,candidate_id=digest(candidate),
        candidate=candidate))
    bill["math_started"]+=1
    target=truth_for_evaluation(task,deepcopy(candidate))
    row=dict(candidate=candidate,candidate_id=digest(candidate),truth=target,
             call_id=call_id)
    if role in ("anchor","online"):
        row["observed"]=observed(candidate,target)
    append(root/"evaluations.jsonl",dict(event="settled",call_id=call_id,
        record_sha256=digest(row),record=row))
    bill["math_settled"]+=1
    return row


def _baseline_brier(test_rows,rates):
    out={}
    for head in ("region","resource","joint"):
        labels=([bool(r["truth"]["feasible"]) for r in test_rows] if head=="joint"
                else [bool(r["truth"][head]<=0) for r in test_rows])
        out[head]=sum((rates[head]-float(y))**2 for y in labels)/len(labels)
    return out


def _test_metrics(test_rows,probabilities,baseline):
    out={}
    for head in ("region","resource","joint"):
        labels=([bool(r["truth"]["feasible"]) for r in test_rows] if head=="joint"
                else [bool(r["truth"][head]<=0) for r in test_rows])
        row=metrics(labels,probabilities[head])
        row["brier_skill_score"]=brier_skill_score(row["brier"],baseline[head])
        positive=sum(labels)
        row["feasible_recall_at_p50"]=(
            sum(bool(y) and p>=0.5 for y,p in zip(labels,probabilities[head]))/positive
            if positive else None)
        row["predicted_feasible_count_at_p50"]=sum(
            p>=0.5 for p in probabilities[head])
        out[head]=row
    return out


def run(output,profile,source):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("PR86-D is write-once; preserve prior attempts")
    spec=plan(profile)
    output.mkdir(parents=True)
    bill=new_bill()
    for key in ("objective_gp_head_started","objective_gp_head_completed",
                "objective_gp_head_errors","objective_gp_kernel_started",
                "objective_gp_kernel_completed","objective_gp_kernel_errors",
                "objective_gp_kernel_candidates_rejected"):
        bill[key]=0
    all_trials=[]
    started=time.perf_counter()
    write(output/"plan.json",spec)
    write(output/"source_receipt.json",source)
    write(output/"environment.json",dict(python=platform.python_version(),
                                         platform=platform.platform()))
    try:
        for unit in spec["units"]:
            task,repeat=unit["task"],unit["repeat"]
            name="%s/r%d"%(task,repeat)
            anchors=[_evaluate(task,c,name+"/anchor/%03d"%i,"anchor",
                               name,output,bill)
                     for i,c in enumerate(candidates(unit["anchor_seed"],
                                                     spec["shared_anchor_count"]))]
            pool=candidates(unit["pool_seed"],spec["candidate_pool"])
            tests=[_evaluate(task,c,name+"/test/%04d"%i,"test",name,output,bill)
                   for i,c in enumerate(candidates(unit["test_seed"],spec["test_count"]))]
            all_ids=[r["candidate_id"] for r in anchors+tests]+[digest(c) for c in pool]
            if len(all_ids)!=len(set(all_ids)):
                raise ValueError("Nested anchors, candidate pool and test set overlap")
            unit_trials=[]
            for anchor_size in ANCHOR_SIZES:
                initial=anchors[:anchor_size]
                rates=anchor_prevalence([r["observed"] for r in initial])
                baseline=_baseline_brier(tests,rates)
                online_budget=spec["online_per_method_by_anchor_size"][str(anchor_size)]
                checkpoints=set(spec["checkpoints_by_anchor_size"][str(anchor_size)])
                for method_index,method in enumerate(METHODS):
                    history=list(initial)
                    available=list(pool)
                    proposals,online,checkpoint_rows=[],[],[]
                    arm_seed=unit["pool_seed"]+1009*method_index
                    first_objective_fit_n=None
                    first_feasible_n=(anchor_size if outcome_state(
                        [r["observed"] for r in history])["feasible_found"] else None)
                    last_risk_receipt=last_objective_receipt=None
                    for step in range(online_budget+1):
                        current=[deepcopy(r["observed"]) for r in history]
                        state=outcome_state(current)
                        incumbent=state["best_exact_feasible"]
                        queries=available+[r["candidate"] for r in tests]
                        fit_seed=unit["pool_seed"]+1009*len(history)
                        risk,rr=risk_predict(current,queries,"native_censored",
                                             fit_seed,bill)
                        last_risk_receipt=rr
                        if risk is None or rr.get("status")!="OK":
                            raise RuntimeError("PR86-D risk GP failed: %r"%rr)
                        nc=len(available)
                        candidate_forecast=dict(probabilities={
                            h:risk["probabilities"][h][:nc]
                            for h in ("region","resource","joint")})
                        test_forecast={h:risk["probabilities"][h][nc:]
                                       for h in ("region","resource","joint")}
                        objective,orr=predict_objective(
                            current,available,fit_seed+4096,bill)
                        last_objective_receipt=orr
                        if first_objective_fit_n is None and orr["status"]=="OK":
                            first_objective_fit_n=len(history)
                        if state["n"] in checkpoints:
                            checkpoint_rows.append(dict(
                                anchor_size=anchor_size,n=state["n"],method=method,
                                observed_prefix_sha256=digest(current),
                                outcome=state,initial_outcome=outcome_state(
                                    [r["observed"] for r in initial]),
                                first_objective_gp_fit_n=first_objective_fit_n,
                                first_feasible_n=first_feasible_n,
                                anchor_prevalence=rates,
                                anchor_baseline_brier=baseline,
                                test_candidate_ids=[r["candidate_id"] for r in tests],
                                test_probabilities_raw=test_forecast,
                                risk_metrics=_test_metrics(tests,test_forecast,baseline),
                                risk_fit_receipt=rr,objective_fit_receipt=orr))
                        if step==online_budget:
                            break
                        available_before=digest(available)
                        forecast_sha=digest(candidate_forecast)
                        index,decision=choose(method,available,candidate_forecast,
                            objective,incumbent,arm_seed,step)
                        selected=available.pop(index)
                        decision.update(
                            anchor_size=anchor_size,
                            method_fit_risk=rr,method_fit_objective=orr,
                            available_sha256=available_before,
                            history_observed_sha256=digest(current),
                            candidate_forecast=candidate_forecast,
                            objective_forecast=objective,
                            forecast_sha256=forecast_sha,
                            candidate_pool_sha256=digest(pool),
                            selected_candidate=selected,
                            pre_outcome=True,no_test_truth_accessed=True,
                            no_unselected_truth_accessed=True)
                        proposals.append(decision)
                        append(output/"progress.jsonl",dict(event="proposal",
                            task=task,repeat=repeat,anchor_size=anchor_size,
                            method=method,step=step,proposal=decision))
                        call=(name+"/online/n0%02d/"%anchor_size+
                              method+"/%03d"%step)
                        record=_evaluate(task,selected,call,"online",
                                         name,output,bill)
                        observation=dict(step=step,candidate_id=record["candidate_id"],
                                         record=record)
                        online.append(observation)
                        append(output/"progress.jsonl",dict(event="observation",
                            task=task,repeat=repeat,anchor_size=anchor_size,
                            method=method,step=step,candidate_id=record["candidate_id"],
                            call_id=call,observed=record["observed"]))
                        history.append(record)
                        if first_feasible_n is None and outcome_state(
                            [r["observed"] for r in history])["feasible_found"]:
                            first_feasible_n=len(history)
                    trial=dict(task=task,repeat=repeat,anchor_size=anchor_size,
                        method=method,seed=arm_seed,
                        anchor_ids=[r["candidate_id"] for r in initial],
                        masked_anchor_ids=[r["candidate_id"] for r in anchors[anchor_size:]],
                        initial_outcome=outcome_state([r["observed"] for r in initial]),
                        first_objective_gp_fit_n=first_objective_fit_n,
                        first_feasible_n=first_feasible_n,
                        proposals=proposals,online_observations=online,
                        checkpoints=checkpoint_rows,final_n=len(history),
                        final_outcome=outcome_state([r["observed"] for r in history]),
                        final_risk_fit_receipt=last_risk_receipt,
                        final_objective_fit_receipt=last_objective_receipt,
                        online_calls=len(online),risk_refits=len(proposals)+1)
                    unit_trials.append(trial)
                    all_trials.append(trial)
                    append(output/"progress.jsonl",dict(event="trial_complete",
                        task=task,repeat=repeat,anchor_size=anchor_size,
                        method=method,initial_outcome=trial["initial_outcome"],
                        final_outcome=trial["final_outcome"]))
            write(output/"units"/name/"unit.json",dict(
                task=task,repeat=repeat,anchors=anchors,pool=pool,tests=tests,
                trials=unit_trials))
        if bill["math_started"]!=spec["total_math_calls"] or (
           bill["math_settled"]!=spec["total_math_calls"]):
            raise ValueError("PR86-D mathematical evaluation budget mismatch")
        if bill["gp_head_started"]!=spec["expected_risk_heads"] or (
           bill["gp_head_completed"]!=spec["expected_risk_heads"]):
            raise ValueError("PR86-D risk head count mismatch")
        if bill["gp_kernel_started"]!=spec["expected_risk_kernel_constructions"] or (
           bill["gp_kernel_completed"]!=spec["expected_risk_kernel_constructions"]):
            raise ValueError("PR86-D risk kernel count mismatch")
        for key in ("gp_head_errors","gp_kernel_errors","kernel_candidates_rejected",
                    "objective_gp_head_errors","objective_gp_kernel_errors",
                    "objective_gp_kernel_candidates_rejected"):
            if bill[key]!=0:
                raise ValueError("Nonzero GP error/rejection: "+key)
        if bill["objective_gp_head_started"]!=bill["objective_gp_head_completed"]:
            raise ValueError("Objective GP head settlement mismatch")
        if bill["objective_gp_kernel_started"]!=bill["objective_gp_kernel_completed"] or (
           bill["objective_gp_kernel_started"]!=8*bill["objective_gp_head_started"]):
            raise ValueError("Objective GP kernel bill mismatch")
        expected_trials=len(spec["units"])*len(METHODS)*len(ANCHOR_SIZES)
        if len(all_trials)!=expected_trials:
            raise ValueError("PR86-D trajectory count mismatch")
        report=dict(schema=spec["schema"],code_sha=source["code_sha"],
            profile=profile,status="COMPLETE_PR86D_INITIAL_SAMPLE_ABLATION",
            interpretation="Nested initial sample allocation under fixed per-arm n=48; within-known-task-family replication.",
            bill=bill,units=len(spec["units"]),trials=len(all_trials),
            risk_refits=spec["expected_risk_refits"],
            anchor_sizes=list(ANCHOR_SIZES),
            final_outcomes=[{k:t[k] for k in
                ("task","repeat","anchor_size","method","final_n",
                 "initial_outcome","first_objective_gp_fit_n","first_feasible_n",
                 "final_outcome")} for t in all_trials],
            elapsed_seconds=time.perf_counter()-started)
        write(output/"report.json",report)
        seal(output)
        return report
    except BaseException as error:
        write(output/"failure.json",dict(code_sha=source.get("code_sha"),
            error_type=type(error).__name__,message=str(error),
            math_started=bill["math_started"],math_settled=bill["math_settled"],
            unknown_calls=bill["math_started"]-bill["math_settled"],
            completed_trials=len(all_trials),traceback=traceback.format_exc()))
        seal(output)
        raise
