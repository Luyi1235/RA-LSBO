"""PR86-C paired objective outcomes; no GP fitting or fresh simulator calls."""
import csv
import math
import statistics
from pathlib import Path

from risk_learning.io import read, write, seal, check_manifest
from pr86c_protocol import plan, METHODS


def compare_objective(left,right):
    """Lexicographic observed-feasibility then exact feasible objective; no imputation."""
    a,b=left["best_exact_feasible"],right["best_exact_feasible"]
    if a is None and b is None:
        return 0, None
    if a is None:
        return -1, None
    if b is None:
        return 1, None
    difference=float(a)-float(b)
    return (1 if difference < -1e-12 else -1 if difference > 1e-12 else 0), difference


def _average(values):
    return statistics.mean(values) if values else None


def _csv_write(path,headers,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root,output,expected_sha):
    root=Path(root).resolve()
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("PR86-C analysis is write-once")
    check_manifest(root)
    spec=read(root/"plan.json")
    source=read(root/"source_receipt.json")
    report=read(root/"report.json")
    if spec!=plan(spec["profile"]) or source["code_sha"]!=expected_sha or (
       report["status"]!="COMPLETE_PR86C_OBJECTIVE_AWARE"):
        raise ValueError("Invalid/frozen PR86-C analysis source")
    output.mkdir(parents=True)
    outcomes=[]
    risk_curves=[]
    for unit in spec["units"]:
        name="%s/r%d"%(unit["task"],unit["repeat"])
        data=read(root/"units"/name/"unit.json")
        for trial in data["trials"]:
            for cp in trial["checkpoints"]:
                state=cp["outcome"]
                outcomes.append(dict(task=unit["task"],repeat=unit["repeat"],
                    method=trial["method"],n=cp["n"],
                    feasible_found=int(state["feasible_found"]),
                    best_exact_feasible=state["best_exact_feasible"],
                    exact_feasible_count=state["exact_feasible_count"],
                    exact_objective_count=state["exact_objective_count"],
                    censored_region_count=state["censored_region_count"],
                    missing_objective_count=state["missing_objective_count"]))
                for head in ("joint","region","resource"):
                    m=cp["risk_metrics"][head]
                    risk_curves.append(dict(task=unit["task"],repeat=unit["repeat"],
                        method=trial["method"],n=cp["n"],head=head,
                        brier=m["brier"],brier_skill=m["brier_skill_score"],
                        auroc=m["auroc"],auprc=m["auprc"],
                        ece=m["ece"],false_feasible=m["false_feasible_rate"],
                        feasible_recall=m["feasible_recall_at_p50"]))
    aggregate=[]
    for task in spec["tasks"] if "tasks" in spec else sorted({r["task"] for r in outcomes}):
        for method in METHODS:
            for n in spec["checkpoints"]:
                rows=[r for r in outcomes if r["task"]==task and
                      r["method"]==method and r["n"]==n]
                if not rows:
                    continue
                feasible=[r for r in rows if r["feasible_found"]]
                aggregate.append(dict(task=task,method=method,n=n,repeats=len(rows),
                     feasible_discovery_fraction=len(feasible)/len(rows),
                     best_objective_mean_conditional=_average(
                         [r["best_exact_feasible"] for r in feasible]),
                     exact_feasible_count_mean=_average(
                         [r["exact_feasible_count"] for r in rows]),
                     exact_objective_count_mean=_average(
                         [r["exact_objective_count"] for r in rows])))
    comparisons=[]
    candidate="objective_weighted_boundary"
    for task in sorted({r["task"] for r in outcomes}):
        for baseline in METHODS:
            if baseline==candidate:
                continue
            for n in spec["checkpoints"]:
                left={r["repeat"]:r for r in outcomes if r["task"]==task and
                      r["method"]==candidate and r["n"]==n}
                right={r["repeat"]:r for r in outcomes if r["task"]==task and
                       r["method"]==baseline and r["n"]==n}
                if left.keys()!=right.keys():
                    raise ValueError("Non-matched objective trial pairs")
                wins=losses=ties=0
                both_feasible_deltas=[]
                comparison_by_repeat=[]
                for repeat in sorted(left):
                    cmp,delta=compare_objective(left[repeat],right[repeat])
                    wins+=cmp==1
                    losses+=cmp==-1
                    ties+=cmp==0
                    if delta is not None:
                        both_feasible_deltas.append(delta)
                    comparison_by_repeat.append(dict(
                        repeat=repeat,comparison=cmp,
                        objective_delta_if_both_feasible=delta,
                        left_feasible=left[repeat]["feasible_found"],
                        right_feasible=right[repeat]["feasible_found"]))
                comparisons.append(dict(task=task,n=n,method=candidate,
                    comparator=baseline,repeats=len(left),wins=wins,losses=losses,
                    ties=ties,mean_objective_delta_if_both_feasible=_average(
                        both_feasible_deltas),
                    both_feasible_count=len(both_feasible_deltas),
                    per_repeat=comparison_by_repeat))
    macro=[]
    for method in METHODS:
        for n in spec["checkpoints"]:
            per_task=[r for r in aggregate if r["method"]==method and r["n"]==n]
            macro.append(dict(method=method,n=n,
                tasks=len(per_task),
                mean_task_feasible_discovery=_average(
                   [r["feasible_discovery_fraction"] for r in per_task]),
                mean_task_conditional_best_objective=_average(
                   [r["best_objective_mean_conditional"] for r in per_task
                    if r["best_objective_mean_conditional"] is not None]),
                caution="Conditional means across tasks have distinct scales; do not rank methods on pooled raw objective"))
    result=dict(schema="pr86c_objective_analysis_v1",
        status="COMPLETE_PR86C_ANALYSIS",code_sha=source["code_sha"],
        protocol=spec["schema"],profile=spec["profile"],
        primary_candidate=candidate,primary_outcomes=outcomes,
        task_stratified=aggregate,macro_discovery=macro,
        paired_lexicographic=comparisons,
        interpretation_limits=[
            "Feasible discovery is compared before objective magnitude; no artificial failure penalty.",
            "Objective difference is used only where both paired arms have exact feasible outcomes.",
            "Each mathematical task has its own objective scale; use task-stratified pairs.",
            "Results on familiar six tasks vs withheld four tasks must be distinguished.",
            "Pooled task averages of objective values are descriptive only, not inferential.",
            "No genuine flight optimization, VAE or simulator-completion model was tested.",
        ])
    write(output/"analysis.json",result)
    oh=["task","repeat","method","n","feasible_found","best_exact_feasible",
        "exact_feasible_count","exact_objective_count","censored_region_count",
        "missing_objective_count"]
    rh=["task","repeat","method","n","head","brier","brier_skill","auroc","auprc",
        "ece","false_feasible","feasible_recall"]
    ch=["task","n","method","comparator","repeats","wins","losses","ties",
        "mean_objective_delta_if_both_feasible","both_feasible_count"]
    _csv_write(output/"objective_learning_curves.csv",oh,outcomes)
    _csv_write(output/"risk_learning_curves.csv",rh,risk_curves)
    _csv_write(output/"paired_objective.csv",ch,[
        {k:r[k] for k in ch} for r in comparisons])
    seal(output)
    return dict(status=result["status"],outcome_rows=len(outcomes),
        risk_rows=len(risk_curves),paired_rows=len(comparisons))
