"""Task-stratified PR86-B sequential acquisition analysis; no fitting/evaluations."""
import csv, statistics
from pathlib import Path
from risk_learning.io import read, write, seal, check_manifest, file_sha
from pr86b_protocol import plan, METHODS

def analyze(root, output, expected_sha):
    root=Path(root).resolve(); output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("PR86-B analysis output is write-once")
    check_manifest(root)
    spec=read(root/"plan.json"); source=read(root/"source_receipt.json")
    if source["code_sha"]!=expected_sha:
        raise ValueError("Code SHA mismatch")
    grouped={}
    for unit in spec["units"]:
        name="%s/r%d"%(unit["task"],unit["repeat"])
        data=read(root/"units"/name/"unit.json")
        for trial in data["trials"]:
            for cp in trial["checkpoints"]:
                for calibration in ("raw_metrics","anchor_calibrated_metrics"):
                    for head in ("joint","region","resource"):
                        key=(unit["task"],trial["method"],cp["n"],calibration,head)
                        grouped.setdefault(key,[]).append(cp[calibration][head])
    curves=[]
    for (task,method,n,calibration,head),rows in sorted(grouped.items()):
        fields=("brier","brier_skill_score","log_loss","auroc","auprc","ece","false_feasible_rate")
        summary={}
        for field in fields:
            values=[r.get(field) for r in rows if r.get(field) is not None]
            summary[field]=dict(n=len(values),values=values,
                mean=statistics.fmean(values) if values else None,
                median=statistics.median(values) if values else None)
        curves.append(dict(task=task,method=method,n=n,calibration=calibration,head=head,
                           repeats=len(rows),metrics=summary))
    # Paired task-specific comparison on anchor-calibrated joint Brier.
    lookup={}
    for unit in spec["units"]:
        name="%s/r%d"%(unit["task"],unit["repeat"])
        data=read(root/"units"/name/"unit.json")
        for trial in data["trials"]:
            for cp in trial["checkpoints"]:
                lookup[(unit["task"],unit["repeat"],trial["method"],cp["n"])]=cp["anchor_calibrated_metrics"]["joint"]["brier"]
    pairs=[]
    for task in spec["tasks"]:
        for n in spec["checkpoints"]:
            for comparator in ("iid_random","maximin","latent_uncertainty","boundary"):
                wlt=[0,0,0];deltas=[]
                for unit in spec["units"]:
                    if unit["task"]!=task: continue
                    a=lookup[(task,unit["repeat"],"calibration_aware_voi",n)]
                    b=lookup[(task,unit["repeat"],comparator,n)]
                    delta=a-b
                    deltas.append(delta)
                    wlt[0 if delta<0 else 1 if delta>0 else 2]+=1
                pairs.append(dict(task=task,n=n,left="calibration_aware_voi",right=comparator,
                    brier_wins_losses_ties=wlt,paired_brier_delta=statistics.fmean(deltas) if deltas else None,
                    per_repeat_deltas=deltas))
    result=dict(status="COMPLETE_PR86B_ANALYSIS",code_sha=expected_sha,
        input_manifest_sha256=file_sha(root/"manifest.json"),task_stratified_curves=curves,
        paired_comparisons=pairs,
        interpretation_limits=[
            "Report each task separately; do not pool task-specific raw margin scales.",
            "Brier Skill Score is relative to the Laplace-smoothed prevalence estimated from common random anchors.",
            "The anchor-intercept correction uses only random anchor labels and unlabeled query covariates.",
            "BALD is an approximate probit mutual-information acquisition, not exact expected Brier reduction.",
            "The joint probability is the product of marginal region/resource probabilities.",
            "This is risk-model acquisition evidence, not aircraft optimization or a safety guarantee."
        ])
    output.mkdir(parents=True)
    write(output/"analysis.json",result)
    with (output/"learning_curves.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["task","method","n","calibration","head","repeats","brier_mean","brier_skill_score_mean",
                "auroc_mean","auprc_mean","ece_mean","false_feasible_rate_mean"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in curves:
            m=row["metrics"]
            w.writerow(dict(task=row["task"],method=row["method"],n=row["n"],
                calibration=row["calibration"],head=row["head"],repeats=row["repeats"],
                brier_mean=m["brier"]["mean"],brier_skill_score_mean=m["brier_skill_score"]["mean"],
                auroc_mean=m["auroc"]["mean"],auprc_mean=m["auprc"]["mean"],
                ece_mean=m["ece"]["mean"],false_feasible_rate_mean=m["false_feasible_rate"]["mean"]))
    seal(output)
    return dict(status="COMPLETE_PR86B_ANALYSIS",curve_rows=len(curves),paired_rows=len(pairs))
