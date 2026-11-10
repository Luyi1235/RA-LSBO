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
    reference_path=Path(__file__).resolve().parent/"docs"/"PR86A_N96_REFERENCE.csv"
    reference_lines=[line for line in reference_path.read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
    historical=list(csv.DictReader(reference_lines))
    if len(historical)!=12 or any(r["method"]!="native_censored" or r["n"]!="96" for r in historical):
        raise ValueError("Pinned PR86-A n=96 reference roster is invalid")
    reference_lookup={(r["task"],r["policy"]):r for r in historical}
    reference_deltas=[]
    for row in curves:
        if row["n"]!=48 or row["head"]!="joint":
            continue
        current=row["metrics"]["brier"]["mean"]
        for policy in ("iid_uniform","coverage_maximin"):
            ref=reference_lookup[(row["task"],policy)]
            ref_brier=float(ref["joint_brier_mean"])
            reference_deltas.append(dict(task=row["task"],method=row["method"],
                calibration=row["calibration"],current_n=48,
                historical_policy=policy,historical_n=96,
                current_brier_mean=current,historical_brier_mean=ref_brier,
                brier_delta_current_minus_historical=current-ref_brier,
                current_repeats=row["repeats"],historical_repeats=int(ref["repeats"]),
                comparison="descriptive_only_independent_PR86A_test_seeds"))
    result=dict(status="COMPLETE_PR86B_ANALYSIS",code_sha=expected_sha,
        input_manifest_sha256=file_sha(root/"manifest.json"),
        historical_reference=dict(source_result_head="38f2f3a469d833c176f1a6bbe2eab8257d59c6c7",
            source_code_sha="a5d5f20a4854de30939b590cb69126d1d5690ed5",
            file_sha256=file_sha(reference_path),rows=historical),
        task_stratified_curves=curves,paired_comparisons=pairs,
        descriptive_n48_vs_historical_n96=reference_deltas,
        interpretation_limits=[
            "Report each task separately; do not pool task-specific raw margin scales.",
            "Brier Skill Score is relative to the Laplace-smoothed prevalence estimated from common random anchors.",
            "The anchor-intercept correction uses only random anchor labels and unlabeled query covariates.",
            "BALD is an approximate probit mutual-information acquisition, not exact expected Brier reduction.",
            "The joint probability is the product of marginal region/resource probabilities.",
            "PR86-B versus PR86-A n=96 comparisons use independent held-out seeds and are descriptive, not paired inference.",
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
