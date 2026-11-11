"""PR86-D analysis of nested initial-sample allocation; no new truth calls or fitting."""
import csv
from pathlib import Path

from risk_learning.io import read, write, seal, check_manifest
from pr86d_protocol import plan, METHODS, ANCHOR_SIZES
from pr86c_analysis import compare_objective


def _mean(values):
    values=[float(v) for v in values if v is not None]
    return sum(values)/len(values) if values else None


def _write_csv(path,headers,rows):
    with path.open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def analyze(root,output,expected_sha):
    root=Path(root).resolve()
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError("PR86-D analysis is write-once")
    check_manifest(root)
    spec=read(root/"plan.json")
    source=read(root/"source_receipt.json")
    report=read(root/"report.json")
    if spec!=plan(spec["profile"]) or source["code_sha"]!=expected_sha or (
       report["status"]!="COMPLETE_PR86D_INITIAL_SAMPLE_ABLATION"):
        raise ValueError("Invalid PR86-D analysis source")
    output.mkdir(parents=True)
    outcomes=[]
    risk_rows=[]
    initial_rows=[]
    paired_sizes=[]
    cross_method=[]
    for unit in spec["units"]:
        name="%s/r%d"%(unit["task"],unit["repeat"])
        data=read(root/"units"/name/"unit.json")
        by_key={(t["anchor_size"],t["method"]):t for t in data["trials"]}
        for trial in data["trials"]:
            n0=trial["anchor_size"]
            initial=trial["initial_outcome"]
            initial_rows.append(dict(task=unit["task"],repeat=unit["repeat"],
                anchor_size=n0,method=trial["method"],
                exact_objective_count=initial["exact_objective_count"],
                objective_gp_trainable=int(initial["exact_objective_count"]>=
                                            spec["objective_min_exact_labels"]),
                exact_feasible_count=initial["exact_feasible_count"],
                feasible_found=int(initial["feasible_found"]),
                first_objective_gp_fit_n=trial["first_objective_gp_fit_n"],
                first_feasible_n=trial["first_feasible_n"]))
            for cp in trial["checkpoints"]:
                state=cp["outcome"]
                outcomes.append(dict(task=unit["task"],repeat=unit["repeat"],
                    anchor_size=n0,method=trial["method"],n=cp["n"],
                    feasible_found=int(state["feasible_found"]),
                    best_exact_feasible=state["best_exact_feasible"],
                    exact_feasible_count=state["exact_feasible_count"],
                    exact_objective_count=state["exact_objective_count"],
                    missing_objective_count=state["missing_objective_count"],
                    objective_gp_trainable=int(state["exact_objective_count"]>=
                                                spec["objective_min_exact_labels"]),
                    first_objective_gp_fit_n=cp["first_objective_gp_fit_n"],
                    first_feasible_n=cp["first_feasible_n"]))
                for head in ("joint","region","resource"):
                    m=cp["risk_metrics"][head]
                    risk_rows.append(dict(task=unit["task"],repeat=unit["repeat"],
                        anchor_size=n0,method=trial["method"],n=cp["n"],head=head,
                        brier=m["brier"],brier_skill=m["brier_skill_score"],
                        auroc=m["auroc"],auprc=m["auprc"],ece=m["ece"],
                        false_feasible=m["false_feasible_rate"],
                        feasible_recall=m["feasible_recall_at_p50"]))
        for method in METHODS:
            a=by_key[(24,method)]["final_outcome"]
            b=by_key[(12,method)]["final_outcome"]
            cmp,delta=compare_objective(a,b)
            paired_sizes.append(dict(task=unit["task"],repeat=unit["repeat"],
                method=method,left_anchor_size=24,right_anchor_size=12,
                comparison=cmp,objective_delta_24_minus_12_if_both_feasible=delta,
                feasible_24=int(a["feasible_found"]),feasible_12=int(b["feasible_found"]),
                exact_objective_count_delta=a["exact_objective_count"]-
                                           b["exact_objective_count"]))
        for n0 in ANCHOR_SIZES:
            for baseline in ("iid_random","boundary","constrained_ei"):
                left=by_key[(n0,"objective_weighted_boundary")]["final_outcome"]
                right=by_key[(n0,baseline)]["final_outcome"]
                cmp,delta=compare_objective(left,right)
                cross_method.append(dict(task=unit["task"],repeat=unit["repeat"],
                    anchor_size=n0,method="objective_weighted_boundary",
                    comparator=baseline,comparison=cmp,
                    objective_delta_if_both_feasible=delta,
                    method_feasible=int(left["feasible_found"]),
                    comparator_feasible=int(right["feasible_found"])))

    initial_summary=[]
    for n0 in ANCHOR_SIZES:
        rows=[r for r in initial_rows if r["anchor_size"]==n0]
        unique={(r["task"],r["repeat"]):r for r in rows}
        unit_rows=list(unique.values())
        initial_summary.append(dict(anchor_size=n0,units=len(unit_rows),
            objective_gp_trainable_units=sum(r["objective_gp_trainable"] for r in unit_rows),
            objective_gp_trainable_fraction=_mean(
                [r["objective_gp_trainable"] for r in unit_rows]),
            exact_feasible_incumbent_units=sum(r["feasible_found"] for r in unit_rows),
            exact_objective_labels_mean=_mean(
                [r["exact_objective_count"] for r in unit_rows]),
            exact_feasible_count_mean=_mean(
                [r["exact_feasible_count"] for r in unit_rows])))

    paired_summary=[]
    for task in sorted({r["task"] for r in paired_sizes}):
        for method in METHODS:
            rows=[r for r in paired_sizes if r["task"]==task and r["method"]==method]
            both=[r["objective_delta_24_minus_12_if_both_feasible"] for r in rows
                  if r["objective_delta_24_minus_12_if_both_feasible"] is not None]
            paired_summary.append(dict(task=task,method=method,pairs=len(rows),
                n0_24_wins=sum(r["comparison"]==1 for r in rows),
                n0_24_losses=sum(r["comparison"]==-1 for r in rows),
                ties=sum(r["comparison"]==0 for r in rows),
                mean_exact_objective_count_delta=_mean(
                    [r["exact_objective_count_delta"] for r in rows]),
                mean_objective_delta_24_minus_12_if_both_feasible=_mean(both),
                both_feasible_pairs=len(both)))

    checkpoint_summary=[]
    for n0 in ANCHOR_SIZES:
        checkpoints=spec["checkpoints_by_anchor_size"][str(n0)]
        for method in METHODS:
            for n in checkpoints:
                rows=[r for r in outcomes if r["anchor_size"]==n0 and
                      r["method"]==method and r["n"]==n]
                checkpoint_summary.append(dict(anchor_size=n0,method=method,n=n,
                    units=len(rows),
                    feasible_discovery_fraction=_mean([r["feasible_found"] for r in rows]),
                    objective_gp_trainable_fraction=_mean(
                        [r["objective_gp_trainable"] for r in rows]),
                    exact_objective_labels_mean=_mean(
                        [r["exact_objective_count"] for r in rows]),
                    exact_feasible_count_mean=_mean(
                        [r["exact_feasible_count"] for r in rows])))

    result=dict(schema="pr86d_initial_sample_analysis_v1",
        status="COMPLETE_PR86D_ANALYSIS",code_sha=source["code_sha"],
        profile=spec["profile"],primary_factor="nested_initial_anchor_count_12_vs_24",
        fixed_total_observations_per_arm=spec["total_observations_per_arm"],
        initial_trainability=initial_summary,
        paired_initial_size_summary=paired_summary,
        checkpoint_summary=checkpoint_summary,
        initial_rows=initial_rows,objective_outcomes=outcomes,
        paired_initial_size=paired_sizes,paired_method_comparisons=cross_method,
        interpretation_limits=[
            "The 12-anchor cohort sees only the first 12 of the common nested 24 anchors.",
            "The additional 12 anchor labels are masked from the 12-anchor optimizer.",
            "Both cohorts have the same total of 48 observations; online budgets differ (36 vs24).",
            "Objective-GP trainability means >=2 exact objective labels, not necessarily a feasible incumbent.",
            "Exact objective labels require region<=0; exact feasible objectives additionally require resource<=0.",
            "Missing objective labels are never imputed.",
            "The ten task functions are known; only candidate/test seed blocks are new.",
            "Do not pool raw objective magnitudes across different task scales.",
            "No aircraft, RCAIDE, VAE or simulator-completion risk evidence.",
        ])
    write(output/"analysis.json",result)
    _write_csv(output/"initial_trainability.csv",
        ["task","repeat","anchor_size","method","exact_objective_count",
         "objective_gp_trainable","exact_feasible_count","feasible_found",
         "first_objective_gp_fit_n","first_feasible_n"],initial_rows)
    _write_csv(output/"objective_learning_curves.csv",
        ["task","repeat","anchor_size","method","n","feasible_found",
         "best_exact_feasible","exact_feasible_count","exact_objective_count",
         "missing_objective_count","objective_gp_trainable",
         "first_objective_gp_fit_n","first_feasible_n"],outcomes)
    _write_csv(output/"risk_learning_curves.csv",
        ["task","repeat","anchor_size","method","n","head","brier",
         "brier_skill","auroc","auprc","ece","false_feasible",
         "feasible_recall"],risk_rows)
    _write_csv(output/"paired_initial_size.csv",
        ["task","repeat","method","left_anchor_size","right_anchor_size",
         "comparison","objective_delta_24_minus_12_if_both_feasible",
         "feasible_24","feasible_12","exact_objective_count_delta"],paired_sizes)
    _write_csv(output/"paired_methods.csv",
        ["task","repeat","anchor_size","method","comparator","comparison",
         "objective_delta_if_both_feasible","method_feasible","comparator_feasible"],
        cross_method)
    seal(output)
    return dict(status=result["status"],initial_rows=len(initial_rows),
        outcome_rows=len(outcomes),risk_rows=len(risk_rows),
        paired_initial_size_rows=len(paired_sizes),
        paired_method_rows=len(cross_method))
