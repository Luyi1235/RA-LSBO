"""Summarize registered refinement comparisons without any truth queries."""

import argparse
from collections import Counter
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.report_partial_bo_development import paired_order
from ra_lsbo.partial_bo.records import DesignRecord
from tools.run_partial_bo_development import lines, read, write_new
from tools.run_partial_refinement_study import ARMS, SEEDS, TASKS

LABELS = ("Grouped latent + local", "Grouped one-hot + local", "Grouped latent global",
          "Legacy kernel + local", "Original LSBO")
COLORS = ("#147d73", "#7652a3", "#b56825", "#b54153", "#3d4248")


def probability_scores(probability, labels, training_rate):
    probability, labels = np.asarray(probability, float), np.asarray(labels, bool)
    error = (probability-labels)**2
    positive = float(np.mean(error[labels])) if labels.any() else None
    negative = float(np.mean(error[~labels])) if (~labels).any() else None
    clipped = np.clip(probability, 1e-12, 1-1e-12)
    return {"n": len(labels), "positive": int(labels.sum()), "brier": float(error.mean()),
        "positive_brier": positive, "negative_brier": negative,
        "balanced_brier": (positive+negative)/2 if positive is not None and negative is not None else None,
        "log_loss": float(-np.mean(labels*np.log(clipped)+(~labels)*np.log(1-clipped))),
        "smoothed_training_rate": training_rate,
        "training_rate_constant_brier": float(np.mean((training_rate-labels)**2))}


def report(folder):
    results = read(folder/"results.json")
    control = read(folder/"lsbo_full_budget_control/results.json")
    if not control["passed"]:
        raise ValueError("Secondary control audit incomplete")
    rows = results["summaries"]+control["summaries"]
    methods = (*ARMS, "original_lsbo_full_budget")
    tasks = [task for task in TASKS if any(row["task"] == task for row in rows)]
    aggregate, contrasts, search, panel_scores = [], [], [], []
    for task in tasks:
        for method in methods:
            selected = sorted((row for row in rows if row["task"] == task and row["method"] == method), key=lambda row: row["seed"])
            aggregate.append({"task": task, "method": method,
                "online_calls": sum(row["online_calls"] for row in selected),
                "online_feasible": sum(row["usable_feasible_online"] for row in selected),
                "endpoint_feasible_trials": sum(row["endpoint_best"] is not None for row in selected),
                "endpoints": [row["endpoint_best"] for row in selected],
                "first_online_feasible": [row["first_online_feasible"] for row in selected],
                "seconds": sum(row["seconds"] for row in selected) if all("seconds" in row for row in selected) else None,
                "mean_relative_improvement": float(np.mean([row["relative_improvement"] for row in selected]))
                if selected and all(row["relative_improvement"] is not None for row in selected) else None,
                "partial_objectives": sum(row["partial_objective_online"] for row in selected),
                "hidden_failures": sum(row["hidden_failure_online"] for row in selected)})
            if method.startswith("original_lsbo"):
                continue
            proposals = [entry for seed in SEEDS for entry in lines(
                folder/"optimization"/task/(method+"_"+str(seed))/"proposals.jsonl")]
            reports = [entry for proposal in proposals for entry in proposal["metadata"]["local_reports"]]
            kernels = {}
            for proposal in proposals:
                for name, head in proposal["metadata"]["heads"].items():
                    selected_kernel = head["selected"]
                    key = (str(selected_kernel["length"]) if "length" in selected_kernel else
                           str((selected_kernel["continuous_length"], selected_kernel["discrete_length"])))
                    kernels.setdefault(name, Counter())[key] += 1
            search.append({"task": task, "method": method, "proposals": len(proposals),
                "selected_from_local": sum(proposal["metadata"]["selected_from_local_augmentation"] for proposal in proposals),
                "local_solves": len(reports), "local_solver_successes": sum(entry["success"] for entry in reports),
                "local_posterior_queries": sum(entry["unique_posterior_queries"] for entry in reports),
                "kernel_selections": kernels,
                "fallbacks": Counter(str(proposal["metadata"]["fallback"]) for proposal in proposals)})
        lookup = {(row["seed"], row["method"]): row for row in rows if row["task"] == task}
        for comparator in methods[1:]:
            order = [paired_order(lookup[seed, ARMS[0]]["endpoint_best"], lookup[seed, comparator]["endpoint_best"])
                     for seed in SEEDS]
            contrasts.append({"task": task, "control": comparator,
                "full_wins": order.count(-1), "ties": order.count(0), "full_losses": order.count(1),
                "interpretation": "raw numerical endpoint order, not engineering significance"})
        for seed in SEEDS:
            panel = read(folder/"common_panel"/(task+"_"+str(seed)+".json"))
            initial = read(folder/"optimization"/task/(ARMS[0]+"_"+str(seed))/"manifest.json")["initial"]
            training_rate = (sum(DesignRecord.from_dict(packet).usable_feasible for packet in initial)+.5)/(len(initial)+1)
            panel_scores.extend({"task": task, "seed": seed, "method": entry["method"],
                "joint_proxy_brier": entry["joint_proxy_brier"],
                "descriptive_class_scores": probability_scores(entry["probabilities"], entry["labels"], training_rate)}
                for entry in panel["forecasts"])
    write_new(folder/"analysis.json", {"aggregate": aggregate, "paired_order": contrasts, "search": search,
        "common_panel_scores": panel_scores, "qualification": results["qualification"],
        "new_model_calls": control["campaign_new_model_calls"], "new_rcaide_calls": 0,
        "primary_calls": results["new_model_calls"], "secondary_control_calls": control["new_model_calls"],
        "secondary_control": "only stopping threshold changed; verified prefixes reused; registered after observing primary early stop",
        "class_score_analysis": "supplementary post-hoc description; not used for model selection or superiority gates",
        "interpretation": "three-seed development only; no stable superiority or causal flight-safety claim"})
    figure(folder, tasks, rows, methods)
    print({"aggregate": aggregate, "paired_order": contrasts}, flush=True)


def figure(folder, tasks, rows, methods):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.3*len(tasks), 5.2), squeeze=False, constrained_layout=True)
    markers = ("o", "s", "^", "v", "D", "P")
    labels = (*LABELS, "LSBO full budget")
    colors = (*COLORS, "#2876ab")
    for index, task in enumerate(tasks):
        axis = axes[0, index]
        for arm, (method, color) in enumerate(zip(methods, colors)):
            for seed_index, seed in enumerate(SEEDS):
                row = next(row for row in rows if (row["task"], row["method"], row["seed"]) == (task, method, seed))
                value = row["endpoint_best"]
                position = seed_index+(arm-2.5)*.12
                if value is None:
                    axis.annotate("no feasible", (position, .96), xycoords=("data", "axes fraction"),
                                  ha="center", va="top", rotation=90, fontsize=8, color=color)
                else:
                    axis.scatter(position, value/(1000. if task == "aircraft_140km" else 1.), color=color,
                                 marker=markers[arm], s=42)
        axis.set_xticks(range(3), [str(seed) for seed in SEEDS])
        axis.set_xlabel("Paired seed; 12 initial + up to 12 online calls")
        axis.set_ylabel("Best feasible mission energy (kWh)" if task == "aircraft_140km" else "Best feasible Branin objective")
        axis.set_title("Reduced electric aircraft, 140 km" if task == "aircraft_140km" else "Hierarchical Branin extension")
        if task != "aircraft_140km":
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=.2)
    handles = [Line2D([], [], linestyle="none", marker=marker, color=color, label=label)
               for marker, color, label in zip(markers, colors, labels)]
    axes[0, -1].legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle("Grouped mixed GP and local refinement: development comparisons", fontsize=12)
    fig.savefig(folder/"refinement_endpoints.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    report(parser.parse_args().folder)
