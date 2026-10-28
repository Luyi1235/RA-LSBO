"""Comparison figures that retain infeasible and censored outcomes."""

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import ARMS


LABELS = {"ra_lsbo": "RA-LSBO", "random": "Random", "logcei": "LogCEI",
          "scbo_adapted": "SCBO (adapted)", "sego_utb_adapted": "SEGOMOE/UTB (adapted)",
          "nsga2_adapted": "NSGA-II (adapted)"}
COLORS = {"ra_lsbo": "#bf3636", "random": "#777777", "logcei": "#176d97",
          "scbo_adapted": "#32875d", "sego_utb_adapted": "#a16a15",
          "nsga2_adapted": "#775799"}


def trajectory_summary(rows):
    budgets = sorted({point["online_budget"] for row in rows for point in row["curve"]})
    values, fractions, counts = [], [], []
    for budget in budgets:
        points = [point for row in rows for point in row["curve"] if point["online_budget"] == budget]
        finite = [point["best_feasible"] for point in points if point["best_feasible"] is not None]
        values.append(float(np.median(finite)) if finite else np.nan)
        counts.append(len(finite))
        fractions.append(len(finite) / len(points) if points else np.nan)
    return np.array(budgets), np.array(values), np.array(fractions), np.array(counts)


def make_figures(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in result["summaries"]:
        grouped[(row["case"], row["method"])].append(row)
    cases = list(dict.fromkeys(row["case"] for row in result["summaries"]))
    figures = []
    for case in cases:
        methods = [method for method in ARMS if (case, method) in grouped]
        fig, (convergence, feasibility) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                                       layout="constrained")
        for method in methods:
            budget, median, fraction, _ = trajectory_summary(grouped[(case, method)])
            convergence.plot(budget, median, label=LABELS[method], color=COLORS[method],
                             linewidth=2.5 if method == "ra_lsbo" else 1.6, drawstyle="steps-post")
            feasibility.plot(budget, fraction * 100, color=COLORS[method],
                             linewidth=2.5 if method == "ra_lsbo" else 1.6, drawstyle="steps-post")
        convergence.set_title(f"{case}: performance and feasible discovery")
        convergence.set_ylabel("Median best feasible objective\n(feasible runs only; lower is better)")
        convergence.legend(ncol=2, fontsize=8)
        feasibility.set_ylabel("Runs with feasible result (%)")
        feasibility.set_xlabel("Sequential evaluations per method (excludes shared initial samples)")
        feasibility.set_ylim(-3, 103)
        for axis in (convergence, feasibility):
            axis.grid(alpha=.2)
        path = output / f"{case}_convergence.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figures.append(path)

        fig, axis = plt.subplots(figsize=(10, 5), layout="constrained")
        for index, method in enumerate(methods):
            rows = grouped[(case, method)]
            values = [row["best_feasible"] for row in rows if row["best_feasible"] is not None]
            if values:
                offsets = np.linspace(-.16, .16, len(values)) if len(values) > 1 else [0.]
                axis.scatter(np.array(offsets) + index, values, color=COLORS[method], alpha=.7, s=30)
                axis.plot([index-.24, index+.24], [np.median(values)] * 2, color="black", linewidth=2)
            axis.text(index, .02, f"{len(values)}/{len(rows)} feasible", transform=axis.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=8)
        axis.set_xticks(range(len(methods)), [LABELS[method] for method in methods], rotation=18, ha="right")
        axis.set_title(f"{case}: final objective by seed")
        axis.set_ylabel("Best feasible objective (lower is better)")
        axis.grid(axis="y", alpha=.2)
        path = output / f"{case}_final_objective.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figures.append(path)

        if any("flight_risk" in row for method in methods for row in grouped[(case, method)]):
            fig, axis = plt.subplots(figsize=(10, 5), layout="constrained")
            for index, method in enumerate(methods):
                rows = grouped[(case, method)]
                risks = [row["flight_risk"]["soc_failure_fraction_bounds"] for row in rows
                         if row.get("flight_risk", {}).get("status") == "evaluated"]
                if risks:
                    lower, upper = np.mean(risks, axis=0)
                    axis.vlines(index, lower * 100, upper * 100, color=COLORS[method], linewidth=5)
                    axis.scatter([index, index], [lower * 100, upper * 100], color=COLORS[method], s=30)
                axis.text(index, .02, f"{len(risks)}/{len(rows)} evaluated", transform=axis.get_xaxis_transform(),
                          ha="center", va="bottom", fontsize=8)
            axis.set_xticks(range(len(methods)), [LABELS[method] for method in methods], rotation=18, ha="right")
            axis.set_title(f"{case}: post-optimization SOC scenario risk")
            axis.set_ylabel("Mean observed violation fraction bounds (%)")
            axis.set_ylim(-3, 103)
            axis.grid(axis="y", alpha=.2)
            path = output / f"{case}_soc_risk.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            figures.append(path)
    return figures
