"""Budget trajectories that retain stopped and unsuccessful seeds."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "passed":
        raise ValueError("Benchmark must be complete")
    styles = {
        "ra": ("#555555", "RA original"),
        "ra_fresh_discovery": ("#008775", "RA fresh learning"),
        "ra_constrained_discovery": ("#9c6213", "RA constraint discovery"),
        "cbo": ("#c54f50", "CBO"), "lsbo": ("#467cac", "Workspace LSBO"),
    }
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    curves = []
    for row_index, scenario in enumerate(report["scenarios"]):
        for algorithm in report["algorithms"]:
            color, label = styles[algorithm]
            runs = [row for row in report["runs"]
                    if row["scenario"] == scenario and row["algorithm"] == algorithm]
            histories = [[json.loads(line) for line in
                          (Path(run["directory"]) / "observations.jsonl").read_text().splitlines()]
                         for run in runs]
            budget = runs[0]["settings"]["budget"]
            start = runs[0]["settings"]["bootstrap"]
            allowance = list(range(start, budget+1))
            means, fractions = [], []
            for count in allowance:
                best = []
                for history in histories:
                    values = [row["risk"]["objective"] for row in history[:count]
                              if row["risk"]["feasible_label"] is True]
                    best.append(min(values) if values else None)
                means.append(float(np.mean(best)) if None not in best else None)
                fractions.append(sum(value is not None for value in best)/len(best))
            axes[row_index, 0].plot(allowance, [np.nan if x is None else x for x in means],
                                    color=color, label=label, linewidth=1.8)
            axes[row_index, 1].step(allowance, fractions, color=color, label=label, where="post")
            curves.append({"scenario": scenario, "algorithm": algorithm, "call_allowance": allowance,
                           "mean_best_only_if_all_seeds_feasible": means, "fraction_with_feasible": fractions})
        axes[row_index, 0].set_title(scenario.replace("_", " ")+": mean best, only when all seeds feasible", fontsize=10)
        axes[row_index, 0].ticklabel_format(axis="y", useOffset=False)
        axes[row_index, 1].set_title("Fraction of seeds with a feasible design", fontsize=10)
        axes[row_index, 1].set_ylim(-.05, 1.05)
        for axis in axes[row_index]:
            axis.set_xlabel("Call allowance (stopped runs retain their incumbent)")
            axis.set_xlim(start, budget)
            axis.grid(alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 1].legend(fontsize=8, loc="lower right")
    figure.suptitle("Budget trajectories with all seeds retained\n"
                    "A missing objective curve means at least one seed has no feasible design", fontsize=13)
    figure.savefig(report_path.parent / "budget_curves.png", dpi=180)
    plt.close(figure)
    with (report_path.parent / "budget_curves.json").open("w", encoding="utf-8") as stream:
        json.dump(curves, stream, indent=2, allow_nan=False)
    print("Budget curves:", report_path.parent / "budget_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    plot(args.report)
