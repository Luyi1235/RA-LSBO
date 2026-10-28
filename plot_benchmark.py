"""Plot every paired run, preserving early stops and missing feasible solutions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(report_path):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "passed":
        raise ValueError("Only plot a completed, source-verified benchmark")
    algorithms = report["algorithms"]
    scenarios = report["scenarios"]
    labels = {"ra": "RA-LSBO", "ra_no_boundary": "No boundary", "cbo": "CBO",
              "lsbo": "Workspace LSBO", "random": "Random"}
    colors = {"ra": "#177e89", "ra_no_boundary": "#aa6c20", "cbo": "#335da8",
              "lsbo": "#36834a", "random": "#9c516f"}
    fig, axes = plt.subplots(2, len(scenarios), figsize=(15, 8), squeeze=False, layout="constrained")
    for column, scenario in enumerate(scenarios):
        top, bottom = axes[:, column]
        missing = []
        for position, algorithm in enumerate(algorithms):
            runs = [run for run in report["runs"]
                    if run["scenario"] == scenario and run["algorithm"] == algorithm]
            jitter = np.linspace(-.16, .16, len(runs))
            feasible = sum(run["best_feasible_objective"] is not None for run in runs)
            if feasible < len(runs):
                missing.append(f"{labels[algorithm]}: {feasible}/{len(runs)} feasible")
            for offset, run in zip(jitter, runs):
                if run["best_feasible_objective"] is not None:
                    top.scatter(position+offset, run["best_feasible_objective"],
                                color=colors[algorithm], s=42)
                bottom.scatter(position+offset, run["evaluations"], color=colors[algorithm], s=42,
                               marker="x" if run["status"] != "complete" else "o")
        top.set(title=scenario.replace("_", " "), ylabel="Final best feasible objective")
        bottom.set(ylabel="Charged synthetic calls", ylim=(0, report["runs"][0]["settings"]["budget"]+8))
        bottom.axhline(report["runs"][0]["settings"]["budget"], color="#777777", linestyle="--", linewidth=1)
        if missing:
            top.set_title(scenario.replace("_", " ") + "\n" + "; ".join(missing), fontsize=10)
        for axis in (top, bottom):
            axis.set_xticks(range(len(algorithms)), [labels[name] for name in algorithms],
                            rotation=25, ha="right")
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
    fig.suptitle("Paired synthetic pilot: each dot is one seed; x marks an early stop", fontsize=15)
    output = report_path.parent / "paired_results.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print("Figure:", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    plot(parser.parse_args().report)
