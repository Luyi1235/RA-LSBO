"""Render already audited study results; never evaluate the controlled tasks."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(folder):
    result = json.loads((folder/"result.json").read_text(encoding="utf-8"))
    audit = json.loads((folder/"audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "passed":
        raise ValueError("Audited results required")
    records = [json.loads(s) for s in (folder/"completed.jsonl").read_text(
        encoding="utf-8").splitlines() if s.strip()]
    selected = result["selection"]["selected_family"]
    groups = {
        "screen": ("mixed_eicb", "ensemble_boundary", "calibrated_eicb", "lsbo"),
        "heldout": ("selected_core", "selected_integrated", "lsbo"),
    }
    labels = {
        "mixed_eicb": "Mixed GP",
        "ensemble_boundary": "Boundary ensemble",
        "calibrated_eicb": "Calibrated mixed GP",
        "lsbo": "Original LSBO",
        "selected_core": "Selected core",
        "selected_integrated": "Core + latent + stages",
    }
    colors = {
        "mixed_eicb": "#147d64", "ensemble_boundary": "#ab4665",
        "calibrated_eicb": "#bc9229", "lsbo": "#505050",
        "selected_core": "#147d64", "selected_integrated": "#375daf",
    }
    tasks = ("complete", "partial_sparse")
    task_names = ("Complete outputs", "Sparse / partial outputs")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    for column, (phase, title) in enumerate((
            ("screen", "Development seeds"), ("heldout", "Held-out seeds"))):
        methods = groups[phase]
        x = np.arange(2)
        width = .8/len(methods)
        for i, method in enumerate(methods):
            values = []
            costs = []
            for task in tasks:
                rows = [r for r in records if (
                    r["phase"], r["task"], r["method"]) == (phase, task, method)]
                calls = sum(r["online_calls"] for r in rows)
                feasible = sum(r["online_feasible_count"] for r in rows)
                values.append(feasible/calls if calls else 0.)
                costs.append(np.mean([r["restricted_first_feasible_cost"] for r in rows]))
            positions = x-.4+width*(i+.5)
            bars = axes[0, column].bar(positions, values, width=width*.9,
                                      color=colors[method], label=labels[method])
            for bar, value in zip(bars, values):
                axes[0, column].annotate(
                    f"{value:.0%}", (bar.get_x()+bar.get_width()/2, value),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
            axes[1, column].bar(positions, costs, width=width*.9, color=colors[method])
        for row in range(2):
            ax = axes[row, column]
            ax.set_xticks(x, task_names)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color="#e3e3e3", linewidth=.6)
        axes[0, column].set_title(title)
        axes[0, column].set_ylim(0, 1.08)
        axes[0, column].set_ylabel("Observed feasible calls / actual online calls")
        axes[0, column].legend(loc="upper right", frameon=False, fontsize=8)
        axes[1, column].set_ylabel("Mean restricted first-feasible cost")
    fig.suptitle(f"Independent risk-core selection: {selected}", fontsize=14)
    fig.supxlabel(
        f"{result['new_truth_labels']} controlled labels, 0 physical MDA calls; "
        "4 paired seeds per task and phase. Descriptive results, not safety guarantees.",
        fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(folder/("comparison."+suffix), dpi=180)
    plt.close(fig)
    print(folder/"comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    render(parser.parse_args().output.resolve())
