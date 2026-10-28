"""Additional comparable-scale figures from audited results; zero new MDA."""

import argparse
import hashlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.mechanism_system import METHODS
from ra_lsbo.system_experiment import save_new


COLORS = ("#087f72", "#d84a65", "#2866ba", "#9964b8", "#d88212", "#424b52", "#1b97af", "#7d8734")
STYLES = ("-", "--", "-.", ":", "-", "--", "-.", ":")


def plot(analysis_path, output=None):
    result = load_json(analysis_path.read_bytes())
    verify_sources(result["input_sha256"])
    output = output or analysis_path.parent
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for kind in ("controlled", "physical"):
        if kind not in result:
            continue
        trials = result[kind]["trials"]
        seeds = sorted({r["seed"] for r in trials})
        curves = {}
        for trial in trials:
            initial = trial["initial_best"]
            if initial is None or initial <= 0:
                raise ValueError("Relative figure requires a positive feasible shared start")
            curves[trial["method"], trial["seed"]] = (
                [p["calls"] for p in trial["curve"]],
                [100*(initial-p["best"])/initial for p in trial["curve"]])
        peak = max(v for x, y in curves.values() for v in y)
        span = max(peak, .01)
        fig, axes = plt.subplots(1, len(seeds), figsize=(12, 4.5), sharey=True, squeeze=False)
        for ax, seed in zip(axes[0], seeds):
            for method in [*METHODS[1:], METHODS[0]]:
                i = METHODS.index(method)
                x, y = curves[method, seed]
                ax.step(x, y, where="post", color=COLORS[i], linestyle=STYLES[i],
                        linewidth=2 if i == 0 else 1.5, label=method)
                ax.plot(x[-1], y[-1], marker="o", markersize=4, color=COLORS[i])
            ax.set(title="Seed "+str(seed), xlabel="Actual new calls", ylim=(-.025*span, 1.08*span))
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].set_ylabel("Improvement from shared start (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        order = [labels.index(m) for m in METHODS]
        fig.legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="lower center", ncol=4, frameon=False, fontsize=8)
        fig.suptitle(kind.capitalize()+": same scale, actual endpoints, all repetitions", fontsize=12)
        fig.tight_layout(rect=(0, .18, 1, .94))
        path = output/(kind+"_improvement.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
        names = list(METHODS)
        aggregates = {r["method"]: r for r in result[kind]["aggregate"]}
        feasible = [aggregates[m]["feasible_calls"] for m in names]
        calls = [aggregates[m]["calls"] for m in names]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        x = np.arange(len(names))
        axes[0].bar(x, feasible, color=COLORS, label="Confirmed feasible")
        axes[0].bar(x, np.asarray(calls)-feasible, bottom=feasible, color="#d5d8d8",
                    label="No confirmed feasible result")
        axes[0].set(xticks=x, xticklabels=names, ylabel="Actual new calls, all three seeds")
        axes[0].tick_params(axis="x", rotation=40, labelsize=8)
        trusted = ("ra_full", "ra_no_mechanisms", "ra_onehot_risk", "ra_no_boundary")
        charge = [sum(r["ledger"]["risk_charge"] for r in trials if r["method"] == m) for m in trusted]
        actual = [sum(r["ledger"]["observed_nonfeasible"] for r in trials if r["method"] == m) for m in trusted]
        x = np.arange(len(trusted))
        axes[1].bar(x-.18, charge, width=.36, color="#087f72", label="Charged estimated risk")
        axes[1].bar(x+.18, actual, width=.36, color="#d84a65", label="No confirmed feasible result")
        axes[1].set(xticks=x, xticklabels=trusted, ylabel="Total over three seeds")
        axes[1].tick_params(axis="x", rotation=30, labelsize=8)
        axes[1].set_title("Empirical risk debits are not a safety bound", fontsize=10)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=.18)
        handles = axes[0].get_legend_handles_labels()[0]+axes[1].get_legend_handles_labels()[0]
        fig.legend(handles, ["Feasible outcomes (left)", "Other outcomes (left)",
                             "Estimated risk debits (right)", "Other calls observed (right)"],
                   loc="lower center", ncol=2, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, .14, 1, 1))
        path = output/(kind+"_outcomes.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    save_new(output/"evidence_plot_provenance.json", {
        "additional_mda_calls": 0, "input_analysis": str(analysis_path),
        "analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in written},
        "interpretation": "Descriptive, common scales; no extrapolation, independence or coverage claim"})
    (output/"evidence_plot.source.py").write_bytes(Path(__file__).read_bytes())
    print("Additional evidence figures", [p.name for p in written], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(args.analysis.resolve(), None if args.output is None else args.output.resolve())
