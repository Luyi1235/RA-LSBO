"""Generate scientific figures and digital data for the October 25 studies."""

import argparse
import csv
import hashlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.system_experiment import save_new
from tools.qualify_compact_mda import wilson

METHODS = ("posterior_full", "feasibility_first", "lsbo")
LABELS = ("Posterior sampling", "Feasibility first", "Original LSBO")
COLORS = ("#b64243", "#16847e", "#375b89")


def csv_rows(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(output):
    output.mkdir(parents=True, exist_ok=False)
    study = ROOT/"runs/discovery_policy_20261025"
    joint = ROOT/"runs/joint_constraints_20261025"
    audit = load_json((study/"audit.json").read_bytes())
    if audit["status"] != "passed":
        raise ValueError("Completed acquisition replay audit required")
    rows = [load_json(line) for line in (study/"completed.jsonl").read_bytes().splitlines()]
    zero = audit["zero_initial_seeds"]
    lookup = {(r["seed"], r["method"]): r for r in rows}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    curves = []
    for method, label, color in zip(METHODS, LABELS, COLORS):
        values, lower, upper = [], [], []
        for budget in range(7):
            events = sum(lookup[(s, method)]["first_feasible_online_call"] is not None
                         and lookup[(s, method)]["first_feasible_online_call"] <= budget for s in zero)
            lo, hi = wilson(events, len(zero))
            values.append(events/len(zero))
            lower.append(lo)
            upper.append(hi)
            curves.append({"method": method, "additional_calls": budget, "initial_calls": 6,
                           "eligible_paired_seeds": len(zero), "discovered": events,
                           "fraction": events/len(zero), "pointwise_wilson_lower": lo,
                           "pointwise_wilson_upper": hi})
        axes[0].step(range(7), values, where="post", color=color, label=label)
        axes[0].fill_between(range(7), lower, upper, step="post", color=color, alpha=.07)
    axes[0].set(xlim=(0, 6), ylim=(0, 1.02), xlabel="Additional evaluations after 6 shared initial",
                ylabel="Fraction with first feasible design", title="A  Discovery from zero (15 paired seeds)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    for seed in zero:
        values = [lookup[(seed, method)]["restricted_first_feasible_cost"] for method in METHODS]
        axes[1].plot(range(3), values, color=".82", alpha=.65, linewidth=.7)
    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        for j, seed in enumerate(zero):
            row = lookup[(seed, method)]
            axes[1].scatter(i+(j-(len(zero)-1)/2)*.013, row["restricted_first_feasible_cost"],
                            color=color, marker="o" if row["first_feasible_event"] else "x",
                            s=24, alpha=.8)
        mean = np.mean([lookup[(s, method)]["restricted_first_feasible_cost"] for s in zero])
        axes[1].plot([i-.16, i+.16], [mean, mean], color="black", linewidth=2)
    axes[1].set(ylim=(.5, 6.5), ylabel="min(first-feasible call, 6)", yticks=range(1, 7),
                xticks=range(3), xticklabels=("Posterior", "Feasibility\nfirst", "LSBO"),
                title="B  Paired discovery cost")
    axes[1].text(.5, -.22, "Circle: event; cross: budget-censored; black: mean",
                 ha="center", transform=axes[1].transAxes, fontsize=8)
    for offset, field, color, label in ((-.17, "online_feasible_count", "#16847e", "Feasible"),
                                        (.17, "hidden_failure_count", "#b64243", "Hidden failure")):
        for i, method in enumerate(METHODS):
            values = np.asarray([r[field] for r in rows if r["method"] == method])
            rng = np.random.default_rng(20261025)
            means = values[rng.integers(20, size=(20000, 20))].mean(1)
            lo, hi = np.quantile(means, [.025, .975])
            mean = values.mean()
            axes[2].bar(i+offset, mean, width=.30, color=color,
                        label=label if i == 0 else None)
            axes[2].errorbar(i+offset, mean, yerr=[[mean-lo], [hi-mean]], color="black",
                            capsize=2, linewidth=.8)
            axes[2].text(i+offset, hi+.10, str(int(values.sum()))+"/120", ha="center", fontsize=8)
    axes[2].set(ylim=(0, 4.1), ylabel="Mean calls per run (6-call horizon)",
                xticks=range(3), xticklabels=("Posterior", "Feasibility\nfirst", "LSBO"),
                title="C  All 20 paired seeds")
    axes[2].legend(frameon=False, fontsize=9)
    fig.suptitle("Cold-start policy diagnostic on a previously studied equation", fontsize=12)
    primary = audit["primary_paired_difference_feasibility_first_minus_posterior"]
    fig.text(.5, .012, "Primary paired difference (feasibility first minus posterior): "
             f'{primary["mean_candidate_minus_reference"]:.2f} calls; '
             f'95% seed bootstrap [{primary["paired_bootstrap_percentile_95"][0]:.2f}, '
             f'{primary["paired_bootstrap_percentile_95"][1]:.2f}]; exact p={primary["exact_two_sided_sign_flip_p"]:.2f}. '
             "Development evidence.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .94), w_pad=2)
    fig.savefig(output/"discovery_policy.png", dpi=180)
    fig.savefig(output/"discovery_policy.pdf")
    plt.close(fig)
    csv_rows(output/"discovery_curves.csv", curves)
    result = load_json((joint/"result.json").read_bytes())
    rank0 = load_json((joint/"rank_0.json").read_bytes())
    rank2 = load_json((joint/"rank_2.json").read_bytes())
    matrix = np.asarray(rank2["task_kernel_correlation"])
    names = [name.replace("_", " ") for name in rank2["constraint_names"]]
    fig, axis = plt.subplots(figsize=(7.6, 6.7))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set(xticks=range(6), yticks=range(6), xticklabels=names, yticklabels=names,
             title="Fitted ICM task correlation (rank 2)")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for i in range(6):
        for j in range(6):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                       color="white" if abs(matrix[i, j]) > .65 else "black", fontsize=10)
    fig.colorbar(image, ax=axis, label="Task-kernel correlation", shrink=.8)
    fig.text(.5, .025, "128 historical training designs; model assumption, not verified physical correlation.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(output/"joint_correlations.png", dpi=180)
    fig.savefig(output/"joint_correlations.pdf")
    plt.close(fig)
    a, b = rank0["predictions"], rank2["predictions"]
    if [r["candidate_key"] for r in a] != [r["candidate_key"] for r in b]:
        raise ValueError("Joint-GP held-out candidate order differs")
    scores = [{"candidate_key": x["candidate_key"], "nll_rank0": x["predictive_joint_nll"],
               "nll_rank2": y["predictive_joint_nll"], "brier_rank0": x["brier"],
               "brier_rank2": y["brier"], "joint_probability_rank0": x["raw_joint_feasibility"],
               "joint_probability_rank2": y["raw_joint_feasibility"],
               "observed": x["observed_joint_satisfaction"]} for x, y in zip(a, b)]
    csv_rows(output/"joint_paired_scores.csv", scores)
    inputs = [study/"audit.json", study/"completed.jsonl", joint/"result.json",
              joint/"rank_0.json", joint/"rank_2.json", Path(__file__).resolve()]
    save_new(output/"manifest.json", {
        "scope": "Diagnostic figures, not independent confirmation or optimization superiority",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
                          if p.is_file()},
        "joint_model_metrics": result["models"],
    })
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    plot(parser.parse_args().output.resolve())
