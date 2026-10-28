"""Create a reproducible figure and compact table from frozen validation reports."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def name_of(cohort):
    return "Own holdout" if cohort == "source_holdout" else Path(cohort).parent.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    if any(r["status"] != "passed" for r in reports):
        raise ValueError("Only completed validation reports can be plotted")
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    table = []
    for column, report in enumerate(reports):
        source_name = Path(report["source_run"]).parent.name
        cohorts = list(report["scores"])
        raw, calibrated = [], []
        for cohort in cohorts:
            scores = report["scores"][cohort]["heads"]["final_feasibility"]
            raw.append(scores["raw_all"]["brier"])
            calibrated.append(scores["calibrated_only"]["brier"])
            table.append({
                "source": source_name, "test": name_of(cohort),
                "n": scores["raw_all"]["samples"], "feasible": scores["raw_all"]["positives"],
                "raw_brier": scores["raw_all"]["brier"],
                "calibrated_brier": scores["calibrated_only"]["brier"],
                "raw_log_loss": scores["raw_all"]["log_loss"],
                "calibrated_log_loss": scores["calibrated_only"]["log_loss"],
                "gate": report["scores"][cohort]["gate"],
            })
        x = np.arange(len(cohorts))
        ax = axes[0, column]
        ax.bar(x-.18, raw, width=.36, color="#64748b", label="Raw")
        if all(value is not None for value in calibrated):
            ax.bar(x+.18, calibrated, width=.36, color="#168878", label="Calibrated")
        else:
            ax.text(.98, .94, "Calibration unavailable\n57 feasible / 3 infeasible in calibration",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.set_xticks(x, [name_of(c) for c in cohorts], rotation=18, ha="right")
        ax.set_ylim(0, .16)
        ax.set_ylabel("Brier score (lower is better)")
        ax.set_title(f"Frozen model fitted on {source_name} history")
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        target = "source_holdout"
        score = report["scores"][target]["heads"]["final_feasibility"]
        ax = axes[1, column]
        ax.plot([0, 1], [0, 1], color="#a3a3a3", linestyle="--", linewidth=1)
        for key, label, color in (("raw_all", "Raw", "#64748b"),
                                  ("calibrated_only", "Calibrated", "#168878")):
            bins = [b for b in score[key]["reliability"] if b["samples"]]
            if bins:
                ax.plot([b["mean_probability"] for b in bins],
                        [b["observed_fraction"] for b in bins], "o-", color=color, label=label)
                for b in bins:
                    ax.annotate(str(b["samples"]), (b["mean_probability"], b["observed_fraction"]),
                                xytext=(5, -13 if label == "Raw" else 7),
                                textcoords="offset points", fontsize=8, color=color)
        ax.set(xlim=(-.04, 1.07), ylim=(-.04, 1.10),
               xlabel="Mean predicted feasibility", ylabel="Observed feasible fraction")
        ax.set_title("Own chronological holdout; labels show bin counts")
        ax.legend(loc="lower right")
        ax.grid(alpha=.18)
    fig.suptitle("X57 feasibility calibration across historical search policies\n"
                 "Same v4 evaluator; shared initial designs removed; no new MDA", fontsize=14)
    fig.savefig(args.output / "transfer_validation.png", dpi=180)
    plt.close(fig)
    (args.output / "summary.json").write_text(
        json.dumps({"reports": [str(p.resolve()) for p in args.reports], "rows": table},
                   indent=2, allow_nan=False), encoding="utf-8")
    for row in table:
        print(json.dumps(row, ensure_ascii=True))
    print("Figure:", args.output / "transfer_validation.png")


if __name__ == "__main__":
    main()
