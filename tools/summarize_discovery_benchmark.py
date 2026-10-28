"""Plot all paired seeds, retaining no-feasible runs and unequal actual calls."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compare(report, first, second, common_cost=False):
    if report["status"] != "passed":
        raise ValueError("Benchmark must finish before analysis")
    pairs = []
    for scenario in report["scenarios"]:
        old = {row["seed"]: row for row in report["runs"]
               if row["scenario"] == scenario and row["algorithm"] == first}
        new = {row["seed"]: row for row in report["runs"]
               if row["scenario"] == scenario and row["algorithm"] == second}
        differences, wins, losses, ties = [], 0, 0, 0
        costs = []
        for seed in report["seeds"]:
            if common_cost:
                count = min(old[seed]["evaluations"], new[seed]["evaluations"])
                costs.append(count)
                values = []
                for run in (old[seed], new[seed]):
                    history = [json.loads(line) for line in
                               (Path(run["directory"]) / "observations.jsonl").read_text().splitlines()]
                    feasible = [row["risk"]["objective"] for row in history[:count]
                                if row["risk"]["feasible_label"] is True]
                    values.append(min(feasible) if feasible else None)
                a, b = values
            else:
                a, b = old[seed]["best_feasible_objective"], new[seed]["best_feasible_objective"]
            if a is None and b is None:
                ties += 1
            elif a is None:
                wins += 1
            elif b is None:
                losses += 1
            else:
                differences.append(b-a)
                wins += b < a-1e-9
                losses += b > a+1e-9
                ties += abs(a-b) <= 1e-9
        pairs.append({
            "scenario": scenario, "reference": first, "comparison": second,
            "comparison_wins": wins, "comparison_losses": losses, "ties": ties,
            "cost_basis": "matched_minimum_actual_calls" if common_cost else "final_actual_calls",
            "paired_comparison_calls": costs,
            "paired_best_differences_when_both_feasible": differences,
            "mean_difference_only_if_all_pairs_feasible":
                statistics.mean(differences) if len(differences) == len(report["seeds"]) else None,
            "interpretation": "Lower objective is better; feasible beats no-feasible; no statistical superiority claim.",
        })
    return pairs


def summarize(report):
    pairs = compare(report, "ra", "ra_fresh_discovery")
    if "ra_constrained_discovery" in report["algorithms"]:
        for reference in ("ra_fresh_discovery", "ra", "cbo", "lsbo"):
            for common in (False, True):
                pairs.extend(compare(report, reference, "ra_constrained_discovery", common))
    return pairs


def plot(report, destination):
    styles = {
        "ra": ("#555555", "RA (original)"),
        "ra_fresh_discovery": ("#008775", "RA (fresh discovery)"),
        "ra_constrained_discovery": ("#9c6213", "RA (constraint discovery)"),
        "cbo": ("#c54f50", "CBO"), "lsbo": ("#467cac", "Workspace LSBO"),
    }
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for row_index, scenario in enumerate(report["scenarios"]):
        objective_axis, cost_axis = axes[row_index]
        for method_index, algorithm in enumerate(report["algorithms"]):
            color, label = styles[algorithm]
            runs = sorted((row for row in report["runs"]
                           if row["scenario"] == scenario and row["algorithm"] == algorithm),
                          key=lambda row: row["seed"])
            x = [row["seed"]+(method_index-(len(report["algorithms"])-1)/2)*.12 for row in runs]
            objective = [row["best_feasible_objective"] for row in runs]
            present = [(a, b) for a, b in zip(x, objective) if b is not None]
            objective_axis.scatter([a for a, _ in present], [b for _, b in present],
                                   color=color, s=40, label=label)
            for a, b in zip(x, objective):
                if b is None:
                    objective_axis.text(a, .96-method_index*.06, "no feasible", rotation=90,
                                        transform=objective_axis.get_xaxis_transform(),
                                        color=color, va="top", ha="center", fontsize=8)
            cost_axis.scatter(x, [row["evaluations"] for row in runs],
                              color=color, s=40, label=label)
        objective_axis.set_title(scenario.replace("_", " ") + ": final best (lower is better)", fontsize=11)
        cost_axis.set_title("Actual charged calls, including 88 initial", fontsize=11)
        cost_axis.axhline(112, color="#777777", linestyle=":", linewidth=1)
        cost_axis.set_ylim(85, 115)
        objective_axis.ticklabel_format(axis="y", useOffset=False)
        for axis in (objective_axis, cost_axis):
            axis.set_xticks(report["seeds"])
            axis.set_xlabel("Paired seed")
            axis.grid(axis="y", alpha=.18)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, loc="best")
    figure.suptitle(f"Discovery validation: fixed budgets, seeds {report['seeds'][0]}-{report['seeds'][-1]}\n"
                    "Synthetic diagnostics; all early stops and unsuccessful runs retained", fontsize=14)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    paired = summarize(report)
    with (args.report.parent / "paired_comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(paired, stream, indent=2, allow_nan=False)
    plot(report, args.report.parent / "paired_results.png")
    print(json.dumps(paired, indent=2))
