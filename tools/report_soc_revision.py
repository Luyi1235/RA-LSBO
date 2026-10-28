"""Analyze fixed independent-pilot outcomes; never fit or evaluate a design."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.soc_revision.acquisition import quantities
from tools.report_partial_refinement_study import probability_scores
from tools.run_partial_bo_development import lines, read, sha, write_new
from tools.run_soc_revision_study import ARMS, SEEDS, TASKS


LABELS = ("Full revision", "Kernel average only", "Probability exploitation only",
          "Frozen grouped + local", "Revision without intervals")
COLORS = ("#127c70", "#936c15", "#7652a3", "#42464d", "#b3465d")
MARKERS = ("o", "^", "s", "D", "v")


def ordering(new, old, tolerance):
    if new is None or old is None:
        return 0 if new is old else 1 if new is None else -1
    return 0 if abs(new-old) <= tolerance else -1 if new < old else 1


def analyze(folder):
    results, plan = read(folder/"results.json"), read(folder/"plan.json")
    source_snapshot = folder/"report_source.source"
    with source_snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    rows, aggregate, comparisons, panels, traces = results["summaries"], [], [], [], []
    for task in TASKS:
        for method in ARMS:
            group = sorted((row for row in rows if row["task"] == task and row["method"] == method),
                           key=lambda row: row["seed"])
            aggregate.append({"task": task, "method": method,
                "endpoints": [row["endpoint_best"] for row in group],
                "initial_best": [row["initial_best"] for row in group],
                "first_feasible": [row["first_online_feasible"] for row in group],
                "feasible_runs": sum(row["endpoint_best"] is not None for row in group),
                "online_calls": sum(row["online_calls"] for row in group),
                "online_feasible": sum(row["usable_feasible_online"] for row in group),
                "soc_aborts": sum(row["soc_reserve_aborts"] for row in group),
                "unknown": sum(row["physical_unknown_online"] for row in group),
                "hidden_failures": sum(row["hidden_failure_online"] for row in group),
                "seconds": sum(row["seconds"] for row in group)})
            for seed in SEEDS:
                path = folder/"optimization"/task/(method+"_"+str(seed))
                observations = lines(path/"observations.jsonl")
                for proposal, observation in zip(lines(path/"proposals.jsonl"), observations):
                    meta = proposal["metadata"]
                    index = next(index for index, candidate in enumerate(meta["pool"])
                                 if candidate == proposal["candidate"])
                    values = quantities(meta["forecasts"], meta["best_observed"], meta["kappa"], .5)
                    traces.append({"task": task, "method": method, "seed": seed,
                        "iteration": proposal["iteration"], "mode": meta["mode"],
                        "soc_probability": None if "soc" not in values["probabilities"] else
                            values["probabilities"]["soc"][index],
                        "soc_abort": observation["diagnostics"]["termination_reason"] == "physical_soc_reserve_abort",
                        "selected_from_local": meta["selected_from_local_augmentation"],
                        "local_queries": sum(entry["unique_posterior_queries"] for entry in meta["local_reports"])})
        lookup = {(row["seed"], row["method"]): row for row in rows if row["task"] == task}
        for control in ARMS[1:]:
            raw = [ordering(lookup[seed, ARMS[0]]["endpoint_best"], lookup[seed, control]["endpoint_best"], 0.)
                   for seed in SEEDS]
            practical = [ordering(lookup[seed, ARMS[0]]["endpoint_best"], lookup[seed, control]["endpoint_best"],
                                 plan["practical_ties"][task]) for seed in SEEDS]
            comparisons.append({"task": task, "control": control,
                "raw_wins_ties_losses": [raw.count(value) for value in (-1, 0, 1)],
                "practical_wins_ties_losses": [practical.count(value) for value in (-1, 0, 1)]})
        for seed in SEEDS:
            initial = read(folder/"optimization"/task/(ARMS[0]+"_"+str(seed))/"manifest.json")["initial"]
            rate = (sum(DesignRecord.from_dict(row).usable_feasible for row in initial)+.5)/(len(initial)+1)
            for forecast in read(folder/"common_panel"/(task+"_"+str(seed)+".json")):
                panels.append({"task": task, "seed": seed, "method": forecast["method"],
                    **probability_scores(forecast["probabilities"], forecast["labels"], rate)})
    indexed = {(row["task"], row["method"]): row for row in aggregate}
    order = {row["task"]: row["practical_wins_ties_losses"] for row in comparisons if row["control"] == "frozen_full"}
    air, base = indexed["aircraft_140km", "revised_full"], indexed["aircraft_140km", "frozen_full"]
    bra, bra_base = indexed["hierarchical_branin", "revised_full"], indexed["hierarchical_branin", "frozen_full"]
    checks = {"aircraft_fewer_soc_aborts": air["soc_aborts"] < base["soc_aborts"],
        "aircraft_no_fewer_feasible_runs": air["feasible_runs"] >= base["feasible_runs"],
        "aircraft_more_practical_wins_than_losses": order["aircraft_140km"][0] > order["aircraft_140km"][2],
        "branin_no_fewer_feasible_runs": bra["feasible_runs"] >= bra_base["feasible_runs"],
        "branin_no_more_practical_losses_than_wins": order["hierarchical_branin"][2] <= order["hierarchical_branin"][0]}
    write_new(folder/"analysis.json", {"aggregate": aggregate, "comparisons": comparisons,
        "common_panel_scores": panels, "selected_point_diagnostics": traces,
        "development_gate": {"checks": checks, "passed": all(checks.values())},
        "result_sha256": sha(folder/"results.json"), "tool_sha256": sha(source_snapshot),
        "new_truth_calls": 0, "paper_confirmation": False,
        "scope": "Three independent seeds on development-selected tasks; no final statistical superiority"})
    figure(folder, rows)
    print({"aggregate": aggregate, "comparisons": comparisons, "gate": checks}, flush=True)


def figure(folder, rows):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    for index, task in enumerate(TASKS):
        for method_index, (method, color, marker) in enumerate(zip(ARMS, COLORS, MARKERS)):
            for seed_index, seed in enumerate(SEEDS):
                row = next(row for row in rows if (row["task"], row["method"], row["seed"]) == (task, method, seed))
                position, value = seed_index+(method_index-2)*.13, row["endpoint_best"]
                if value is None:
                    axes[index].annotate("no feasible", (position, .96), xycoords=("data", "axes fraction"),
                        rotation=90, ha="center", va="top", fontsize=7, color=color)
                else:
                    axes[index].scatter(position, value/(1000 if index else 1), marker=marker, color=color, s=35)
                if index == 1:
                    axes[2].scatter(position, row["soc_reserve_aborts"], marker=marker, color=color, s=35)
    for axis in axes:
        axis.set_xticks(range(3), [str(seed) for seed in SEEDS])
        axis.set_xlabel("Paired independent seed")
        axis.grid(axis="y", alpha=.2)
    axes[0].set_title("Hierarchical Branin")
    axes[0].set_ylabel("Best feasible objective")
    axes[0].set_yscale("log")
    axes[1].set_title("Reduced aircraft, 140 km")
    axes[1].set_ylabel("Best feasible mission energy (kWh)")
    axes[2].set_title("Aircraft SOC reserve aborts")
    axes[2].set_ylabel("Aborts out of 12 online calls")
    axes[2].set_ylim(-.5, 12.5)
    handles = [Line2D([], [], linestyle="none", marker=marker, color=color, label=label)
               for marker, color, label in zip(MARKERS, COLORS, LABELS)]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=8)
    fig.savefig(folder/"soc_revision_comparisons.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    analyze(parser.parse_args().folder)
