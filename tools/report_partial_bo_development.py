"""Read-only analysis of frozen development records; no problem oracle."""

import argparse
from collections import Counter
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.records import DesignRecord
from tools.run_partial_bo_development import ARMS, HORIZON, SEEDS, TASKS, lines, read, write_new
from tools.qualify_partial_aircraft_envelope import CAMPAIGN_MAXIMUM, verify

LABELS = {"ra_latent": "RA latent", "ra_onehot": "RA one-hot",
          "ra_no_intervals": "RA no intervals", "original_lsbo": "Original LSBO"}
COLORS = ("#147d73", "#7652a3", "#b56825", "#3d4248")


def paired_order(left, right):
    if left is None or right is None:
        return 0 if left is right else 1 if left is None else -1
    return 0 if np.isclose(left, right, rtol=1e-10, atol=1e-10) else -1 if left < right else 1


def summarize(folder):
    result = read(folder/"results.json")
    rows, aggregate, forecasts = result["summaries"], [], []
    for task in TASKS:
        for method in ARMS:
            selected = [row for row in rows if row["task"] == task and row["method"] == method]
            aggregate.append({"task": task, "method": method, "trials": len(selected),
                "online_calls": sum(row["online_calls"] for row in selected),
                "online_feasible": sum(row["usable_feasible_online"] for row in selected),
                "endpoint_feasible_trials": sum(row["endpoint_best"] is not None for row in selected),
                "endpoints": [row["endpoint_best"] for row in selected],
                "mean_relative_improvement": float(np.mean([row["relative_improvement"] for row in selected]))
                if selected and all(row["relative_improvement"] is not None for row in selected) else None})
            if method == "original_lsbo":
                continue
            complete_errors, joint_errors = [], []
            constraint_errors, scales, modes, fallbacks = {}, {}, Counter(), Counter()
            for manifest in sorted((folder/"optimization"/task).glob(method+"_*/manifest.json")):
                proposals, observations = lines(manifest.parent/"proposals.jsonl"), lines(manifest.parent/"observations.jsonl")
                for proposal, packet in zip(proposals, observations):
                    metadata, record = proposal["metadata"], DesignRecord.from_dict(packet)
                    index = metadata["index"]
                    modes[metadata["mode"]] += 1
                    fallbacks[str(metadata["fallback"])] += 1
                    label = record.observation.completion_label()
                    if label is not None:
                        complete_errors.append((metadata["completion_probability"][index]-label)**2)
                    if record.physical_feasible is not None:
                        joint_errors.append((metadata["joint_probability_independence_proxy"][index]
                                             -int(record.usable_feasible))**2)
                    for name, head in metadata["heads"].items():
                        scales.setdefault(name, Counter())[str(head["selected"]["length"])] += 1
                    for name, forecast in metadata["forecasts"]["constraints"].items():
                        label = record.observation.constraint(name).constraint_satisfaction()
                        if label is None:
                            continue
                        mean, variance = forecast["mean"][index], forecast["variance"][index]
                        probability = ndtr(-mean/np.sqrt(max(variance, 1e-24)))
                        constraint_errors.setdefault(name, []).append((probability-int(label))**2)
            forecasts.append({"task": task, "method": method,
                "prequential_completion_brier": float(np.mean(complete_errors)) if complete_errors else None,
                "prequential_joint_proxy_brier": float(np.mean(joint_errors)) if joint_errors else None,
                "joint_proxy_label_count": len(joint_errors),
                "constraint_brier": {name: {"n": len(values), "brier": float(np.mean(values))}
                                     for name, values in constraint_errors.items()},
                "kernel_length_counts": scales, "modes": modes, "fallbacks": fallbacks,
                "interpretation": "on-policy development scoring; query distributions differ; not common-panel calibration proof"})
    comparisons = []
    for task in TASKS:
        for control in ARMS[1:]:
            orders = []
            for seed in SEEDS:
                a = next(row for row in rows if (row["task"], row["seed"], row["method"]) == (task, seed, "ra_latent"))
                b = next(row for row in rows if (row["task"], row["seed"], row["method"]) == (task, seed, control))
                orders.append(paired_order(a["endpoint_best"], b["endpoint_best"]))
            comparisons.append({"task": task, "control": control, "ra_wins": orders.count(-1),
                                "ties": orders.count(0), "ra_losses": orders.count(1)})
    integration = []
    events = [event for event in lines(folder/"calls.jsonl") if event["event"] == "settled"]
    for index in range(2):
        coarse = next(event["record"] for event in events if event["tag"] == f"reduced_aircraft/anchor/{index}")
        fine = next(event["record"] for event in events if event["tag"] == f"reduced_aircraft/integration/{index}/5.0")
        a, b = dict(coarse["observation"]["constraints"]), dict(fine["observation"]["constraints"])
        integration.append({"anchor": index,
            "margin_absolute_differences_20s_vs_5s": {name: abs(a[name]["value"]-b[name]["value"])
                                                      for name in ("soc", "current", "voltage", "power")},
            "energy_gate_limitation": "DC power fixed within each stage, so Wh is nearly independent of SOC integration step"})
    supplemental = folder/"mission_envelope"
    supplement_plan = read(supplemental/"plan.json")
    verify(folder, supplement_plan)
    additional = lines(supplemental/"calls.jsonl")
    starts = [event for event in additional if event["event"] == "started"]
    settled = [event for event in additional if event["event"] == "settled"]
    if len(starts) != 24 or len(settled) != 24 or len({event["tag"] for event in starts}) != 24:
        raise ValueError("Supplement settlement mismatch")
    for start, end in zip(starts, settled):
        record = DesignRecord.from_dict(end["record"])
        if (start["call_id"] != end["call_id"] or record.candidate != start["candidate"]
                or record.observation.contract_id != start["contract"]
                or digest(record.to_dict()) != end["record_digest"]):
            raise ValueError("Supplement evidence mismatch")
    total = result["new_model_calls"]+len(starts)
    if total > CAMPAIGN_MAXIMUM:
        raise ValueError("Campaign budget exceeded")
    write_new(folder/"analysis.json", {"aggregate": aggregate, "paired_endpoint_order": comparisons,
        "forecasts": forecasts, "integration": integration, "supplement_linkage_passed": True,
        "campaign_new_model_calls": total, "campaign_new_rcaide_calls": 0,
        "no_algorithm_superiority_established": True, "supplement": read(supplemental/"results.json")})
    plot(folder, rows)
    print({"aggregate": aggregate, "comparisons": comparisons, "campaign_calls": total}, flush=True)


def plot(folder, rows):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    offsets = np.linspace(-.27, .27, 4)
    for number, task in enumerate(TASKS):
        axis = axes[number]
        for index, (method, color) in enumerate(zip(ARMS, COLORS)):
            for seed_index, seed in enumerate(SEEDS):
                row = next(row for row in rows if (row["task"], row["method"], row["seed"]) == (task, method, seed))
                value = row["endpoint_best"]
                if value is None:
                    axis.annotate("no feasible", (seed_index+offsets[index], .96),
                        xycoords=("data", "axes fraction"), rotation=90, ha="center", va="top", color=color, fontsize=8)
                else:
                    axis.scatter(seed_index+offsets[index], value/(1000. if number else 1.), s=48,
                        marker=("o", "s", "^", "D")[index], color=color,
                        label=LABELS[method] if seed_index == 0 else None)
        axis.set_xticks(range(3), [str(seed) for seed in SEEDS])
        axis.set_xlabel("Paired seed; 12 shared initial + up to 14 online calls")
        axis.set_ylabel("Best feasible mission energy (kWh)" if number else "Best feasible Branin objective")
        axis.set_title("Reduced electric aircraft (100 km)" if number else "Hierarchical Branin extension")
        axis.grid(axis="y", alpha=.2)
        if not number:
            axis.set_yscale("log")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Development only: three paired seeds; no stable RA-LSBO superiority", fontsize=12)
    fig.savefig(folder/"development_endpoints.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
