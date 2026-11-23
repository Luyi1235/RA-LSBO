"""Plot the ledger-verified incumbent energy in each arm."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from summarize_pilot import load


def plot(summary, destination):
    series = summary["arms"]
    initial = summary["shared_initial_starts"]
    baseline = series["lsbo"]["trajectory"]
    sign = series["ra_lsbo_energy_sign"]["trajectory"]
    interval = series["ra_lsbo_energy_interval"]["trajectory"]

    def energy(rows):
        return [row["best_usable_Wh"] for row in rows]

    plt.rcParams.update({"font.size": 10, "axes.labelcolor": "#202631", "xtick.color": "#515965",
                         "ytick.color": "#515965", "text.color": "#202631", "axes.edgecolor": "#aab2bd"})
    fig, ax = plt.subplots(figsize=(8.1, 4.25), layout="constrained")
    ax.axvspan(.5, initial + .5, color="#edf0f3", linewidth=0)
    ax.axvline(initial + .5, color="#aab2bd", linewidth=.8)
    ax.text(.9, .94, "Shared initialization", transform=ax.get_xaxis_transform(),
            color="#515965", fontsize=9, va="top")

    def step(rows, label, color, style):
        ax.step([row["evaluation"] for row in rows], energy(rows), where="post", label=label,
                color=color, linewidth=2.1, linestyle=style)

    step(baseline, "LSBO", "#2b609b", "-")
    if energy(sign) == energy(interval):
        step(sign, "RA-LSBO sign and interval", "#ad4d29", "--")
    else:
        step(sign, "RA-LSBO sign", "#ad4d29", "--")
        step(interval, "RA-LSBO interval", "#267654", (0, (2, 2)))
    ax.set(xlim=(.5, len(baseline) + .5), xlabel="Cumulative evaluations per arm",
           ylabel="Best usable objective (Wh)")
    ax.set_title(f"X-57 | {summary['distance_nmi']} nmi | seed {summary['seed']}", loc="left", pad=12)
    ax.set_xticks([1, 4, initial, *range(initial + 2, len(baseline) + 1, 2)])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", color="#d7dde5", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(load(args.trajectory), args.output)
