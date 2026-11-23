"""Plot the best feasible objective on each diagnostic trajectory."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ARMS = {
    "lsbo": ("Original LSBO", "#454545", "-"),
    "ra_lsbo_energy_sign": ("RA-LSBO sign", "#0072B2", "--"),
    "ra_lsbo_energy_interval": ("RA-LSBO interval", "#D55E00", ":"),
}


def draw(report, output):
    indicators = report["indicators"]
    target = indicators["target_objective_Wh"] / 1000.
    seeds = indicators["seeds"]
    if not seeds:
        raise ValueError("No settled pilot trajectories are available")
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.2 * len(seeds) + .4, 3.65),
                             sharey=True, squeeze=False)
    fig.subplots_adjust(left=.12, right=.98, bottom=.22, top=.81, wspace=.14)
    for (seed, data), ax in zip(seeds.items(), axes[0]):
        ax.set_title(f"Seed {seed}  |  {data['initial_usable']} feasible at initialization", fontsize=10)
        ax.axvspan(1, 8, facecolor="#ECEFF1", zorder=0)
        ax.axvline(8, color="#999999", linewidth=.65, zorder=1)
        ax.axhline(target, color="#797979", linewidth=.9, linestyle="-.", zorder=1)
        for arm, (_, color, style) in ARMS.items():
            if arm not in data["arms"]:
                continue
            values = data["arms"][arm]["best_usable_trajectory_Wh"]
            ys = [v / 1000. if v is not None else float("nan") for v in values]
            ax.step(range(1, len(ys) + 1), ys, where="post", color=color, linestyle=style, linewidth=1.65)
            for i in (next((i for i, v in enumerate(values) if v is not None), None), len(ys) - 1):
                if i is not None and i >= 0 and values[i] is not None:
                    ax.scatter(i + 1, ys[i], marker={"-": "s", "--": "^", ":": "o"}[style],
                               s=18, facecolors="white", edgecolors=color, linewidths=.9, zorder=4)
        ax.set(xlim=(1, 20.2), xticks=(1, 4, 8, 12, 16, 20), xlabel="Cumulative evaluations")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=.5)
    axes[0, 0].set_ylabel("Best feasible mission energy (kWh)")
    mission = report["selected_distance_nmi"]
    fig.suptitle(f"X-57 / RCAIDE pilot  |  {mission:g} nmi", fontsize=13, y=.96)
    handles = [Line2D([0], [0], label=label, color=color, linestyle=style, linewidth=1.8)
               for label, color, style in ARMS.values()]
    handles.append(Line2D([0], [0], label=f"{target:g} kWh target", color="#797979", linestyle="-."))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, .015), frameon=False,
               fontsize=9)
    for suffix in (".pdf", ".png"):
        fig.savefig(output.with_suffix(suffix), dpi=250, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(json.loads(args.report.read_bytes()), args.output)
