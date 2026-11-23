"""Plot first feasible and first quality hits for the zero-feasible cohort."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


METHODS = (
    ("lsbo_full_budget", "LSBO", "#343b48", "-", "o"),
    ("ra_lsbo_energy_sign", "RA-LSBO sign", "#0868ac", "--", "s"),
    ("ra_lsbo_energy_interval", "RA-LSBO interval", "#0a8a76", ":", "D"),
    ("archsbo_screened", "ArchSBO", "#c3532b", "-.", "^"),
    ("logcei_screened_online", "LogCEI", "#825098", (0, (5, 2, 1, 2, 1, 2)), "v"),
)


def plot(report, output):
    group = report["group_metrics"]["0"]
    n = len(group["seeds"])
    if n == 0 or report["status"] != "complete":
        raise ValueError("The plot requires a completed checkpoint with a zero-feasible cohort")
    end = report["checkpoint"]
    grid = range(8, end + 1)
    if any(group["arms"][method]["checkpoint_settled"] != n for method, *_ in METHODS):
        raise ValueError("All five arms must share the same settled comparison denominator")
    fields = ("first_usable_calls", "first_quality_calls")
    titles = ("First feasible design", f"First feasible design at or below {report['quality_target_Wh']:,} Wh")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.97, top=.79, bottom=.27, wspace=.16)
    fig.suptitle(f"X-57 at 47.5 nmi  |  Zero feasible in shared initialization  |  {n} seed"
                 + ("s" if n != 1 else ""), fontsize=12, y=.97)
    for ax, field, title in zip(axes, fields, titles):
        for arm, label, color, style, marker in METHODS:
            hits = group["arms"][arm][field]
            counts = [sum(hit is not None and hit <= step for hit in hits) for step in grid]
            ax.step(grid, counts, where="post", color=color, linestyle=style, linewidth=2, label=label)
            event_x = sorted({hit for hit in hits if hit is not None})
            ax.plot(event_x, [counts[hit - 8] for hit in event_x], linestyle="none", marker=marker,
                    markersize=4, markerfacecolor="white", markeredgecolor=color)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(ylim=(-.12, n + .35), xlim=(7.8, end + .2), title=title, xlabel="Cumulative evaluations")
        ax.set_xticks(list(grid))
        ax.grid(axis="y", color="#dce0e4", linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Seeds that reached the endpoint")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035),
               ncol=len(METHODS), frameon=False, fontsize=9)
    for suffix, options in ((".pdf", {}), (".png", {"dpi": 230})):
        fig.savefig(output.with_suffix(suffix), **options)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.report.read_bytes()), args.output)
