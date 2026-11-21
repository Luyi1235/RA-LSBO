"""Render the paper's end-to-end and external benchmark evidence from fixed commits."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean
import subprocess


ROOT = Path(__file__).resolve().parent
EXTERNAL_BATCH = "External comparison"
ARCHIVES = {
    "Independent confirmation": (
        "ff076555615150a3885611dbf0d288906fca1767",
        "results/aircraft-structured-risk-confirmation-20261121-956b595",
        "ra-lsbo-structured-confirmation-956b595-analysis.json",
        "956b5956538ae17def1edd97125c3538d4a26a6e",
        "VERIFIED_STRUCTURED_RISK_CONFIRMATION", 6912,
    ),
    "End-to-end confirmation": (
        "236c3262230215b5644495ade9d576a478457982",
        "results/aircraft-structured-risk-optimization-20261121-c0151e4",
        "ra-lsbo-structured-optimization-c0151e4-analysis.json",
        "c0151e4238d5acf3f56e56b1f6343d5844ab6263",
        "VERIFIED_STRUCTURED_RISK_OPTIMIZATION", 3840,
    ),
    EXTERNAL_BATCH: (
        "30d63fbc42498166d537e651053181b8d2c859c3",
        "results/aircraft-structured-risk-external-20261121-62750a5",
        "ra-lsbo-structured-external-62750a5-analysis.json",
        "62750a562a868bc208568b30b65c08b848d2c2fb",
        "VERIFIED_STRUCTURED_RISK_EXTERNAL", 3840,
    ),
}
STRUCTURED = "ra_structured_greedy"
CONTROLS = {"lsbo_native": "Original LSBO", "ra_refine_interval": "Interval-risk greedy"}
EXTERNAL_CONTROLS = {"archsbo_screened": "Screened ArchSBO", "logcei_screened_online": "Adapted LogCEI"}
LABELS = {STRUCTURED: "Structured-risk greedy", **CONTROLS, **EXTERNAL_CONTROLS}
STYLES = {STRUCTURED: ("#00796B", "-"), "lsbo_native": ("#394B59", "--"),
          "ra_refine_interval": ("#B75D00", "-."), "archsbo_screened": ("#394B59", "--"),
          "logcei_screened_online": ("#B75D00", "-.")}
INITIAL, BUDGET = 8, 32
TARGETS = {"headwind": 40000., "payload": 38000.}
REPORT = "AIRCRAFT_STRUCTURED_RISK_OPTIMIZATION_EVIDENCE_20261121.md"
FIGURE = "aircraft-structured-risk-attainment.pdf"
EXTERNAL_FIGURE = "aircraft-structured-risk-external-attainment.pdf"


def load_archive(batch, spec):
    commit, folder, filename, code, status, calls = spec
    ref = f"{commit}:{folder}/{filename}"
    result = subprocess.run(["git", "show", ref], cwd=ROOT, check=True, stdout=subprocess.PIPE)
    data = json.loads(result.stdout)
    observed = (data["source_sha"], data["status"], data["paid_calls"], data["rcaide_mda_calls"],
                data["plan"]["initial_evaluations"], data["plan"]["primary"]["budget"])
    if observed != (code, status, calls, 0, INITIAL, BUDGET):
        raise ValueError(f"{batch} does not match the fixed result identity")
    return data


def cold_cases(scenario, section):
    rows = [row["arms"] for row in section["rows"] if row["cold"]]
    summary = section["summary"]["cold"]
    if section["energy_target_Wh"] != TARGETS[scenario] or len(rows) != summary["n"] or not rows:
        raise ValueError(f"Unexpected {scenario} cold-start cohort or energy threshold")
    return rows, summary


def paired_endpoints(section, rows, summary, comparator, *, cohort="cold"):
    key = f"{STRUCTURED}_minus_{comparator}"
    contrast = summary["contrasts"][key]
    time = contrast["energy_target"]["capped_time_delta"]
    at_budget = contrast["quality_by_budget"][str(BUDGET)]
    third = section["accumulation"][cohort]["contrasts"][key]["capped_target_call_delta"]
    target_times = [(row[STRUCTURED]["first_energy_target"], row[comparator]["first_energy_target"])
                    for row in rows]
    time_delta = mean((a if a is not None else BUDGET+1)-(b if b is not None else BUDGET+1)
                      for a, b in target_times)
    hits = tuple(sum(value is not None for value in side) for side in zip(*target_times))
    third_delta = mean(row[STRUCTURED]["accumulation"]["capped_target_call"]
                       - row[comparator]["accumulation"]["capped_target_call"] for row in rows)
    counts = (summary["arms"][STRUCTURED]["hit_energy_target"], summary["arms"][comparator]["hit_energy_target"])
    energies = [(row[STRUCTURED]["quality_by_budget"][str(BUDGET)],
                 row[comparator]["quality_by_budget"][str(BUDGET)]) for row in rows]
    deltas = [a-b for a, b in energies if a is not None and b is not None]
    energy = at_budget["energy_Wh_delta_both_usable"]
    if (hits != counts or time["n"] != len(rows) or third["n"] != len(rows)
            or not math.isclose(time_delta, time["mean"], abs_tol=1e-12)
            or not math.isclose(third_delta, third["mean"], abs_tol=1e-12)
            or energy["n"] != len(deltas)
            or not math.isclose(mean(deltas), energy["mean"], abs_tol=1e-9)):
        raise ValueError(f"Inconsistent archived endpoints for {comparator}")
    return time, at_budget, third, hits


def display_interval(result, scale=1., places=2):
    def number(value):
        return format(value / scale, f"+.{places}f").replace("-", "−")

    lo, hi = result["paired_95"]
    return f"{number(result['mean'])} [{number(lo)}, {number(hi)}]"


def write_report(data, output):
    text = [
        "# Structured RA-LSBO cold-start and optimization evidence", "",
        "The three batches use independent seeds and the same evaluation budget. Cold-start seeds have no "
        "usable design among the eight shared initial evaluations. The end-to-end confirmation designates "
        "both optimization endpoints as primary in headwind cold-start seeds versus original LSBO. The "
        "external comparison designates the same endpoints and population versus screened ArchSBO. External "
        "arms share the engineering screen and evaluation budget, and use their own candidate search.",
        "", f"[Internal target-attainment curves]({FIGURE}) · [External target-attainment curves]({EXTERNAL_FIGURE})",
        "", "## Optimization outcomes", "",
        "Differences are structured-risk greedy minus the named comparator. Times include the initial evaluations "
        "and assign 33 evaluations to a seed that does not reach its prespecified target by 32. Terminal energy "
        "means are paired among seeds where both methods found a usable design. Feasibility-first win/loss/tie "
        "counts include every seed and use each batch's registered comparison rule. Intervals are the archived "
        "95% paired bootstrap intervals for means. Negative differences favor structured-risk greedy.",
        "", "| Batch | Scenario (cold pairs) | Comparator | Target hits (structured / comparator) | "
        "Δ target evaluations [95% CI] | Δ best feasible energy at 32, kWh [95% CI] | "
        "Energy pairs | Terminal W/L/T |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for batch, archive in data.items():
        controls = EXTERNAL_CONTROLS if batch == EXTERNAL_BATCH else CONTROLS
        for scenario in TARGETS:
            section = archive["scenarios"][scenario]
            rows, summary = cold_cases(scenario, section)
            for comparator, label in controls.items():
                time, at_budget, _, hits = paired_endpoints(section, rows, summary, comparator)
                energy = at_budget["energy_Wh_delta_both_usable"]
                order = at_budget["feasibility_first_order"]
                text.append(f"| {batch} | {scenario.title()} ({len(rows)}) | {label} | "
                            f"{hits[0]}/{len(rows)} vs. {hits[1]}/{len(rows)} | {display_interval(time)} | "
                            f"{display_interval(energy, 1000.)} | {energy['n']} | "
                            f"{order['wins']}/{order['losses']}/{order['ties']} |")
    text.extend([
        "", "The end-to-end headwind primary contrasts against original LSBO favor the structured arm in point "
        "estimates. Both mean intervals include zero. The prespecified terminal order is six wins, two losses, "
        "and one tie (exact sign test p=0.289). Payload has favorable target-time and terminal-energy mean "
        "intervals against original LSBO. The incremental optimization effect against interval-risk greedy "
        "varies by scenario and endpoint.", "",
        "In the external headwind primary comparison against screened ArchSBO, the mean target-time difference "
        "is −6.90 evaluations, with six wins, three losses, and one tie (exact sign test p=0.508). Terminal "
        "feasibility-first order gives nine wins and one loss (p=0.0215). In payload, terminal mean energy "
        "is lower in point estimate against ArchSBO and the interval includes zero. The adapted LogCEI "
        "comparison favors the structured method in both scenarios on both mean endpoints.", "",
        "## Cold-start evidence", "",
        "| Scenario | Δ Brier at evaluation 12, independent confirmation | Independent Δ evaluations to "
        "third usable | End-to-end Δ evaluations to third usable |",
        "|---|---:|---:|---:|",
    ])
    for scenario in TARGETS:
        independent = data["Independent confirmation"]["scenarios"][scenario]["accumulation"]["cold"]["contrasts"]
        end = data["End-to-end confirmation"]["scenarios"][scenario]["accumulation"]["cold"]["contrasts"]
        key = f"{STRUCTURED}_minus_ra_refine_interval"
        base = independent[key]
        brier = base["heldout_completion"]["12"]["configuration_balanced_brier_delta"]
        text.append(f"| {scenario.title()} | {display_interval(brier, places=4)} | "
                    f"{display_interval(base['capped_target_call_delta'])} | "
                    f"{display_interval(end[key]['capped_target_call_delta'])} |")
    text.extend(["", "These endpoints compare structured-risk with interval-risk greedy. The Brier score uses the "
                 "independent held-out panel from the independent-confirmation batch.", "",
                 "## Archive correction", ""])
    for batch, controls in (("End-to-end confirmation", CONTROLS), (EXTERNAL_BATCH, EXTERNAL_CONTROLS)):
        source = ARCHIVES[batch]
        archive_url = f"https://github.com/Luyi1235/RA-LSBO/blob/{source[0]}/{source[1]}"
        text.append(f"The [{batch} README]({archive_url}/README.md) labels the evaluation count to the third usable "
                    "design as time to the energy target in its cold-start and all-seed summaries.")
        text.extend(["", "| Scenario | Cohort | Comparator | README values (third usable) | Correct energy-target Δ evaluations |",
                     "|---|---|---|---:|---:|"])
        for scenario in TARGETS:
            section = data[batch]["scenarios"][scenario]
            cold_rows, _ = cold_cases(scenario, section)
            cohorts = (("cold", "Cold start", cold_rows), ("all", "All seeds", [row["arms"] for row in section["rows"]]))
            for cohort, name, rows in cohorts:
                for comparator, label in controls.items():
                    time, _, third, _ = paired_endpoints(section, rows, section["summary"][cohort], comparator, cohort=cohort)
                    text.append(f"| {scenario.title()} | {name} | {label} | {display_interval(third)} | {display_interval(time)} |")
        text.append("")
    text.extend(["", "## Fixed sources", ""])
    for batch, (commit, folder, filename, *_rest) in ARCHIVES.items():
        text.append(f"- [{batch} analysis JSON](https://github.com/Luyi1235/RA-LSBO/blob/{commit}/{folder}/{filename})")
    (output/REPORT).write_bytes(("\n".join(text)+"\n").encode("utf-8"))


def write_figures(data, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    budgets = range(INITIAL, BUDGET+1)

    def draw_panel(axis, archive, scenario, controls):
        cases, _ = cold_cases(scenario, archive["scenarios"][scenario])
        for arm in (STRUCTURED, *controls):
            times = [case[arm]["first_energy_target"] for case in cases]
            attained = [sum(time is not None and time <= budget for time in times)/len(times)
                        for budget in budgets]
            color, line = STYLES[arm]
            axis.step(budgets, attained, where="post", label=LABELS[arm], color=color, linestyle=line, linewidth=1.8)
        axis.set_title(f"{scenario.title()} | ≤ {TARGETS[scenario]/1000:.0f} kWh | n = {len(cases)}")
        axis.grid(axis="y", alpha=.25, linewidth=.6)
        axis.set_xlim(INITIAL, BUDGET)
        axis.set_ylim(0., 1.04)
        axis.set_xticks([8, 12, 16, 20, 24, 28, 32])
        axis.set_yticks([0., .25, .5, .75, 1.])
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1., decimals=0))

    def finish(fig, axis, name, ylabel):
        fig.supylabel(ylabel, x=.015)
        fig.supxlabel("Cumulative aircraft evaluations", y=.113)
        handles, labels = axis.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.52, .018))
        fig.savefig(output/name, metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.125, right=.975, bottom=.21, top=.855, wspace=.14, hspace=.32)
    for column, batch in enumerate(("Independent confirmation", "End-to-end confirmation")):
        for row, scenario in enumerate(TARGETS):
            draw_panel(axes[row, column], data[batch], scenario, CONTROLS)
        bounds = axes[0, column].get_position()
        fig.text((bounds.x0+bounds.x1)/2, .93, batch, ha="center", fontsize=11, weight="bold")
    finish(fig, axes[0, 0], FIGURE, "Cold-start seeds reaching the feasible energy target")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=.125, right=.975, bottom=.27, top=.875, wspace=.14)
    for axis, scenario in zip(axes, TARGETS):
        draw_panel(axis, data[EXTERNAL_BATCH], scenario, EXTERNAL_CONTROLS)
    finish(fig, axes[0], EXTERNAL_FIGURE, "Cold-start target attainment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = {batch: load_archive(batch, spec) for batch, spec in ARCHIVES.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(data, args.output_dir)
    write_figures(data, args.output_dir)


if __name__ == "__main__":
    main()
