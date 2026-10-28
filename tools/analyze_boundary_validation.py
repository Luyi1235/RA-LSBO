"""Analyze held-out boundary forecasts without changing the experiment."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import test_metrics
from ra_lsbo.metrics import binary_metrics


MODELS = ("global_rf", "global_constraint", "local_rf", "local_constraint")
LABELS = ("RF / global calibration", "Constraint score / global calibration",
          "RF / local calibration", "Constraint score / local calibration")
COLORS = ("#555555", "#467cac", "#008775", "#b66b19")
BANDS = ("outer", "near", "inner")


def heldout_records(plan, result, local):
    forecasts = {row["index"]: row for row in local["forecasts"]}
    records = []
    for item in result["results"]:
        if item["index"] <= 60:
            continue
        row = dict(forecasts[item["index"]])
        proposal = plan["proposals"][item["index"]-1]
        risk = item["risk"]
        row.update(
            feasible=None if risk is None else risk["feasible_label"],
            state="precheck_rejected_before_submission" if risk is None else risk["state"],
            evaluated=risk is not None,
            raw_rf=proposal["raw_rf"], raw_constraint=proposal["raw_constraint"])
        records.append(row)
    return records


def paired_difference(records, reference, comparison, draws=2000, seed=20261020):
    rows = [row for row in records if row["feasible"] is not None
            and row.get(reference) is not None and row.get(comparison) is not None]
    groups = [[row for row in rows if row["band"] == band] for band in BANDS]
    groups = [group for group in groups if group]
    if not rows:
        return {"samples": 0, "mean_brier_difference": None, "bootstrap_interval": None}
    differences = [np.asarray([(row[comparison]-int(row["feasible"]))**2 -
                               (row[reference]-int(row["feasible"]))**2 for row in group])
                   for group in groups]
    rng = np.random.default_rng(seed)
    resampled = np.zeros(draws)
    for values in differences:
        resampled += values[rng.integers(0, len(values), size=(draws, len(values)))].sum(axis=1)
    resampled /= len(rows)
    return {
        "reference": reference, "comparison": comparison, "samples": len(rows),
        "mean_brier_difference": float(np.concatenate(differences).mean()),
        "bootstrap_interval": np.quantile(resampled, [.025, .975]).tolist(),
        "bootstrap_draws": draws, "bootstrap_seed": seed,
        "stratum_counts": dict(Counter(row["band"] for row in rows)),
        "interpretation": "Comparison minus reference; exploratory within-sample interval, not a safety bound.",
    }


def analyze(plan, result, local):
    records = heldout_records(plan, result, local)
    metrics, fixed_acceptance, gates = {}, {}, {}
    for name in MODELS:
        valid = [row for row in records if row["feasible"] is not None and row[name] is not None]
        metrics[name] = binary_metrics([row[name] for row in valid], [int(row["feasible"]) for row in valid])
        available = [row for row in records if row["evaluated"] and row[name] is not None]
        accepted = [row for row in available if row[name] >= .8]
        gates[name] = {
            "predicted_test_samples": len(available), "accepted": len(accepted),
            "known_accepted": sum(row["feasible"] is not None for row in accepted),
            "unknown_accepted": sum(row["feasible"] is None for row in accepted),
            "infeasible_among_accepted": sum(row["feasible"] is False for row in accepted),
        }
        raw_name = "raw_rf" if name.endswith("_rf") else "raw_constraint"
        ranking = sorted(available, key=lambda row: (-row[raw_name], row["index"]))
        fixed_acceptance[name] = []
        for count in (10, 20):
            if len(ranking) < count:
                continue
            accepted = ranking[:count]
            fixed_acceptance[name].append({
                "accepted": count,
                "known_accepted": sum(row["feasible"] is not None for row in accepted),
                "unknown_accepted": sum(row["feasible"] is None for row in accepted),
                "infeasible": sum(row["feasible"] is False for row in accepted),
                "indices": [row["index"] for row in accepted]})
    pairs = [paired_difference(records, a, b) for a, b in (
        ("global_rf", "local_rf"), ("global_constraint", "local_constraint"),
        ("global_rf", "global_constraint"), ("local_rf", "local_constraint"))]
    evaluated = [row for row in records if row["evaluated"]]
    completed = len(evaluated)
    known = sum(row["feasible"] is not None for row in evaluated)
    return {
        "campaign_status": result["status"], "stop_reason": result.get("reason"),
        "completed_calibration": sum(item["index"] <= 60 and item["risk"] is not None
                                     for item in result["results"]),
        "planned_test": 60, "completed_test": completed, "known_test_labels": known,
        "unknown_test_labels": completed-known, "unexecuted_test": 60-completed,
        "unsubmitted_precheck_test": len(records)-completed,
        "completed_test_by_stratum": dict(Counter(row["band"] for row in evaluated)),
        "local_calibration": local["diagnostics"], "metrics": metrics,
        "acceptance_at_0_8": gates,
        "threshold_metrics": test_metrics(local["forecasts"], [item for item in result["results"] if item["risk"] is not None]),
        "paired_brier_differences": pairs, "fixed_acceptance": fixed_acceptance,
        "ledger": result["ledger"],
        "limitations": [
            "Local calibration is evaluated only on test points, never on its fitted calibration points.",
            "Target distribution is the frozen balanced boundary mixture, not the whole design space.",
            "Unavailable calibrators, unknown labels and unexecuted points remain explicit.",
            "Positive-slope sigmoid changes probability values, not the underlying raw score ordering.",
            "Bootstrap intervals are exploratory; they do not certify an adaptive optimization policy.",
        ],
    }, records


def plot(report, records, destination):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for name, label, color in zip(MODELS, LABELS, COLORS):
        reliability = report["metrics"][name]["reliability"]
        nonempty = [row for row in reliability if row["samples"]]
        axes[0, 0].plot(
            [row["mean_probability"] for row in nonempty],
            [row["observed_fraction"] for row in nonempty], marker="o", color=color, label=label)
    axes[0, 0].plot([0, 1], [0, 1], linestyle=":", color="#999999")
    axes[0, 0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean forecast", ylabel="Observed feasible fraction",
                   title="Held-out reliability (five fixed bins)")
    axes[0, 0].legend(fontsize=7)
    for method, (name, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
        values = [report["threshold_metrics"][band][name]["brier"] for band in BANDS]
        positions = np.arange(3)+(method-1.5)*.17
        axes[0, 1].bar(positions, [np.nan if value is None else value for value in values],
                       width=.16, color=color, label=label)
    axes[0, 1].set(xticks=np.arange(3), xticklabels=BANDS, ylabel="Brier loss",
                   title="Loss by frozen boundary stratum")
    accepted = [report["acceptance_at_0_8"][name]["accepted"] for name in MODELS]
    bad = [report["acceptance_at_0_8"][name]["infeasible_among_accepted"] for name in MODELS]
    unknown = [report["acceptance_at_0_8"][name]["unknown_accepted"] for name in MODELS]
    positions = np.arange(4)
    axes[1, 0].bar(positions, accepted, color=COLORS, alpha=.4, label="Accepted")
    axes[1, 0].bar(positions, bad, color=COLORS, label="Accepted but infeasible")
    if any(unknown):
        axes[1, 0].bar(positions, unknown, bottom=bad, facecolor="none", edgecolor="#555555",
                       hatch="//", label="Accepted with unknown outcome")
    axes[1, 0].set(xticks=positions, xticklabels=["RF/global", "Constraint/global", "RF/local", "Constraint/local"],
                   ylabel="Count", title="Acceptance at fixed probability 0.8")
    axes[1, 0].tick_params(axis="x", labelsize=8)
    axes[1, 0].legend(fontsize=8)
    pairs = [row for row in report["paired_brier_differences"] if row["samples"]]
    for position, pair in enumerate(pairs):
        mean = pair["mean_brier_difference"]
        lower, upper = pair["bootstrap_interval"]
        axes[1, 1].hlines(position, lower, upper, color="#467cac", linewidth=2)
        axes[1, 1].scatter([mean], [position], color="#008775", s=35)
    axes[1, 1].axvline(0, color="#999999", linestyle=":")
    axes[1, 1].set(
        yticks=range(len(pairs)),
        yticklabels=[row["comparison"]+" - "+row["reference"] for row in pairs],
        xlabel="Paired Brier difference (negative is better)", title="Exploratory stratified bootstrap")
    axes[1, 1].tick_params(axis="y", labelsize=7)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
    figure.suptitle(
        f"Boundary calibration validation: {report['completed_test']}/60 test calls, "
        f"{report['known_test_labels']} known labels\n"
        "Fixed target distribution; no imputed outcomes or safety guarantee", fontsize=13)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    result = load_json((folder / "result.json").read_bytes())
    if result["status"] not in ("complete", "stopped_by_policy", "stopped_evaluator_error"):
        raise ValueError("Wait for the terminal campaign; do not report partial results as final")
    local_path = folder / "test_forecasts.json"
    if not local_path.exists():
        raise ValueError("No local calibration/test forecasts were produced before stopping")
    local_bytes = local_path.read_bytes()
    if hashlib.sha256(local_bytes).hexdigest() != result["test_forecasts_sha256"]:
        raise ValueError("Frozen test forecasts changed")
    audit = load_json((folder / "audit.json").read_bytes())
    if audit["status"] != "passed" or audit["ledger"] != result["ledger"]:
        raise ValueError("Require a passed terminal campaign audit before analysis")
    for filename, expected in audit["input_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Audited campaign inputs changed")
    plan = load_json((folder / "plan.json").read_bytes())
    report, records = analyze(plan, result, load_json(local_path.read_bytes()))
    inputs = [folder / filename for filename in ("plan.json", "result.json", "test_forecasts.json", "audit.json")]
    inputs.extend([Path(__file__), Path(__file__).resolve().parents[1] /
                   "docs" / "BOUNDARY_ANALYSIS_PROTOCOL_20261020.md"])
    report["analysis_source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    plot(report, records, folder / "boundary_validation.png")
    with (folder / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "test": report["completed_test"],
                      "metrics": report["metrics"], "paired": report["paired_brier_differences"]}, indent=2))
    print("Analysis:", folder / "analysis.json")


if __name__ == "__main__":
    main()
