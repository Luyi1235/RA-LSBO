"""Assemble physical provenance and all favorable/unfavorable development evidence."""

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import RiskLedger
from ra_lsbo.system_experiment import save_new


DEVELOPMENT = ROOT/"runs/posterior_development_20261021T082018Z_877bb84b"
PHYSICAL = ROOT/"runs/compact_mda_20261021T081006Z_20eb608d"
METHODS = ("posterior_full", "posterior_no_completion", "posterior_onehot_risk", "lsbo", "random")
LABELS = ("TS + completion", "TS without\ncompletion", "TS + one-hot\nrisk", "Original LSBO", "Random")
COLORS = ("#187b68", "#ab645d", "#9b811c", "#313131", "#8b8b8b")


def lines(path):
    return [load_json(line) for line in path.read_bytes().splitlines()]


def main():
    plan = load_json((DEVELOPMENT/"plan.json").read_bytes())
    verify_sources(plan["source_sha256"])
    audit = load_json((DEVELOPMENT/"audit.json").read_bytes())
    if audit["status"] != "passed":
        raise ValueError("Development audit incomplete")
    physical_plan = load_json((PHYSICAL/"plan.json").read_bytes())
    verify_sources(physical_plan["source_sha256"])
    physical_result = load_json((PHYSICAL/"pilot_result.json").read_bytes())
    transfer = load_json((ROOT/"runs/posterior_transfer_20261021/diagnostics.json").read_bytes())
    rows = lines(PHYSICAL/"observations.jsonl")
    physical = []
    for row in rows:
        worker = Path(row["risk"]["evidence"]["worker_dir"])
        request, response = [load_json((worker/name).read_bytes()) for name in ("request.json", "response.json")]
        metrics = response["metrics"]
        expected_area = row["candidate"]["continuous"]["wing_area"]
        if (request["candidate"] != row["candidate"]
                or not np.isclose(metrics["design_study_realized_wing_area_m2"], expected_area, atol=1e-9)
                or not np.isclose(metrics["cruise_distance_m"], row["distance_nmi"]*1852., atol=1e-9)
                or metrics["reserve_segments_retained"] is not True
                or response["objective"] != row["risk"]["objective"]):
            raise ValueError("Worker did not evaluate the declared physical scenario")
        physical.append({
            "distance_nmi": row["distance_nmi"], "anchor": row["anchor"],
            "requested_and_realized_wing_area_m2": expected_area,
            "stored_energy_depletion_Wh": row["risk"]["objective"],
            "terminal_energy_Wh": metrics.get("terminal_energy_Wh"),
            "final_soc": metrics.get("final_soc"), "state": row["risk"]["state"],
            "violations": {n: v for n, v in row["risk"]["constraints"].items() if v is not None and v > 0},
            "worker_response_sha256": hashlib.sha256((worker/"response.json").read_bytes()).hexdigest()})
    charged = sum(RiskLedger(p).summary()["charged_calls"] for p in PHYSICAL.glob("pilot_*/budget.jsonl"))
    if charged != len(rows) or charged != physical_result["charged_real_calls"]:
        raise ValueError("Physical accounting differs")
    trials = lines(DEVELOPMENT/"completed.jsonl")
    aggregate = []
    for case in plan["cases"]:
        for method in METHODS:
            group = [r for r in trials if r["case"] == case and r["method"] == method]
            observations = [r for seed in plan["seeds"] for r in
                            lines(DEVELOPMENT/case/(method+"_"+str(seed))/"observations.jsonl")]
            aggregate.append({
                "case": case, "method": method, "calls": sum(r["new_calls"] for r in group),
                "feasible": sum(r["new_feasible"] for r in group),
                "hidden_failures": sum(r["risk"]["state"] == "hidden_failure" for r in observations),
                "no_feasible_endpoints": sum(r["best"] is None for r in group),
                "best_by_seed": {str(r["seed"]): r["best"] for r in group},
                "zero_initial_discovery": [{"seed": r["seed"], "first_feasible_call": r["first_new_feasible_call"]}
                                           for r in group if r["initial_best"] is None]})
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), layout="constrained")
    for row_number, case in enumerate(plan["cases"]):
        for j, method in enumerate(METHODS):
            group = [r for r in trials if r["case"] == case and r["method"] == method]
            axes[row_number, 0].bar(j, sum(r["new_feasible"] for r in group), color=COLORS[j], width=.62)
            for k, r in enumerate(group):
                if r["best"] is not None:
                    axes[row_number, 1].scatter(j+(k-1)*.12, r["best"], color=COLORS[j],
                                               marker=("o", "s", "^")[k], s=30)
            missing = sum(r["best"] is None for r in group)
            if missing:
                axes[row_number, 1].text(
                    j, .96, str(missing)+"/3 no feasible", ha="center", va="top",
                    transform=axes[row_number, 1].get_xaxis_transform(), fontsize=8)
        for axis in axes[row_number]:
            axis.set_xticks(range(len(METHODS)), LABELS, fontsize=8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=.2)
        axes[row_number, 0].set_ylim(0, 37)
        axes[row_number, 0].set_ylabel("Feasible online evaluations / 36")
        axes[row_number, 1].set_ylabel("Best feasible objective (lower is better)")
        axes[row_number, 0].set_title(case+" | 20 shared initial labels / seed")
        axes[row_number, 1].set_title("Three paired seeds; 12 online calls / run")
    output = ROOT/"runs/posterior_revision_report_20261021"
    output.mkdir(exist_ok=False)
    figure.savefig(output/"controlled_results.png", dpi=180)
    plt.close(figure)
    ei, p, price = np.array([1., .1]), np.array([.5, .9]), .5
    before = ei/ei.max()*p-price*(1-p)
    extended_ei, extended_p = np.r_[ei, 100.], np.r_[p, 0.]
    after = extended_ei/extended_ei.max()*extended_p-price*(1-extended_p)
    save_new(output/"evidence.json", {
        "status": "development_completed_physical_pilot_completed",
        "new_real_mda_calls": charged, "new_study_complete_budget": 300,
        "old_v1_v5_cumulative_calls_unchanged": 500,
        "new_controlled_truth_labels": audit["total_new_controlled_truth_labels"],
        "full_acquisition_replays": audit["full_acquisition_replays"],
        "controlled_aggregate": aggregate, "physical_engineering_checks": physical,
        "sparse_region_qualified": False,
        "pool_normalization_counterexample": {
            "before_scores": before.tolist(), "after_scores": after.tolist(),
            "before_winner": int(before.argmax()), "after_winner": int(after.argmax()),
            "added_candidate_has_zero_feasibility_and_worse_utility": True,
            "interpretation": "A candidate that cannot win changes the ranking of two existing candidates"},
        "whole_family_transfer": [{
            "anchor": f["anchor"], "view": f["view"],
            "recall_at_half": f["mission_failure_metrics"]["recall_at_half"],
            "brier": f["mission_failure_metrics"]["brier"],
            "latent_variance_at_failures": f["mean_latent_variance_at_failures"]}
            for f in transfer["whole_family_folds"]],
        "historical_unique_brier": transfer["historical_metrics"]["unique_designs"]["brier"],
        "conclusions": [
            "The task mismatch and observed acquisition weaknesses both warrant revision.",
            "Posterior completion improved some matched comparisons, not consistently over original LSBO.",
            "Whole-family failure recall remains zero; high latent variance does not repair recall.",
            "Engineering anchors demonstrate competing physical constraints, not sparse prevalence.",
            "All unfavorable methods and seeds retained; no new-mission label reuse from old data."],
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            DEVELOPMENT/"audit.json", PHYSICAL/"pilot_result.json",
            ROOT/"runs/posterior_transfer_20261021/diagnostics.json",
            ROOT/"runs/uncertainty_rootcause_20261021/diagnostics.json",
            Path(__file__).resolve())}})
    print(output, flush=True)


if __name__ == "__main__":
    main()
