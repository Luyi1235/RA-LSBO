"""Retrospective run-level audit; no new truth calls or model fitting."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.budget import RiskLedger
from ra_lsbo.research_statistics import holm, ordered_endpoint, paired_effect, run_metrics
from ra_lsbo.structured_encoder import history_identity
from ra_lsbo.system_experiment import save_new

CASES = ("five_mode", "sparse_interior")
METHODS = ("posterior_full", "posterior_no_completion", "posterior_onehot_risk", "lsbo", "random")
SEEDS = (81, 82, 83)
METRICS = ("online_feasible_count", "hidden_failure_count", "endpoint_order",
           "endpoint_best", "restricted_first_feasible_cost")


def read_lines(path):
    return [load_json(line) for line in path.read_bytes().splitlines() if line.strip()]


def report(source, output):
    output.mkdir(parents=True, exist_ok=False)
    source_files, runs, trajectories = set(), [], []
    for case in CASES:
        for seed in SEEDS:
            initial_path = source/case/("initial_"+str(seed)+".json")
            source_files.add(initial_path)
            initial = load_json(initial_path.read_bytes())
            for method in METHODS:
                folder = source/case/(method+"_"+str(seed))
                manifest_path, observations_path = folder/"manifest.json", folder/"observations.jsonl"
                manifest = load_json(manifest_path.read_bytes())
                if manifest["initial_history"] != history_identity(initial):
                    raise ValueError("Paired initial histories differ")
                observations = read_lines(observations_path)
                result = run_metrics(initial, observations, 12)
                ledger = RiskLedger(folder/"budget.jsonl").summary()
                if (ledger["charged_calls"] != len(observations)
                        or ledger["unresolved_started_calls"]):
                    raise ValueError("Unsettled or missing charged evaluations")
                summary_path = folder/"summary.json"
                summary = load_json(summary_path.read_bytes())
                if (summary["status"] != "complete" or not result["complete_horizon"]
                        or not result["all_labels_resolved"]
                        or result["endpoint_best"] != summary["best"]
                        or result["online_feasible_count"] != summary["new_feasible"]):
                    raise ValueError("Incomplete run or saved endpoint mismatch; do not drop run")
                source_files.update((manifest_path, observations_path, folder/"budget.jsonl", summary_path))
                for point in result.pop("trajectory"):
                    trajectories.append({"case": case, "seed": seed, "method": method, **point})
                runs.append({"case": case, "seed": seed, "method": method, **result})
    lookup = {(r["case"], r["seed"], r["method"]): r for r in runs}
    comparisons = []
    for case in CASES:
        for reference in METHODS[1:]:
            pairs = [(lookup[(case, seed, "posterior_full")], lookup[(case, seed, reference)])
                     for seed in SEEDS]
            for metric in METRICS:
                selected = pairs
                if metric == "restricted_first_feasible_cost":
                    selected = [(a, b) for a, b in pairs if a["initial_feasible"] == 0]
                eligible = bool(selected)
                if metric == "endpoint_order":
                    differences = [ordered_endpoint(a["endpoint_best"], b["endpoint_best"])
                                   for a, b in selected]
                else:
                    eligible = eligible and all(a[metric] is not None and b[metric] is not None
                                                for a, b in selected)
                    differences = [a[metric]-b[metric] for a, b in selected] if eligible else []
                inference = paired_effect(differences) if eligible else {
                    "exact_two_sided_sign_flip_p": None,
                    "reason_unavailable": "no eligible pairs or at least one endpoint unavailable"}
                comparisons.append({"case": case, "candidate": "posterior_full", "reference": reference,
                                    "metric": metric, "paired_seeds": [a["seed"] for a, _ in selected],
                                    "differences": differences, **inference})
    adjusted = holm([r["exact_two_sided_sign_flip_p"] for r in comparisons])
    for comparison, value in zip(comparisons, adjusted):
        comparison["holm_adjusted_p_40_planned_tests"] = value
    group_metrics = []
    for case in CASES:
        for method in METHODS:
            group = [r for r in runs if r["case"] == case and r["method"] == method]
            zero = [r for r in group if r["initial_feasible"] == 0]
            group_metrics.append({
                "case": case, "method": method, "n_runs": len(group),
                "online_feasible_total": sum(r["online_feasible_count"] for r in group),
                "hidden_failures_total": sum(r["hidden_failure_count"] for r in group),
                "endpoint_mean_all_seeds": None if any(r["endpoint_best"] is None for r in group)
                else float(np.mean([r["endpoint_best"] for r in group])),
                "zero_initial_runs": len(zero),
                "first_feasible_events_from_zero": sum(r["first_feasible_event"] for r in zero),
                "restricted_mean_calls_from_zero": float(np.mean(
                    [r["restricted_first_feasible_cost"] for r in zero])) if zero else None,
                "discovery_survival": [
                    {"online_budget": b, "not_yet_discovered": sum(
                        r["first_feasible_online_call"] is None or r["first_feasible_online_call"] > b
                        for r in zero), "n_runs": len(zero)}
                    for b in range(13)] if zero else [],
            })
    source_files.update((Path(__file__).resolve(), ROOT/"ra_lsbo/research_statistics.py",
                         source/"plan.json", source/"audit.json"))
    evidence = {
        "schema": "smo_retrospective_paired_evidence_v1", "analysis_date": "2026-10-25",
        "study": str(source), "inference_status": "retrospective development analysis",
        "run_count": len(runs), "online_calls": sum(r["online_calls"] for r in runs),
        "new_mda_calls": 0, "new_controlled_truth_calls": 0,
        "independent_replicates_per_task_and_method": 3,
        "discovery_replicates_per_method_sparse_case": 2,
        "multiplicity_family": "40 case/reference/endpoint slots; missing tests retained with p=1",
        "task_pooling": False, "runs": runs, "groups": group_metrics, "comparisons": comparisons,
        "limits": [
            "All outcomes were available before this analysis; this is not preregistration.",
            "Bootstrap resamples paired seeds, not dependent sequential observations.",
            "Three pairs imply minimum two-sided exact label-swap p=0.25 when all differences align.",
            "Small-sample bootstrap intervals are descriptive and may be overconfident.",
            "Exact tests require exchangeability under the null; method labels were not randomized treatments.",
            "Endpoint order ranks observed feasibility before objective without fake penalty targets.",
            "No-feasible runs remain in discovery and ordinal endpoint analyses.",
            "No across-task population inference from two already-used development equations.",
        ],
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source_files)},
    }
    save_new(output/"evidence.json", evidence)
    with (output/"trajectories.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectories[0]))
        writer.writeheader()
        writer.writerows(trajectories)
    with (output/"paired_comparisons.csv").open("x", encoding="utf-8", newline="") as stream:
        keys = ("case", "reference", "metric", "n_pairs", "mean_candidate_minus_reference",
                "paired_bootstrap_percentile_95", "exact_two_sided_sign_flip_p",
                "holm_adjusted_p_40_planned_tests", "reason_unavailable")
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparisons)
    lines = [
        "# SMO run-level evidence audit", "", "Analysis date: 2026-10-25.",
        "Retrospective development evidence; no new physical or controlled truth calls.", "",
        "| Case | Method | Feasible online/36 | Hidden failures/36 | Mean endpoint, all seeds | "
        "Zero-initial discoveries | Restricted mean discovery calls |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in group_metrics:
        endpoint = "unavailable" if r["endpoint_mean_all_seeds"] is None else f'{r["endpoint_mean_all_seeds"]:.6f}'
        cost = "N/A" if r["restricted_mean_calls_from_zero"] is None else f'{r["restricted_mean_calls_from_zero"]:.2f}'
        lines.append(f'| {r["case"]} | {r["method"]} | {r["online_feasible_total"]} | '
                     f'{r["hidden_failures_total"]} | {endpoint} | '
                     f'{r["first_feasible_events_from_zero"]}/{r["zero_initial_runs"]} | {cost} |')
    lines += ["", "Candidate-minus-reference differences are reported in paired_comparisons.csv.",
              "Positive endpoint-order effect means the candidate is worse; positive feasible-count "
              "effect means the candidate is better. Direction depends on the metric.", "",
              "## Limits", ""] + ["- "+s for s in evidence["limits"]]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"runs": len(runs), "comparisons": len(comparisons),
                      "new_mda_calls": 0, "output": str(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.study.resolve(), args.output.resolve())
