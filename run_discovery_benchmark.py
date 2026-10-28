"""Predeclared paired validation of fresh learning during uncalibrated discovery."""

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import uuid

import torch

from ra_lsbo.budget import RiskLedger
from ra_lsbo.discovery_learning import FreshDiscoveryLearning
from ra_lsbo.legacy_benchmark import run_original_lsbo
from ra_lsbo.optimizer import OptimizerSettings, initial_candidates, run_optimizer
from ra_lsbo.synthetic import SyntheticProblem


def verify_run(folder, summary):
    ledger = RiskLedger(folder / "budget.jsonl").summary()
    if ledger != summary["ledger"] or ledger["charged_calls"] != summary["evaluations"]:
        raise ValueError("Ledger mismatch")
    if ledger["unresolved_started_calls"] or ledger["reserved_calls"]:
        raise ValueError("Unsettled calls")
    fit_path = folder / "model_fits.jsonl"
    fits = [json.loads(line) for line in fit_path.read_text().splitlines()] if fit_path.exists() else []
    for fit in fits:
        if max(fit["training_indices"]) >= fit["before_evaluation"]:
            raise ValueError("Future training row")
        if fit["calibration_indices"]:
            if not max(fit["training_indices"]) < min(fit["calibration_indices"]):
                raise ValueError("Training/calibration overlap")
            if max(fit["calibration_indices"]) >= fit["before_evaluation"]:
                raise ValueError("Future calibration row")
        if fit.get("active_regime") == "raw_full_history_discovery":
            if fit["calibration_id"] is not None or fit["calibration_indices"]:
                raise ValueError("Raw discovery presented as calibrated")
            if max(fit["training_indices"]) != fit["before_evaluation"]-1:
                raise ValueError("Newest observation omitted")
    for previous, following in zip(fits, fits[1:]):
        if previous.get("refresh_on_next_observation") and (
                following["before_evaluation"] != previous["before_evaluation"]+1):
            raise ValueError("Discovery model was not refreshed on the next observation")
    return {"ledger_replay": "passed", "chronology": "passed", "model_fits": len(fits)}


def aggregate(runs):
    groups = defaultdict(list)
    for row in runs:
        groups[row["scenario"], row["algorithm"]].append(row)
    result = []
    for (scenario, algorithm), group in groups.items():
        values = [row["best_feasible_objective"] for row in group]
        result.append({
            "scenario": scenario, "algorithm": algorithm, "runs": len(group),
            "runs_with_feasible": sum(value is not None for value in values),
            "best_values_in_seed_order": values,
            "mean_best_only_if_all_feasible": statistics.mean(values) if None not in values else None,
            "mean_calls": statistics.mean(row["evaluations"] for row in group),
            "mean_nonfeasible_calls": statistics.mean(row["ledger"]["observed_nonfeasible"] for row in group),
            "mean_wall_seconds": statistics.mean(row["wall_seconds"] for row in group),
            "early_stops": sum(row["status"] != "complete" for row in group),
        })
    return result


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-project", type=Path, default=root.parent / "LSBO" / "X57-LSBO-refactored")
    parser.add_argument("--output", type=Path, default=root / "runs")
    parser.add_argument("--stage", choices=["fresh", "constraint"], default="fresh")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve() / ("discovery_benchmark_" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    base = args.base_project.resolve()
    protocol = ("CONSTRAINED_DISCOVERY_PROTOCOL_20261017.md" if args.stage == "constraint"
                else "EXECUTION_AND_DISCOVERY_PROTOCOL_20261017.md")
    paths = list((root / "ra_lsbo").glob("*.py")) + list((base / "lsbo").glob("*.py")) + [
        Path(__file__), root / "docs" / protocol]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = {
        "status": "running", "schema": "fresh_discovery_benchmark_v1", "source_sha256": hashes,
        "stage": args.stage, "protocol": protocol,
        "seeds": list(range(15, 20)) if args.stage == "constraint" else list(range(10, 15)),
        "scenarios": ["mixed", "sparse", "no_initial_feasible"],
        "algorithms": (["ra", "ra_fresh_discovery", "ra_constrained_discovery", "cbo", "lsbo"]
                       if args.stage == "constraint" else ["ra", "ra_fresh_discovery", "cbo", "lsbo"]),
        "runs": [],
        "scope": [
            "Same charged initial 88, total 112, RA discovery cap 8 and RA risk charge cap 102.",
            ("Constraint stage additionally changes discovery ranking using the existing raw CBO score."
             if args.stage == "constraint" else
             "Fresh policy only changes learning when final feasibility calibration is unavailable."),
            "No aircraft MDA; synthetic calls are inexpensive deterministic function evaluations.",
            "All early stops and no-feasible runs retained; no post-result seed selection.",
            "Current workspace LSBO is a reduced-effort implementation comparator.",
        ],
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    save()
    try:
        for scenario in report["scenarios"]:
            for seed in report["seeds"]:
                problem = SyntheticProblem(base, scenario)
                settings = OptimizerSettings(seed=seed)
                initial = initial_candidates(problem, settings)
                paired = []
                for algorithm in report["algorithms"]:
                    fresh = algorithm in ("ra_fresh_discovery", "ra_constrained_discovery")
                    specific = replace(
                        settings, algorithm="ra" if fresh else algorithm,
                        discovery_rule="constraint_ei" if algorithm == "ra_constrained_discovery" else "risk")
                    folder = output / scenario / ("seed_" + str(seed)) / algorithm
                    if algorithm == "lsbo":
                        _, summary = run_original_lsbo(problem, base, specific, folder, initial)
                    else:
                        strategy = FreshDiscoveryLearning() if fresh else None
                        _, summary = run_optimizer(problem, base, specific, folder, initial=initial,
                                                   model_strategy=strategy)
                    verification = verify_run(folder, summary)
                    paired.append(summary["initial_candidates_sha256"])
                    report["runs"].append(dict(
                        summary, scenario=scenario, seed=seed, algorithm=algorithm,
                        directory=str(folder), verification=verification))
                    save()
                    print(f"[discovery] {scenario} seed={seed} {algorithm}: "
                          f"{summary['status']}, calls={summary['evaluations']}, "
                          f"best={summary['best_feasible_objective']}", flush=True)
                if len(set(paired)) != 1:
                    raise ValueError("Paired initialization differs")
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha for path, sha in hashes.items()):
            raise ValueError("Frozen benchmark sources changed")
        report["aggregate"] = aggregate(report["runs"])
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print("Discovery report:", output / "report.json", flush=True)


if __name__ == "__main__":
    main()
