"""Paired small research runs; not an aircraft optimization claim."""

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

from ra_lsbo.legacy_benchmark import run_original_lsbo
from ra_lsbo.optimizer import OptimizerSettings, initial_candidates, run_optimizer
from ra_lsbo.synthetic import SyntheticProblem


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-project", type=Path, default=root.parent / "LSBO" / "X57-LSBO-refactored")
    parser.add_argument("--output", type=Path, default=root / "runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--scenarios", nargs="+", choices=["mixed", "sparse", "no_initial_feasible"],
                        default=["mixed", "sparse", "no_initial_feasible"])
    parser.add_argument("--algorithms", nargs="+", choices=["ra", "ra_no_boundary", "cbo", "lsbo", "random"],
                        default=["ra", "ra_no_boundary", "cbo", "lsbo", "random"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output.resolve() / ("synthetic_benchmark_" + stamp + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    base = args.base_project.resolve()
    paths = list((root / "ra_lsbo").glob("*.py")) + list((base / "lsbo").glob("*.py")) + [Path(__file__)]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = {"status": "running", "mode": "paired_small_synthetic_research_benchmark",
              "source_sha256": hashes, "seeds": args.seeds, "scenarios": args.scenarios,
              "algorithms": args.algorithms, "runs": [], "scope": [
                  "All methods share evaluator, initial candidates and total call allowance.",
                  "Initialization and calibration labels consume calls; no free oracle data.",
                  "RA uses held-out calibration; comparators may train on all available observations.",
                  "RA can stop early under its extra risk policy; failed/no-feasible runs retained.",
                  "Original LSBO uses reduced numerical effort, not a tuned publication baseline.",
                  "Three seeds are a software/research pilot, not statistical superiority evidence."]}
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save()
    try:
        for scenario in args.scenarios:
            for seed in args.seeds:
                problem = SyntheticProblem(base, scenario)
                settings = OptimizerSettings(seed=seed)
                initial = initial_candidates(problem, settings)
                for algorithm in args.algorithms:
                    specific = replace(settings, algorithm=algorithm)
                    folder = output / scenario / ("seed_" + str(seed)) / algorithm
                    if algorithm == "lsbo":
                        rows, summary = run_original_lsbo(problem, base, specific, folder, initial)
                    else:
                        rows, summary = run_optimizer(problem, base, specific, folder, initial=initial)
                    report["runs"].append({
                        "scenario": scenario, "seed": seed, "algorithm": algorithm,
                        "directory": str(folder), **summary})
                    save()
                    print(f"[benchmark] {scenario} seed={seed} {algorithm}: "
                          f"{summary['status']}, calls={summary['evaluations']}, "
                          f"best={summary['best_feasible_objective']}", flush=True)
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha for path, sha in hashes.items()):
            raise RuntimeError("Code changed during benchmark")
        groups = defaultdict(list)
        for run in report["runs"]:
            groups[(run["scenario"], run["algorithm"])].append(run)
        report["aggregate"] = []
        for (scenario, algorithm), runs in groups.items():
            best = [run["best_feasible_objective"] for run in runs]
            report["aggregate"].append({
                "scenario": scenario, "algorithm": algorithm, "runs": len(runs),
                "runs_with_feasible": sum(value is not None for value in best),
                "best_values_in_seed_order": best,
                "mean_best_only_if_all_feasible": statistics.mean(best) if all(v is not None for v in best) else None,
                "mean_calls": statistics.mean(run["evaluations"] for run in runs),
                "mean_nonfeasible_calls": statistics.mean(run["ledger"]["observed_nonfeasible"] for run in runs),
                "early_stops": sum(run["status"] != "complete" for run in runs),
            })
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print("Benchmark report:", output / "report.json", flush=True)


if __name__ == "__main__":
    main()
