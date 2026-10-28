"""Check source identities, saved forecasts, paired initialization and ledgers."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.budget import RiskLedger
from ra_lsbo.legacy import normalize_evaluation


def verify_files(root, hashes):
    for name, sha in hashes.items():
        if hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Source hash mismatch: " + str(Path(root) / name))


def verify_smoke(path):
    result = load_json(path.read_bytes())
    risk = result["risk"]
    raw = load_json((Path(risk["evidence"]["worker_dir"]) / "response.json").read_bytes())
    converted = normalize_evaluation(raw, list(risk["constraints"]), risk["analytical_constraints"]).to_dict()
    for key in ("state", "objective", "constraints", "constraint_observed",
                "hidden_labels", "hidden_success", "feasible_label"):
        if converted[key] != risk[key]:
            raise ValueError("Adapter differs from raw worker output: " + key)
    if RiskLedger(path.parent / "budget.jsonl").summary() != result["ledger"]:
        raise ValueError("Smoke ledger replay differs")
    if result["ledger"]["charged_calls"] != 1 or result["ledger"]["unresolved_started_calls"]:
        raise ValueError("Smoke call accounting differs")
    return {"path": str(path), "raw_worker_matches_adapter": True,
            "charged_calls": 1, "ledger_replay": "passed"}


def verify_transfer(path):
    report = load_json(path.read_bytes())
    if report["status"] != "passed":
        raise ValueError("Incomplete transfer report")
    for source, hashes in report["source_sha256"].items():
        verify_files(source, hashes)
    verify_files(Path(report["base_project"]) / "lsbo", report["base_source_sha256"])
    forecast = [load_json(line) for line in (path.parent / "test_forecasts.jsonl").read_bytes().splitlines()]
    scored = [load_json(line) for line in (path.parent / "scored_test_predictions.jsonl").read_bytes().splitlines()]
    if len(forecast) != len(scored) or not forecast:
        raise ValueError("Forecast count differs")
    for before, after in zip(forecast, scored):
        if "observed" in before or before != {key: value for key, value in after.items() if key != "observed"}:
            raise ValueError("Forecast was changed after labels became available")
    return {"path": str(path), "forecasts": len(forecast), "forecast_preservation": "passed",
            "source_hashes": "passed", "cohorts": len(report["cohorts"])}


def verify_benchmark(path):
    report = load_json(path.read_bytes())
    if report["status"] != "passed":
        raise ValueError("Incomplete benchmark")
    verify_files(Path("."), report["source_sha256"])
    groups = {}
    errors = []
    for run in report["runs"]:
        summary = RiskLedger(Path(run["directory"]) / "budget.jsonl").summary()
        if summary != run["ledger"] or summary["charged_calls"] != run["evaluations"]:
            raise ValueError("Benchmark ledger mismatch: " + run["directory"])
        if summary["unresolved_started_calls"]:
            raise ValueError("Unsettled benchmark call")
        groups.setdefault((run["scenario"], run["seed"]), set()).add(run["initial_candidates_sha256"])
        if run["status"] in ("stopped_with_error", "stopped_evaluator_error"):
            errors.append({"directory": run["directory"], "stop": run["stop"]})
    if any(len(values) != 1 for values in groups.values()):
        raise ValueError("Paired initializations differ")
    return {"path": str(path), "runs": len(report["runs"]), "paired_groups": len(groups),
            "ledger_replay": "passed", "source_hashes": "passed", "execution_errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, nargs="+", required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"smoke": verify_smoke(args.smoke),
              "transfer": [verify_transfer(path) for path in args.transfer]}
    if args.benchmark:
        report["benchmark"] = verify_benchmark(args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
