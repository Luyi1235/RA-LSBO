"""Score actual past-only risk forecasts; do not rescore calibration fit blocks."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.calibration import CalibrationPolicy, SigmoidCalibrator, observed_target
from ra_lsbo.mechanism_system import METHODS
from ra_lsbo.metrics import binary_metrics
from ra_lsbo.system_experiment import save_new
from tools.run_system_validation import read_lines


def rebuild_calibrators(fit):
    result = {}
    for name, diagnostic in (fit["calibration"] or {}).items():
        rows = [r for r in fit["calibration_records"] if r["labels"][name] is not None]
        base = "hidden_success" if name == "any_hidden_failure" else name
        calibrator = SigmoidCalibrator(
            [r["risk_probabilities_raw"][name] for r in rows], [r["labels"][name] for r in rows],
            CalibrationPolicy(**diagnostic["policy"]),
            base_fitted=fit["risk_diagnostics"][base]["status"] == "fitted_raw")
        if any(diagnostic[k] != value for k, value in calibrator.diagnostics.items()):
            raise ValueError("Recorded and reconstructed calibration differ")
        result[name] = calibrator
    return result


def metric(values):
    return dict(
        samples=len(values), positive=sum(y for p, y in values),
        negative=sum(1-y for p, y in values),
        scores=binary_metrics([p for p, y in values], [y for p, y in values]))


def score(folder, output):
    plan = load_json((folder/"plan.json").read_bytes())
    report = load_json((folder/"report.json").read_bytes())
    audit = load_json((folder/"independent_audit.json").read_bytes())
    if report["status"] != "terminal" or audit["status"] != "passed":
        raise ValueError("Require terminal independently audited study")
    verify_sources(plan["source_sha256"])
    verify_sources(audit["input_sha256"])
    all_values, skipped, details = defaultdict(list), Counter(), []
    checked = 0
    for run in report["physical"]:
        directory = Path(run["directory"])
        proposals = {r["evaluation"]: r for r in read_lines(directory/"proposals.jsonl")}
        fits = read_lines(directory/"model_fits.jsonl")
        cache, values = {}, defaultdict(list)
        for row in read_lines(directory/"observations.jsonl"):
            proposal = proposals[row["evaluation"]]
            forecast = proposal["selection"].get("forecast")
            if forecast is None:
                continue
            fit = max((f for f in fits if f["before_evaluation"] <= row["evaluation"]),
                      key=lambda f: f["before_evaluation"])
            if max(fit["training_indices"]) >= row["evaluation"] or any(
                    index >= row["evaluation"] for index in fit["calibration_indices"]):
                raise ValueError("Future label reached forecast")
            if fit["before_evaluation"] not in cache:
                cache[fit["before_evaluation"]] = rebuild_calibrators(fit)
            calibrators = cache[fit["before_evaluation"]]
            raw = dict(forecast["raw"])
            raw["any_hidden_failure"] = 1.-raw["hidden_success"]
            for name, probability in raw.items():
                base = "hidden_success" if name == "any_hidden_failure" else name
                label = observed_target(row["risk"], name)
                if fit["risk_diagnostics"][base]["status"] != "fitted_raw":
                    skipped[(run["method"], name)] += 1
                    continue
                if label is None:
                    continue
                values[(name, "raw")].append((probability, label))
                all_values[(run["method"], name, "raw")].append((probability, label))
                cal = None if name not in calibrators else calibrators[name].predict([probability])
                if cal is not None:
                    if name == "final_feasibility" and not np.isclose(
                            cal[0], forecast["calibrated_final"], rtol=1e-10, atol=1e-12):
                        raise ValueError("Actual selected-point calibration differs")
                    values[(name, "calibrated")].append((cal[0], label))
                    all_values[(run["method"], name, "calibrated")].append((cal[0], label))
            checked += 1
        details.append(dict(method=run["method"], seed=run["seed"],
            actual_calls=run["evaluations"], forecasts=sum(
                p["selection"].get("forecast") is not None for p in proposals.values()),
            metrics=[dict(head=name, scoring=kind, **metric(v)) for (name, kind), v in values.items()]))
    result = dict(
        status="completed", additional_mda_calls=0, selected_forecasts_checked=checked,
        interpretation="Adaptive selected-point diagnostics with observed label masks; not cross-method reliability ranking",
        no_scoring_calibrated_predictions_on_their_fit_block=True,
        unfitted_head_probabilities_excluded=True,
        aggregate=[dict(method=method, head=name, scoring=kind, **metric(v))
                   for (method, name, kind), v in all_values.items()],
        unfitted_head_predictions_skipped=[dict(method=m, head=n, count=c) for (m, n), c in skipped.items()],
        runs=details,
        input_audit_sha256=hashlib.sha256((folder/"independent_audit.json").read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=False)
    save_new(output/"risk_forecasts.json", result)
    (output/"scorer.source.py").write_bytes(Path(__file__).read_bytes())
    print("Risk forecasts checked", checked, "additional MDA", 0, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    score(args.folder.resolve(), args.output.resolve())
