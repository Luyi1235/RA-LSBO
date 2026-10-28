"""Independently audit sources, worker truth, chronology, checkpoints and accounting."""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import RiskLedger, digest
from ra_lsbo.calibration import CalibrationPolicy, SigmoidCalibrator, observed_target
from ra_lsbo.features import history_identity
from ra_lsbo.legacy import normalize_evaluation
from ra_lsbo.mechanism_system import FiveModeProblem, HIDDEN_MODES, METHODS, mechanism_support
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem


def lines(path):
    return [load_json(line) for line in Path(path).read_bytes().splitlines()] if Path(path).exists() else []


def check_worker(problem, row):
    risk = row["risk"]
    directory = risk["evidence"].get("worker_dir")
    if directory is None:
        if risk["state"] in ("unknown", "infrastructure_error"):
            return {}
        raise ValueError("Physical outcome has no worker provenance")
    directory = Path(directory)
    request_path, response_path = directory/"request.json", directory/"response.json"
    request = load_json(request_path.read_bytes())
    if request["candidate"] != row["candidate"] or request["config"] != problem.config:
        raise ValueError("Actual worker request differs from submitted candidate or physics")
    if not response_path.exists():
        if risk["state"] in ("unknown", "infrastructure_error"):
            return {str(request_path): hashlib.sha256(request_path.read_bytes()).hexdigest()}
        raise ValueError("Physical response missing")
    converted = normalize_evaluation(load_json(response_path.read_bytes()), problem.required_constraints,
                                     problem.analytical_constraints).to_dict()
    for key in ("raw_status", "state", "objective", "objective_observed", "constraints",
                "constraint_observed", "analytical_constraints", "hidden_success", "hidden_labels",
                "failure_modes", "feasible_label", "mda_evaluated", "budget_charged", "elapsed_seconds"):
        if converted[key] != risk[key]:
            raise ValueError("Raw physical response/record differs: "+key)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (request_path, response_path)}


def verify_trial(folder, plan, spec):
    summary = load_json((folder/"summary.json").read_bytes())
    manifest = load_json((folder/"manifest.json").read_bytes())
    history = load_json(Path(manifest["history_path"]).read_bytes())
    observations, proposals, fits = (lines(folder/name) for name in
                                    ("observations.jsonl", "proposals.jsonl", "model_fits.jsonl"))
    ledger = RiskLedger(folder/"budget.jsonl")
    if (ledger.summary() != summary["ledger"] or ledger.summary()["charged_calls"] != len(observations)
            or ledger.summary()["unresolved_started_calls"] or ledger.summary()["reserved_calls"]
            or history_identity(history) != manifest["history_id"]):
        raise ValueError("Run ledger, outcomes or shared history differ")
    if len(observations) > manifest["settings"]["calls"] or len(proposals) not in (
            len(observations), len(observations)+1):
        raise ValueError("Unexpected proposal/outcome count")
    physical = manifest["kind"] == "physical"
    problem = (X57LiveProblem(spec["base_project"], spec["config_path"], folder/"workers", allow_mda=False)
               if physical else FiveModeProblem(spec["base_project"]))
    if problem.contract_id != manifest["contract_id"]:
        raise ValueError("Run contract differs")
    raw_hashes, checkpoints = {}, 0
    all_rows = list(history)
    seen = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in all_rows}
    tickets = {ticket["candidate_key"]: ticket for ticket in ledger.tickets.values()
               if ticket["state"] == "settled"}
    if len(tickets) != len(observations):
        raise ValueError("Ledger/outcome key sets differ")
    fit_by_eval = {f["before_evaluation"]: f for f in fits}
    latest_fit = None
    for position, row in enumerate(observations):
        submitted = proposals[position]
        if (row["evaluation"] != len(all_rows)+1 or submitted["evaluation"] != row["evaluation"]
                or submitted["history_id"] != history_identity(all_rows)
                or row["candidate"] != submitted["proposal"]["candidate"]):
            raise ValueError("Prediction/history/candidate chronology differs")
        if row["evaluation"] in fit_by_eval:
            latest_fit = fit_by_eval[row["evaluation"]]
        evidence = submitted["proposal"]["evidence"]
        if latest_fit is not None and evidence.get("model_id") != latest_fit["model_id"]:
            raise ValueError("Proposal uses a stale or foreign risk model")
        if evidence["basis"] == "empirical_calibrated" and (
                latest_fit is None or evidence["calibration_id"] != latest_fit["calibration_id"]):
            raise ValueError("Calibration identity is not bound to this prediction")
        if submitted["proposal"]["mode"] == "development":
            threshold = .8+.1*position/max(1, manifest["settings"]["calls"]-1)
            if evidence["probability"] < threshold:
                raise ValueError("Scheduled development threshold bypassed")
        ticket = tickets[digest(row["candidate"])]
        if ticket["risk"] != row["risk"] or any(
                ticket[key] != submitted["proposal"][key] for key in ("mode", "source", "evidence")):
            raise ValueError("Ledger truth/proposal differs")
        key = problem.space.key(problem.candidate_class(**row["candidate"]))
        if key in seen:
            raise ValueError("Previously evaluated physical design repeated within run")
        seen.add(key)
        if physical:
            raw_hashes.update(check_worker(problem, row))
            directory = row["risk"]["evidence"].get("worker_dir")
            if directory is not None:
                # This is a timestamp consistency check, not an external witnessing service.
                timestamp = datetime.fromisoformat(submitted["predicted_utc"]).timestamp()
                if timestamp > (Path(directory)/"request.json").stat().st_mtime+.1:
                    raise ValueError("Forecast timestamp follows actual worker submission")
        elif problem.evaluate(row["candidate"]) != row["risk"]:
            raise ValueError("Controlled function outcome changed")
        all_rows.append(row)
    for fit in fits:
        past = all_rows[:fit["before_evaluation"]-1]
        if not past or max(fit["training_indices"]) >= fit["before_evaluation"]:
            raise ValueError("Future training labels used")
        train_set = set(fit["training_indices"])
        training = [r for r in past if r["evaluation"] in train_set]
        if len(training) != len(train_set) or history_identity(training) != fit["training_history_id"]:
            raise ValueError("Training prefix identity differs")
        if fit["calibration_indices"]:
            if (max(train_set) >= min(fit["calibration_indices"])
                    or max(fit["calibration_indices"]) >= fit["before_evaluation"]):
                raise ValueError("Training/calibration/future overlap")
        expected_targets = [r["evaluation"] for r in training if r["risk"]["objective_observed"]]
        if expected_targets != fit["objective_indices"]:
            raise ValueError("Objective masking differs")
        restored = StructuredFeatures.load_checkpoint(
            folder/fit["encoder_checkpoint"], spec["base_project"], training)
        if restored.snapshot_id != fit["representation_snapshot"]:
            raise ValueError("Saved encoder snapshot differs")
        checkpoints += 1
        calibration = {r["evaluation"]: r for r in past if r["evaluation"] in fit["calibration_indices"]}
        records = fit["calibration_records"]
        if [r["evaluation"] for r in records] != fit["calibration_indices"]:
            raise ValueError("Calibration records are incomplete")
        for record in records:
            source = calibration[record["evaluation"]]
            if any(record["labels"][name] != observed_target(source["risk"], name)
                   for name in record["labels"]):
                raise ValueError("Calibration used foreign or imputed labels")
        if fit["calibration"] is not None:
            for name, diagnostic in fit["calibration"].items():
                known = [r for r in records if r["labels"][name] is not None]
                base = "hidden_success" if name == "any_hidden_failure" else name
                calibrator = SigmoidCalibrator(
                    [r["risk_probabilities_raw"][name] for r in known],
                    [r["labels"][name] for r in known],
                    CalibrationPolicy(**diagnostic["policy"]),
                    base_fitted=fit["risk_diagnostics"][base]["status"] == "fitted_raw")
                for field, value in calibrator.diagnostics.items():
                    if diagnostic[field] != value:
                        raise ValueError("Calibration parameter/threshold reconstruction differs")
    if summary["mechanism_support"] != mechanism_support(observations):
        raise ValueError("Mechanism support differs")
    return {"ledger": "passed", "worker_or_control_outcomes": len(observations),
            "checkpoints": checkpoints, "chronology": "passed", "raw_sha256": raw_hashes}


def audit(folder):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    report = load_json((folder/"report.json").read_bytes())
    verify_sources(plan["source_sha256"])
    for sha in set(plan["source_sha256"].values()):
        if hashlib.sha256((folder/"source_snapshot"/(sha+".source")).read_bytes()).hexdigest() != sha:
            raise ValueError("Source snapshot changed")
    if report["status"] != "terminal" or report["new_mda_calls"] > 336:
        raise ValueError("Campaign not terminal or beyond complete scope")
    verified, raw_hashes = [], {}
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                             folder/"audit_precheck", allow_mda=False)
    probe_calls = 0
    for path in sorted((folder/"probes").glob("probe_*/result.json")):
        value = load_json(path.read_bytes())
        ledger = RiskLedger(path.parent/"budget.jsonl").summary()
        if ledger != value["ledger"] or ledger["unresolved_started_calls"]:
            raise ValueError("Probe ledger differs")
        probe_calls += ledger["charged_calls"]
        if value["risk"] is not None and value["risk"]["state"] != "precheck_rejected":
            raw_hashes.update(check_worker(problem, value))
    calls = probe_calls
    for kind in ("physical", "controlled"):
        expected = {(method, seed) for method in plan["methods"] for seed in plan["seeds"]}
        found = set()
        for run in report[kind]:
            found.add((run["method"], run["seed"]))
            if run["status"] == "process_error":
                raise ValueError("Incomplete process needs accounting audit before accepting results")
            path = Path(run["directory"])
            result = verify_trial(path, plan, plan["physical"])
            raw_hashes.update(result.pop("raw_sha256"))
            verified.append(dict(result, kind=kind, method=run["method"], seed=run["seed"]))
            if kind == "physical":
                calls += run["ledger"]["charged_calls"]
        if found != expected:
            raise ValueError("Missing method/seed run")
    if calls != report["new_mda_calls"]:
        raise ValueError("Campaign true call count differs")
    paths = [path for path in folder.rglob("*") if path.is_file()
             and "source_snapshot" not in path.parts
             and not any(part in ("workers", "solver_work") for part in path.parts)
             and path.suffix in (".json", ".jsonl", ".pt")]
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    inputs.update(raw_hashes)
    return {"status": "passed", "new_mda_calls": calls, "probe_calls": probe_calls,
            "runs": verified, "raw_worker_input_response_files": len(raw_hashes),
            "input_sha256": inputs, "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = audit(args.folder.resolve())
    save_new(args.folder.resolve()/"independent_audit.json", result)
    print("Audit", result["status"], "MDA", result["new_mda_calls"], "runs", len(result["runs"]))
