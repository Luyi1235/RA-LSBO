"""Frozen boundary-distribution calibration trial, requiring separate large-budget approval."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from .audit import build_snapshot, load_json
from .budget import BudgetPolicy, RiskLedger, digest
from .calibration import SigmoidCalibrator
from .constraint_risk import fit_frozen_models
from .legacy import normalize_evaluation
from .optimizer import append_json, dispatch
from .selection import candidate_pool
from .x57_live import X57LiveProblem


BANDS = [("outer", -1., 0.), ("near", 0., 1.), ("inner", 1., 2.)]
POLICY = BudgetPolicy(120, 120., 0, 120, 0, .8, 90)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def verify_sources(hashes):
    for filename, expected in hashes.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source changed: " + filename)


def validate_plan(plan):
    if (plan.get("schema") != "ra_boundary_validation_plan_v1" or len(plan["proposals"]) != 120
            or plan["budget_policy"] != asdict(POLICY) or plan["worker_timeout_seconds"] != 1800):
        raise ValueError("Campaign differs from the reviewed 120-call scope")
    seen, counts = set(), {}
    for index, item in enumerate(plan["proposals"], 1):
        phase = "calibration" if index <= 60 else "test"
        if item["index"] != index or item["phase"] != phase or item["band"] not in dict((x[0], x) for x in BANDS):
            raise ValueError("Frozen phase or stratum differs")
        key = digest(item["candidate"])
        if key in seen or item["proposal"]["candidate"] != item["candidate"]:
            raise ValueError("Duplicate or changed candidate")
        seen.add(key)
        counts[phase, item["band"]] = counts.get((phase, item["band"]), 0)+1
        if item["proposal"]["mode"] != "discovery" or item["proposal"]["evidence"]["basis"] != "raw":
            raise ValueError("Validation requires full-charge raw discovery")
        for field in ("raw_rf", "raw_constraint", "global_rf", "global_constraint"):
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError("Invalid frozen probability")
    if any(counts.get((phase, band)) != 20 for phase in ("calibration", "test") for band, _, _ in BANDS):
        raise ValueError("Fixed stratum quotas differ")


def prepare_boundary_validation(previous_pilot, transfer_report, output_root):
    previous_pilot, transfer_report = Path(previous_pilot).resolve(), Path(transfer_report).resolve()
    previous = load_json((previous_pilot / "plan.json").read_bytes())
    previous_result = load_json((previous_pilot / "result.json").read_bytes())
    if previous_result["status"] not in ("complete", "stopped_by_policy"):
        raise ValueError("Previous physical pilot must be terminal without evaluator errors")
    history, features, risks, responses, old_cal, combined, new_cal = fit_frozen_models(previous)
    if new_cal.diagnostics["final_feasibility"]["status"] != "fitted_empirical":
        raise ValueError("Frozen global calibrator unavailable")
    root = Path(__file__).resolve().parents[1]
    folder = Path(output_root).resolve() / ("boundary_validation_request_" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    config_path = folder / "frozen_config.json"
    config_path.write_bytes(Path(previous["config_path"]).read_bytes())
    problem = X57LiveProblem(previous["base_project"], config_path, folder / "workers", allow_mda=False)
    if problem.contract_id != previous["contract_id"]:
        raise ValueError("Physical evaluator changed")
    source_hashes = {}
    exclusions = []
    seen = {digest(problem.canonical(row["candidate"])) for row in history}
    for filename, sha in previous["historical_source_sha256"].items():
        source_hashes[str(Path(previous["historical_source"]) / filename)] = sha
    transfer = load_json(transfer_report.read_bytes())
    for source in transfer["cohorts"]:
        if source == "source_holdout":
            continue
        rows, _, audit, _ = build_snapshot(source)
        if audit["evaluator_contract_sha256"] != previous["historical_contract_sha256"]:
            raise ValueError("Exclusion source has another physical contract")
        seen.update(digest(problem.canonical(row["candidate"])) for row in rows)
        exclusions.append({"source": source, "records": len(rows)})
        for filename, sha in audit["source_sha256"].items():
            source_hashes[str(Path(source) / filename)] = sha
    seen.update(digest(item["proposal"]["candidate"]) for item in previous["proposals"])
    excluded_unique = len(seen)
    groups = {name: [] for name, _, _ in BANDS}
    rng = np.random.default_rng(20261019)
    examined = 0
    for _ in range(40):
        # One integrity check around each cheap pool avoids hashing the same
        # physical assets hundreds of times; real dispatch retains its checks.
        problem.verify_frozen()
        pool, _ = candidate_pool(
            problem.space, rng, 512, seen, history[:240],
            lambda raw: problem.problem.precheck(problem.candidate_class(**raw)))
        candidates = [item["candidate"] for item in pool]
        response = responses.predict(candidates)
        raw_rf = risks.predict(candidates)["final_feasibility"]
        raw_combined = combined.predict(candidates)["final_feasibility"]
        global_rf = old_cal.predict(candidates)["risk_probabilities_calibrated"]["final_feasibility"]
        global_combined = new_cal.predict(candidates)["risk_probabilities_calibrated"]["final_feasibility"]
        margins = np.min([
            -np.asarray(response[name]["mean"])/np.maximum(np.asarray(response[name]["std"]), 1e-12)
            for name in combined.required_heads], axis=0)
        for position, item in enumerate(pool):
            seen.add(digest(item["candidate"]))
            examined += 1
            for name, lower, upper in BANDS:
                if lower <= margins[position] < upper and len(groups[name]) < 40:
                    groups[name].append({
                        "candidate": item["candidate"], "source": item["source"], "band": name,
                        "minimum_standardized_margin": float(margins[position]),
                        "raw_rf": raw_rf[position], "raw_constraint": raw_combined[position],
                        "global_rf": global_rf[position], "global_constraint": global_combined[position],
                    })
                    break
        problem.verify_frozen()
        if all(len(group) == 40 for group in groups.values()):
            break
    if any(len(group) != 40 for group in groups.values()):
        save(folder / "preparation_failed.json", {"band_counts": {k: len(v) for k, v in groups.items()},
                                                  "mda_calls": 0})
        raise ValueError("Fixed candidate strata could not be filled; no MDA executed")
    for group in groups.values():
        rng.shuffle(group)
    proposals = []
    for phase, offset in (("calibration", 0), ("test", 20)):
        for position in range(20):
            for name, _, _ in BANDS:
                item = groups[name][offset+position]
                candidate = item["candidate"]
                proposals.append(dict(
                    item, index=len(proposals)+1, phase=phase,
                    proposal={"candidate": candidate, "source": "frozen_boundary_" + name,
                              "mode": "discovery", "evidence": {
                                  "basis": "raw", "probability": item["raw_constraint"],
                                  "model_id": combined.model_id, "calibration_id": None,
                                  "candidate_key": digest(candidate)}}))
    with RiskLedger(folder / "planning_reservations.jsonl",
                    replace(POLICY, max_observed_nonfeasible=120)) as ledger:
        for item in proposals:
            proposal = item["proposal"]
            ledger.reserve(proposal["candidate"], proposal["mode"], proposal["evidence"], proposal["source"])
        planning_summary = ledger.summary()
    for filename, sha in features.base_hashes.items():
        source_hashes[str(Path(previous["base_project"]) / "lsbo" / filename)] = sha
    for filename, sha in problem.source_hashes.items():
        source_hashes[str(Path(previous["base_project"]) / filename)] = sha
    files = list((root / "ra_lsbo").glob("*.py")) + [
        root / "run_boundary_validation.py", root / "docs" / "BOUNDARY_VALIDATION_PROTOCOL_20261017.md",
        previous_pilot / "plan.json", previous_pilot / "result.json", transfer_report]
    source_hashes.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    verify_sources(source_hashes)
    measured = [item["risk"]["elapsed_seconds"] for item in previous_result["results"]]
    plan = {
        "schema": "ra_boundary_validation_plan_v1", "base_project": previous["base_project"],
        "config_path": str(config_path), "contract_id": problem.contract_id,
        "source_sha256": source_hashes, "budget_policy": asdict(POLICY),
        "worker_timeout_seconds": problem.config["runtime"]["timeout_seconds"],
        "training_indices": features.training_indices, "historical_calibration_indices": old_cal.calibration_indices,
        "historical_fit_and_calibration_calls": 240, "historical_source": previous["historical_source"],
        "global_rf_model_id": risks.model_id, "global_constraint_model_id": combined.model_id,
        "global_rf_calibration": old_cal.diagnostics["final_feasibility"],
        "global_constraint_calibration": new_cal.diagnostics["final_feasibility"],
        "bands": BANDS, "rng_seed": 20261019, "proposals": proposals,
        "excluded_unique_candidates": excluded_unique, "exclusion_sources": exclusions,
        "cheap_candidates_examined": examined, "planning_reservations": planning_summary,
        "estimated_solver_hours_from_previous_pilot": np.mean(measured)*120/3600,
        "nominal_sum_of_worker_timeouts_hours": 120*problem.config["runtime"]["timeout_seconds"]/3600,
        "notes": ["60 calibration + 60 untouched test, each with 20 points per fixed stratum.",
                  "All real calls use worst-case debit 1; global probabilities are forecasts only.",
                  "New local test predictions must be saved before the first test MDA.",
                  "This is distribution-specific calibration validation, not optimization comparison.",
                  "Planning reservations are not real submissions; execution uses a separate ledger."],
    }
    validate_plan(plan)
    save(folder / "plan.json", plan)
    save(folder / "request.json", {
        "schema": "ra_boundary_validation_request_v1", "operator_approved": False,
        "plan_path": str(folder / "plan.json"),
        "plan_sha256": hashlib.sha256((folder / "plan.json").read_bytes()).hexdigest(),
        "purpose": "Explicitly approve up to 120 new MDA calls and the reviewed calibration-stage stop rules.",
        "authorization_required": "User excluded large campaigns of 100 or more calls from automatic approval.",
    })
    # Retain exact executable sources for reproducibility after future edits.
    snapshot = folder / "source_snapshot"
    snapshot.mkdir()
    for filename, sha in source_hashes.items():
        destination = snapshot / (sha + ".source")
        if not destination.exists():
            destination.write_bytes(Path(filename).read_bytes())
    save(snapshot / "index.json", source_hashes)
    return folder, plan


def fit_local_calibration(plan, results):
    if len(results) != 60:
        raise ValueError("Local calibration requires the completed first 60 outcomes")
    calibrators, diagnostics = {}, {}
    for name, score_name in (("local_rf", "raw_rf"), ("local_constraint", "raw_constraint")):
        valid = [(item[score_name], result["risk"]["feasible_label"])
                 for item, result in zip(plan["proposals"][:60], results)
                 if result["risk"]["feasible_label"] is not None]
        fitted = SigmoidCalibrator([p for p, _ in valid], [int(y) for _, y in valid])
        calibrators[name], diagnostics[name] = fitted, fitted.diagnostics
    forecasts = []
    probabilities = {
        name: fitted.predict([item["raw_rf" if name == "local_rf" else "raw_constraint"]
                              for item in plan["proposals"][60:]])
        for name, fitted in calibrators.items()}
    for position, item in enumerate(plan["proposals"][60:]):
        forecasts.append({
            "index": item["index"], "candidate_key": digest(item["candidate"]), "band": item["band"],
            "global_rf": item["global_rf"], "global_constraint": item["global_constraint"],
            **{name: None if values is None else values[position] for name, values in probabilities.items()},
        })
    return {"training_indices": list(range(1, 61)), "diagnostics": diagnostics, "forecasts": forecasts}


def test_metrics(forecasts, results):
    observed = {item["index"]: item["risk"]["feasible_label"] for item in results if item["index"] > 60}
    records = [dict(row, feasible=observed[row["index"]]) for row in forecasts
               if row["index"] in observed and observed[row["index"]] is not None]
    output = {}
    for band in ("all", "outer", "near", "inner"):
        group = [row for row in records if band == "all" or row["band"] == band]
        output[band] = {}
        for model in ("global_rf", "global_constraint", "local_rf", "local_constraint"):
            values = [row for row in group if row[model] is not None]
            accepted = [row for row in values if row[model] >= .8]
            losses = [-(int(row["feasible"])*np.log(np.clip(row[model], 1e-12, 1-1e-12)) +
                        (1-int(row["feasible"]))*np.log1p(-np.clip(row[model], 1e-12, 1-1e-12)))
                      for row in values]
            output[band][model] = {
                "observed_test_samples": len(group), "scored_samples": len(values),
                "brier": float(np.mean([(row[model]-int(row["feasible"]))**2 for row in values])) if values else None,
                "log_loss": float(np.mean(losses)) if losses else None,
                "accepted_at_0_8": len(accepted),
                "infeasible_among_accepted": sum(not row["feasible"] for row in accepted),
            }
    return output


def execute_boundary_validation(request_path):
    request_path = Path(request_path).resolve()
    request = load_json(request_path.read_bytes())
    if request.get("schema") != "ra_boundary_validation_request_v1" or request.get("operator_approved") is not True:
        raise ValueError("Explicit separate approval for the 120-call campaign is required")
    plan_path = Path(request["plan_path"]).resolve()
    if plan_path.parent != request_path.parent:
        raise ValueError("Plan and request must share a reviewed directory")
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != request["plan_sha256"]:
        raise ValueError("Reviewed plan changed")
    plan = load_json(plan_path.read_bytes())
    validate_plan(plan)
    verify_sources(plan["source_sha256"])
    problem = X57LiveProblem(plan["base_project"], plan["config_path"],
                             request_path.parent / "workers", allow_mda=True)
    if problem.contract_id != plan["contract_id"]:
        raise ValueError("Frozen physical contract changed")
    report = {"status": "running", "plan_sha256": request["plan_sha256"],
              "contract_id": plan["contract_id"], "results": []}
    local = None
    with RiskLedger(request_path.parent / "budget.jsonl", POLICY) as ledger:
        try:
            for item in plan["proposals"]:
                if item["index"] == 61:
                    local = fit_local_calibration(plan, report["results"])
                    save(request_path.parent / "test_forecasts.json", local)
                    report["test_forecasts_sha256"] = hashlib.sha256(
                        (request_path.parent / "test_forecasts.json").read_bytes()).hexdigest()
                    if not any(value["status"] == "fitted_empirical" for value in local["diagnostics"].values()):
                        report.update(status="stopped_by_policy", reason="local_calibration_unavailable")
                        break
                proposal = item["proposal"]
                allowed = ledger.assess(proposal["candidate"], proposal["mode"],
                                         proposal["evidence"], proposal["source"])
                if not allowed["allowed"]:
                    report.update(status="stopped_by_policy", reason=allowed["reason"])
                    break
                append_json(request_path.parent / "submitted_proposals.jsonl", item)
                risk = dispatch(problem, ledger, proposal)
                report["results"].append({"index": item["index"], "phase": item["phase"],
                                           "band": item["band"], "proposal": proposal, "risk": risk})
                if risk is None or risk["state"] == "precheck_rejected":
                    report.update(status="stopped_by_policy", reason="final_precheck_changed")
                    break
                save(request_path.parent / "result.json", dict(report, ledger=ledger.summary()))
                print(f"[boundary validation] {item['index']}/120 {item['phase']} {risk['state']}", flush=True)
                if risk["state"] in ("unknown", "infrastructure_error"):
                    report.update(status="stopped_evaluator_error", reason=risk["raw_status"])
                    break
            else:
                report["status"] = "complete"
        finally:
            if report["status"] == "running":
                report["status"] = "interrupted"
            report["ledger"] = ledger.summary()
            if local is not None and all(item["risk"] is not None for item in report["results"]):
                report["test_metrics"] = test_metrics(local["forecasts"], report["results"])
            save(request_path.parent / "result.json", report)
    if RiskLedger(request_path.parent / "budget.jsonl").summary() != report["ledger"]:
        raise RuntimeError("Ledger replay differs")
    return report
