"""Read-only compatibility checks before spending any real MDA budget."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from .audit import build_snapshot, load_json


def differences(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            path = prefix + "." + key if prefix else key
            if key not in before or key not in after:
                result.append({"field": path, "historical": before.get(key), "current": after.get(key)})
            else:
                result.extend(differences(before[key], after[key], path))
        return result
    return [] if before == after else [{"field": prefix, "historical": before, "current": after}]


def inspect_x57(source, base_project, config_path, calibration_report=None, active_run=None):
    source, base, config_path = (Path(path).resolve() for path in (source, base_project, config_path))
    _, _, audit, contract = build_snapshot(source)
    config_bytes = config_path.read_bytes()
    config = load_json(config_bytes)
    if not contract["evaluator_source_sha256"]:
        raise ValueError("No saved evaluator source identity; do not assume compatibility")
    current_hashes, changed = {}, []
    for name, expected in contract["evaluator_source_sha256"].items():
        path = (base / name).resolve()
        if base not in path.parents:
            raise ValueError("Evaluator path escapes base project")
        current = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        current_hashes[name] = current
        if current != expected:
            changed.append({"file": name, "historical_sha256": expected, "current_sha256": current})
    config_changes = differences(contract["problem_config"], config)
    calibration = {"provided": False}
    if calibration_report is not None:
        raw = load_json(Path(calibration_report).read_bytes())
        if (raw.get("status") != "passed" or raw.get("source_sha256") != audit["source_sha256"]
                or raw.get("evaluator_contract_sha256") != audit["evaluator_contract_sha256"]):
            raise ValueError("Calibration report does not match this accepted source run")
        score = raw["test_scores"]["final_feasibility"]
        calibration = {
            "provided": True, "path": str(Path(calibration_report).resolve()),
            "report_sha256": hashlib.sha256(Path(calibration_report).read_bytes()).hexdigest(),
            "status": raw["calibration_diagnostics"]["final_feasibility"]["status"],
            "test_samples": score["calibrated_only"]["samples"],
            "raw_brier": score["raw_on_calibrated_cases"]["brier"],
            "calibrated_brier": score["calibrated_only"]["brier"],
            "brier_change": score["calibrated_minus_raw_brier"],
            "log_loss_change": score["calibrated_minus_raw_log_loss"],
            "independent_run_validation": False,
        }
    active = None
    if active_run is not None:
        active_path = Path(active_run).resolve() / "summary.json"
        data = load_json(active_path.read_bytes())
        active = {"path": str(active_path.parent), "recorded_status": data.get("status"),
                  "recorded_evaluations": data.get("evaluations"),
                  "note": "Summary snapshot only; no process was stopped or modified."}
    reasons = []
    if changed:
        reasons.append("evaluator_source_changed")
    if config_changes:
        reasons.append("physical_config_changed")
    if calibration.get("brier_change") is not None and calibration["brier_change"] > 0:
        reasons.append("heldout_calibration_brier_worsened")
    if calibration.get("log_loss_change") is not None and calibration["log_loss_change"] > 0:
        reasons.append("heldout_calibration_log_loss_worsened")
    if calibration.get("status") != "fitted_empirical":
        reasons.append("final_feasibility_calibration_unavailable")
    if active is not None and active["recorded_status"] == "running":
        reasons.append("existing_experiment_records_running")
    reasons.extend(["no_independent_run_calibration_validation", "real_mda_budget_requires_explicit_configuration"])
    # Verify the read-only inputs remained stable during inspection.
    if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != sha
           for name, sha in audit["source_sha256"].items()):
        raise RuntimeError("Historical source changed during preflight")
    if config_path.read_bytes() != config_bytes:
        raise RuntimeError("Current config changed during preflight")
    for name, sha in current_hashes.items():
        path = base / name
        latest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if latest != sha:
            raise RuntimeError("Evaluator source changed during preflight")
    return {
        "status": "inspected_no_mda", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source), "source_sha256": audit["source_sha256"],
        "historical_contract_sha256": audit["evaluator_contract_sha256"],
        "current_config_path": str(config_path), "current_config": config,
        "current_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "current_evaluator_sha256": current_hashes, "changed_evaluator_files": changed,
        "config_changes": config_changes, "historical_data_reusable_without_revalidation": not changed and not config_changes,
        "calibration_evidence": calibration, "existing_experiment": active,
        "real_mda_enabled": False, "readiness_issues": reasons,
        "scope": "No physical imports, no candidate evaluation, no source or experiment modifications.",
    }


def export_preflight(output_root, **arguments):
    report = inspect_x57(**arguments)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(output_root).resolve() / ("x57_preflight_" + stamp + "_" + uuid.uuid4().hex[:8])
    source = Path(arguments["source"]).resolve()
    if output == source or source in output.parents:
        raise ValueError("Preflight output cannot be inside historical source")
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return output, report
