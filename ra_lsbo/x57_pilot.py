"""Prepare a frozen eight-candidate pilot; execution requires a separate approval."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import uuid

import numpy as np

from .audit import build_snapshot, load_json
from .budget import BudgetPolicy, RiskLedger, digest
from .calibration import CalibratedRiskHeads
from .features import LatentFeatures
from .optimizer import append_json, dispatch
from .preflight import inspect_x57
from .responses import ResponseHeads
from .risk_model import RiskHeads
from .selection import candidate_pool, select_candidate
from .transfer_validation import candidate_key, unseen_rows
from .x57_live import X57LiveProblem


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prepare_pilot(source, base_project, output_root):
    source, base, root = [Path(p).resolve() for p in (source, base_project, output_root)]
    if root == source or source in root.parents:
        raise ValueError("Output must be outside the historical run")
    rows, _, audit, contract = build_snapshot(source)
    if len(rows) < 240:
        raise ValueError("Require the predeclared 180 training and 60 calibration observations")
    folder = root / ("x57_pilot_request_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                     + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    config_path = folder / "frozen_config.json"
    save_json(config_path, contract["problem_config"])
    preflight = inspect_x57(source, base, config_path)
    if not preflight["historical_data_reusable_without_revalidation"]:
        raise ValueError("Historical evaluator source/configuration differs from current evaluator")
    if contract["python"] != sys.version:
        raise ValueError("Historical Python version differs")
    for name, version in contract["dependencies"].items():
        if version != "not installed or not packaged" and importlib.metadata.version(name) != version:
            raise ValueError("Historical dependency version differs: " + name)
    problem = X57LiveProblem(base, config_path, folder / "workers", allow_mda=False)
    if (digest([asdict(v) for v in problem.space.variables]) != digest(contract["design_variables"])
            or list(problem.required_constraints) != contract["required_constraints"]
            or list(problem.analytical_constraints) != contract["analytical_constraints"]):
        raise ValueError("Historical variable or constraint schema differs")
    history = rows[:240]
    features = LatentFeatures(base, contract["design_variables"], seed=0).fit(history[:180], epochs=60)
    known = {candidate_key(features, row["candidate"]) for row in history[:180]}
    calibration, removed = unseen_rows(history[180:], features, known)
    risks = RiskHeads(history[:180], features, seed=0)
    calibrated = CalibratedRiskHeads(risks, calibration)
    if calibrated.diagnostics["final_feasibility"]["status"] != "fitted_empirical":
        raise ValueError("Pilot requires an available empirical calibrator")
    responses = ResponseHeads(history[:180], features, seed=0, steps=45)
    policy = BudgetPolicy(8, 4.8, 0, 0, 4, .8, 3)
    rng = np.random.default_rng(20261017)
    # Exclude all historical candidates, including held-out designs. Their
    # coordinates are used only for duplicate exclusion, never for model fitting.
    seen = {digest(problem.canonical(row["candidate"])) for row in rows}
    proposals = []
    # A future sequential plan can contain eight points, but they are not eight
    # simultaneous live reservations. Execution retains the three-outcome stop.
    planning_policy = replace(policy, max_observed_nonfeasible=8)
    with RiskLedger(folder / "planning_reservations.jsonl", planning_policy) as ledger:
        for index in range(8):
            pool, rejected = candidate_pool(
                problem.space, rng, 256, seen, history, problem.precheck)
            proposal, selection = select_candidate(
                pool, history, risks, responses, calibrated, ledger, index+1,
                boundary_every=2)
            if proposal is None:
                raise ValueError("Unable to prepare eight candidates within the fixed risk policy")
            ledger.reserve(proposal["candidate"], proposal["mode"], proposal["evidence"], proposal["source"])
            seen.add(digest(proposal["candidate"]))
            proposals.append({"proposal": proposal, "selection": selection,
                              "precheck_rejections": rejected, "precheck": problem.precheck(proposal["candidate"])})
        reservation_summary = ledger.summary()
    problem.verify_frozen()
    for name, sha in audit["source_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError("History changed during pilot preparation")
    for name, sha in features.base_hashes.items():
        if hashlib.sha256((base / "lsbo" / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError("Representation source changed during pilot preparation")
    plan = {
        "schema": "ra_x57_frozen_pilot_plan_v1", "base_project": str(base),
        "config_path": str(config_path), "contract_id": problem.contract_id,
        "evaluator_sha256": problem.source_hashes, "historical_source": str(source),
        "historical_source_sha256": audit["source_sha256"],
        "historical_contract_sha256": audit["evaluator_contract_sha256"],
        "historical_calls_used_for_fit_and_calibration": 240,
        "training_indices": features.training_indices,
        "calibration_indices": calibrated.calibration_indices, "calibration_excluded": removed,
        "representation_snapshot": features.snapshot_id, "model_id": risks.model_id,
        "calibration_id": calibrated.calibration_id, "calibration_diagnostics": calibrated.diagnostics,
        "base_source_sha256": features.base_hashes,
        "executor_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))},
        "budget_policy": asdict(policy), "proposals": proposals,
        "planning_only_budget_policy": asdict(planning_policy),
        "planning_reservations": reservation_summary,
        "worker_timeout_seconds": problem.config["runtime"]["timeout_seconds"],
        "notes": [
            "Eight frozen proposals for prospective adapter/probability validation, not an adaptive BO trial.",
            "Planning reservations are never submitted; execution creates a separate ledger with the three-outcome stop.",
            "Uses historical training/calibration from one v4 contract; the v3 smoke is separate.",
            "Historical labels are not free optimization evaluations and are disclosed separately.",
            "Risk charge is an empirical expected-nonfeasible score, not a guaranteed bound.",
            "Stop after three nonfeasible/unknown outcomes or the first infrastructure/unknown outcome.",
            "No retry, threshold relaxation, automatic continuation, or concurrent workers.",
            "RCAIDE is not packaged; local evaluator and saved assets are hashed, not a certified environment.",
        ],
    }
    save_json(folder / "plan.json", plan)
    request = {
        "schema": "ra_x57_frozen_pilot_request_v1", "operator_approved": False,
        "plan_path": str(folder / "plan.json"),
        "plan_sha256": hashlib.sha256((folder / "plan.json").read_bytes()).hexdigest(),
        "purpose": "Approve at most eight sequential calls, 1800 seconds each, without automatic retries.",
    }
    save_json(folder / "request.json", request)
    return folder, plan


def execute_pilot(request_path):
    request_path = Path(request_path).resolve()
    request = load_json(request_path.read_bytes())
    if request.get("schema") != "ra_x57_frozen_pilot_request_v1" or request.get("operator_approved") is not True:
        raise ValueError("Explicit approval of this frozen pilot is required")
    plan_path = Path(request["plan_path"]).resolve()
    if plan_path.parent != request_path.parent:
        raise ValueError("Plan and request must share the reviewed directory")
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != request["plan_sha256"]:
        raise ValueError("Reviewed plan changed")
    plan = load_json(plan_path.read_bytes())
    expected_policy = BudgetPolicy(8, 4.8, 0, 0, 4, .8, 3)
    if (plan.get("schema") != "ra_x57_frozen_pilot_plan_v1"
            or plan["budget_policy"] != asdict(expected_policy)
            or len(plan["proposals"]) != 8 or plan["worker_timeout_seconds"] != 1800):
        raise ValueError("Pilot scope differs from the reviewed eight-call protocol")
    for directory, hashes in (
        (Path(plan["historical_source"]), plan["historical_source_sha256"]),
        (Path(plan["base_project"]) / "lsbo", plan["base_source_sha256"]),
        (Path(__file__).parent, plan["executor_source_sha256"]),
    ):
        if any(hashlib.sha256((directory / name).read_bytes()).hexdigest() != sha for name, sha in hashes.items()):
            raise RuntimeError("Frozen source changed; reprepare before approval")
    problem = X57LiveProblem(plan["base_project"], plan["config_path"],
                             request_path.parent / "workers", allow_mda=True)
    if problem.contract_id != plan["contract_id"]:
        raise RuntimeError("Frozen physical contract changed")
    report = {"status": "running", "results": [], "plan_sha256": request["plan_sha256"],
              "contract_id": plan["contract_id"]}
    with RiskLedger(request_path.parent / "budget.jsonl", expected_policy) as ledger:
        try:
            for item in plan["proposals"]:
                proposal = item["proposal"]
                assessment = ledger.assess(proposal["candidate"], proposal["mode"],
                                           proposal["evidence"], proposal["source"])
                if not assessment["allowed"]:
                    report.update(status="stopped_by_policy", reason=assessment["reason"])
                    break
                append_json(request_path.parent / "submitted_proposals.jsonl", item)
                risk = dispatch(problem, ledger, proposal)
                report["results"].append({"proposal": proposal, "risk": risk})
                if risk is None or risk["state"] == "precheck_rejected":
                    report.update(status="stopped_by_policy", reason="final_precheck_changed")
                    break
                if risk["state"] in ("unknown", "infrastructure_error"):
                    report.update(status="stopped_evaluator_error", reason=risk["raw_status"])
                    break
                save_json(request_path.parent / "result.json", dict(report, ledger=ledger.summary()))
            else:
                report["status"] = "complete"
        finally:
            report["ledger"] = ledger.summary()
            if report["status"] == "running":
                report["status"] = "interrupted"
            save_json(request_path.parent / "result.json", report)
    return report
