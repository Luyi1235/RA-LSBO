"""Research RA-LSBO loop; all expensive submissions cross one ledger boundary."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

import numpy as np

from .budget import BudgetPolicy, BudgetRejected, RiskLedger, digest
from .calibration import CalibratedRiskHeads
from .features import LatentFeatures
from .legacy import normalize_evaluation
from .records import RiskEvaluation, finite_number
from .responses import ResponseHeads
from .risk_model import RiskHeads
from .selection import candidate_pool, incumbent, select_candidate


@dataclass(frozen=True)
class OptimizerSettings:
    budget: int = 112
    bootstrap: int = 88
    calibration_window: int = 40
    update_every: int = 8
    pool_size: int = 128
    epochs: int = 10
    gp_steps: int = 12
    seed: int = 0
    algorithm: str = "ra"
    discovery_rule: str = "risk"

    def __post_init__(self):
        for name in ("budget", "bootstrap", "calibration_window", "update_every",
                     "pool_size", "epochs", "gp_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer optimizer settings required")
        if not self.calibration_window + 3 <= self.bootstrap <= self.budget:
            raise ValueError("Require training >=3, nonempty calibration, bootstrap <= budget")
        if self.algorithm not in ("ra", "ra_no_boundary", "cbo", "unconstrained_ei", "random", "lsbo"):
            raise ValueError("Unknown research algorithm")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Nonnegative integer seed required")
        if self.discovery_rule not in ("risk", "constraint_ei"):
            raise ValueError("Unknown discovery acquisition rule")
        if self.discovery_rule != "risk" and self.algorithm not in ("ra", "ra_no_boundary"):
            raise ValueError("Discovery acquisition rules are only supported for RA")


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def initial_candidates(problem, settings):
    rng, result, seen = np.random.default_rng(settings.seed), [], set()
    for _ in range(100000):
        candidate = problem.space.sample(rng)
        if getattr(problem, "scenario", "") == "no_initial_feasible":
            raw = candidate.to_dict()
            raw["continuous"]["x"] *= .08
            candidate = problem.space.canonical(problem.candidate_class(**raw))
        raw = problem.canonical(candidate.to_dict())
        if digest(raw) in seen or any(value > 0 for value in problem.precheck(raw).values()):
            continue
        seen.add(digest(raw))
        result.append(raw)
        if len(result) == settings.bootstrap:
            return result
    raise RuntimeError("Unable to generate a complete shared initialization")


def dispatch(problem, ledger, proposal):
    candidate = problem.canonical(proposal["candidate"])
    if candidate != proposal["candidate"]:
        raise ValueError("Final candidate changed; rescore before approval")
    checks = problem.precheck(candidate)
    if any(not finite_number(value) or value > 0 for value in checks.values()):
        return None
    token = ledger.reserve(candidate, proposal["mode"], proposal["evidence"], proposal["source"])
    ledger.start(token)
    try:
        risk = problem.evaluate(candidate)
        RiskEvaluation(**risk)
    except Exception as error:
        risk = normalize_evaluation({
            "objective": None, "solver_success": False, "constraints": {},
            "status": "exception:" + type(error).__name__,
            "metrics": {"failure_origin": "RA_dispatch", "failure_message": str(error)},
        }, problem.required_constraints, problem.analytical_constraints).to_dict()
    ledger.settle(token, risk)
    return risk


def default_policy(settings):
    ra = settings.algorithm in ("ra", "ra_no_boundary")
    return BudgetPolicy(
        max_calls=settings.budget,
        max_risk_charge=min(settings.budget, settings.bootstrap+14) if ra else settings.budget,
        max_bootstrap_calls=settings.bootstrap, max_discovery_calls=8 if ra else 0,
        max_boundary_calls=8 if ra and settings.algorithm == "ra" else 0,
        allow_baseline=not ra, max_observed_nonfeasible=settings.budget)


def run_optimizer(problem, base_project, settings, output, policy=None, initial=None,
                  model_strategy=None):
    if settings.algorithm == "lsbo":
        raise ValueError("Use the original-LSBO adapter for that comparator")
    if model_strategy is not None and settings.algorithm not in ("ra", "ra_no_boundary"):
        raise ValueError("Discovery model strategies are only supported for RA")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    policy = policy or default_policy(settings)
    if policy.max_calls != settings.budget:
        raise ValueError("Optimizer and ledger call budgets differ")
    initial = initial if initial is not None else initial_candidates(problem, settings)
    if len(initial) != settings.bootstrap:
        raise ValueError("Shared initialization has the wrong length")
    ledger = RiskLedger(output / "budget.jsonl", policy)
    rows, seen, models = [], set(), None
    rng = np.random.default_rng(settings.seed+100000)
    started, status, reason = time.perf_counter(), "running", None
    manifest = {
        "mode": "research_optimization_empirical_risk_not_safety_certification",
        "settings": asdict(settings), "budget_policy": asdict(policy),
        "evaluator_contract_sha256": problem.contract_id,
        "initial_candidates_sha256": digest(initial),
        "design_variables": [asdict(variable) for variable in problem.space.variables],
        "model_strategy": None if model_strategy is None else type(model_strategy).__name__,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    summary = {}
    try:
        while ledger.summary()["charged_calls"] < settings.budget:
            if len(rows) < settings.bootstrap:
                proposal = {
                    "candidate": initial[len(rows)], "source": "shared_initialization", "mode": "bootstrap",
                    "evidence": {"basis": "unavailable", "probability": None},
                }
                info = {"reason": "predeclared_bootstrap"}
            else:
                pool_rows = [] if settings.algorithm == "random" else rows
                pool, rejected = candidate_pool(problem.space, rng, settings.pool_size, seen, pool_rows, problem.precheck)
                iteration = len(rows)-settings.bootstrap
                if settings.algorithm == "random":
                    proposal = None if not pool else dict(pool[0], mode="baseline",
                        evidence={"basis": "unavailable", "probability": None})
                    info = {"reason": "random_pool", "precheck_rejections": rejected}
                else:
                    if (models is None or iteration % settings.update_every == 0
                            or (model_strategy is not None and model_strategy.should_refresh(rows))):
                        variables = manifest["design_variables"]
                        fit_metadata = {}
                        if model_strategy is None:
                            ra = settings.algorithm in ("ra", "ra_no_boundary")
                            training = rows[:-settings.calibration_window] if ra else rows
                            features = LatentFeatures(base_project, variables, settings.seed).fit(training, settings.epochs)
                            risks = RiskHeads(training, features, settings.seed)
                            responses = ResponseHeads(training, features, settings.seed, settings.gp_steps)
                            calibrated = CalibratedRiskHeads(risks, rows[-settings.calibration_window:]) if ra else None
                            models = risks, responses, calibrated
                        else:
                            models, fit_metadata = model_strategy.fit(rows, variables, base_project, settings)
                            risks, responses, calibrated = models
                            features = risks.features
                        append_json(output / "model_fits.jsonl", {
                            **fit_metadata,
                            "before_evaluation": len(rows)+1, "training_indices": features.training_indices,
                            "representation_snapshot": features.snapshot_id, "model_id": risks.model_id,
                            "base_source_sha256": features.base_hashes,
                            "calibration_indices": [] if calibrated is None else calibrated.calibration_indices,
                            "calibration_id": None if calibrated is None else calibrated.calibration_id,
                            "calibration": None if calibrated is None else calibrated.diagnostics,
                        })
                    proposal, info = select_candidate(
                        pool, rows, *models, ledger, iteration, settings.algorithm,
                        discovery_rule=settings.discovery_rule)
                    info["precheck_rejections"] = rejected
                if proposal is None:
                    status, reason = "stopped_by_policy", info
                    break
            # This log precedes dispatch and contains no outcome.
            append_json(output / "proposals.jsonl", {"evaluation": len(rows)+1, "proposal": proposal, "selection": info})
            try:
                risk = dispatch(problem, ledger, proposal)
            except BudgetRejected as error:
                status, reason = "stopped_by_policy", {"reason": str(error)}
                break
            if risk is None or risk["state"] == "precheck_rejected":
                status, reason = "stopped_by_policy", {"reason": "final_precheck_changed"}
                break
            row = {
                "evaluation": len(rows)+1, "candidate": proposal["candidate"], "risk": risk,
                "phase": proposal["mode"], "proposal": proposal,
                "evaluator_contract_sha256": problem.contract_id,
            }
            rows.append(row)
            seen.add(digest(row["candidate"]))
            append_json(output / "observations.jsonl", row)
            if risk["state"] in ("unknown", "infrastructure_error"):
                status, reason = "stopped_evaluator_error", {"reason": risk["raw_status"]}
                break
        else:
            status = "complete"
    except BaseException as error:
        status, reason = "stopped_with_error", {"reason": f"{type(error).__name__}: {error}"}
        raise
    finally:
        summary = {
            "status": status, "stop": reason, "evaluations": len(rows),
            "feasible_evaluations": sum(row["risk"]["feasible_label"] is True for row in rows),
            "best_feasible_objective": incumbent(rows),
            "first_feasible_evaluation": next((row["evaluation"] for row in rows
                                               if row["risk"]["feasible_label"] is True), None),
            "states": dict(Counter(row["risk"]["state"] for row in rows)),
            "modes": dict(Counter(row["phase"] for row in rows)),
            "ledger": ledger.summary(), "wall_seconds": time.perf_counter()-started,
            "settings": asdict(settings), "initial_candidates_sha256": digest(initial),
            "model_strategy": manifest["model_strategy"],
        }
        ledger.close()
        replayed = RiskLedger(output / "budget.jsonl")
        if replayed.summary() != summary["ledger"]:
            raise RuntimeError("Ledger replay disagrees with live accounting")
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return rows, summary
