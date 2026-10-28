"""Read-only use of the existing LSBO optimizer for a synthetic comparator."""

from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import time

from .budget import RiskLedger, digest
from .features import load_base
from .optimizer import append_json, default_policy, dispatch
from .selection import incumbent


def run_original_lsbo(problem, base_project, settings, output, initial):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    modules, _ = load_base(base_project)
    original = importlib.import_module(modules["space"].__package__ + ".optimizer")
    rows, started = [], time.perf_counter()
    ledger = RiskLedger(output / "budget.jsonl", default_policy(settings))

    class Adapter:
        name = "RA_synthetic_" + problem.scenario
        space = problem.space
        required_constraints = problem.required_constraints
        analytical_constraints = problem.analytical_constraints

        def precheck(self, candidate):
            return problem.precheck(candidate.to_dict())

        def evaluate(self, candidate):
            mode = "bootstrap" if len(rows) < settings.bootstrap else "baseline"
            proposal = {"candidate": problem.canonical(candidate.to_dict()), "mode": mode,
                        "source": "original_lsbo", "evidence": {"basis": "unavailable", "probability": None}}
            risk = dispatch(problem, ledger, proposal)
            if risk is None:
                raise RuntimeError("Original optimizer submitted a cheap-rejected candidate")
            row = {"evaluation": len(rows)+1, "candidate": proposal["candidate"], "risk": risk,
                   "phase": mode, "evaluator_contract_sha256": problem.contract_id}
            rows.append(row)
            append_json(output / "observations.jsonl", row)
            return modules["records"].Evaluation(
                objective=risk["objective"], solver_success=risk["objective_observed"],
                constraints=risk["constraints"], status=risk["raw_status"])

    configuration = original.Settings(
        budget=settings.budget, initial=settings.bootstrap, seed=settings.seed,
        pretrain_epochs=settings.epochs, update_epochs=settings.epochs,
        update_every=settings.update_every, gp_steps=settings.gp_steps,
        branches=3, points_per_branch=4, restarts=1, raw_samples=8, local_branches=0,
        max_consecutive_solver_failures=settings.budget, coverage_interval=0)
    engine = original.Optimizer(Adapter(), configuration, output / "original")
    status, reason = "complete", None
    (output / "manifest.json").write_text(json.dumps({
        "settings": asdict(settings), "original_settings": asdict(configuration),
        "evaluator_contract_sha256": problem.contract_id,
        "initial_candidates_sha256": digest(initial),
        "scope": "Existing original optimizer, reduced numerical effort, shared charged initialization.",
    }, indent=2, allow_nan=False), encoding="utf-8")
    try:
        with (output / "original_console.log").open("x", encoding="utf-8") as console, redirect_stdout(console):
            for candidate in initial:
                engine.evaluate(modules["space"].Candidate(**candidate), "initial",
                                {"source": "shared_initialization"})
            engine.run()
    except Exception as error:
        status, reason = "stopped_with_error", {"reason": f"{type(error).__name__}: {error}"}
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
        }
        ledger.close()
        if RiskLedger(output / "budget.jsonl").summary() != summary["ledger"]:
            raise RuntimeError("Original-comparator ledger replay differs")
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return rows, summary
