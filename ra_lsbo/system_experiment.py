"""Paired warm-start experiments with mechanism-aware ranking and durable outcomes."""

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .audit import load_json
from .budget import BudgetPolicy, BudgetRejected, RiskLedger, digest
from .features import history_identity, load_base
from .mechanism_system import FullSystemLearning, FiveModeProblem, METHODS, SystemSettings, mechanism_support
from .optimizer import append_json, dispatch
from .records import HIDDEN_MODES
from .selection import candidate_pool, constraint_acquisition, incumbent, log_expected_improvement, normalized
from .x57_live import X57LiveProblem


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def update_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sample_rows(problem, count, seed):
    rng, rows, seen = np.random.default_rng(seed), [], set()
    for _ in range(count*1000):
        candidate = problem.canonical(problem.space.sample(rng).to_dict())
        key = problem.space.key(problem.candidate_class(**candidate))
        if key in seen or any(v > 0 for v in problem.precheck(candidate).values()):
            continue
        risk = problem.evaluate(candidate)
        rows.append({"evaluation": len(rows)+1, "candidate": candidate, "risk": risk,
                     "evaluator_contract_sha256": problem.contract_id})
        seen.add(key)
        if len(rows) == count:
            return rows
    raise RuntimeError("Unable to sample the predeclared controlled history")


def policy_for(method, settings):
    trusted = method.startswith("ra_") and method != "ra_no_calibration"
    return BudgetPolicy(
        settings.calls, settings.calls/2 if trusted else float(settings.calls),
        0, settings.discovery_cap if trusted else 0,
        settings.boundary_cap if trusted and method != "ra_no_boundary" else 0,
        .8, settings.failure_cap, allow_baseline=not trusted)


def select_system_candidate(pool, rows, models, ledger, iteration, method, settings):
    if not pool:
        return None, {"reason": "empty_pool"}
    risks, responses, calibrated = models
    candidates = [p["candidate"] for p in pool]
    raw = risks.predict(candidates)
    predicted = responses.predict(candidates)
    calibrated_values = (None if calibrated is None else
                         calibrated.predict(candidates)["risk_probabilities_calibrated"]["final_feasibility"])
    log_ei = log_expected_improvement(responses, candidates, incumbent(rows))
    relative_ei = (np.exp(log_ei-np.max(log_ei))
                   if np.isfinite(log_ei).any() else np.zeros(len(pool)))
    objective = predicted.get("objective")
    sd = np.zeros(len(pool)) if objective is None else np.asarray(objective["std"])
    boundary = np.zeros(len(pool))
    for name, values in predicted.items():
        if name.startswith("constraint:") and values is not None:
            p = np.asarray(values["p_satisfied_raw"])
            boundary = np.maximum(boundary, 4*p*(1-p))
    mechanism_uncertainty = risks.mechanism_uncertainty(raw)
    information = boundary+mechanism_uncertainty
    boundary_score = information*(.25+relative_ei+.25*normalized(sd))
    discovery_score = np.asarray(raw["final_feasibility"])+.15*information+.05*normalized(sd)
    modes = []
    if method == "cbo":
        modes = [("baseline", constraint_acquisition(raw, predicted, log_ei))]
    elif method == "ra_no_calibration":
        # Raw scores remain raw and consume a full debit; no calibrated identity.
        modes = [("baseline", log_ei+np.log(np.clip(raw["final_feasibility"], 1e-12, 1.)))]
    else:
        development = ("development", log_ei)
        boundary_mode = ("boundary", boundary_score)
        if calibrated_values is not None and incumbent(rows) is not None:
            modes = [development]
            if method != "ra_no_boundary" and iteration < (settings.calls+1)//2:
                modes = ([boundary_mode, development] if iteration % 4 == 0
                         else [development, boundary_mode])
        modes.append(("discovery", discovery_score))
    threshold = .8+.1*iteration/max(1, settings.calls-1)
    rejections = Counter()
    for mode, score in modes:
        for index in np.argsort(-score, kind="stable"):
            if not np.isfinite(score[index]):
                rejections["nonfinite_acquisition"] += 1
                continue
            p_cal = None if calibrated_values is None else float(calibrated_values[index])
            if mode == "development" and (p_cal is None or p_cal < threshold):
                rejections["scheduled_development_threshold"] += 1
                continue
            uses_calibration = mode in ("development", "boundary") and p_cal is not None
            evidence = {
                "basis": "empirical_calibrated" if uses_calibration else "raw",
                "probability": p_cal if uses_calibration else float(raw["final_feasibility"][index]),
                "model_id": risks.model_id,
                "calibration_id": calibrated.calibration_id if uses_calibration else None,
                "candidate_key": digest(candidates[index])}
            item = pool[int(index)]
            assessment = ledger.assess(item["candidate"], mode, evidence, item["source"])
            if not assessment["allowed"]:
                rejections[assessment["reason"]] += 1
                continue
            forecast = {
                "raw": {name: values[index] for name, values in raw.items()},
                "calibrated_final": p_cal,
                "responses": {name: None if values is None else
                              {key: value[index] for key, value in values.items()}
                              for name, values in predicted.items()},
                "active_mechanisms": list(risks.active_mechanisms)}
            return dict(item, mode=mode, evidence=evidence), {
                "reason": "approved", "score": float(score[index]), "forecast": forecast,
                "development_threshold": threshold, "constraint_boundary": float(boundary[index]),
                "mechanism_uncertainty": float(mechanism_uncertainty[index]),
                "risk_charge": assessment["charge"], "rejections": dict(rejections)}
    return None, {"reason": "no_approved_candidate", "rejections": dict(rejections)}


class OriginalLSBOWarmStart:
    """Run the unmodified original proposal/evaluation update using settled outcomes."""

    def __init__(self, problem, base_project, rows, seed, settings, output):
        modules, _ = load_base(base_project)
        original = importlib.import_module(modules["space"].__package__+".optimizer")
        self.modules, self.pending = modules, None
        self.problem = problem
        owner = self

        class SettledAdapter:
            name = getattr(problem, "name", "X57 paired validation")
            space = problem.space
            required_constraints = problem.required_constraints
            analytical_constraints = problem.analytical_constraints
            coverage_variables = getattr(problem, "coverage_variables", ())
            solver_domain_metric = getattr(problem, "solver_domain_metric", None)

            def precheck(self, candidate):
                return problem.precheck(candidate.to_dict())

            def evaluate(self, candidate):
                if owner.pending is None or owner.pending[0] != problem.space.key(candidate):
                    raise RuntimeError("Original update needs an already settled, matching physical outcome")
                result = owner.pending[1]
                owner.pending = None
                return result

        # Preserve the original search effort, screening and proposal defaults.
        config = original.Settings(
            budget=len(rows)+settings.calls, initial=len(rows), seed=seed,
            gp_steps=settings.gp_steps)
        self.engine = original.Optimizer(SettledAdapter(), config, output/"original")
        self.engine.observations = [self.observation(r) for r in rows]
        self.engine.seen = {problem.space.key(o.candidate) for o in self.engine.observations}
        self.settings = asdict(config)

    def observation(self, row):
        records, space = self.modules["records"], self.modules["space"]
        risk = row["risk"]
        result = records.Evaluation(
            risk["objective"], risk["objective_observed"], risk["constraints"], risk["raw_status"],
            dict(risk["evidence"]))
        return records.Observation(row["evaluation"], space.Candidate(**row["candidate"]), result,
                                   row.get("phase", "shared_history"), row.get("proposal", {}))

    def propose(self):
        phase = self.engine.search_phase()
        candidate, info = self.engine.propose(phase)
        return {"candidate": self.problem.canonical(candidate.to_dict()), "mode": "baseline",
                "source": "original_lsbo:"+info.get("source", "proposal"),
                "evidence": {"basis": "unavailable", "probability": None}}, {
                    "reason": "original_lsbo", "original_phase": phase, "original_proposal": info}

    def accept(self, row, selection):
        observation = self.observation(row)
        self.pending = self.problem.space.key(observation.candidate), observation.result
        self.engine.evaluate(observation.candidate, selection["original_phase"],
                             selection["original_proposal"])
        if self.pending is not None:
            raise RuntimeError("Original optimizer did not consume the settled outcome")


def signal_stop(path, reason):
    try:
        save_new(path, reason)
    except FileExistsError:
        pass


def run_system_trial(spec, method, seed, settings_dict, history_path, output, stop_path):
    torch.set_num_threads(1)
    settings = SystemSettings(**settings_dict)
    output, stop_path = Path(output), Path(stop_path)
    output.mkdir(parents=True, exist_ok=False)
    rows = load_json(Path(history_path).read_bytes())
    original_count = len(rows)
    base = spec["base_project"]
    problem = (FiveModeProblem(base) if spec["kind"] == "controlled" else
               X57LiveProblem(base, spec["config_path"], output/"workers", allow_mda=True))
    if problem.contract_id != spec["contract_id"]:
        raise ValueError("Physical/control contract changed before trial")
    variables = [asdict(v) for v in problem.space.variables]
    seen = {digest(problem.canonical(r["candidate"])) for r in rows}
    initial_id = history_identity(rows)
    rng = np.random.default_rng(seed+200000)
    learner = None if method in ("random", "lsbo") else FullSystemLearning(method, settings, seed)
    policy = policy_for(method, settings)
    manifest = {
        "method": method, "seed": seed, "settings": asdict(settings), "policy": asdict(policy),
        "kind": spec["kind"], "contract_id": problem.contract_id,
        "history_path": str(history_path), "history_id": initial_id,
        "historical_calls": original_count, "design_variables": variables,
        "mechanism_support": mechanism_support(rows)}
    save_new(output/"manifest.json", manifest)
    start, status, reason, models, engine = time.perf_counter(), "running", None, None, None
    new_rows = []
    ledger = RiskLedger(output/"budget.jsonl", policy)
    try:
        if method == "lsbo":
            engine = OriginalLSBOWarmStart(problem, base, rows, seed, settings, output)
            save_new(output/"original_settings.json", engine.settings)
        for iteration in range(settings.calls):
            if stop_path.exists():
                status, reason = "stopped_global", load_json(stop_path.read_bytes())
                break
            if method == "lsbo":
                proposal, selection = engine.propose()
            else:
                # Physical identity is checked around a batch; cheap pool checks do not rehash assets.
                cheap = (lambda raw: problem.problem.precheck(problem.candidate_class(**raw))
                         if isinstance(problem, X57LiveProblem) else problem.precheck(raw))
                if isinstance(problem, X57LiveProblem):
                    problem.verify_frozen()
                pool, rejected = candidate_pool(
                    problem.space, rng, settings.pool_size, seen, rows, cheap)
                if method == "random":
                    proposal = None if not pool else dict(
                        pool[0], mode="baseline", evidence={"basis": "unavailable", "probability": None})
                    selection = {"reason": "random"}
                else:
                    if models is None or iteration % settings.update_every == 0:
                        models, metadata = learner.fit(
                            rows, variables, base,
                            sorted(set(problem.required_constraints)-set(problem.analytical_constraints)))
                        metadata["before_evaluation"] = len(rows)+1
                        metadata["fit_number"] = iteration//settings.update_every
                        checkpoint = "encoder_fit_"+str(metadata["fit_number"])+".pt"
                        learner.features.save_checkpoint(output/checkpoint)
                        metadata["encoder_checkpoint"] = checkpoint
                        append_json(output/"model_fits.jsonl", metadata)
                    proposal, selection = select_system_candidate(
                        pool, rows, models, ledger, iteration, method, settings)
                selection["precheck_rejections"] = rejected
            if proposal is None:
                status, reason = "stopped_by_policy", selection
                break
            if stop_path.exists():
                status, reason = "stopped_global", load_json(stop_path.read_bytes())
                break
            append_json(output/"proposals.jsonl", {
                "evaluation": len(rows)+1, "new_call": iteration+1,
                "predicted_utc": datetime.now(timezone.utc).isoformat(),
                "proposal": proposal, "selection": selection, "history_id": history_identity(rows)})
            try:
                risk = dispatch(problem, ledger, proposal)
            except BudgetRejected as error:
                status, reason = "stopped_by_policy", {"reason": str(error)}
                break
            if risk is None or risk["state"] == "precheck_rejected":
                status, reason = "stopped_by_policy", {"reason": "final_precheck_changed"}
                break
            row = {"evaluation": len(rows)+1, "candidate": proposal["candidate"], "risk": risk,
                   "phase": proposal["mode"], "proposal": proposal,
                   "evaluator_contract_sha256": rows[0]["evaluator_contract_sha256"],
                   "source_journal_sha256": rows[0].get("source_journal_sha256")}
            rows.append(row)
            new_rows.append(row)
            seen.add(digest(row["candidate"]))
            append_json(output/"observations.jsonl", row)
            if engine is not None:
                engine.accept(row, selection)
            update_json(output/"progress.json", {
                "method": method, "seed": seed, "new_calls": len(new_rows),
                "best": incumbent(rows), "last_state": risk["state"], "ledger": ledger.summary()})
            print(method, seed, len(new_rows), risk["state"], incumbent(rows), flush=True)
            if risk["state"] in ("unknown", "infrastructure_error"):
                status, reason = "stopped_evaluator_error", {"state": risk["state"], "status": risk["raw_status"]}
                signal_stop(stop_path, dict(reason, method=method, seed=seed))
                break
            if ledger.summary()["observed_nonfeasible"] >= settings.failure_cap:
                status, reason = "stopped_by_policy", {"reason": "observed_failure_stop"}
                break
        else:
            status = "complete"
    except Exception as error:
        status, reason = "stopped_model_error", {"reason": f"{type(error).__name__}: {error}"}
        import traceback
        (output/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        ledger.close()
        replayed = RiskLedger(output/"budget.jsonl").summary()
        if replayed["charged_calls"] != len(new_rows) or replayed["unresolved_started_calls"]:
            signal_stop(stop_path, {"reason": "Unresolved or unrecorded expensive call", "directory": str(output)})
            status = "stopped_accounting_error"
        summary = {
            "status": status, "stop": reason, "method": method, "seed": seed, "kind": spec["kind"],
            "evaluations": len(new_rows), "historical_calls": original_count, "history_id": initial_id,
            "initial_best": incumbent(rows[:original_count]), "best_feasible_objective": incumbent(rows),
            "feasible_evaluations": sum(r["risk"]["feasible_label"] is True for r in new_rows),
            "states": dict(Counter(r["risk"]["state"] for r in new_rows)),
            "mechanism_support": mechanism_support(new_rows),
            "ledger": replayed, "wall_seconds": time.perf_counter()-start,
            "directory": str(output)}
        save_new(output/"summary.json", summary)
    return summary
