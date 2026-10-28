"""Independent conditional-uniform qualification; never use optimizer rankings."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import BudgetPolicy, RiskLedger, digest
from ra_lsbo.optimizer import append_json, dispatch
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem


def wilson(successes, count):
    if type(successes) is not int or type(count) is not int or not 0 <= successes <= count or count < 1:
        raise ValueError("Valid binomial counts required")
    z = float(norm.ppf(.975))
    p, denominator = successes/count, 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    half = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0., center-half), min(1., center+half)]


def study_charged(folder):
    # Parent study contains physical ledgers only. Child ledger recovery also
    # validates all hash chains and includes started-but-unsettled calls.
    return sum(RiskLedger(path).summary()["charged_calls"]
               for path in folder.rglob("budget.jsonl"))


def conditional_candidates(problem, seed, count):
    rng, accepted, seen, passed = np.random.default_rng(seed), [], set(), 0
    problem.verify_frozen()
    # Fixed denominator also measures rejection by the analytical constraints.
    for _ in range(4000):
        raw = problem.canonical(problem.space.sample(rng).to_dict())
        checks = problem.problem.precheck(problem.candidate_class(**raw))
        if any(not np.isfinite(v) or v > 0 for v in checks.values()):
            continue
        passed += 1
        key = digest(raw)
        if key not in seen and len(accepted) < count:
            accepted.append(raw)
            seen.add(key)
    problem.verify_frozen()
    if len(accepted) != count:
        raise ValueError("Not enough analytical-pass conditional-uniform designs")
    return accepted, {"unconditional_draws": 4000, "analytical_pass": passed,
                      "analytical_pass_fraction": passed/4000,
                      "analytical_pass_wilson_95": wilson(passed, 4000)}


def prepare(study, phase):
    pilot = load_json((study/"plan.json").read_bytes())
    result = load_json((study/"pilot_result.json").read_bytes())
    verify_sources(pilot["source_sha256"])
    if (result["status"] != "complete" or result["charged_real_calls"] != 6
            or pilot["entire_new_study_max_real_calls"] != 300):
        raise ValueError("Complete six-call engineering pilot required")
    distances, count = [55, 65], 30
    if phase == "confirmation":
        qualification = load_json((study/"qualification/result.json").read_bytes())
        if qualification["status"] != "complete" or qualification["selected_distance_nmi"] is None:
            raise ValueError("Qualification gate did not select a challenge")
        distances, count = [qualification["selected_distance_nmi"]], 60
    output = study/phase
    output.mkdir(exist_ok=False)
    tasks = []
    for distance in distances:
        task = pilot["tasks"][str(distance)]
        problem = X57LiveProblem(pilot["base_project"], task["config_path"], output/"disabled_workers", False)
        if problem.contract_id != task["contract_id"]:
            raise ValueError("Physical task changed")
        seed = (91000 if phase == "qualification" else 192000)+distance
        candidates, precheck = conditional_candidates(problem, seed, count)
        tasks.append({"distance_nmi": distance, "contract_id": problem.contract_id,
                      "config_path": task["config_path"], "seed": seed,
                      "candidates": candidates, "precheck_sampling": precheck})
    paths = [Path(__file__).resolve(), ROOT/"ra_lsbo/x57_live.py", ROOT/"ra_lsbo/budget.py",
             ROOT/"ra_lsbo/optimizer.py"]
    save_new(output/"plan.json", {
        "schema": "compact_conditional_uniform_v1", "phase": phase,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study": str(study), "study_max_calls": 300, "phase_max_calls": 60,
        "prior_charged_real_calls": study_charged(study), "per_task_calls": count,
        "selection_rule": "shortest challenge with 1..6 feasible of 30; no unknown outcomes",
        "confirmation_rule": ">=1 feasible and 95% Wilson upper <=0.20 at n=60; no unknown outcomes",
        "base_project": pilot["base_project"], "pilot_plan_sha256": hashlib.sha256(
            (study/"plan.json").read_bytes()).hexdigest(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "tasks": tasks})
    print(output/"plan.json", flush=True)


def execute(study, phase):
    output = study/phase
    plan = load_json((output/"plan.json").read_bytes())
    if (plan["schema"] != "compact_conditional_uniform_v1"
            or plan["study_max_calls"] != 300 or plan["phase_max_calls"] != 60
            or plan["phase"] != phase or str(study) != plan["study"]
            or hashlib.sha256((study/"plan.json").read_bytes()).hexdigest() != plan["pilot_plan_sha256"]):
        raise ValueError("Frozen task qualification scope differs")
    verify_sources(plan["source_sha256"])
    if study_charged(study) != plan["prior_charged_real_calls"]:
        raise ValueError("Study advanced after qualification preparation")
    lock = study/"study_active.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(phase)
    try:
        save_new(output/"execution_started.json", {"utc": datetime.now(timezone.utc).isoformat()})
        status, outcomes = "complete", []
        for task in plan["tasks"]:
            problem = X57LiveProblem(plan["base_project"], task["config_path"],
                                     output/("workers_"+str(task["distance_nmi"])), True)
            if problem.contract_id != task["contract_id"]:
                raise ValueError("Physical contract changed")
            ledger_path = output/("task_"+str(task["distance_nmi"]))/"budget.jsonl"
            n = plan["per_task_calls"]
            with RiskLedger(ledger_path, BudgetPolicy(n, float(n), n, 0, 0)) as ledger:
                for candidate in task["candidates"]:
                    verify_sources(plan["source_sha256"])
                    if study_charged(study) >= 300 or len(outcomes) >= 60:
                        raise RuntimeError("Global physical study budget reached")
                    proposal = {"candidate": candidate, "source": "independent_conditional_uniform",
                                "mode": "bootstrap", "evidence": {"basis": "unavailable"}}
                    append_json(output/"submissions.jsonl", {
                        "phase_call": len(outcomes)+1, "task": task["distance_nmi"],
                        "candidate_key": digest(candidate), "proposal": proposal})
                    result = dispatch(problem, ledger, proposal)
                    if result is None:
                        raise RuntimeError("Qualified analytical-pass point was rejected")
                    row = {"evaluation": len(outcomes)+1, "candidate": candidate, "risk": result,
                           "evaluator_contract_sha256": problem.contract_id,
                           "distance_nmi": task["distance_nmi"]}
                    outcomes.append(row)
                    append_json(output/"observations.jsonl", row)
                    print("Qualification", len(outcomes), task["distance_nmi"], result["state"], flush=True)
                    if result["feasible_label"] is None:
                        status = "stopped_unknown"
                        break
            if status != "complete":
                break
        scores = []
        for task in plan["tasks"]:
            rows = [r for r in outcomes if r["distance_nmi"] == task["distance_nmi"]]
            successes = sum(r["risk"]["feasible_label"] is True for r in rows)
            scores.append({"distance_nmi": task["distance_nmi"], "count": len(rows),
                           "feasible": successes, "wilson_95": wilson(successes, len(rows)) if rows else None})
        selected, qualified = None, False
        if status == "complete" and phase == "qualification":
            eligible = [s["distance_nmi"] for s in scores if s["count"] == 30 and 1 <= s["feasible"] <= 6]
            selected = min(eligible) if eligible else None
        elif status == "complete" and phase == "confirmation":
            score = scores[0]
            selected = score["distance_nmi"]
            qualified = score["count"] == 60 and score["feasible"] >= 1 and score["wilson_95"][1] <= .20
        report = {"status": status, "scores": scores, "selected_distance_nmi": selected,
                  "sparse_region_qualified": qualified, "phase_calls": len(outcomes),
                  "entire_study_charged": study_charged(study),
                  "selection_did_not_use_optimizer_rankings": True}
        save_new(output/"result.json", report)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--phase", choices=("qualification", "confirmation"), default="qualification")
    args = parser.parse_args()
    (prepare if args.action == "prepare" else execute)(args.study.resolve(), args.phase)
