"""Recover a zero-dispatch phase using verified historical source blobs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import BudgetPolicy, RiskLedger, digest
from ra_lsbo.optimizer import append_json, dispatch
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem
from tools.qualify_compact_mda import study_charged, wilson


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_entries(index, original):
    original = Path(original).resolve()
    entries = {}
    for filename, expected in index.items():
        try:
            relative = Path(filename).resolve().relative_to(original)
        except ValueError:
            continue
        if relative.parts[0] in ("lsbo", "problems"):
            entries[relative] = expected
    if not {"lsbo/space.py", "problems/x57/worker.py"}.issubset(
            {path.as_posix() for path in entries}):
        raise ValueError("Incomplete historical evaluator snapshot")
    return entries


def materialize(snapshot, original, destination):
    snapshot, destination = Path(snapshot), Path(destination)
    entries = source_entries(load_json((snapshot/"index.json").read_bytes()), original)
    # Validate every blob before creating a runnable checkout.
    for expected in entries.values():
        if sha(snapshot/(expected+".source")) != expected:
            raise ValueError("Corrupt source blob: "+expected)
    destination.mkdir(parents=True, exist_ok=False)
    for relative, expected in entries.items():
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((snapshot/(expected+".source")).read_bytes())
    verify_sources({str(destination/path): expected for path, expected in entries.items()})
    return {str(destination/path): expected for path, expected in entries.items()}


def prepare(study):
    old = study/"qualification"
    frozen = load_json((old/"plan.json").read_bytes())
    verify_sources(frozen["source_sha256"])
    if (study/"study_active.lock").exists() or (old/"result.json").exists():
        raise ValueError("Another phase is active or original phase has a result")
    if any((old/name).exists() for name in ("submissions.jsonl", "observations.jsonl")):
        raise ValueError("Original phase dispatched work; zero-dispatch recovery forbidden")
    if list(old.rglob("budget.jsonl")) or study_charged(study) != 6:
        raise ValueError("Original phase has charges or study has advanced")
    log = old/"execution_console_20261025.log"
    if not log.exists() or "Physical contract changed" not in log.read_text(encoding="utf-8"):
        raise ValueError("Missing evidence of pre-dispatch contract rejection")
    output = study/"qualification_recovered_20261025"
    snapshot = ROOT/"runs/system_validation_v5_20261020T192312Z_5f6531eb/source_snapshot"
    base = output/"frozen_base"
    hashes = materialize(snapshot, frozen["base_project"], base)
    tasks = []
    for task in frozen["tasks"]:
        problem = X57LiveProblem(base, task["config_path"], output/("workers_"+str(task["distance_nmi"])))
        if problem.contract_id != task["contract_id"]:
            raise ValueError("Recovered physical contract still differs; no MDA allowed")
        if any(any(value > 0 for value in problem.precheck(raw).values())
               for raw in task["candidates"]):
            raise ValueError("Frozen candidate no longer passes analytical checks")
        tasks.append(task)
    dependencies = dict(frozen["source_sha256"])
    dependencies.update(hashes)
    dependencies[str(Path(__file__).resolve())] = sha(__file__)
    dependencies[str(snapshot/"index.json")] = sha(snapshot/"index.json")
    dependencies.update({task["config_path"]: sha(task["config_path"]) for task in tasks})
    save_new(output/"plan.json", {
        **frozen, "schema": "compact_snapshot_recovery_v1",
        "recovered_utc": datetime.now(timezone.utc).isoformat(),
        "original_plan": str(old/"plan.json"), "original_plan_sha256": sha(old/"plan.json"),
        "rejected_attempt_console_sha256": sha(log),
        "recovery_reason": "11 frozen original-workspace files changed; zero calls dispatched",
        "base_project": str(base), "source_sha256": dependencies, "tasks": tasks,
        "interpreter": sys.executable, "python_version": sys.version,
        "runtime_dependencies": problem.dependencies,
        "unchanged_scope": "same 60 candidates, mission contracts, gates, seed order and total cap300",
    })
    print(output/"plan.json", flush=True)


def execute(study):
    output = study/"qualification_recovered_20261025"
    plan = load_json((output/"plan.json").read_bytes())
    verify_sources(plan["source_sha256"])
    if sha(plan["original_plan"]) != plan["original_plan_sha256"]:
        raise ValueError("Original qualification plan changed")
    if (plan["schema"] != "compact_snapshot_recovery_v1"
            or plan["study_max_calls"] != 300 or plan["phase_max_calls"] != 60
            or study_charged(study) != plan["prior_charged_real_calls"]):
        raise ValueError("Recovery scope or physical charges differ")
    lock = study/"study_active.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(output))
    try:
        save_new(output/"execution_started.json", {"utc": datetime.now(timezone.utc).isoformat()})
        outcomes, status = [], "complete"
        for task in plan["tasks"]:
            distance = task["distance_nmi"]
            problem = X57LiveProblem(plan["base_project"], task["config_path"],
                                     output/("workers_"+str(distance)), True)
            if problem.contract_id != task["contract_id"]:
                raise ValueError("Recovered contract changed")
            n = plan["per_task_calls"]
            with RiskLedger(output/("task_"+str(distance))/"budget.jsonl",
                            BudgetPolicy(n, float(n), n, 0, 0)) as ledger:
                for raw in task["candidates"]:
                    verify_sources(plan["source_sha256"])
                    if study_charged(study) >= 300 or len(outcomes) >= 60:
                        raise RuntimeError("Frozen physical cap reached")
                    proposal = {"candidate": raw, "source": "independent_conditional_uniform",
                                "mode": "bootstrap", "evidence": {"basis": "unavailable"}}
                    append_json(output/"submissions.jsonl", {
                        "phase_call": len(outcomes)+1, "task": distance,
                        "candidate_key": digest(raw), "proposal": proposal})
                    risk = dispatch(problem, ledger, proposal)
                    if risk is None:
                        raise RuntimeError("Frozen analytical-pass candidate rejected")
                    row = {"evaluation": len(outcomes)+1, "candidate": raw, "risk": risk,
                           "distance_nmi": distance, "evaluator_contract_sha256": problem.contract_id}
                    outcomes.append(row)
                    append_json(output/"observations.jsonl", row)
                    print("Qualification", len(outcomes), distance, risk["state"], flush=True)
                    if risk["feasible_label"] is None:
                        status = "stopped_unknown"
                        break
            if status != "complete":
                break
        scores = []
        for task in plan["tasks"]:
            rows = [row for row in outcomes if row["distance_nmi"] == task["distance_nmi"]]
            count = sum(row["risk"]["feasible_label"] is True for row in rows)
            scores.append({"distance_nmi": task["distance_nmi"], "count": len(rows),
                           "feasible": count, "wilson_95": wilson(count, len(rows)) if rows else None})
        eligible = [s["distance_nmi"] for s in scores
                    if status == "complete" and s["count"] == 30 and 1 <= s["feasible"] <= 6]
        report = {"status": status, "scores": scores,
                  "selected_distance_nmi": min(eligible) if eligible else None,
                  "sparse_region_qualified": False, "phase_calls": len(outcomes),
                  "entire_study_charged": study_charged(study),
                  "selection_did_not_use_optimizer_rankings": True,
                  "recovery_plan_sha256": sha(output/"plan.json")}
        save_new(output/"result.json", report)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else execute)(args.study.resolve())
