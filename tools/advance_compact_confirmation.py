"""One-shot qualification-to-confirmation workflow under the existing cap300."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import BudgetPolicy, RiskLedger, digest
from ra_lsbo.optimizer import append_json, dispatch
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem
from tools.qualify_compact_mda import conditional_candidates, study_charged, wilson


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confirmation_distance(result):
    if result["status"] != "complete" or result["phase_calls"] != 60:
        return None
    scores = result["scores"]
    if (len(scores) != 2 or {s["distance_nmi"] for s in scores} != {55, 65}
            or any(s["count"] != 30 or not 0 <= s["feasible"] <= 30 for s in scores)):
        raise ValueError("Incomplete qualification tasks")
    eligible = [s["distance_nmi"] for s in scores if 1 <= s["feasible"] <= 6]
    selected = min(eligible) if eligible else None
    if result["selected_distance_nmi"] != selected:
        raise ValueError("Qualification selection differs from frozen shortest-distance rule")
    return selected


def prepare(study):
    prior = study/"qualification_recovered_20261025"
    result = load_json((prior/"result.json").read_bytes())
    distance = confirmation_distance(result)
    if distance is None:
        raise ValueError("No passed qualification gate; confirmation forbidden")
    recovery = load_json((prior/"plan.json").read_bytes())
    verify_sources(recovery["source_sha256"])
    pilot = load_json((study/"plan.json").read_bytes())
    verify_sources(pilot["source_sha256"])
    if (study/"study_active.lock").exists() or study_charged(study) != 66:
        raise ValueError("Qualification is not settled at the expected cumulative cost66")
    if any(RiskLedger(p).summary()["unresolved_started_calls"] for p in study.rglob("budget.jsonl")):
        raise ValueError("Unresolved physical calls forbid confirmation")
    task = next(t for t in recovery["tasks"] if t["distance_nmi"] == distance)
    output = study/"confirmation_recovered_20261025"
    problem = X57LiveProblem(recovery["base_project"], task["config_path"], output/"workers", False)
    if problem.contract_id != task["contract_id"]:
        raise ValueError("Confirmation physical identity differs")
    candidates, precheck = conditional_candidates(problem, 192000+distance, 60)
    prior_keys = {digest(row["candidate"]) for path in
                  [study/"observations.jsonl", prior/"observations.jsonl"] if path.exists()
                  for row in [load_json(line) for line in path.read_bytes().splitlines()]}
    if any(digest(raw) in prior_keys for raw in candidates):
        raise ValueError("Independent confirmation contains a repeated design")
    output.mkdir(exist_ok=False)
    sources = dict(recovery["source_sha256"])
    sources.update(pilot["source_sha256"])
    sources.update({str(Path(__file__).resolve()): sha(__file__), str(prior/"result.json"): sha(prior/"result.json"),
                    task["config_path"]: sha(task["config_path"])})
    save_new(output/"plan.json", {
        "schema": "compact_recovered_confirmation_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_cap": 300, "phase_cap": 60, "prior_study_charged": 66,
        "base_project": recovery["base_project"], "config_path": task["config_path"],
        "contract_id": problem.contract_id, "distance_nmi": distance,
        "seed": 192000+distance, "candidates": candidates, "precheck_sampling": precheck,
        "rule": "60 resolved calls; >=1 feasible; 95% Wilson upper <=0.20",
        "source_sha256": sources, "new_algorithm_comparison": "held for method evidence review"})
    return output


def execute(study, output):
    plan = load_json((output/"plan.json").read_bytes())
    verify_sources(plan["source_sha256"])
    if (plan["schema"] != "compact_recovered_confirmation_v1" or plan["study_cap"] != 300
            or plan["phase_cap"] != 60 or len(plan["candidates"]) != 60
            or study_charged(study) != plan["prior_study_charged"]):
        raise ValueError("Confirmation scope or charges differ")
    lock = study/"study_active.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(output))
    try:
        save_new(output/"execution_started.json", {"utc": datetime.now(timezone.utc).isoformat()})
        problem = X57LiveProblem(plan["base_project"], plan["config_path"], output/"workers", True)
        if problem.contract_id != plan["contract_id"]:
            raise ValueError("Confirmation evaluator changed")
        outcomes, status = [], "complete"
        with RiskLedger(output/"budget.jsonl", BudgetPolicy(60, 60., 60, 0, 0)) as ledger:
            for candidate in plan["candidates"]:
                verify_sources(plan["source_sha256"])
                if study_charged(study) >= 300:
                    raise RuntimeError("Whole physical study cap reached")
                proposal = {"candidate": candidate, "mode": "bootstrap",
                            "source": "independent_conditional_uniform_confirmation",
                            "evidence": {"basis": "unavailable"}}
                append_json(output/"submissions.jsonl", {"phase_call": len(outcomes)+1, "proposal": proposal})
                risk = dispatch(problem, ledger, proposal)
                if risk is None:
                    raise RuntimeError("Frozen analytical-pass confirmation point rejected")
                row = {"evaluation": len(outcomes)+1, "candidate": candidate, "risk": risk,
                       "evaluator_contract_sha256": problem.contract_id, "distance_nmi": plan["distance_nmi"]}
                append_json(output/"observations.jsonl", row)
                outcomes.append(row)
                print("Confirmation", len(outcomes), risk["state"], flush=True)
                if risk["feasible_label"] is None:
                    status = "stopped_unknown"
                    break
        feasible = sum(r["risk"]["feasible_label"] is True for r in outcomes)
        interval = wilson(feasible, len(outcomes)) if outcomes else None
        qualified = (status == "complete" and len(outcomes) == 60
                     and feasible >= 1 and interval[1] <= .20)
        result = {"status": status, "distance_nmi": plan["distance_nmi"],
                  "count": len(outcomes), "feasible": feasible, "wilson_95": interval,
                  "sparse_region_qualified": qualified, "entire_study_charged": study_charged(study),
                  "plan_sha256": sha(output/"plan.json")}
        save_new(output/"result.json", result)
        return result
    finally:
        lock.unlink()


def watch(study):
    workflow = study/"qualification_confirmation_workflow_20261025"
    launched = load_json((workflow/"launch_request.json").read_bytes())
    if sha(__file__) != launched["source_sha256"]:
        raise ValueError("Workflow source changed after launch")
    save_new(workflow/"watch_started.json", {"utc": datetime.now(timezone.utc).isoformat()})
    prior = study/"qualification_recovered_20261025"
    deadline = time.monotonic()+12*3600
    while not (prior/"result.json").exists() or (study/"study_active.lock").exists():
        if time.monotonic() >= deadline:
            save_new(workflow/"result.json", {"status": "waiting_limit_reached", "new_calls": 0,
                                             "note": "No retry or interruption of qualification"})
            return
        time.sleep(5)
    qualification = load_json((prior/"result.json").read_bytes())
    if sha(__file__) != launched["source_sha256"]:
        raise ValueError("Workflow source changed while waiting; no physical progression")
    if confirmation_distance(qualification) is None:
        save_new(workflow/"result.json", {
            "status": "stopped_qualification_gate", "qualification": qualification,
            "new_confirmation_calls": 0, "automatic_retry": False})
        return
    output = prepare(study)
    confirmation = execute(study, output)
    save_new(workflow/"result.json", {
        "status": "confirmation_finished", "confirmation": confirmation,
        "optimization_comparison": "not_started_pending_method_evidence_review",
        "entire_study_cap": 300, "automatic_retry": False})


def launch(study):
    import os
    workflow = study/"qualification_confirmation_workflow_20261025"
    workflow.mkdir(exist_ok=False)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "watch", "--study", str(study)]
    save_new(workflow/"launch_request.json", {
        "command": command, "source_sha256": sha(__file__),
        "scope": "one-shot current-study continuation, not a scheduled automation"})
    with (workflow/"console.log").open("xb") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                 env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                      "OPENBLAS_NUM_THREADS": "1"})
    save_new(workflow/"launched.json", {"pid": child.pid, "command": command,
                                       "source_sha256": sha(__file__),
                                       "scope": "one-shot current-study continuation, not a scheduled automation"})
    print("One-shot confirmation continuation PID", child.pid, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "watch", "prepare"))
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    try:
        {"launch": launch, "watch": watch, "prepare": prepare}[args.action](args.study.resolve())
    except Exception as error:
        if args.action == "watch":
            path = args.study.resolve()/"qualification_confirmation_workflow_20261025/result.json"
            if not path.exists():
                save_new(path, {"status": "stopped_workflow_error",
                                "error": f"{type(error).__name__}: {error}", "automatic_retry": False})
        raise
