"""Registered low-cost development study; never launch an RCAIDE evaluator."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_lsbo.budget import digest
from ra_lsbo.mechanism_system import SystemSettings
from ra_lsbo.partial_bo.problems import HierarchicalBranin, SimplifiedElectricAircraft, candidate_pool
from ra_lsbo.partial_bo.records import DesignRecord, without_intervals
from ra_lsbo.partial_bo.selector import PartialBOSettings, PartialLatentBO
from ra_lsbo.system_experiment import OriginalLSBOWarmStart

BASE = (ROOT/"runs/compact_mda_20261021T081006Z_20eb608d/"
        "qualification_recovered_20261025/frozen_base")
TASKS = ("hierarchical_branin", "reduced_aircraft")
ARMS = ("ra_latent", "ra_onehot", "ra_no_intervals", "original_lsbo")
SEEDS = (8101, 8102, 8103)
INITIAL, HORIZON, POOL = 12, 14, 192
MAXIMUM = 2*(2+30+3*(INITIAL+len(ARMS)*HORIZON))+4
STOP_MESSAGE = "Repeated missing/failed objective returns; stopped before wasting remaining budget"


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, sort_keys=True, indent=2, allow_nan=False)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(clean(value), sort_keys=True, allow_nan=False)+"\n")
        stream.flush()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()] if Path(path).exists() else []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def problem_for(task, step=20.):
    if task == "hierarchical_branin":
        return HierarchicalBranin(BASE)
    if task == "reduced_aircraft":
        return SimplifiedElectricAircraft(BASE, integration_step=step)
    raise ValueError("Unregistered development problem")


def arm_settings(method):
    return PartialBOSettings(horizon=HORIZON, latent=method != "ra_onehot")


def observed_history(records, method):
    return [without_intervals(record) for record in records] if method == "ra_no_intervals" else records


def verify_sources(plan):
    changed = [path for path, identity in plan["sources"].items() if sha(path) != identity]
    if changed:
        raise RuntimeError("Registered source changed: "+repr(changed))


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=False)
    sources = [*sorted((ROOT/"ra_lsbo").rglob("*.py")), *sorted(BASE.rglob("*.py")),
               Path(__file__).resolve(), ROOT/"tests/test_partial_latent_bo.py",
               ROOT/"tests/test_partial_bo_study.py",
               ROOT/"docs/PARTIAL_LATENT_BO_PROTOCOL_20261026.md"]
    identities = {str(path): sha(path) for path in sources}
    snapshot = folder/"source_snapshot"
    snapshot.mkdir()
    for path, identity in identities.items():
        destination = snapshot/(identity+".source")
        if not destination.exists():
            destination.write_bytes(Path(path).read_bytes())
    tasks = {}
    for index, task in enumerate(TASKS):
        problem = problem_for(task)
        qualification, rejected = candidate_pool(problem, 91001+index, 30)
        initial = {str(seed): candidate_pool(problem, seed+100000*index, INITIAL)[0] for seed in SEEDS}
        tasks[task] = {"contract": problem.contract_id, "settings": problem.settings,
                       "anchors": problem.anchors(), "qualification": qualification,
                       "qualification_exact_screen_rejections": rejected, "initial": initial}
    write_new(folder/"plan.json", {
        "schema": "partial_latent_bo_development_v1", "registered_utc": datetime.now(timezone.utc).isoformat(),
        "maximum_new_model_calls": MAXIMUM, "maximum_new_rcaide_mda_calls": 0,
        "physical_workers": 0, "sequential_execution": True,
        "budget_formula": "2*(2 anchors+30 qualification+3*(12 shared initial+4*14 online))+4 integration checks=476",
        "tasks": tasks, "arms": ARMS, "seeds": SEEDS, "initial": INITIAL,
        "horizon": HORIZON, "pool": POOL, "sources": identities,
        "settings": asdict(PartialBOSettings(horizon=HORIZON)),
        "native_gp_steps": 25, "native_stop": "absorbing, remaining calls not reassigned",
        "qualification_gate": "at least one prespecified usable anchor; no domain/parameter retuning",
        "integration_gate": "both aircraft anchors same terminal status and physical labels; 20s vs 5s energy relative error <=0.01",
        "sparse_claim": "descriptive random panel with Wilson interval; no qualification from selected anchors",
        "packages": {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "torch", "gpytorch", "botorch")}, "python": sys.version,
        "pool_seed_rule": "seed*10000+iteration+100000000*task_index; common rule, per-arm seen exclusion",
        "endpoint_order": "usable feasible observed objective; no-feasible kept explicit, never penalized by invented objective",
        "qualification_training": False, "anchors_training": False})


class CallLedger:
    def __init__(self, folder, maximum):
        self.path, self.maximum = folder/"calls.jsonl", maximum

    def evaluate(self, problem, candidate, tag):
        events = lines(self.path)
        started = {event["call_id"]: event for event in events if event["event"] == "started"}
        settled = {event["call_id"]: event for event in events if event["event"] == "settled"}
        if set(started) != set(settled):
            raise RuntimeError("Unsettled model call: reconcile it before another call")
        if any(event["tag"] == tag for event in started.values()):
            raise RuntimeError("Refusing an automatic retry of a registered tag")
        if len(started) >= self.maximum:
            raise RuntimeError("Registered complete-study budget exhausted")
        number = len(started)+1
        call_id = digest([number, tag, problem.contract_id, candidate])
        append(self.path, {"event": "started", "call_id": call_id, "number": number,
            "tag": tag, "contract": problem.contract_id, "candidate": candidate,
            "new_rcaide_mda": False})
        record = problem.evaluate(candidate)
        if record.candidate != candidate or record.observation.contract_id != problem.contract_id:
            raise ValueError("Evaluator returned another design or contract; leave call unsettled")
        append(self.path, {"event": "settled", "call_id": call_id, "number": number,
            "tag": tag, "record_digest": digest(record.to_dict()), "record": record.to_dict()})
        return record


def wilson(count, total):
    z, rate = 1.959963984540054, count/total
    center = (rate+z*z/(2*total))/(1+z*z/total)
    half = z*np.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/(1+z*z/total)
    return [max(0., center-half), min(1., center+half)]


def qualify(folder, plan, ledger):
    results = {}
    for task in TASKS:
        problem, specification = problem_for(task), plan["tasks"][task]
        anchors = [ledger.evaluate(problem, candidate, f"{task}/anchor/{index}")
                   for index, candidate in enumerate(specification["anchors"])]
        panel = [ledger.evaluate(problem, candidate, f"{task}/qualification/{index}")
                 for index, candidate in enumerate(specification["qualification"])]
        feasible = sum(record.usable_feasible for record in panel)
        result = {"anchor_usable_feasible": [record.usable_feasible for record in anchors],
            "random_count": len(panel), "random_usable_feasible": feasible,
            "random_usable_rate_wilson95": wilson(feasible, len(panel)),
            "random_physical_infeasible": sum(record.physical_feasible is False for record in panel),
            "random_physical_unknown": sum(record.physical_feasible is None for record in panel),
            "random_model_domain_failures": sum(bool(record.diagnostics.get("model_domain_failure")) for record in panel),
            "passed": any(record.usable_feasible for record in anchors)}
        if task == "reduced_aircraft":
            convergence = []
            for index, anchor in enumerate(anchors):
                finer = [ledger.evaluate(problem_for(task, step), specification["anchors"][index],
                         f"{task}/integration/{index}/{step}") for step in (10., 5.)]
                values = [record.observation.objective for record in [anchor, *finer]]
                numeric = all(value.kind == "exact" for value in values)
                relative = abs(values[0].value-values[-1].value)/max(1., abs(values[-1].value)) if numeric else None
                consistent = len({(record.observation.status,
                    tuple((name, value.constraint_satisfaction()) for name, value in record.observation.constraints),
                    record.diagnostics.get("termination_reason")) for record in [anchor, *finer]}) == 1
                convergence.append({"anchor": index, "objective_exact": numeric,
                    "coarse_vs_fine_relative_error": relative, "status_consistent": consistent,
                    "passed": bool(numeric and consistent and relative <= .01)})
            result["integration_convergence"] = convergence
            result["passed"] &= all(row["passed"] for row in convergence)
        results[task] = result
        print(json.dumps({"qualification": task, **result}), flush=True)
    write_new(folder/"qualification.json", results)
    return results


def metrics(initial, online, stopped=False):
    history = initial+online
    feasible = [record.observation.objective.value for record in history if record.usable_feasible]
    first = next((index+1 for index, record in enumerate(online) if record.usable_feasible), None)
    initial_best = min((record.observation.objective.value for record in initial if record.usable_feasible), default=None)
    best = min(feasible, default=None)
    return {"online_calls": len(online), "usable_feasible_online": sum(record.usable_feasible for record in online),
        "physical_infeasible_online": sum(record.physical_feasible is False for record in online),
        "physical_unknown_online": sum(record.physical_feasible is None for record in online),
        "hidden_failure_online": sum(record.observation.status == "hidden_failure" for record in online),
        "numeric_failure_online": sum(bool(record.diagnostics.get("numerical_failure")) for record in online),
        "partial_objective_online": sum(record.observation.objective.kind != "exact" for record in online),
        "initial_best": initial_best, "endpoint_best": best, "first_online_feasible": first,
        "policy_termination_absorbing": stopped,
        "relative_improvement": (initial_best-best)/abs(initial_best) if initial_best and best is not None else None}


def run_trial(folder, plan, ledger, task, seed, method, initial):
    problem = problem_for(task)
    output = folder/"optimization"/task/(method+"_"+str(seed))
    output.mkdir(parents=True, exist_ok=False)
    native = OriginalLSBOWarmStart(problem, BASE,
        [record.legacy_row(index) for index, record in enumerate(initial)], seed,
        SystemSettings(calls=HORIZON, failure_cap=HORIZON, gp_steps=plan["native_gp_steps"]), output
        ) if method == "original_lsbo" else None
    selector = None if native else PartialLatentBO(problem, BASE, seed, arm_settings(method))
    write_new(output/"manifest.json", {"task": task, "seed": seed, "method": method,
        "initial": [record.to_dict() for record in initial], "contract": problem.contract_id,
        "settings": native.settings if native else asdict(selector.settings)})
    history, online, stopped = list(initial), [], False
    start = time.monotonic()
    for iteration in range(HORIZON):
        pool, rejections = [], 0
        if native:
            proposal, metadata = native.propose()
            candidate = proposal["candidate"]
        else:
            pool, rejections = candidate_pool(problem, seed*10000+iteration+100000000*TASKS.index(task),
                POOL, {record.observation.design_id for record in history})
            candidate, metadata = selector.propose(observed_history(history, method), pool, iteration)
        append(output/"proposals.jsonl", {"iteration": iteration, "candidate": candidate, "pool": pool,
            "exact_screen_rejections": rejections, "metadata": metadata,
            "truth_prefix_digest": digest([record.to_dict() for record in history])})
        record = ledger.evaluate(problem, candidate, f"{task}/{seed}/{method}/{iteration}")
        append(output/"observations.jsonl", record.to_dict())
        online.append(record)
        history.append(record)
        if native:
            try:
                native.accept(record.legacy_row(len(history)-1), metadata)
            except RuntimeError as error:
                if str(error) != STOP_MESSAGE:
                    raise
                stopped = True
                break
        else:
            selector.accept(record)
    summary = {"task": task, "seed": seed, "method": method, **metrics(initial, online, stopped),
               "seconds": time.monotonic()-start}
    write_new(output/"summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def execute(folder):
    plan = read(folder/"plan.json")
    verify_sources(plan)
    if (folder/"calls.jsonl").exists():
        raise RuntimeError("Execution is single-pass; use audit for an existing ledger")
    ledger = CallLedger(folder, plan["maximum_new_model_calls"])
    qualification = qualify(folder, plan, ledger)
    summaries = []
    for task in TASKS:
        if not qualification[task]["passed"]:
            continue
        problem = problem_for(task)
        for seed in SEEDS:
            initial = [ledger.evaluate(problem, candidate, f"{task}/{seed}/initial/{index}") for index, candidate
                       in enumerate(plan["tasks"][task]["initial"][str(seed)])]
            for method in ARMS:
                summaries.append(run_trial(folder, plan, ledger, task, seed, method, initial))
    write_new(folder/"results.json", {"qualification": qualification, "summaries": summaries,
        "new_model_calls": sum(event["event"] == "started" for event in lines(ledger.path)),
        "new_rcaide_mda_calls": 0, "confirmatory_evidence": False})


def audit(folder):
    plan = read(folder/"plan.json")
    verify_sources(plan)
    events = lines(folder/"calls.jsonl")
    starts = [event for event in events if event["event"] == "started"]
    settled = [event for event in events if event["event"] == "settled"]
    if len(starts) > MAXIMUM or len(starts) != len(settled):
        raise ValueError("Budget or settlement mismatch")
    if len({event["tag"] for event in starts}) != len(starts):
        raise ValueError("Repeated call tags")
    lookup = {event["tag"]: event for event in settled}
    for started in starts:
        result = lookup[started["tag"]]
        record = DesignRecord.from_dict(result["record"])
        if (started["call_id"] != result["call_id"] or record.candidate != started["candidate"]
                or record.observation.contract_id != started["contract"]
                or digest(record.to_dict()) != result["record_digest"]):
            raise ValueError("Call/evidence linkage mismatch")
    replays = 0
    for manifest in sorted((folder/"optimization").glob("*/*/manifest.json")):
        spec, output = read(manifest), manifest.parent
        task, seed, method = spec["task"], spec["seed"], spec["method"]
        problem = problem_for(task)
        history = [DesignRecord.from_dict(record) for record in spec["initial"]]
        for index, record in enumerate(history):
            if digest(record.to_dict()) != lookup[f"{task}/{seed}/initial/{index}"]["record_digest"]:
                raise ValueError("Initial truth differs from ledger")
            if record.candidate != plan["tasks"][task]["initial"][str(seed)][index]:
                raise ValueError("Initial design differs from registration")
        native = OriginalLSBOWarmStart(problem, BASE,
            [record.legacy_row(index) for index, record in enumerate(history)], seed,
            SystemSettings(calls=HORIZON, failure_cap=HORIZON, gp_steps=plan["native_gp_steps"]),
            output/"audit_replay") if method == "original_lsbo" else None
        selector = None if native else PartialLatentBO(problem, BASE, seed, arm_settings(method))
        proposals = lines(output/"proposals.jsonl")
        observations = lines(output/"observations.jsonl")
        if len(proposals) != len(observations):
            raise ValueError("Pending proposal without settled truth")
        for iteration, (saved, packet) in enumerate(zip(proposals, observations)):
            if saved["truth_prefix_digest"] != digest([record.to_dict() for record in history]):
                raise ValueError("Training prefix leakage or mutation")
            if native:
                proposal, metadata = native.propose()
                candidate = proposal["candidate"]
            else:
                pool, rejected = candidate_pool(problem,
                    seed*10000+iteration+100000000*TASKS.index(task), POOL,
                    {record.observation.design_id for record in history})
                if pool != saved["pool"] or rejected != saved["exact_screen_rejections"]:
                    raise ValueError("Pool generation changed")
                candidate, metadata = selector.propose(observed_history(history, method), pool, iteration)
            if candidate != saved["candidate"] or digest(clean(metadata)) != digest(saved["metadata"]):
                raise ValueError("Proposal/posterior replay mismatch")
            record = DesignRecord.from_dict(packet)
            if digest(packet) != lookup[f"{task}/{seed}/{method}/{iteration}"]["record_digest"]:
                raise ValueError("Online truth differs from ledger")
            history.append(record)
            if native:
                try:
                    native.accept(record.legacy_row(len(history)-1), metadata)
                except RuntimeError as error:
                    if str(error) != STOP_MESSAGE or iteration != len(proposals)-1:
                        raise
            else:
                selector.accept(record)
            replays += 1
        expected = metrics(history[:INITIAL], history[INITIAL:],
                           read(output/"summary.json")["policy_termination_absorbing"])
        if any(read(output/"summary.json")[key] != value for key, value in expected.items()):
            raise ValueError("Endpoint summary differs from settled records")
        print(f"replayed {task}/{method}/{seed}", flush=True)
    result = {"passed": True, "source_bindings": len(plan["sources"]),
        "started_calls": len(starts), "settled_calls": len(settled), "proposal_replays": replays,
        "new_model_calls_during_audit": 0, "new_rcaide_mda_calls": 0}
    write_new(folder/"audit.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "audit"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "prepare":
        prepare(args.folder)
    elif args.action == "execute":
        execute(args.folder)
    else:
        audit(args.folder)


if __name__ == "__main__":
    main()
