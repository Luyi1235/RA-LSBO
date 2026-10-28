"""Frozen 492-call mechanism pilot; never invoke an RCAIDE MDA evaluator."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.problems import candidate_pool
from ra_lsbo.partial_bo.records import DesignRecord, without_intervals
from ra_lsbo.partial_refinement.optimizer import RefinementSettings, RefiningLatentBO
from ra_lsbo.partial_refinement.posterior import RefinementPosterior
from ra_lsbo.soc_revision.acquisition import quantities
from ra_lsbo.soc_revision.optimizer import RevisionSettings, RiskRefiningBO
from ra_lsbo.soc_revision.posterior import RiskPosterior
from tools.run_partial_refinement_study import problem_for, TASKS
from tools.run_partial_bo_development import BASE, CallLedger, append, clean, lines, metrics, read, sha, verify_sources, write_new

OLD = ROOT/"archive/historical_research/2026-10-26/results/partial_refinement_20261026"
SEEDS = (8401, 8402, 8403)
ARMS = ("revised_full", "average_only", "probability_only", "frozen_full", "revised_no_intervals")
INITIAL, HORIZON, POOL, PANEL = 12, 12, 192, 30
MAXIMUM = len(TASKS)*len(SEEDS)*(INITIAL+len(ARMS)*HORIZON)+len(TASKS)*PANEL


def settings_for(method):
    if method not in ARMS:
        raise ValueError("Unregistered method")
    if method == "frozen_full":
        return RefinementSettings(horizon=HORIZON)
    return RevisionSettings(horizon=HORIZON, average_kernel=method != "probability_only",
        probability_exploitation=method != "average_only", use_intervals=method != "revised_no_intervals")


def optimizer_for(problem, method, seed):
    constructor = RefiningLatentBO if method == "frozen_full" else RiskRefiningBO
    return constructor(problem, BASE, seed, settings_for(method))


def prepare(folder):
    verify_sources(read(OLD/"plan.json"))
    if not read(OLD/"audit.json")["passed"] or not read(OLD/"qualification.json")["passed"]:
        raise ValueError("Frozen study and aircraft qualification must have passed")
    files = [*sorted((ROOT/"ra_lsbo").rglob("*.py")), *sorted(BASE.rglob("*.py")),
        Path(__file__).resolve(), ROOT/"tools/run_partial_bo_development.py",
        ROOT/"tools/run_partial_refinement_study.py", ROOT/"tools/qualify_partial_aircraft_envelope.py",
        ROOT/"tools/diagnose_soc_revisits.py", ROOT/"runs/soc_diagnosis_20261026/diagnosis.json",
        ROOT/"docs/SOC_REVISION_PROTOCOL_20261026.md", OLD/"plan.json", OLD/"audit.json",
        OLD/"qualification.json", *sorted((ROOT/"tests").glob("test_soc_revision*.py"))]
    bindings = {str(path): sha(path) for path in files}
    design = {}
    old_ids = {digest(event["record"]["candidate"]) for event in lines(OLD/"calls.jsonl")
               if event["event"] == "settled"}
    for number, task in enumerate(TASKS):
        problem = problem_for(task)
        panel = candidate_pool(problem, 104026+number, PANEL)[0]
        initial = {str(seed): candidate_pool(problem, seed+340000+100000*number, INITIAL)[0]
                   for seed in SEEDS}
        candidates = panel+[candidate for rows in initial.values() for candidate in rows]
        identities = [digest(candidate) for candidate in candidates]
        if len(set(identities)) != len(identities) or old_ids.intersection(identities):
            raise ValueError("Fresh panels and initial sets must be disjoint")
        design[task] = {"contract": problem.contract_id, "panel": panel, "initial": initial}
    folder.mkdir(parents=True, exist_ok=False)
    (folder/"source_snapshot").mkdir()
    for path, identity in bindings.items():
        snapshot = folder/"source_snapshot"/(identity+".source")
        if not snapshot.exists():
            snapshot.write_bytes(Path(path).read_bytes())
    write_new(folder/"plan.json", {"schema": "soc_revision_pilot_v1", "sources": bindings,
        "maximum_new_model_calls": MAXIMUM, "maximum_new_rcaide_mda_calls": 0,
        "physical_workers": 0, "seeds": SEEDS, "arms": ARMS, "tasks": design,
        "initial": INITIAL, "horizon": HORIZON, "pool": POOL, "panel": PANEL,
        "settings": {method: asdict(settings_for(method)) for method in ARMS},
        "budget_formula": "2*3*(12+5*12)+2*30=492",
        "new_independent_seed_data": True, "independent_task_selection": False,
        "paper_confirmation": False, "practical_ties": {"aircraft_140km": 40., "hierarchical_branin": .01}})
    print({"prepared": str(folder), "budget": MAXIMUM, "bindings": len(bindings)}, flush=True)


def panel_forecasts(problem, initial, panel, seed):
    forecasts = []
    for method in ARMS:
        settings = settings_for(method)
        records = [without_intervals(row) for row in initial] if method == "revised_no_intervals" else initial
        posterior = (RefinementPosterior if method == "frozen_full" else RiskPosterior)(
            problem, BASE, records, seed, settings)
        predictions = posterior.predict([row.candidate for row in panel])
        values = quantities(predictions, None, 2., .5)
        forecasts.append({"method": method, "probabilities": values["joint"].tolist(),
            "labels": [row.usable_feasible for row in panel],
            "per_constraint_probability": {name: value.tolist() for name, value in values["probabilities"].items()},
            "diagnostics": posterior.diagnostics(),
            "initial_digest": digest([row.to_dict() for row in records]),
            "panel_digest": digest([row.to_dict() for row in panel])})
    return forecasts


def summary(initial, online):
    return {**metrics(initial, online), "soc_reserve_aborts": sum(
        row.diagnostics.get("termination_reason") == "physical_soc_reserve_abort" for row in online)}


def run_trial(folder, ledger, task, method, seed, initial):
    problem = problem_for(task)
    output = folder/"optimization"/task/(method+"_"+str(seed))
    output.mkdir(parents=True, exist_ok=False)
    optimizer = optimizer_for(problem, method, seed)
    write_new(output/"manifest.json", {"task": task, "method": method, "seed": seed,
        "contract": problem.contract_id, "initial": [row.to_dict() for row in initial],
        "settings": asdict(optimizer.settings)})
    history, online, start = list(initial), [], time.monotonic()
    for iteration in range(HORIZON):
        candidates, rejected = candidate_pool(problem, seed*10000+iteration+100000000*TASKS.index(task),
            POOL, {row.observation.design_id for row in history})
        candidate, metadata = optimizer.propose(history, candidates, iteration)
        append(output/"proposals.jsonl", {"iteration": iteration, "candidate": candidate,
            "global_pool": candidates, "global_exact_rejections": rejected,
            "prefix_digest": digest([row.to_dict() for row in history]), "metadata": metadata})
        record = ledger.evaluate(problem, candidate, f"{task}/{seed}/{method}/{iteration}")
        append(output/"observations.jsonl", record.to_dict())
        history.append(record)
        online.append(record)
        optimizer.accept(record)
    result = {"task": task, "method": method, "seed": seed, **summary(initial, online),
              "seconds": time.monotonic()-start}
    write_new(output/"summary.json", result)
    print(result, flush=True)
    return result


def execute(folder):
    plan = read(folder/"plan.json")
    verify_sources(plan)
    if plan["maximum_new_model_calls"] != MAXIMUM or (folder/"calls.jsonl").exists():
        raise ValueError("Registered budget or single-pass execution violated")
    ledger, results = CallLedger(folder, MAXIMUM), []
    (folder/"common_panel").mkdir()
    for task in TASKS:
        problem = problem_for(task)
        if problem.contract_id != plan["tasks"][task]["contract"]:
            raise ValueError("Physical contract changed")
        panel = [ledger.evaluate(problem, candidate, f"{task}/panel/{index}")
                 for index, candidate in enumerate(plan["tasks"][task]["panel"])]
        for seed in SEEDS:
            initial = [ledger.evaluate(problem, candidate, f"{task}/{seed}/initial/{index}")
                       for index, candidate in enumerate(plan["tasks"][task]["initial"][str(seed)])]
            forecasts = panel_forecasts(problem, initial, panel, seed)
            write_new(folder/"common_panel"/(task+"_"+str(seed)+".json"), forecasts)
            for method in ARMS:
                results.append(run_trial(folder, ledger, task, method, seed, initial))
    write_new(folder/"results.json", {"summaries": results,
        "new_model_calls": sum(event["event"] == "started" for event in lines(ledger.path)),
        "new_rcaide_mda_calls": 0, "paper_confirmation": False})


def audit(folder):
    plan = read(folder/"plan.json")
    verify_sources(plan)
    for identity in plan["sources"].values():
        if sha(folder/"source_snapshot"/(identity+".source")) != identity:
            raise ValueError("Frozen snapshot changed")
    events = lines(folder/"calls.jsonl")
    starts = [row for row in events if row["event"] == "started"]
    settled = [row for row in events if row["event"] == "settled"]
    if len(events) != 2*MAXIMUM or len(starts) != MAXIMUM or len(settled) != MAXIMUM:
        raise ValueError("Budget or settlement mismatch")
    if len({row["tag"] for row in starts}) != MAXIMUM:
        raise ValueError("Repeated charged tag")
    lookup = {row["tag"]: row for row in settled}
    for index, start in enumerate(starts, 1):
        end = lookup[start["tag"]]
        record = DesignRecord.from_dict(end["record"])
        expected = digest([index, start["tag"], start["contract"], start["candidate"]])
        if (events[2*index-2] != start or events[2*index-1] != end or start["number"] != index
                or start["call_id"] != expected or end["call_id"] != expected
                or start["candidate"] != record.candidate
                or start["contract"] != record.observation.contract_id
                or digest(record.to_dict()) != end["record_digest"]):
            raise ValueError("Call/evidence identity or sequence changed")
    replays, panel_replays, summaries = 0, 0, []
    for task in TASKS:
        problem = problem_for(task)
        panel = [DesignRecord.from_dict(lookup[f"{task}/panel/{index}"]["record"]) for index in range(PANEL)]
        if [row.candidate for row in panel] != plan["tasks"][task]["panel"]:
            raise ValueError("Independent panel changed")
        for seed in SEEDS:
            initial = [DesignRecord.from_dict(lookup[f"{task}/{seed}/initial/{index}"]["record"])
                       for index in range(INITIAL)]
            if [row.candidate for row in initial] != plan["tasks"][task]["initial"][str(seed)]:
                raise ValueError("Initial labels/designs changed")
            if digest(panel_forecasts(problem, initial, panel, seed)) != digest(read(
                    folder/"common_panel"/(task+"_"+str(seed)+".json"))):
                raise ValueError("Common-panel posterior replay differs")
            panel_replays += len(ARMS)
            for method in ARMS:
                output = folder/"optimization"/task/(method+"_"+str(seed))
                manifest = read(output/"manifest.json")
                if digest(manifest["initial"]) != digest([row.to_dict() for row in initial]):
                    raise ValueError("Trial prefix differs from registered initial data")
                optimizer, history, online = optimizer_for(problem, method, seed), list(initial), []
                proposals, packets = lines(output/"proposals.jsonl"), lines(output/"observations.jsonl")
                if len(proposals) != HORIZON or len(packets) != HORIZON:
                    raise ValueError("Incomplete online trajectory")
                for iteration, (saved, packet) in enumerate(zip(proposals, packets)):
                    if saved["prefix_digest"] != digest([row.to_dict() for row in history]):
                        raise ValueError("History leakage or mutation")
                    pool, rejected = candidate_pool(problem, seed*10000+iteration+100000000*TASKS.index(task),
                        POOL, {row.observation.design_id for row in history})
                    candidate, metadata = optimizer.propose(history, pool, iteration)
                    if (saved["global_pool"] != pool or saved["global_exact_rejections"] != rejected
                            or candidate != saved["candidate"] or digest(clean(metadata)) != digest(saved["metadata"])):
                        raise ValueError("Candidate, mixture posterior or continuous search replay differs")
                    row = DesignRecord.from_dict(packet)
                    if digest(packet) != lookup[f"{task}/{seed}/{method}/{iteration}"]["record_digest"]:
                        raise ValueError("Online label differs from settled call")
                    history.append(row)
                    online.append(row)
                    optimizer.accept(row)
                    replays += 1
                registered = read(output/"summary.json")
                if any(registered[key] != value for key, value in summary(initial, online).items()):
                    raise ValueError("Trial summary changed")
                summaries.append(registered)
                print(f"replayed {task}/{seed}/{method}", flush=True)
    results = read(folder/"results.json")
    if digest(summaries) != digest(results["summaries"]) or results["new_model_calls"] != MAXIMUM:
        raise ValueError("Aggregate differs from audited trajectories")
    write_new(folder/"audit.json", {"passed": True, "settled_calls": len(settled),
        "proposal_replays": replays, "common_panel_model_replays": panel_replays,
        "source_bindings": len(plan["sources"]), "new_truth_calls_in_audit": 0,
        "new_rcaide_mda_calls": 0})
    print(read(folder/"audit.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "audit"))
    parser.add_argument("folder", type=Path)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    {"prepare": prepare, "execute": execute, "audit": audit}[arguments.action](arguments.folder)
