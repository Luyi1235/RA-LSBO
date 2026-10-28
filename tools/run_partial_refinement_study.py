"""Frozen grouped-kernel/local-search development study, at most 430 new calls."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import ndtr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra_lsbo.budget import digest
from ra_lsbo.mechanism_system import SystemSettings
from ra_lsbo.partial_bo.problems import HierarchicalBranin, candidate_pool
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.partial_refinement import RefinementSettings, RefiningLatentBO
from ra_lsbo.partial_refinement.posterior import RefinementPosterior
from ra_lsbo.system_experiment import OriginalLSBOWarmStart
from tools.qualify_partial_aircraft_envelope import MissionEnvelopeAircraft
from tools.run_partial_bo_development import (
    BASE, CallLedger, STOP_MESSAGE, append, clean, lines, metrics, read, sha, verify_sources, wilson, write_new)

OLD = ROOT/"archive/historical_research/2026-10-26/results/partial_latent_development_20261026"
TASKS = ("hierarchical_branin", "aircraft_140km")
SEEDS = (8201, 8202, 8203)
ARMS = ("ra_grouped_local", "ra_onehot_local", "ra_grouped_global", "ra_legacy_local", "original_lsbo")
INITIAL, HORIZON, POOL = 12, 12, 192
MAXIMUM = 2*3*5*HORIZON+3*INITIAL+30+4


def problem_for(task, step=20.):
    if task == "hierarchical_branin":
        return HierarchicalBranin(BASE)
    if task != "aircraft_140km":
        raise ValueError("Unregistered task")
    problem = MissionEnvelopeAircraft(140.)
    if step != 20.:
        parent = problem.contract_id
        problem.settings = {**problem.settings, "integration_step_s": step}
        problem.contract_id = digest({"parent": parent, "step": step, "runner_sha256": sha(__file__)})
    return problem


def settings_for(method):
    if method not in ARMS[:-1]:
        raise ValueError("Unknown adapted method")
    return RefinementSettings(horizon=HORIZON, latent=method != "ra_onehot_local",
        kernel="legacy" if method == "ra_legacy_local" else "grouped",
        local=method != "ra_grouped_global")


def prepare(folder):
    verify_sources(read(OLD/"plan.json"))
    old_truth = {event["tag"]: event for event in lines(OLD/"calls.jsonl") if event["event"] == "settled"}
    envelope = [event for event in lines(OLD/"mission_envelope/calls.jsonl") if event["event"] == "settled"]
    anchors = [event["record"] for event in envelope if event["tag"].startswith("range_140.0/")
               and DesignRecord.from_dict(event["record"]).usable_feasible]
    if len(anchors) != 2:
        raise ValueError("Expected two independently discovered aircraft anchors")
    folder.mkdir(parents=True, exist_ok=False)
    files = [*sorted((ROOT/"ra_lsbo").rglob("*.py")), *sorted(BASE.rglob("*.py")), Path(__file__).resolve(),
        ROOT/"tools/run_partial_bo_development.py", ROOT/"tools/qualify_partial_aircraft_envelope.py",
        ROOT/"tests/test_partial_refinement.py", ROOT/"tests/test_grouped_partial_gp.py",
        ROOT/"tests/test_partial_refinement_study.py",
        ROOT/"docs/PARTIAL_REFINEMENT_PROTOCOL_20261026.md", OLD/"plan.json", OLD/"calls.jsonl",
        OLD/"mission_envelope/calls.jsonl"]
    sources = {str(path): sha(path) for path in files}
    snapshot = folder/"source_snapshot"
    snapshot.mkdir()
    for path, identity in sources.items():
        destination = snapshot/(identity+".source")
        if not destination.exists():
            destination.write_bytes(Path(path).read_bytes())
    initial_branin = {str(seed): [old_truth[f"hierarchical_branin/{old_seed}/initial/{index}"]["record"]
        for index in range(INITIAL)] for seed, old_seed in zip(SEEDS, (8101, 8102, 8103))}
    aircraft = problem_for("aircraft_140km")
    write_new(folder/"plan.json", {"schema": "partial_refinement_study_v1",
        "maximum_new_model_calls": MAXIMUM, "maximum_new_rcaide_mda_calls": 0, "physical_workers": 0,
        "budget_formula": "2 tasks*3 seeds*5 arms*12 online+3*12 new aircraft initial+30 qualification+4 integration=430",
        "seeds": SEEDS, "arms": ARMS, "horizon": HORIZON, "pool": POOL, "initial": INITIAL,
        "sources": sources, "parent_study": str(OLD), "native_gp_steps": 25,
        "reused_branin_initial": initial_branin,
        "reused_branin_panel": [old_truth[f"hierarchical_branin/qualification/{index}"]["record"] for index in range(30)],
        "reused_aircraft_anchors": anchors,
        "aircraft_qualification": candidate_pool(aircraft, 94026, 30)[0],
        "aircraft_initial": {str(seed): candidate_pool(aircraft, seed+250000, INITIAL)[0] for seed in SEEDS},
        "contracts": {task: problem_for(task).contract_id for task in TASKS},
        "settings": {arm: asdict(settings_for(arm)) for arm in ARMS[:-1]},
        "aircraft_gate": "both inherited anchors remain feasible; <=6/30 independently random feasible; no physical-unknown results; integration consistency",
        "integration_gate": "20s vs5s status/constraint labels same; SOC/current/voltage margin abs difference <=0.001; power <=0.01; energy relative <=0.01",
        "qualification_interpretation": "development sparsity gate is point-estimate based; Wilson interval retained, not a certified rate",
        "selection_training": "no qualification or anchor labels; Branin prefixes reused for development, not heldout confirmation",
        "pool_seed_rule": "seed*10000+iteration+100000000*task_index; exclude own prefix only",
        "common_panel": "all methods forecast same independent panel from shared initial only, no tuning",
        "unused_budget": "never transfer skipped aircraft/native-stop calls", "execution": "single pass sequential"})


def convergence_result(index, anchor, finer):
    variants = [anchor, *finer]
    labels = [(record.observation.status, record.diagnostics.get("termination_reason"),
               tuple((name, value.constraint_satisfaction()) for name, value in record.observation.constraints))
              for record in variants]
    a, b = dict(anchor.observation.constraints), dict(finer[-1].observation.constraints)
    numeric = all(value.kind == "exact" for record in variants for _, value in record.observation.constraints) and all(
        record.observation.objective.kind == "exact" for record in variants)
    errors = {name: abs(a[name].value-b[name].value) for name in ("soc", "current", "voltage", "power")} if numeric else {}
    energy_error = abs(anchor.observation.objective.value-finer[-1].observation.objective.value)/max(
        1., abs(finer[-1].observation.objective.value)) if numeric else None
    passed = (numeric and labels[0] == labels[1] == labels[2] and energy_error <= .01
        and all(errors[name] <= (.01 if name == "power" else .001) for name in errors))
    return {"anchor": index, "passed": bool(passed), "margin_errors": errors, "energy_relative_error": energy_error}


def qualify(folder, plan, ledger):
    problem = problem_for("aircraft_140km")
    panel = [ledger.evaluate(problem, candidate, f"aircraft_140km/qualification/{index}")
             for index, candidate in enumerate(plan["aircraft_qualification"])]
    anchors = [DesignRecord.from_dict(packet) for packet in plan["reused_aircraft_anchors"]]
    convergence = []
    for index, anchor in enumerate(anchors):
        finer = [ledger.evaluate(problem_for("aircraft_140km", step), anchor.candidate,
            f"aircraft_140km/integration/{index}/{step}") for step in (10., 5.)]
        convergence.append(convergence_result(index, anchor, finer))
    feasible = sum(record.usable_feasible for record in panel)
    unknown = sum(record.physical_feasible is None for record in panel)
    result = {"random_count": 30, "random_usable_feasible": feasible, "random_physical_unknown": unknown,
        "wilson95": wilson(feasible, 30), "convergence": convergence,
        "passed": feasible <= 6 and unknown == 0 and all(record.usable_feasible for record in anchors)
                  and all(row["passed"] for row in convergence),
        "new_model_calls": 34, "certified_sparse_rate": False}
    write_new(folder/"qualification.json", result)
    print({"aircraft_qualification": result}, flush=True)
    return result, panel


def common_panel(folder, task, seed, initial, panel):
    forecasts = []
    for method in ("ra_grouped_local", "ra_onehot_local", "ra_legacy_local"):
        posterior = RefinementPosterior(problem_for(task), BASE, initial, seed, settings_for(method))
        prediction = posterior.predict([record.candidate for record in panel])
        probability = ndtr(-prediction["completion"]["mean"]/np.sqrt(1+prediction["completion"]["variance"]))
        for forecast in prediction["constraints"].values():
            probability *= ndtr(-forecast["mean"]/np.sqrt(np.maximum(forecast["variance"], 1e-24)))
        labels = [record.usable_feasible for record in panel]
        forecasts.append({"method": method, "prediction": prediction, "diagnostics": posterior.diagnostics(),
            "joint_proxy_brier": float(np.mean((probability-np.asarray(labels, float))**2)),
            "probabilities": probability.tolist(), "labels": labels,
            "interpretation": "common-panel scoring, independent-head probability proxy, not certified calibration"})
    target = folder/"common_panel"
    target.mkdir(parents=True, exist_ok=True)
    write_new(target/(task+"_"+str(seed)+".json"), {"task": task, "seed": seed, "forecasts": forecasts,
        "initial_digest": digest([record.to_dict() for record in initial]),
        "panel_digest": digest([record.to_dict() for record in panel])})


def run_trial(folder, plan, ledger, task, seed, method, initial):
    problem = problem_for(task)
    output = folder/"optimization"/task/(method+"_"+str(seed))
    output.mkdir(parents=True, exist_ok=False)
    native = OriginalLSBOWarmStart(problem, BASE, [row.legacy_row(index) for index, row in enumerate(initial)],
        seed, SystemSettings(calls=HORIZON, failure_cap=HORIZON, gp_steps=plan["native_gp_steps"]), output
        ) if method == "original_lsbo" else None
    optimizer = None if native else RefiningLatentBO(problem, BASE, seed, settings_for(method))
    write_new(output/"manifest.json", {"task": task, "seed": seed, "method": method,
        "initial": [record.to_dict() for record in initial], "contract": problem.contract_id,
        "settings": native.settings if native else asdict(optimizer.settings)})
    history, online, stopped, start = list(initial), [], False, time.monotonic()
    for iteration in range(HORIZON):
        global_pool, rejected = [], 0
        if native:
            proposal, metadata = native.propose()
            candidate = proposal["candidate"]
        else:
            global_pool, rejected = candidate_pool(problem, seed*10000+iteration+100000000*TASKS.index(task),
                POOL, {record.observation.design_id for record in history})
            candidate, metadata = optimizer.propose(history, global_pool, iteration)
        append(output/"proposals.jsonl", {"iteration": iteration, "candidate": candidate,
            "global_pool": global_pool, "global_exact_rejections": rejected, "metadata": metadata,
            "prefix_digest": digest([record.to_dict() for record in history])})
        record = ledger.evaluate(problem, candidate, f"{task}/{seed}/{method}/{iteration}")
        append(output/"observations.jsonl", record.to_dict())
        history.append(record)
        online.append(record)
        if native:
            try:
                native.accept(record.legacy_row(len(history)-1), metadata)
            except RuntimeError as error:
                if str(error) != STOP_MESSAGE:
                    raise
                stopped = True
                break
        else:
            optimizer.accept(record)
    result = {"task": task, "seed": seed, "method": method, **metrics(initial, online, stopped),
              "seconds": time.monotonic()-start}
    write_new(output/"summary.json", result)
    print(result, flush=True)
    return result


def execute(folder):
    plan = read(folder/"plan.json")
    verify_sources(plan)
    if plan["maximum_new_model_calls"] != MAXIMUM or (folder/"calls.jsonl").exists():
        raise RuntimeError("Unregistered budget or existing execution")
    ledger = CallLedger(folder, MAXIMUM)
    qualification, aircraft_panel = qualify(folder, plan, ledger)
    results = []
    for task in TASKS:
        if task == "aircraft_140km" and not qualification["passed"]:
            continue
        problem = problem_for(task)
        for seed in SEEDS:
            initial = ([DesignRecord.from_dict(packet) for packet in plan["reused_branin_initial"][str(seed)]]
                if task == "hierarchical_branin" else [ledger.evaluate(problem, candidate,
                    f"{task}/{seed}/initial/{index}") for index, candidate in enumerate(plan["aircraft_initial"][str(seed)])])
            panel = [DesignRecord.from_dict(packet) for packet in plan["reused_branin_panel"]] if task == "hierarchical_branin" else aircraft_panel
            common_panel(folder, task, seed, initial, panel)
            for method in ARMS:
                results.append(run_trial(folder, plan, ledger, task, seed, method, initial))
    write_new(folder/"results.json", {"summaries": results, "qualification": qualification,
        "new_model_calls": sum(event["event"] == "started" for event in lines(ledger.path)),
        "new_rcaide_mda_calls": 0, "confirmatory_evidence": False})


def audit(folder):
    plan = read(folder/"plan.json")
    verify_sources(plan)
    for identity in plan["sources"].values():
        if sha(folder/"source_snapshot"/(identity+".source")) != identity:
            raise ValueError("Source snapshot changed")
    events = lines(folder/"calls.jsonl")
    starts = [event for event in events if event["event"] == "started"]
    settled = [event for event in events if event["event"] == "settled"]
    if len(starts) != len(settled) or len(starts) > MAXIMUM or len({event["tag"] for event in starts}) != len(starts):
        raise ValueError("Budget, retry or settlement mismatch")
    lookup = {event["tag"]: event for event in settled}
    for start in starts:
        end = lookup[start["tag"]]
        record = DesignRecord.from_dict(end["record"])
        if (start["call_id"] != end["call_id"] or start["candidate"] != record.candidate
                or start["contract"] != record.observation.contract_id or digest(record.to_dict()) != end["record_digest"]):
            raise ValueError("Call/evidence linkage changed")
    parent = {event["tag"]: event for event in lines(OLD/"calls.jsonl") if event["event"] == "settled"}
    for seed, old_seed in zip(SEEDS, (8101, 8102, 8103)):
        for index, packet in enumerate(plan["reused_branin_initial"][str(seed)]):
            if digest(packet) != parent[f"hierarchical_branin/{old_seed}/initial/{index}"]["record_digest"]:
                raise ValueError("Historical training labels changed")
    panel = [DesignRecord.from_dict(lookup[f"aircraft_140km/qualification/{index}"]["record"]) for index in range(30)]
    qualification = read(folder/"qualification.json")
    if qualification["random_usable_feasible"] != sum(record.usable_feasible for record in panel):
        raise ValueError("Qualification labels changed")
    for index, packet in enumerate(plan["reused_aircraft_anchors"]):
        anchor = DesignRecord.from_dict(packet)
        finer = [DesignRecord.from_dict(lookup[f"aircraft_140km/integration/{index}/{step}"]["record"])
                 for step in (10., 5.)]
        if convergence_result(index, anchor, finer) != qualification["convergence"][index]:
            raise ValueError("Integration gate differs from settled evidence")
    expected_gate = (qualification["random_usable_feasible"] <= 6 and
        all(record.physical_feasible is not None for record in panel) and
        all(DesignRecord.from_dict(packet).usable_feasible for packet in plan["reused_aircraft_anchors"]) and
        all(entry["passed"] for entry in qualification["convergence"]))
    if qualification["passed"] != expected_gate:
        raise ValueError("Qualification decision changed")
    replays, summaries, local_selected = 0, [], 0
    for manifest in sorted((folder/"optimization").glob("*/*/manifest.json")):
        specification, output = read(manifest), manifest.parent
        task, seed, method = specification["task"], specification["seed"], specification["method"]
        if task == "aircraft_140km" and not qualification["passed"]:
            raise ValueError("Unqualified aircraft optimization")
        history = [DesignRecord.from_dict(packet) for packet in specification["initial"]]
        expected = plan["reused_branin_initial"][str(seed)] if task == "hierarchical_branin" else [
            lookup[f"{task}/{seed}/initial/{index}"]["record"] for index in range(INITIAL)]
        if digest(expected) != digest(specification["initial"]):
            raise ValueError("Trial initial data mismatch")
        problem = problem_for(task)
        native = OriginalLSBOWarmStart(problem, BASE, [row.legacy_row(index) for index, row in enumerate(history)],
            seed, SystemSettings(calls=HORIZON, failure_cap=HORIZON, gp_steps=plan["native_gp_steps"]),
            output/"audit_replay") if method == "original_lsbo" else None
        optimizer = None if native else RefiningLatentBO(problem, BASE, seed, settings_for(method))
        proposals, packets = lines(output/"proposals.jsonl"), lines(output/"observations.jsonl")
        if len(proposals) != len(packets):
            raise ValueError("Unsettled proposal")
        stopped = False
        for iteration, (saved, packet) in enumerate(zip(proposals, packets)):
            if saved["prefix_digest"] != digest([record.to_dict() for record in history]):
                raise ValueError("Future data leakage or history mutation")
            if native:
                proposal, metadata = native.propose()
                candidate = proposal["candidate"]
            else:
                candidates, rejected = candidate_pool(problem, seed*10000+iteration+100000000*TASKS.index(task),
                    POOL, {record.observation.design_id for record in history})
                if candidates != saved["global_pool"] or rejected != saved["global_exact_rejections"]:
                    raise ValueError("Global candidate stream changed")
                candidate, metadata = optimizer.propose(history, candidates, iteration)
                local_selected += metadata["selected_from_local_augmentation"]
            if candidate != saved["candidate"] or digest(clean(metadata)) != digest(saved["metadata"]):
                raise ValueError("Posterior, continuous search or selection replay differs")
            record = DesignRecord.from_dict(packet)
            if digest(packet) != lookup[f"{task}/{seed}/{method}/{iteration}"]["record_digest"]:
                raise ValueError("Online observation differs from settled call")
            history.append(record)
            if native:
                try:
                    native.accept(record.legacy_row(len(history)-1), metadata)
                except RuntimeError as error:
                    if str(error) != STOP_MESSAGE or iteration != len(proposals)-1:
                        raise
                    stopped = True
            else:
                optimizer.accept(record)
            replays += 1
        expected_summary = metrics(history[:INITIAL], history[INITIAL:], stopped)
        summary = read(output/"summary.json")
        if any(summary[key] != value for key, value in expected_summary.items()):
            raise ValueError("Summary differs from settled records")
        summaries.append(summary)
        print(f"replayed {task}/{seed}/{method}", flush=True)
    panel_replays = 0
    for task in TASKS:
        if task == "aircraft_140km" and not qualification["passed"]:
            continue
        for seed in SEEDS:
            initial = [DesignRecord.from_dict(packet) for packet in
                read(folder/"optimization"/task/("ra_grouped_local_"+str(seed))/"manifest.json")["initial"]]
            shared_panel = [DesignRecord.from_dict(packet) for packet in plan["reused_branin_panel"]] if task == "hierarchical_branin" else panel
            common_panel(folder/"audit_replay", task, seed, initial, shared_panel)
            file = task+"_"+str(seed)+".json"
            if digest(read(folder/"common_panel"/file)) != digest(read(folder/"audit_replay/common_panel"/file)):
                raise ValueError("Common-panel forecasts changed")
            panel_replays += 3
    registered = read(folder/"results.json")
    if (registered["new_model_calls"] != len(starts) or digest(sorted(summaries, key=lambda row: (
            row["task"], row["seed"], row["method"]))) != digest(sorted(registered["summaries"], key=lambda row: (
                row["task"], row["seed"], row["method"])))):
        raise ValueError("Study aggregate differs from trial summaries")
    result = {"passed": True, "source_bindings": len(plan["sources"]), "settled_calls": len(settled),
        "proposal_replays": replays, "common_panel_model_replays": panel_replays,
        "local_augmented_selected": local_selected, "new_truth_calls_in_audit": 0, "new_rcaide_mda_calls": 0}
    write_new(folder/"audit.json", result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "audit"))
    parser.add_argument("folder", type=Path)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    {"prepare": prepare, "execute": execute, "audit": audit}[arguments.action](arguments.folder)


if __name__ == "__main__":
    main()
