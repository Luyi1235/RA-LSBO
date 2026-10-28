"""Complete the fixed comparisons after architecture-aware training/calibration."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch
import uuid

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import RiskLedger, digest
from ra_lsbo.mechanism_system import FullSystemLearning, METHODS, SystemSettings, mechanism_support
from ra_lsbo.optimizer import append_json
from ra_lsbo.partial_observation import enable_partial_lift_normalization
from ra_lsbo.selection import candidate_pool
from ra_lsbo.system_experiment import save_new, signal_stop
from ra_lsbo.x57_live import X57LiveProblem
from tools import run_system_validation as original
from tools.run_system_validation_v2 import collect_batch
from tools.run_system_validation_v4 import validate_plan as validate_previous
from tools.snapshot_experiment_sources import snapshot


TARGET = "mission_solver_nonconvergence"
CAP, TRAINING_CALLS, CALIBRATION_CALLS = 500, 140, 72
RADII = (.002, .01, .03, .06)
QUOTAS = {"hidden_neighborhood": 24, "outer": 12, "near": 12, "inner": 24}


def validate_plan(plan):
    if (plan.get("schema") != "ra_system_validation_v5"
            or plan.get("operator_approved") is not True
            or plan.get("complete_plan_mda_cap") != CAP
            or plan.get("probe_call_cap_including_carried") != TRAINING_CALLS
            or plan.get("fresh_calibration_calls") != CALIBRATION_CALLS
            or plan.get("calibration_quotas") != QUOTAS
            or plan.get("methods") != list(METHODS) or plan.get("seeds") != [31, 32, 33]
            or plan.get("settings") != asdict(replace(SystemSettings(), calibration_window=72))
            or plan.get("parallel_workers") != 4
            or not 80 <= plan.get("carried_mda_calls", -1) < TRAINING_CALLS
            or plan.get("remaining_mda_cap") != CAP-plan["carried_mda_calls"]
            or sum(plan["prior_call_cost_by_version"].values()) != plan["carried_mda_calls"]
            or len(plan.get("carried_probes", [])) != plan["carried_mda_calls"]
            or len(plan.get("probes", [])) != TRAINING_CALLS):
        raise ValueError("Complete 500-call, unchanged-algorithm scope differs")
    if TRAINING_CALLS+CALIBRATION_CALLS+len(METHODS)*3*plan["settings"]["calls"] != CAP:
        raise ValueError("Maximum cumulative budget differs")
    if [p["index"] for p in plan["probes"]] != list(range(1, TRAINING_CALLS+1)):
        raise ValueError("Training acquisition indices differ")
    if len({digest(p["candidate"]) for p in plan["probes"]}) != TRAINING_CALLS:
        raise ValueError("Duplicate planned training candidate")
    for index, item in enumerate(plan["carried_probes"], 1):
        if item["index"] != index or item["candidate"] != plan["probes"][index-1]["candidate"]:
            raise ValueError("Carried chronology/candidate differs")
    anchors = {r["index"]: r for r in plan["carried_probes"]
               if r["risk"]["hidden_labels"][TARGET] == 1}
    for item in plan["probes"][plan["carried_mda_calls"]:]:
        if (item.get("anchor_index") not in anchors or item.get("radius") not in RADII
                or item["candidate"]["discrete"] == anchors[item["anchor_index"]]["candidate"]["discrete"]
                or sum(item["candidate"]["discrete"][k] != v for k, v in
                       anchors[item["anchor_index"]]["candidate"]["discrete"].items()) != 1):
            raise ValueError("New training design is not a single-factor architecture neighbor")


def legal_neighbor(problem, center, discrete, radius, seen, rng):
    center_unit = problem.space.unit(problem.candidate_class(**center))
    for _ in range(10000):
        value = problem.space.from_unit(np.clip(
            center_unit+rng.uniform(-radius, radius, len(center_unit)), 0., 1.), discrete)
        raw = problem.canonical(value.to_dict())
        key = problem.space.key(problem.candidate_class(**raw))
        if key not in seen and all(np.isfinite(v) and v <= 0 for v in problem.problem.precheck(value).values()):
            seen.add(key)
            return raw
    raise ValueError("Cannot freeze a unique legal local candidate")


def freeze_architectures(problem, initial, carried, carried_items):
    seen = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in initial+carried}
    rng = np.random.default_rng(20261025)
    anchors = {r["index"]: r for r in carried if r["index"] in (5, 19)}
    configurations = problem.space.configurations()
    groups, used = [], set()
    # Change airfoil and blade count first; both preserve the active coordinate mask.
    for factor in ("wing_airfoil", "blade_count", "has_winglet", "propeller_count"):
        for index, anchor in anchors.items():
            old = anchor["candidate"]["discrete"]
            choices = [d for d in configurations if d[factor] != old[factor]
                       and all(d[k] == v for k, v in old.items() if k != factor)]
            for discrete in reversed(choices) if factor == "blade_count" else choices:
                if digest(discrete) not in used:
                    used.add(digest(discrete))
                    groups.append((index, discrete, factor))
                    break
        if len(groups) >= min(4, (TRAINING_CALLS-len(carried))//3):
            break
    count = min(4, max(1, (TRAINING_CALLS-len(carried))//3))
    groups = groups[:count]
    if not groups:
        raise ValueError("No architecture neighbor available")
    items = list(carried_items)
    for slot in range(TRAINING_CALLS-len(carried)):
        anchor_index, discrete, factor = groups[slot % len(groups)]
        anchor = anchors[anchor_index]
        radius = RADII[(slot//len(groups)) % len(RADII)]
        center = problem.space.from_unit(
            problem.space.unit(problem.candidate_class(**anchor["candidate"])), discrete).to_dict()
        candidate = legal_neighbor(problem, center, discrete, radius, seen, rng)
        items.append(dict(index=len(items)+1, candidate=candidate, anchor_index=anchor_index,
                          architecture_group=slot % len(groups), changed_factor=factor,
                          center_candidate=problem.canonical(center), radius=radius,
                          source="new_architecture_failure_stress:"+str(anchor_index)+":"+factor+":"+str(radius)))
    return items


def prepare(predecessor, output_root):
    predecessor = predecessor.resolve()
    old = load_json((predecessor/"plan.json").read_bytes())
    report = load_json((predecessor/"report.json").read_bytes())
    audit = load_json((predecessor/"independent_audit.json").read_bytes())
    validate_previous(old)
    if (report["status"] != "terminal" or audit["status"] != "passed"
            or any(r["evaluations"] for r in report["physical"])):
        raise ValueError("Require audited predecessor before any real optimizer outcomes")
    verify_sources(old["source_sha256"])
    verify_sources(audit["input_sha256"])
    carried, items = [], []
    for stage, filename in (("probe", "probe_results.jsonl"), ("calibration", "calibration_results.jsonl")):
        for prior in original.read_lines(predecessor/filename):
            if prior["risk"] is None:
                continue
            if prior["risk"]["state"] in ("unknown", "infrastructure_error"):
                raise ValueError("Unresolved predecessor response")
            row = deepcopy(prior)
            row.update(index=len(carried)+1, carried_stage=stage, carried_index=prior["index"],
                       predecessor_record_sha256=digest(prior))
            carried.append(row)
            proposal = load_json((Path(row["directory"])/"proposal.json").read_bytes())
            item = (deepcopy(old["probes"][prior["index"]-1]) if stage == "probe" else {})
            items.append(dict(item, index=row["index"], candidate=row["candidate"], source=proposal["source"]))
    if len(carried) != report["complete_new_mda_calls"] or not len(carried) < TRAINING_CALLS:
        raise ValueError("Prior cost leaves no architecture training capacity")
    folder = output_root.resolve()/("system_validation_v5_"
        +datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    config = folder/"frozen_config.json"
    config.write_bytes(Path(old["physical"]["config_path"]).read_bytes())
    initial = load_json((predecessor/"initial_history.json").read_bytes())
    save_new(folder/"initial_history.json", initial)
    problem = X57LiveProblem(old["physical"]["base_project"], config, folder/"planning", allow_mda=False)
    if problem.contract_id != old["physical"]["contract_id"]:
        raise ValueError("Physical contract differs")
    probes = freeze_architectures(problem, initial, carried, items)
    project = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), project/"tools/audit_system_validation_v5.py",
             project/"tools/verify_system_v5_request.py", project/"tools/analyze_system_validation_v5.py",
             project/"docs/SYSTEM_VALIDATION_V5_PROTOCOL_20261021.md",
             predecessor/"plan.json", predecessor/"report.json", predecessor/"independent_audit.json",
             predecessor/"physical_STOP.json", config, folder/"initial_history.json"]
    sources = dict(old["source_sha256"])
    sources.update(audit["input_sha256"])
    sources.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    plan = dict(old, schema="ra_system_validation_v5", predecessor=str(predecessor),
                complete_plan_mda_cap=CAP, carried_mda_calls=len(carried),
                remaining_mda_cap=CAP-len(carried), probe_call_cap_including_carried=TRAINING_CALLS,
                physical=dict(old["physical"], config_path=str(config)), carried_probes=carried,
                probes=probes, calibration_quotas=QUOTAS, source_sha256=sources,
                prior_call_cost_by_version=report["call_cost_by_version"],
                scope="Architecture-aware training acquisition and fresh hidden-aware calibration; "
                      "unchanged algorithms before all real optimizer outcomes; cumulative <=500 MDA")
    validate_plan(plan)
    verify_sources(sources)
    save_new(folder/"plan.json", plan)
    snapshot(folder/"plan.json")
    print("Prepared", folder, "prior", len(carried), "architecture calls", TRAINING_CALLS-len(carried),
          "maximum all-version MDA", CAP, flush=True)
    return folder


def freeze_calibration(folder, plan, problem, training):
    learner = FullSystemLearning("cbo", SystemSettings(**plan["settings"]), 0)
    models, metadata = learner.fit(training, [asdict(v) for v in problem.space.variables],
                                  plan["physical"]["base_project"],
                                  sorted(set(problem.required_constraints)-set(problem.analytical_constraints)))
    learner.features.save_checkpoint(folder/"calibration_planning_encoder.pt")
    save_new(folder/"calibration_planning_model.json", metadata)
    responses = models[1]
    seen = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in training}
    rng = np.random.default_rng(20261026)
    anchors, configurations = [], set()
    for row in training:
        if row["risk"]["hidden_labels"][TARGET] == 1:
            key = digest(row["candidate"]["discrete"])
            if key not in configurations:
                anchors.append(row)
                configurations.add(key)
    if not anchors:
        raise ValueError("No observed failure for hidden-aware held-out design")
    groups = {name: [] for name in QUOTAS}
    for slot in range(QUOTAS["hidden_neighborhood"]):
        anchor = anchors[slot % len(anchors)]
        radius = RADII[(slot//len(anchors)) % len(RADII)]
        candidate = legal_neighbor(problem, anchor["candidate"], anchor["candidate"]["discrete"],
                                   radius, seen, rng)
        groups["hidden_neighborhood"].append(dict(
            candidate=candidate, band="hidden_neighborhood", anchor_evaluation=anchor["evaluation"],
            radius=radius, source="fresh_calibration_hidden_neighborhood"))
    for _ in range(60):
        pool, _ = candidate_pool(problem.space, rng, 512,
            {digest(problem.canonical(r["candidate"])) for r in training}
            | {digest(item["candidate"]) for values in groups.values() for item in values},
            training, lambda raw: problem.problem.precheck(problem.candidate_class(**raw)))
        candidates = [p["candidate"] for p in pool]
        if not candidates:
            continue
        predicted = responses.predict(candidates)
        margins = np.min([-np.asarray(v["mean"])/np.maximum(np.asarray(v["std"]), 1e-12)
                          for name, v in predicted.items() if name.startswith("constraint:") and v is not None], axis=0)
        for i, item in enumerate(pool):
            band = "outer" if margins[i] < 0 else "near" if margins[i] < 1 else "inner"
            key = problem.space.key(problem.candidate_class(**item["candidate"]))
            if len(groups[band]) < QUOTAS[band] and key not in seen:
                seen.add(key)
                groups[band].append(dict(item, band=band, standardized_margin=float(margins[i]),
                                         source="fresh_calibration_"+band))
        if all(len(groups[k]) == QUOTAS[k] for k in QUOTAS):
            break
    if any(len(groups[k]) != QUOTAS[k] for k in QUOTAS):
        raise ValueError("Cannot fill frozen calibration quotas")
    proposals = []
    for slot in range(24):
        for band in QUOTAS:
            if slot < len(groups[band]):
                proposals.append(dict(groups[band][slot], index=len(proposals)+1))
    save_new(folder/"fresh_calibration_plan.json", dict(
        training_indices=learner.features.training_indices, training_history_id=learner.features.history_id,
        representation_snapshot=learner.features.snapshot_id, quotas=QUOTAS, proposals=proposals,
        new_calibration_labels_available=False, source_sha256=plan["source_sha256"]))
    sha = hashlib.sha256((folder/"fresh_calibration_plan.json").read_bytes()).hexdigest()
    return [dict(item, source=item["source"]+":"+sha) for item in proposals]


def initialized_pool(*args, **kwargs):
    return ProcessPoolExecutor(*args, initializer=enable_partial_lift_normalization, **kwargs)


def rows(results):
    return [dict(evaluation=r["index"], candidate=r["candidate"], risk=r["risk"])
            for r in results if r["risk"] is not None and r["risk"]["state"] != "precheck_rejected"]


def execute(folder):
    torch.set_num_threads(1)
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    verify_sources(plan["source_sha256"])
    preflight = load_json((folder/"preflight.json").read_bytes())
    if preflight["status"] != "passed" or preflight["plan_sha256"] != hashlib.sha256(
            (folder/"plan.json").read_bytes()).hexdigest():
        raise ValueError("Missing bound zero-MDA preflight")
    save_new(folder/"execution_started.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest()))
    initial = load_json((folder/"initial_history.json").read_bytes())
    stop, probes = folder/"physical_STOP.json", list(plan["carried_probes"])
    for result in probes:
        append_json(folder/"probe_results.jsonl", result)
    with initialized_pool(max_workers=4) as pool:
        for offset in range(len(probes), TRAINING_CALLS, 4):
            if stop.exists():
                break
            probes.extend(collect_batch(pool, folder/"probes", plan["physical"],
                plan["probes"][offset:min(offset+4, TRAINING_CALLS)], stop, folder/"probe_results.jsonl"))
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                             folder/"planning", allow_mda=False)
    training, excluded = original.combine_history(problem, [initial, rows(probes)])
    save_new(folder/"precalibration_training.json", training)
    calibration = []
    if not stop.exists():
        proposals = freeze_calibration(folder, plan, problem, training)
        with initialized_pool(max_workers=4) as pool:
            for offset in range(0, CALIBRATION_CALLS, 4):
                if stop.exists():
                    break
                calibration.extend(collect_batch(pool, folder/"calibration", plan["physical"],
                    proposals[offset:offset+4], stop, folder/"calibration_results.jsonl"))
    shared, duplicates = original.combine_history(problem, [initial, rows(probes), rows(calibration)])
    save_new(folder/"shared_history.json", shared)
    save_new(folder/"mechanism_data_report.json", dict(
        scope="Current physical contract; directed training/calibration counts are not population prevalence",
        training_support=mechanism_support(training), probe_support=mechanism_support(rows(probes)),
        calibration_support=mechanism_support(rows(calibration)), probe_calls_including_carried=len(rows(probes)),
        training_acquisition_calls=len(rows(probes)), fresh_calibration_calls=len(rows(calibration)),
        carried_old_calibration_as_training_calls=sum(r.get("carried_stage") == "calibration" for r in probes),
        duplicates_removed=excluded+duplicates, generalization_not_proven_by_positive_count=True,
        training_support_flags={k: v["positive"] >= 20 and v["negative"] >= 20
                                for k, v in mechanism_support(training).items()}))
    if len(rows(calibration)) != CALIBRATION_CALLS:
        signal_stop(stop, {"reason": "Fresh calibration incomplete; no optimization submissions"})
    with patch.object(original, "ProcessPoolExecutor", initialized_pool):
        physical = original.run_trials(folder, plan, "physical",
            {seed: folder/"shared_history.json" for seed in plan["seeds"]}, plan["settings"])
    ledgers = [p for root in ("probes", "calibration", "physical")
               for p in (folder/root).rglob("budget.jsonl")]
    new_calls = sum(RiskLedger(p).summary()["charged_calls"] for p in ledgers)
    total = plan["carried_mda_calls"]+new_calls
    if total > CAP:
        raise RuntimeError("Cumulative all-version MDA cap exceeded")
    verify_sources(plan["source_sha256"])
    controlled = load_json((Path(plan["controlled_source"])/"report.json").read_bytes())["controlled"]
    save_new(folder/"report.json", dict(
        schema=plan["schema"], status="terminal", physical=physical, controlled=controlled,
        controlled_source=plan["controlled_source"], carried_mda_calls=plan["carried_mda_calls"],
        new_current_mda_calls=new_calls, new_v5_mda_calls=new_calls,
        call_cost_by_version=dict(plan["prior_call_cost_by_version"], v5=new_calls),
        complete_new_mda_calls=total, maximum_complete_mda_calls=CAP,
        probe_calls_including_carried=len(rows(probes)), fresh_calibration_calls=len(rows(calibration)),
        label_contract_id=plan["label_contract_id"], partial_observation_rule=plan["partial_observation_rule"],
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print("Terminal report", folder/"report.json", "all-version MDA", total, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"runs")
    args = parser.parse_args()
    if (args.prepare is None) == (args.execute is None):
        parser.error("Choose exactly one of --prepare/--execute")
    if args.prepare:
        prepare(args.prepare, args.output)
    else:
        execute(args.execute.resolve())
