"""Enrich observed physical failure neighborhoods before untouched comparisons."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import uuid

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import RiskLedger, digest
from ra_lsbo.mechanism_system import METHODS, SystemSettings, mechanism_support
from ra_lsbo.optimizer import append_json
from ra_lsbo.system_experiment import save_new, signal_stop
from ra_lsbo.x57_live import X57LiveProblem
from tools.run_system_validation import combine_history, read_lines, run_trials
from tools.run_system_validation_v2 import collect_batch, freeze_calibration
from tools.snapshot_experiment_sources import snapshot


RADII = (.002, .01, .03, .06)
TARGET = "mission_solver_nonconvergence"


def validate_plan(plan):
    if (plan.get("schema") != "ra_system_validation_v3" or plan.get("operator_approved") is not True
            or plan.get("complete_plan_mda_cap") != 480
            or plan.get("probe_call_cap_including_carried") != 120
            or plan.get("fresh_calibration_calls") != 72
            or plan.get("methods") != list(METHODS) or plan.get("seeds") != [31, 32, 33]
            or plan.get("settings") != asdict(replace(SystemSettings(), calibration_window=72))
            or plan.get("parallel_workers") != 4 or plan.get("target_mode") != TARGET
            or plan.get("target_positive") != 20 or plan.get("target_negative") != 20
            or plan.get("neighborhood_radii") != list(RADII)
            or type(plan.get("carried_mda_calls")) is not int
            or not 4 <= plan["carried_mda_calls"] <= 120
            or plan.get("remaining_mda_cap") != 480-plan["carried_mda_calls"]
            or len(plan.get("carried_probes", [])) != plan["carried_mda_calls"]
            or len(plan.get("probes", [])) != 120):
        raise ValueError("Scope differs from complete 480-call targeted-data amendment")
    if [p["index"] for p in plan["probes"]] != list(range(1, 121)):
        raise ValueError("Probe indices differ")
    if len({digest(p["candidate"]) for p in plan["probes"]}) != 120:
        raise ValueError("Duplicated diagnostic candidate")
    if [p["index"] for p in plan["carried_probes"]] != list(range(1, plan["carried_mda_calls"]+1)):
        raise ValueError("Carried diagnostic chronology differs")
    if any(p["candidate"] != plan["probes"][p["index"]-1]["candidate"] for p in plan["carried_probes"]):
        raise ValueError("Carried candidate differs")
    anchors = {p["index"]: p for p in plan["carried_probes"]
               if p["risk"]["hidden_labels"][TARGET] == 1}
    if not anchors:
        raise ValueError("No observed current-contract mission failure anchor")
    for item in plan["probes"][plan["carried_mda_calls"]:]:
        if item.get("anchor_index") not in anchors or item.get("radius") not in RADII:
            raise ValueError("Neighborhood anchor/radius differs")
        anchor = anchors[item["anchor_index"]]
        if (item["candidate"]["discrete"] != anchor["candidate"]["discrete"]
                or item.get("source") != "verified_failure_neighborhood:"
                +str(item["anchor_index"])+":"+str(item["radius"])):
            raise ValueError("Neighborhood source/configuration differs")


def freeze_neighborhoods(problem, initial, carried, previous_probes):
    probes = [previous_probes[r["index"]-1] for r in carried]
    anchors = [r for r in carried if r["risk"]["hidden_labels"][TARGET] == 1]
    if not anchors:
        raise ValueError("Observed positive required before targeted continuation")
    seen = {problem.space.key(problem.candidate_class(**r["candidate"])) for r in initial+carried}
    rng = np.random.default_rng(20261023)
    for attempt in range(50000):
        if len(probes) == 120:
            break
        slot = len(probes)-len(carried)
        anchor = anchors[slot % len(anchors)]
        radius = RADII[(slot//len(anchors)) % len(RADII)]
        center = problem.candidate_class(**anchor["candidate"])
        unit = np.clip(problem.space.unit(center)+rng.uniform(-radius, radius, len(problem.space.continuous)),
                       0., 1.)
        candidate = problem.space.from_unit(unit, center.discrete)
        raw = problem.canonical(candidate.to_dict())
        key = problem.space.key(candidate)
        if key in seen or any(v > 0 for v in problem.problem.precheck(candidate).values()):
            continue
        seen.add(key)
        probes.append({"index": len(probes)+1, "candidate": raw, "anchor_index": anchor["index"],
                       "radius": radius, "source": "verified_failure_neighborhood:"
                       +str(anchor["index"])+":"+str(radius)})
    if len(probes) != 120:
        raise ValueError("Unable to freeze all legal neighborhood candidates")
    return probes


def prepare(predecessor, output_root):
    predecessor = predecessor.resolve()
    previous = load_json((predecessor/"plan.json").read_bytes())
    report = load_json((predecessor/"report.json").read_bytes())
    audit = load_json((predecessor/"independent_audit.json").read_bytes())
    if (report["status"] != "terminal" or audit["status"] != "passed"
            or report["fresh_calibration_calls"] != 0
            or any(r["evaluations"] for r in report["physical"])):
        raise ValueError("Require audited pre-optimization/pre-calibration predecessor")
    verify_sources(previous["source_sha256"])
    verify_sources(audit["input_sha256"])
    carried = [r for r in read_lines(predecessor/"probe_results.jsonl") if r["risk"] is not None]
    if (any(r["risk"]["state"] in ("unknown", "infrastructure_error") for r in carried)
            or audit["complete_new_mda_calls"] != len(carried)):
        raise ValueError("Unresolved outcomes or unexpected prior physical costs")
    folder = output_root.resolve()/("system_validation_v3_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                     +"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    config = folder/"frozen_config.json"
    config.write_bytes(Path(previous["physical"]["config_path"]).read_bytes())
    initial = load_json((predecessor/"initial_history.json").read_bytes())
    save_new(folder/"initial_history.json", initial)
    problem = X57LiveProblem(previous["physical"]["base_project"], config, folder/"planning", allow_mda=False)
    if problem.contract_id != previous["physical"]["contract_id"]:
        raise ValueError("Physical contract changed")
    probes = freeze_neighborhoods(problem, initial, carried, previous["probes"])
    sources = dict(previous["source_sha256"])
    sources.update(audit["input_sha256"])
    project = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), project/"tools/audit_system_validation_v3.py",
             project/"docs/SYSTEM_VALIDATION_V3_PROTOCOL_20261021.md",
             predecessor/"plan.json", predecessor/"report.json", predecessor/"independent_audit.json",
             predecessor/"physical_STOP.json", config, folder/"initial_history.json"]
    sources.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    plan = {
        "schema": "ra_system_validation_v3", "operator_approved": True, "complete_plan_mda_cap": 480,
        "carried_mda_calls": len(carried), "remaining_mda_cap": 480-len(carried),
        "probe_call_cap_including_carried": 120, "fresh_calibration_calls": 72,
        "methods": list(METHODS), "seeds": [31, 32, 33], "parallel_workers": 4,
        "settings": asdict(replace(SystemSettings(), calibration_window=72)),
        "target_mode": TARGET, "target_positive": 20, "target_negative": 20,
        "neighborhood_radii": list(RADII),
        "predecessor": str(predecessor), "controlled_source": report["controlled_source"],
        "physical": dict(previous["physical"], config_path=str(config)),
        "probes": probes, "carried_probes": carried, "source_sha256": sources,
        "prior_call_cost_by_version": {"v1": report["carried_mda_calls"], "v2": report["new_v2_mda_calls"]},
        "scope": "Targeted training enrichment before any real optimizer outcomes; no efficacy tuning"}
    validate_plan(plan)
    verify_sources(sources)
    save_new(folder/"plan.json", plan)
    snapshot(folder/"plan.json")
    print("Prepared", folder, "carried", len(carried), "complete cap", 480, flush=True)
    return folder


def execute(folder):
    torch.set_num_threads(1)
    folder = folder.resolve()
    plan = load_json((folder/"plan.json").read_bytes())
    validate_plan(plan)
    verify_sources(plan["source_sha256"])
    save_new(folder/"execution_started.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": hashlib.sha256((folder/"plan.json").read_bytes()).hexdigest()})
    initial = load_json((folder/"initial_history.json").read_bytes())
    stop, probes = folder/"physical_STOP.json", list(plan["carried_probes"])
    for result in probes:
        append_json(folder/"probe_results.jsonl", result)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for offset in range(len(probes), 120, 4):
            support = mechanism_support(initial+[r for r in probes if r["risk"] is not None])[TARGET]
            if support["positive"] >= 20 and support["negative"] >= 20:
                save_new(folder/"probe_support_target_stop.json", {
                    "completed_probes": len([r for r in probes if r["risk"] is not None]),
                    "support": support, "not_a_generalization_guarantee": True})
                break
            if stop.exists():
                break
            probes.extend(collect_batch(pool, folder/"probes", plan["physical"],
                                        plan["probes"][offset:offset+4], stop, folder/"probe_results.jsonl"))
    problem = X57LiveProblem(plan["physical"]["base_project"], plan["physical"]["config_path"],
                             folder/"planning", allow_mda=False)
    probe_rows = [{"evaluation": r["index"], "candidate": r["candidate"], "risk": r["risk"]}
                  for r in probes if r["risk"] is not None and r["risk"]["state"] != "precheck_rejected"]
    training, excluded = combine_history(problem, [initial, probe_rows])
    save_new(folder/"precalibration_training.json", training)
    calibration = []
    if not stop.exists():
        proposals = freeze_calibration(folder, plan, problem, training)
        with ProcessPoolExecutor(max_workers=4) as pool:
            for offset in range(0, 72, 4):
                if stop.exists():
                    break
                calibration.extend(collect_batch(
                    pool, folder/"calibration", plan["physical"], proposals[offset:offset+4],
                    stop, folder/"calibration_results.jsonl"))
    cal_rows = [{"evaluation": r["index"], "candidate": r["candidate"], "risk": r["risk"]}
                for r in calibration if r["risk"] is not None and r["risk"]["state"] != "precheck_rejected"]
    shared, duplicates = combine_history(problem, [initial, probe_rows, cal_rows])
    save_new(folder/"shared_history.json", shared)
    save_new(folder/"mechanism_data_report.json", {
        "scope": "Current physical contract; targeted training prevalence is not population prevalence",
        "training_support": mechanism_support(training), "probe_support": mechanism_support(probe_rows),
        "calibration_support": mechanism_support(cal_rows),
        "probe_calls_including_carried": len(probe_rows), "fresh_calibration_calls": len(cal_rows),
        "duplicates_removed": excluded+duplicates,
        "training_support_flags": {name: c["positive"] >= 20 and c["negative"] >= 20
                                   for name, c in mechanism_support(training).items()},
        "neighborhood_anchors": sorted({p["anchor_index"] for p in plan["probes"] if "anchor_index" in p}),
        "generalization_not_proven_by_positive_count": True})
    if len(cal_rows) != 72:
        signal_stop(stop, {"reason": "Fresh calibration incomplete; no optimization submissions"})
    physical = run_trials(folder, plan, "physical",
                          {seed: folder/"shared_history.json" for seed in plan["seeds"]}, plan["settings"])
    ledgers = [*sorted((folder/"probes").rglob("budget.jsonl")),
               *sorted((folder/"calibration").rglob("budget.jsonl")),
               *sorted((folder/"physical").rglob("budget.jsonl"))]
    new_calls = sum(RiskLedger(path).summary()["charged_calls"] for path in ledgers)
    total = plan["carried_mda_calls"]+new_calls
    if total > 480:
        raise RuntimeError("Complete all-version budget exceeded")
    verify_sources(plan["source_sha256"])
    controlled = load_json((Path(plan["controlled_source"])/"report.json").read_bytes())["controlled"]
    save_new(folder/"report.json", {
        "schema": plan["schema"], "status": "terminal", "physical": physical,
        "controlled_source": plan["controlled_source"], "controlled": controlled,
        "carried_mda_calls": plan["carried_mda_calls"], "new_current_mda_calls": new_calls,
        "new_v3_mda_calls": new_calls,
        "call_cost_by_version": dict(plan["prior_call_cost_by_version"], v3=new_calls),
        "complete_new_mda_calls": total, "maximum_complete_mda_calls": 480,
        "probe_calls_including_carried": len(probe_rows), "fresh_calibration_calls": len(cal_rows),
        "completed_utc": datetime.now(timezone.utc).isoformat()})
    print("Terminal report", folder/"report.json", "complete all-version MDA", total, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"runs")
    args = parser.parse_args()
    if (args.prepare is None) == (args.execute is None):
        parser.error("Choose exactly one of --prepare/--execute")
    if args.prepare is not None:
        prepare(args.prepare, args.output)
    else:
        execute(args.execute)
