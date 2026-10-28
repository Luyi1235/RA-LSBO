"""Resume the fixed experiment under an explicit partial-observation contract."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import digest
from ra_lsbo.partial_observation import (
    RULE_ID, enable_partial_lift_normalization, normalize_with_partial_lift)
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem
from tools import run_system_validation as original
from tools import run_system_validation_v3 as previous
from tools.snapshot_experiment_sources import snapshot


OLD_VALIDATE = previous.validate_plan
TARGET = previous.TARGET
ADAPTER = Path(__file__).resolve().parents[1]/"ra_lsbo/partial_observation.py"


def validate_plan(plan):
    if (plan.get("schema") != "ra_system_validation_v4"
            or plan.get("partial_observation_rule") != RULE_ID
            or plan.get("carried_mda_calls") != 68
            or plan.get("prior_call_cost_by_version") != {"v1": 4, "v2": 48, "v3": 16}
            or plan.get("partial_annotation_indices") != [66]
            or plan.get("adapter_sha256") != hashlib.sha256(ADAPTER.read_bytes()).hexdigest()
            or plan.get("label_contract_id") != digest({
                "physical_contract": plan["physical"]["contract_id"],
                "rule": RULE_ID, "adapter_sha256": plan["adapter_sha256"]})):
        raise ValueError("Partial-observation continuation contract differs")
    compatible = dict(plan, schema="ra_system_validation_v3")
    OLD_VALIDATE(compatible)
    for result in plan["carried_probes"]:
        if result["index"] in plan["partial_annotation_indices"]:
            if (result["original_risk"]["state"] != "unknown"
                    or result["risk"]["state"] != "incomplete"
                    or result["risk"]["feasible_label"] is not None
                    or result["risk"]["evidence"]["partial_observation_proof"]["rule"] != RULE_ID):
                raise ValueError("Carried annotation changed feasibility or scope")
        elif "original_risk" in result:
            raise ValueError("Unexpected retrospective annotation")


def prepare(predecessor, output_root):
    predecessor = predecessor.resolve()
    old = load_json((predecessor/"plan.json").read_bytes())
    report = load_json((predecessor/"report.json").read_bytes())
    audit = load_json((predecessor/"independent_audit.json").read_bytes())
    OLD_VALIDATE(old)
    if (audit["status"] != "passed" or report["status"] != "terminal"
            or report["complete_new_mda_calls"] != 68 or report["fresh_calibration_calls"] != 0
            or any(r["evaluations"] for r in report["physical"])):
        raise ValueError("Require audited 68-call, pre-comparison predecessor")
    verify_sources(old["source_sha256"])
    verify_sources(audit["input_sha256"])
    carried = deepcopy(original.read_lines(predecessor/"probe_results.jsonl"))
    folder = output_root.resolve()/("system_validation_v4_"
        +datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    config = folder/"frozen_config.json"
    config.write_bytes(Path(old["physical"]["config_path"]).read_bytes())
    initial_path = folder/"initial_history.json"
    initial_path.write_bytes((predecessor/"initial_history.json").read_bytes())
    problem = X57LiveProblem(old["physical"]["base_project"], config, folder/"planning", allow_mda=False)
    if problem.contract_id != old["physical"]["contract_id"]:
        raise ValueError("Physical contract changed")
    annotated = []
    for result in carried:
        if result["risk"]["state"] in ("unknown", "infrastructure_error"):
            original_risk = deepcopy(result["risk"])
            worker = Path(original_risk["evidence"]["worker_dir"])
            converted = normalize_with_partial_lift(
                load_json((worker/"response.json").read_bytes()), problem.required_constraints,
                problem.analytical_constraints).to_dict()
            if converted["state"] != "incomplete" or not converted["objective_observed"]:
                raise ValueError("Unresolved predecessor outcome")
            converted["evidence"]["worker_dir"] = str(worker)
            result["original_risk"] = original_risk
            result["original_result_sha256"] = digest(
                original.read_lines(predecessor/"probe_results.jsonl")[result["index"]-1])
            result["risk"] = converted
            annotated.append(result["index"])
    adapter_sha = hashlib.sha256(ADAPTER.read_bytes()).hexdigest()
    project = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), ADAPTER, project/"tools/audit_system_validation_v4.py",
             project/"tools/verify_system_v4_request.py",
             project/"docs/SYSTEM_VALIDATION_V4_PROTOCOL_20261021.md",
             predecessor/"plan.json", predecessor/"report.json", predecessor/"independent_audit.json",
             predecessor/"physical_STOP.json", config, initial_path]
    sources = dict(old["source_sha256"])
    sources.update(audit["input_sha256"])
    sources.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    plan = dict(
        old, schema="ra_system_validation_v4", predecessor=str(predecessor),
        carried_mda_calls=68, remaining_mda_cap=412, carried_probes=carried,
        physical=dict(old["physical"], config_path=str(config)), source_sha256=sources,
        prior_call_cost_by_version=report["call_cost_by_version"], partial_observation_rule=RULE_ID,
        adapter_sha256=adapter_sha, partial_annotation_indices=annotated,
        label_contract_id=digest({"physical_contract": problem.contract_id, "rule": RULE_ID,
                                  "adapter_sha256": adapter_sha}),
        scope="Same unexecuted candidates and algorithms; audited objective-only partial observation")
    validate_plan(plan)
    verify_sources(sources)
    save_new(folder/"plan.json", plan)
    snapshot(folder/"plan.json")
    print("Prepared", folder, "carried", 68, "complete cap", 480, flush=True)
    return folder


def initialized_pool(*args, **kwargs):
    return ProcessPoolExecutor(*args, initializer=enable_partial_lift_normalization, **kwargs)


def resumed_trials(*args, **kwargs):
    with patch.object(original, "ProcessPoolExecutor", initialized_pool):
        return original.run_trials(*args, **kwargs)


def execute(folder):
    def write_result(path, value):
        if Path(path).name == "report.json":
            value = dict(value)
            value["new_v4_mda_calls"] = value.pop("new_v3_mda_calls")
            plan = load_json((folder/"plan.json").read_bytes())
            value["call_cost_by_version"] = dict(
                plan["prior_call_cost_by_version"], v4=value["new_v4_mda_calls"])
            value["label_contract_id"] = plan["label_contract_id"]
            value["partial_observation_rule"] = RULE_ID
        return save_new(path, value)
    with patch.object(previous, "validate_plan", validate_plan), \
            patch.object(previous, "ProcessPoolExecutor", initialized_pool), \
            patch.object(previous, "run_trials", resumed_trials), \
            patch.object(previous, "save_new", write_result):
        previous.execute(folder)


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
