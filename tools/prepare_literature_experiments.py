"""Prepare literature-led experiments without evaluating any task."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_risk_redesign_sources import retrieve

SBARCH_COMMIT = "b78705044ad45653e5e77d2f0e1dda67581d6096"
TURBOFAN_COMMIT = "2a6c82b107720e5c281908387c9c3045e3dad749"
MAIN_ARMS = ("original_lsbo", "reference_hierarchical_gp_pov", "ra_lsbo_full", "random")
ABLATION_ARMS = ("ra_without_latent", "ra_without_mechanisms",
                 "ra_without_calibration", "ra_without_boundary_budget")


@dataclass(frozen=True)
class Case:
    identifier: str
    upstream_class: str
    expected_dimension: int
    role: str
    has_hierarchy: bool
    has_hidden_constraints: bool


CASES = (
    Case("hierarchical_goldstein", "HierarchicalGoldstein", 11,
         "representation and explicit constraints", True, False),
    Case("hierarchical_alimo_edge", "HierAlimoEdge", 7,
         "hierarchy and optimum near hidden boundary", True, True),
    Case("hierarchical_mueller08", "HierMueller08", 14,
         "hierarchy and high hidden-failure frequency", True, True),
    Case("hierarchical_rosenbrock_hc", "HierarchicalRosenbrockHC", 13,
         "hierarchy, numeric and hidden constraints", True, True),
    Case("continuous_alimo_edge_control", "AlimoEdge", 2,
         "negative control without hierarchy", False, True),
)


def comparison_budget(dimensions, seeds, arms, online_calls):
    if not dimensions or any(type(value) is not int or value < 1 for value in dimensions):
        raise ValueError("Positive task dimensions required")
    if any(type(value) is not int or value < 1 for value in (seeds, arms, online_calls)):
        raise ValueError("Positive integer experiment sizes required")
    initial = sum(5*dimension*seeds for dimension in dimensions)
    online = len(dimensions)*seeds*arms*online_calls
    return {"shared_initial_labels": initial, "online_labels": online,
            "total_labels": initial+online}


def build_proposal():
    public_budget = comparison_budget([case.expected_dimension for case in CASES], 16, 8, 50)
    engine_budget = comparison_budget([15], 16, 4, 50)
    total = public_budget["total_labels"]+engine_budget["total_labels"]+192
    return {
        "schema": "literature_experiment_design_v1",
        "status": "design_only_not_registered_for_execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "excluded_primary_tasks": ["FiveModeProblem", "SparseInteriorFiveModeProblem", "old X57 tasks"],
        "public_cases": [asdict(case) for case in CASES],
        "dimension_note": "Upstream documented dimensions; runtime verification required",
        "main_arms": list(MAIN_ARMS), "public_ablation_arms": list(ABLATION_ARMS),
        "public_design": {
            "paired_seeds": list(range(9101, 9117)), "online_calls_per_arm": 50,
            "initial_labels_per_seed": "5 * verified dimension; shared and charged once",
            "sampling": "unconditional hierarchical sampling; no outcome-based initial selection",
            "latent_pretraining": {"unlabelled_design_limit": 5000, "truth_calls": 0,
                                   "information": "grammar and masks only; no old pretrained weights"},
            "ablation_control": "one component toggled, common original-space search effort"},
        "engineering_case": {
            "name": "independent_turbofan_architecture_mda", "upstream_class": "SimpleTurbofanArch",
            "physical_model": "OpenTurbofanArchitecting and OpenMDAO/PyCycle",
            "expected_dimension": 15, "constraint_count": None,
            "requirement_rule": "pinned published requirements; no optimizer-ranking-driven retuning",
            "qualification_labels": 128, "qualification_batches": [64, 64],
            "qualification_gate": {"feasible_count_each_batch": [1, 6], "unknown_outcomes": 0,
                                   "zero_feasible": "reject qualification, not proof of emptiness"},
            "independent_prediction_labels": 64,
            "prediction_panel_rule": "post-selection independent designs; no training or tuning",
            "paired_seeds": list(range(10101, 10117)), "online_calls_per_arm": 50,
            "shared_initial_labels_per_seed": 75,
            "mechanism_rule": "audited causes only; no multi-cause claim with fewer than two supported causes",
            "partial_outputs": "retain source-valid outputs only; never invent margins or signs"},
        "statistics": {
            "unit": "paired run seed, not individual adaptive calls",
            "primary": "feasibility-first ordered final incumbent under common call cap",
            "discovery": "restricted first-feasible cost; report natural zero-feasible initial strata",
            "primary_family": ["RA versus original LSBO", "RA versus reference hierarchical GP-PoV"],
            "secondary_family": "four single-component ablations; separate Holm family",
            "uncertainty": "paired task-stratified bootstrap and exact tests; fixed-task inference",
            "risk": "hidden termination, explicit violations, and infrastructure errors separated",
            "prediction": "identical training prefixes and independent common queries",
            "policy_stop": "absorbing policy termination; no fictional calls"},
        "budget_estimate": {
            "public_controlled_labels": public_budget, "engineering_comparison_mda": engine_budget,
            "engineering_qualification_mda": 128, "engineering_independent_prediction_mda": 64,
            "total_physical_mda": engine_budget["total_labels"]+192,
            "total_labels_all_tasks": total, "standing_approval_limit": 500,
            "requires_additional_budget_authorization": total > 500,
            "execution_authorized_by_this_proposal": False, "current_task_evaluations": 0,
            "note": "reference publication-scale estimate; no split execution to evade approval"},
        "readiness_gates": ["isolated upstream environment", "runtime masks and dimensions verified",
                            "generic LSBO adapter verified", "native stage evidence audited",
                            "independent qualification", "algorithms and full budget registered",
                            "explicit authorization for any complete plan above 500"],
        "source_versions": {"SBArchOpt": SBARCH_COMMIT, "OpenTurbofanArchitecting": TURBOFAN_COMMIT},
        "source_urls": source_catalog(),
    }


def source_catalog():
    sbarch = "https://raw.githubusercontent.com/jbussemaker/SBArchOpt/"+SBARCH_COMMIT
    turbofan = "https://raw.githubusercontent.com/jbussemaker/OpenTurbofanArchitecting/"+TURBOFAN_COMMIT
    return {
        "architecture_hidden_constraints_2025": "https://arxiv.org/html/2504.08721v1",
        "benchmarking_guidelines_2025": "https://researchonline.ljmu.ac.uk/id/eprint/27017/1/Benchmarking%20in%20Optimization.pdf",
        "ci_bo_benchmark_2026": "https://arxiv.org/html/2603.14434v2",
        "becbo_2024": "https://proceedings.mlr.press/v235/tian24g.html",
        "cbob_2024": "https://ojs.aaai.org/index.php/AAAI/article/view/29488",
        "sbarch_documentation": "https://sbarchopt.readthedocs.io/en/latest/test_problems/",
        "sbarch_hierarchical": sbarch+"/sb_arch_opt/problems/hierarchical.py",
        "sbarch_hidden_constraints": sbarch+"/sb_arch_opt/problems/hidden_constraints.py",
        "sbarch_turbofan": sbarch+"/sb_arch_opt/problems/turbofan_arch.py",
        "turbofan_readme": turbofan+"/README.md",
        "turbofan_dependencies": turbofan+"/setup.py",
        "turbofan_simple_problem": turbofan+"/open_turb_arch/tests/examples/simple_problem.py",
    }


def dependency_inventory():
    packages = ("sb_arch_opt", "pymoo", "open_turb_arch", "openmdao", "pycycle")
    result = {}
    for package in packages:
        specification = importlib.util.find_spec(package)
        result[package] = {"available": specification is not None,
                           "origin": None if specification is None else specification.origin}
    return result


def prepare(output, capture_sources=False):
    output.mkdir(parents=True, exist_ok=False)
    proposal = build_proposal()
    readiness = {
        "status": "not_ready_for_evaluation", "dependencies": dependency_inventory(),
        "generic_lsbo_adapter_verified": False, "engineering_task_qualified": False,
        "runtime_dimensions_verified": False, "new_truth_evaluations": 0,
        "new_physical_mda_calls": 0,
        "planner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution_entry_point": None,
    }
    for name, record in (("proposal.json", proposal), ("readiness.json", readiness)):
        with (output/name).open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
    if capture_sources:
        source_folder = output/"primary_sources"
        source_folder.mkdir()
        manifest = []
        for name, url in source_catalog().items():
            record = {"name": name, "url": url,
                      "retrieved_utc": datetime.now(timezone.utc).isoformat()}
            try:
                payload, metadata = retrieve(url)
                target = source_folder/(name+".source")
                target.write_bytes(payload)
                record.update(metadata, status="retrieved", bytes=len(payload),
                              sha256=hashlib.sha256(payload).hexdigest(), path=str(target))
            except Exception as error:
                record.update(status="unavailable", error=type(error).__name__+": "+str(error))
            manifest.append(record)
            print(name, record["status"], flush=True)
        with (source_folder/"manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
    print(output/"proposal.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--capture-sources", action="store_true")
    arguments = parser.parse_args()
    prepare(arguments.output.resolve(), arguments.capture_sources)
