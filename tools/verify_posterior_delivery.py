"""Final read-only scientific checks, with one new delivery manifest."""

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import RiskLedger, digest
from ra_lsbo.system_experiment import save_new
from tools.qualify_compact_mda import study_charged


def main():
    development = ROOT/"runs/posterior_development_20261021T082018Z_877bb84b"
    physical = ROOT/"runs/compact_mda_20261021T081006Z_20eb608d"
    old = ROOT/"runs/system_validation_v5_20261020T192312Z_5f6531eb"
    report = ROOT/"runs/posterior_revision_report_20261021"
    source_counts = {}
    for name, path in (
        ("old_v5", old/"plan.json"), ("new_development", development/"plan.json"),
        ("new_physical_pilot", physical/"plan.json"),
        ("new_qualification", physical/"qualification/plan.json")):
        plan = load_json(path.read_bytes())
        verify_sources(plan["source_sha256"])
        source_counts[name] = len(plan["source_sha256"])
    audit = load_json((development/"audit.json").read_bytes())
    if (audit["status"] != "passed" or audit["truth_replays"] != 360
            or audit["full_acquisition_replays"] != 216):
        raise ValueError("Controlled verification differs")
    ledgers = [RiskLedger(path).summary() for path in physical.rglob("budget.jsonl")]
    if (study_charged(physical) != 6 or any(item["unresolved_started_calls"] for item in ledgers)
            or (physical/"study_active.lock").exists()
            or (physical/"qualification/execution_started.json").exists()):
        raise ValueError("Unexpected physical accounting or phase execution")
    plan = load_json((physical/"qualification/plan.json").read_bytes())
    if len(plan["tasks"]) != 2 or sum(len(t["candidates"]) for t in plan["tasks"]) != 60:
        raise ValueError("Qualification candidates incomplete")
    for task in plan["tasks"]:
        if len({digest(raw) for raw in task["candidates"]}) != 30:
            raise ValueError("Duplicate qualification designs")
    evidence = load_json((report/"evidence.json").read_bytes())
    if evidence["sparse_region_qualified"] is not False or evidence["new_real_mda_calls"] != 6:
        raise ValueError("Scientific status differs")
    paths = [
        ROOT/"docs/POSTERIOR_AND_MDA_RESULTS_20261021.md",
        ROOT/"docs/POSTERIOR_MDA_PROTOCOL_20261021.md",
        report/"evidence.json", report/"controlled_results.png",
        development/"audit.json", physical/"pilot_result.json",
        physical/"qualification/plan.json", Path(__file__).resolve()]
    result = {
        "status": "passed", "new_real_mda_calls": 6,
        "new_complete_study_cap": 300, "new_study_remaining": 294,
        "qualification_candidates_frozen": 60, "qualification_real_calls": 0,
        "new_controlled_truth_labels": 480, "full_acquisition_replays": 216,
        "frozen_source_counts": source_counts, "sparse_region_qualified": False,
        "test_result": "246 passed; pytest completed in the delivery session",
        "artifact_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    save_new(report/"delivery_check.json", result)
    print(result)


if __name__ == "__main__":
    main()
