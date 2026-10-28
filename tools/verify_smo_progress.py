"""Read-only evidence checks plus a dated snapshot of ongoing physical work."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo.audit import load_json
from ra_lsbo.boundary_validation import verify_sources
from ra_lsbo.budget import RiskLedger
from ra_lsbo.system_experiment import save_new
from ra_lsbo.x57_live import X57LiveProblem
from tools.qualify_compact_mda import study_charged


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(output, test_log):
    test_text = test_log.read_text(encoding="utf-8")
    matches = re.findall(r"\b(\d+) passed\b", test_text)
    if not matches or int(matches[-1]) < 269 or re.search(r"\b\d+ (?:failed|errors?)\b", test_text):
        raise ValueError("Archived full-suite test result is missing or failed")
    passed = int(matches[-1])
    physical = ROOT/"runs/compact_mda_20261021T081006Z_20eb608d"
    phase = physical/"qualification_recovered_20261025"
    development = ROOT/"runs/discovery_policy_20261025"
    joint = ROOT/"runs/joint_constraints_20261025"
    old = ROOT/"runs/system_validation_v5_20261020T192312Z_5f6531eb"
    frozen_counts = {}
    for name, path in (("pilot", physical/"plan.json"), ("qualification", physical/"qualification/plan.json"),
                       ("recovery", phase/"plan.json"), ("discovery", development/"plan.json"),
                       ("joint", joint/"plan.json")):
        plan = load_json(path.read_bytes())
        verify_sources(plan["source_sha256"])
        frozen_counts[name] = len(plan["source_sha256"])
    original = load_json((physical/"qualification/plan.json").read_bytes())
    recovery = load_json((phase/"plan.json").read_bytes())
    if recovery["tasks"] != original["tasks"]:
        raise ValueError("Recovered qualification points, order or contracts differ")
    for task in recovery["tasks"]:
        problem = X57LiveProblem(recovery["base_project"], task["config_path"], phase/"unused", False)
        if problem.contract_id != task["contract_id"]:
            raise ValueError("Recovered physical contract no longer matches")
    audit = load_json((development/"audit.json").read_bytes())
    if (audit["status"] != "passed" or audit["new_truth_labels"] != 480
            or audit["online_truth_checks"] != 360 or audit["full_acquisition_checks"] != 240
            or audit["native_lsbo_proposal_checks"] != 120):
        raise ValueError("Development truth or full-proposal replay incomplete")
    if len((development/"completed.jsonl").read_bytes().splitlines()) != 60:
        raise ValueError("Missing planned development runs")
    result = load_json((joint/"result.json").read_bytes())
    if (result["status"] != "complete" or result["unique_holdout_count"] != 262
            or result["new_mda_calls"] != 0
            or any(m["checkpoint_roundtrip"] != "passed" for m in result["models"])):
        raise ValueError("Joint constraint diagnostic differs")
    old_plan = load_json((old/"plan.json").read_bytes())
    drift = [{"path": path, "frozen_sha256": expected,
              "current_sha256": sha(path) if Path(path).exists() else None}
             for path, expected in old_plan["source_sha256"].items()
             if not Path(path).exists() or sha(path) != expected]
    observations = [load_json(line) for line in (phase/"observations.jsonl").read_bytes().splitlines()]
    ledgers = [RiskLedger(path).summary() for path in physical.rglob("budget.jsonl")]
    charged = study_charged(physical)
    if charged > 300:
        raise ValueError("Entire physical study budget exceeded")
    terminal = load_json((phase/"result.json").read_bytes()) if (phase/"result.json").exists() else None
    active = (physical/"study_active.lock").exists()
    phase_status = (terminal["status"] if terminal else "in_progress" if active
                    else "interrupted_without_terminal_result")
    files = [
        ROOT/"docs/SMO_RESEARCH_ALIGNMENT_20261025.md", ROOT/"docs/SMO_PROGRESS_20261025.md",
        development/"plan.json", development/"audit.json", joint/"plan.json", joint/"result.json",
        ROOT/"runs/smo_evidence_20261025/evidence.json",
        ROOT/"runs/smo_evidence_20261025/primary_sources/sources.json",
        ROOT/"runs/smo_evidence_20261025/figures/manifest.json",
        physical/"qualification/plan.json", phase/"plan.json", Path(__file__).resolve(),
        test_log, ROOT/"tools/advance_compact_confirmation.py",
    ]
    workflow = physical/"qualification_confirmation_workflow_20261025"
    workflow_result = load_json((workflow/"result.json").read_bytes()) if (workflow/"result.json").exists() else None
    workflow_started = (workflow/"watch_started.json").exists()
    for name in ("launch_request.json", "launched.json", "watch_started.json"):
        if (workflow/name).exists():
            files.append(workflow/name)
    save_new(output, {
        "schema": "smo_progress_delivery_snapshot_v1", "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "software_evidence_checks": "passed", "submission_ready": False,
        "tests": str(passed)+" passed; full pytest output archived",
        "python": {"executable": sys.executable, "version": sys.version},
        "runtime_versions": {name: importlib.metadata.version(name) for name in
                             ("numpy", "scipy", "torch", "botorch", "gpytorch", "scikit-learn")},
        "frozen_source_counts": frozen_counts,
        "original_workspace_source_drift": drift,
        "new_controlled_truth_labels": 480, "controlled_runs": 60,
        "full_acquisition_replays": 240, "native_proposal_replays": 120,
        "joint_gp_historical_holdout_unique": 262,
        "physical_qualification_status": phase_status,
        "physical_qualification_results_at_snapshot": len(observations),
        "physical_qualification_budgeted_calls": 60,
        "physical_study_charged_at_snapshot": charged,
        "physical_study_cap": 300, "physical_study_remaining_at_snapshot": 300-charged,
        "new_real_mda_started_this_session_at_snapshot": charged-6,
        "unresolved_started_calls_at_snapshot": sum(s["unresolved_started_calls"] for s in ledgers),
        "qualification_final_result": terminal,
        "one_shot_confirmation_continuation_started": workflow_started,
        "one_shot_workflow_result_at_snapshot": workflow_result,
        "sparse_region_qualified": False,
        "snapshot_limit": "Physical phase may advance after this file; consult its terminal result and live ledgers.",
        "artifact_sha256": {str(p): sha(p) for p in files},
    })
    print("Evidence checks passed; qualification", phase_status,
          "settled", len(observations), "study charged", charged, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    args = parser.parse_args()
    verify(args.output.resolve(), args.test_log.resolve())
