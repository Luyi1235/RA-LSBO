"""Resume with the pinned runner and audit carried-forward worker timings separately."""

import json
from pathlib import Path
import sys

from run import matches


def carried_forward(output):
    manifest = json.loads((output / "recovery_manifest.json").read_bytes())
    original = Path(manifest["original_output"])
    for name in ("calls.jsonl", "run_plan.json"):
        if not matches(original / name, manifest["source_sha256"][name].lower()):
            raise ValueError(f"Original campaign file changed: {original / name}")
    if not matches(output / "run_plan.json", manifest["recovery_plan_sha256"]):
        raise ValueError("Recovery plan differs from its manifest")

    excluded = set(manifest["infrastructure_task_ids"])
    with (original / "calls.jsonl").open(encoding="utf-8") as ledger:
        tasks = {row["task_id"] for row in map(json.loads, ledger)
                 if row["event"] == "worker_started" and row["task_id"] not in excluded}
    if len(tasks) != manifest["replayed_worker_starts"]:
        raise ValueError("Unexpected number of carried-forward worker starts")
    for task in tasks:
        if (output / task / "calls.jsonl").read_bytes() != (original / task / "calls.jsonl").read_bytes():
            raise ValueError(f"Carried-forward call differs from the original: {task}")
    return manifest, tasks


def recover_report(runner, runner_report, campaign, errors, trajectories, services, execution_code_sha):
    manifest, tasks = carried_forward(campaign.output)
    concurrency = runner.elapsed_concurrency
    recovery_cost = {}

    def recovery_peak(rows):
        historical = [row for row in rows if row["task_id"] in tasks]
        recent = [row for row in rows if row["task_id"] not in tasks]
        historical_peak = concurrency(historical)
        if historical_peak > manifest["original_workers"]:
            raise ValueError("Carried-forward calls exceeded the original worker limit")
        peak = concurrency(recent)
        recovery_cost.update(carried_forward_worker_starts=len(historical),
                             carried_forward_peak_workers=historical_peak,
                             recovery_worker_starts=len(recent),
                             recovery_peak_workers=peak,
                             original_infrastructure_attempts=manifest["omitted_charged_infrastructure_attempts"])
        return peak

    runner.elapsed_concurrency = recovery_peak
    try:
        status = runner_report(campaign, errors, trajectories, services, execution_code_sha)
    finally:
        runner.elapsed_concurrency = concurrency
    path = campaign.output / "report.json"
    report = runner.load(path)
    runner.save(path, {**report, **recovery_cost})
    return status


if __name__ == "__main__":
    checkout = Path(sys.argv.pop(1)).resolve()
    sys.path.insert(0, str(checkout))
    import run_x57_five_arm as runner

    runner_report = runner.report

    def report(*args):
        return recover_report(runner, runner_report, *args)

    runner.report = report
    runner.main()
