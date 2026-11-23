"""Replay native ArchSBO's first proposal on the six archived X-57 prefixes."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

from ra_lsbo.budget import digest
from ra_lsbo.x57_cold_worker import sha256
from ra_lsbo.x57_external import X57External, native_kpls_minimum, native_kpls_shortfall
from ra_lsbo.x57_live import X57LiveProblem
from run_x57_five_arm import BASELINE_VERSIONS, check_threads, code_sha

ARCHIVE_SHA = "bcc54e6ecfda8a68fdcf93238d13b8a07d023daa"
INPUT_SHA256 = {
    "run_plan.json": "1160101863a748cc0eda4283cd8c8cf1b269bcf432f441493114f9ada5b8531a",
    "calls.jsonl": "da57219a60b112165fce52ed2a048f2710872ee060541be24a084e5967022f1b",
    "distance_47.5.json": "9902197262e862b13a3b3ac4a75c6f2c7892c91d8a69d6ce1af8aa884f6e6c3d",
}


def replay(archive, physical_base, output):
    for filename, expected in INPUT_SHA256.items():
        if sha256(archive / filename) != expected:
            raise ValueError(f"Archive input changed: {filename}")
    plan = json.loads((archive / "run_plan.json").read_bytes())
    problem = X57LiveProblem(physical_base, archive / "distance_47.5.json", output.parent)
    if digest(problem.source_hashes) != plan["worker_manifest_sha256"]:
        raise ValueError("The physical evaluator differs from the frozen plan")

    rows = [json.loads(line) for line in (archive / "calls.jsonl").read_bytes().splitlines()]
    dimension = len(problem.space.variables)
    minimum = native_kpls_minimum(dimension)
    results = []
    for seed in plan["seeds"]:
        initial = [r for r in rows if r["seed"] == seed and r["arm"] == "shared" and r["event"] == "settled"]
        expected = plan["rosters"][str(seed)]["records"]
        if [r["candidate_id"] for r in initial] != [r["candidate_id"] for r in expected]:
            raise ValueError(f"Shared prefix differs from the archived plan for seed {seed}")
        adapter = X57External(problem, "archsbo_native", seed, initial, plan["total_budget_per_arm"])
        complete = adapter.native_complete_rows()
        row = {"seed": seed, "initial": len(initial), "complete_training_rows": complete}
        try:
            candidate, info = adapter.ask()
        except ValueError as error:
            if not native_kpls_shortfall(error, minimum, complete):
                raise
            row.update(status="insufficient_complete_rows", error=str(error))
        else:
            key = problem.space.key(problem.candidate_class(**candidate))
            row.update(status="proposed", candidate_id=key, proposal=info)
        results.append(row)

    packages = {name: version(name) for name in (*BASELINE_VERSIONS, "numpy", "scipy")}
    report = {
        "code_sha": code_sha(), "archive_sha": ARCHIVE_SHA, "input_sha256": INPUT_SHA256, "python": sys.version,
        "packages": packages, "nominal_dimension": dimension, "kpls_minimum_complete_rows": minimum,
        "can_propose_for_all_prefixes": all(r["status"] == "proposed" for r in results),
        "prefixes": results, "new_worker_starts": 0, "new_rcaide_calls": 0,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--physical-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_threads()
    report = replay(args.archive, args.physical_base, args.output)
    count = sum(row["status"] == "proposed" for row in report["prefixes"])
    print(f"Native ArchSBO produced first proposals for {count}/{len(report['prefixes'])} prefixes. RCAIDE calls: 0.")


if __name__ == "__main__":
    main()
