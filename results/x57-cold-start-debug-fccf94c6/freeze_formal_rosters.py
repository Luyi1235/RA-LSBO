"""Freeze the independent initialization rosters without running an evaluator."""

import argparse
import json
from pathlib import Path

from ra_lsbo.budget import digest
from ra_lsbo.x57_cold_worker import dump, sha256
from ra_lsbo.x57_live import X57LiveProblem, evaluator_files
from run_x57_engineering_energy_pilot import points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("physical_base", type=Path)
    args = parser.parse_args()
    directory = args.archive
    protocol_path = directory / "formal_protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    mission = protocol["mission"]
    config = directory / mission["config"]
    if (sha256(config) != mission["config_sha256"]
            or digest(evaluator_files(args.physical_base)) != mission["worker_manifest_sha256"]
            or sha256(directory / "campaign_report.json") != protocol["pilot_report_sha256"]):
        raise ValueError("Formal roster inputs changed")
    campaign = json.loads((directory / "campaign_plan.json").read_bytes())
    pilot = directory / f"seed-{campaign['pilot_seeds'][0]}" / "pilot_plan.json"
    excluded = set(json.loads(pilot.read_bytes())["historical_candidate_exclusions"])
    excluded.update(row["candidate_id"] for line in (directory / "calls.jsonl").read_bytes().splitlines()
                    if (row := json.loads(line))["event"] == "worker_started")
    exclusion_hash = digest(sorted(excluded))
    problem = X57LiveProblem(args.physical_base, config, directory)
    main_count = protocol["shared_designs"]["main_count"]
    small_count = protocol["shared_designs"]["sensitivity_count"]
    rosters = {}
    for seed in protocol["seeds"]:
        _, rows, sampling = points(problem, seed, excluded, main_count)
        excluded.update(row["candidate_id"] for row in rows)
        rosters[seed] = {"main_sha256": digest(rows), "sensitivity_prefix_sha256": digest(rows[:small_count]),
                         "records": rows, "sampling": sampling}
    dump(directory / protocol["shared_designs"]["roster_file"],
         {"protocol_sha256": sha256(protocol_path), "initial_exclusions_sha256": exclusion_hash,
          "rosters": rosters, "new_worker_calls": 0})


if __name__ == "__main__":
    main()
