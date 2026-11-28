"""Summarize matched early performance and later outcomes from the archived pilot."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean


def analyze(folder):
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        check, name = line.split()
        if sha256((folder / name).read_bytes()).hexdigest() != check:
            raise ValueError(f"Archive hash differs: {name}")
    report = json.loads((folder / "report.json").read_bytes())
    plan = json.loads((folder / "run_plan.json").read_bytes())
    with (folder / "calls.jsonl").open(encoding="utf-8") as stream:
        settled = [row for row in map(json.loads, stream) if row["event"] == "settled"]
    if len(settled) != len({r["task_id"] for r in settled}) or len(settled) != 480:
        raise ValueError("Invalid campaign settlements")
    result = {"source_commit": "e5fcec2710f1a8f3b5326974a76fff23f67ed917", "arms": {}}
    zero = [s for s in report["per_seed"] if report["initial_groups"][s] == "0"]
    for arm in plan["arms"]:
        rows = [s[arm] for s in report["per_seed"].values()]
        hits = [r["first_quality_call"] for r in rows]
        later = [r for r in settled if r["arm"] == arm and r["evaluation"] > 16]
        exposed = [r for r in settled if r["arm"] == arm and r["proposal"].get("recovery_candidate_id")]
        result["arms"][arm] = {
            "zero_initial_seeds": {s: report["per_seed"][s][arm]["first_usable_call"] for s in zero},
            "hits_by_16": sum(h is not None and h <= 16 for h in hits),
            "hits_by_32": sum(h is not None for h in hits),
            "mean_first_hit_if_all_hit": mean(hits) if None not in hits else None,
            "mean_best_Wh_if_all_feasible": mean(r["best_usable_Wh"] for r in rows)
            if all(r["best_usable_Wh"] is not None for r in rows) else None,
            "outcome_counts_17_to_32": dict(Counter(r["feedback"]["outcome"] for r in later)),
            "violations_among_completed_infeasible_17_to_32": dict(Counter(
                k for r in later if r["feedback"]["outcome"] == "completed_constraint_violation"
                for k, v in r["feedback"]["constraints"].items() if v is not None and v > 0)),
            "recovery_candidates_offered": len(exposed),
            "recovery_candidates_selected": [[r["seed"], r["evaluation"]] for r in exposed
                                            if r["candidate_id"] == r["proposal"]["recovery_candidate_id"]],
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(analyze(parser.parse_args().archive), indent=2, allow_nan=False))
