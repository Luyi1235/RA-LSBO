"""Describe cold-start decision divergence in sealed PR119/120 ledgers."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from run_aircraft_ra_lsbo_comparison import file_hash, paired_interval, read, read_ledger


FULL = "ra_to_lsbo"
MASKED = "ra_stop_status_to_lsbo"


def first_usable(records):
    return next((step for step, record in enumerate(records, 1) if record.usable_feasible),
                len(records) + 1)


def compare_paths(full, masked):
    first_full, first_masked = first_usable(full), first_usable(masked)
    limit = min(first_full, first_masked, len(full))
    split = next((index for index in range(limit)
                  if full[index].candidate != masked[index].candidate), None)
    row = {"full_first_capped": first_full, "masked_first_capped": first_masked,
           "first_distinct_call_before_or_at_discovery": None,
           "different_layout_at_split": None, "full_only_usable_at_split": None,
           "masked_only_usable_at_split": None}
    if split is not None:
        a, b = full[split], masked[split]
        row.update(first_distinct_call_before_or_at_discovery=split + 1,
                   different_layout_at_split=(a.candidate["discrete"]["prop_layout"]
                                              != b.candidate["discrete"]["prop_layout"]),
                   full_only_usable_at_split=a.usable_feasible and not b.usable_feasible,
                   masked_only_usable_at_split=b.usable_feasible and not a.usable_feasible)
    return row


def audit(folder):
    seal = read(folder / "seal.json")
    for name in ("plan.json", "calls.jsonl"):
        if file_hash(folder / name) != seal[name]:
            raise ValueError(f"File differs from sealed input: {folder / name}")
    header = read(folder / "plan.json")
    spec = header["experiment"]
    if not {FULL, MASKED}.issubset(spec["arms"]):
        raise ValueError("The full and masked stop-evidence arms are required")
    calls = read_ledger(folder, header["contract"])
    horizon = spec["horizon"]
    seeds = []
    for seed in spec["seeds"]:
        initial = [calls[f"{seed}/initial/{i}"][1] for i in range(spec["initial"])]
        if any(record.usable_feasible for record in initial):
            continue
        full, masked = ([calls[f"{seed}/{arm}/{i}"][1] for i in range(horizon)]
                        for arm in (FULL, MASKED))
        seeds.append({"seed": seed, **compare_paths(full, masked)})
    if not seeds:
        raise ValueError(f"No initially unsolved seeds in {folder}")
    deltas = [row["full_first_capped"] - row["masked_first_capped"] for row in seeds]
    curves = [{"online_calls": k,
               "full_discovered": sum(row["full_first_capped"] <= k for row in seeds),
               "masked_discovered": sum(row["masked_first_capped"] <= k for row in seeds)}
              for k in range(1, horizon + 1)]
    split_rows = [row for row in seeds if row["first_distinct_call_before_or_at_discovery"] is not None]
    return {"profile": spec["name"], "source_sha": header["source_sha"],
            "cold_seeds": len(seeds), "mean_first_call_delta": mean(deltas),
            "paired_95": paired_interval(deltas), "online_discovery_curve": curves,
            "seeds_with_distinct_choice": len(split_rows),
            "layout_changed_at_first_split": sum(row["different_layout_at_split"] for row in split_rows),
            "full_only_usable_at_first_split": sum(row["full_only_usable_at_split"] for row in split_rows),
            "masked_only_usable_at_first_split": sum(row["masked_only_usable_at_split"] for row in split_rows),
            "first_split_histogram": dict(Counter(row["first_distinct_call_before_or_at_discovery"]
                                                  for row in split_rows)), "seeds": seeds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path, nargs="+", help="Archived formal profile directories")
    args = parser.parse_args()
    print(json.dumps([audit(folder) for folder in args.profiles], indent=2))
