"""Verify both feedback-policy runs and compare their paired discovery effects."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median

from aircraft_risk_discovery import discovery_pairs
from aircraft_risk_feedback import summarize_discovery
from aircraft_risk_protocol import TERMINATION_PAIRS
from aircraft_risk_termination import coarsen
from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.records import DesignRecord
from run_aircraft_risk_cold_start import discovery_outcome, read, source_commit, verify_evidence


def compare_discovery(native, structural):
    left = {row["seed"]: row for row in native}
    right = {row["seed"]: row for row in structural}
    if left.keys() != right.keys():
        raise ValueError("Unmatched factorial discovery seeds")
    rows = []
    for seed in sorted(left):
        a, b = left[seed], right[seed]
        if a["initial_usable"] != b["initial_usable"]:
            raise ValueError("Feedback policies disagree about initial usable designs")
        if not a["initial_usable"]:
            rows.append({"seed": seed, "native_delta_calls": a["capped_calls_delta"],
                         "structural_delta_calls": b["capped_calls_delta"],
                         "interaction_delta_calls": b["capped_calls_delta"]-a["capped_calls_delta"]})
    d = [row["interaction_delta_calls"] for row in rows]
    loo = [(sum(d)-value)/(len(d)-1) for value in d] if len(d) > 1 else []
    return {"initial_unsolved": len(rows), "mean_interaction_delta_calls": mean(d) if d else None,
            "median_interaction_delta_calls": median(d) if d else None,
            "leave_one_out_range": [min(loo), max(loo)] if loo else None, "seed_pairs": rows}


def loaded_calls(folder):
    records, categories = {}, Counter()
    with (folder/"calls.jsonl").open(encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            if row["event"] != "settled":
                continue
            record = row["record"]
            identity = record["observation"]["design_id"]
            previous = records.setdefault(identity, record)
            if digest(previous) != digest(record):
                raise ValueError("Repeated aircraft evaluation returned conflicting evidence")
            if "/heldout/" in row["tag"]:
                diag = record["diagnostics"]
                categories[diag.get("controlled_trigger") or diag.get("termination_reason", "unreported")] += 1
    return records, dict(categories)


def compare(native, structural, expected):
    folders = {"native": native, "structural-first": structural}
    plans = {key: read(folder/"plan.json") for key, folder in folders.items()}
    spec = plans["native"]
    profiles = (plans["native"]["profile"], plans["structural-first"]["profile"])
    if profiles not in {("termination-native-smoke", "termination-structural-smoke"),
                        ("termination-native", "termination-structural")}:
        raise ValueError("Use a registered native and structural path at the same scale")
    for field in ("seeds", "initial", "total", "pool", "heldout", "arms", "budget"):
        if spec[field] != plans["structural-first"][field]:
            raise ValueError("Termination experiment schedules differ")
    pairs, runs, recorded = {}, {}, {}
    for key, folder in folders.items():
        report, _, trajectories = verify_evidence(folder, expected)
        records, triggers = loaded_calls(folder)
        pairs[key] = discovery_pairs(trajectories, spec["total"], TERMINATION_PAIRS)
        recorded[key] = records
        runs[key] = {"report": report, "heldout_termination_counts": triggers,
                     "first_discovery": summarize_discovery(pairs[key])}
    first, second = (read(folder/"contract.json") for folder in folders.values())
    extra = {"termination_feedback", "structural_gate_order", "feedback_source_sha256"}
    if {k: v for k, v in second["settings"].items() if k not in extra} != first["settings"]:
        raise ValueError("Underlying aircraft contracts differ")
    if digest([first["id"], second["settings"]]) != second["id"]:
        raise ValueError("Controlled observation contract is not bound to the native evaluator")
    analytical = {name for name, _ in next(iter(recorded["native"].values()))["observation"]["constraints"]} - set(first["heads"])
    fixed = set()
    for seed in spec["seeds"]:
        a, b = (read(folder/str(seed)/"roster.json") for folder in folders.values())
        if a != b:
            raise ValueError("Factorial candidate rosters differ")
        fixed.update(digest(candidate) for role in ("initial", "heldout") for candidate in a[role]["designs"])
    a, b = (recorded[key] for key in folders)
    overlap = a.keys() & b.keys()
    if not fixed <= overlap:
        raise ValueError("Missing paired fixed observations")
    for key in overlap:
        expected_view = coarsen(DesignRecord.from_dict(a[key]), analytical, second["id"]).to_dict()
        if digest(expected_view) != digest(b[key]):
            raise ValueError("Paired output does not match the controlled coarsening")
        if (discovery_outcome(a[key]) == "usable") != (discovery_outcome(b[key]) == "usable"):
            raise ValueError("Feedback policies changed usability")
    return {"status": "VERIFIED_TERMINATION_FEEDBACK_FACTORIAL", "source_sha": expected,
            "matched_fixed_designs": len(fixed), "matched_evaluated_designs": len(overlap),
            "runs": runs, "interaction": compare_discovery(pairs["native"], pairs["structural-first"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--structural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    for source in (args.native.resolve(), args.structural.resolve()):
        if args.output.resolve() == source or source in args.output.resolve().parents:
            parser.error("Write the factorial report outside both source runs")
    sha = source_commit(args.expected_sha)
    result = compare(args.native, args.structural, sha)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(result["status"])


if __name__ == "__main__":
    main()
