"""Compare enriched structural feedback with the sealed PR107 trigger-only run."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

from analyze_aircraft_termination_factorial import compare
from run_aircraft_risk_cold_start import read, source_commit

PR107_SOURCE = "8980cb5850234e1feb6290b05d985d588dd6bf92"


def staged_prefix(folder, seed):
    rows = (folder/str(seed)/"discovery_staged.jsonl").read_bytes().splitlines()
    return [(row["candidate"], row["scores"]) for row in map(json.loads, rows)
            if row["event"] == "proposal" and "discovery" in row]


def matched_staged_prefix(before, after):
    if len(before) != len(after):
        return False
    for (a, scores_a), (b, scores_b) in zip(before, after):
        if a != b or len(scores_a) != len(scores_b) or any(
                not math.isclose(x, y, rel_tol=1e-11, abs_tol=1e-13)
                for x, y in zip(scores_a, scores_b)):
            return False
    return True


def margin_effect(reference, margins, seeds, trigger_only, enriched):
    for seed in seeds:
        if not matched_staged_prefix(staged_prefix(trigger_only, seed), staged_prefix(enriched, seed)):
            raise ValueError(f"Staged cold-start control differs at seed {seed}")
    before = {r["seed"]: r for r in reference["interaction"]["seed_pairs"]}
    after = {r["seed"]: r for r in margins["interaction"]["seed_pairs"]}
    if before.keys() != after.keys():
        raise ValueError("Margins experiment changed the initially unsolved cohort")

    # With the staged path fixed, the change in the paired contrast is the effect
    # of retaining margins within the constraint-GP arm.
    differences = [{"seed": seed, "capped_calls_delta":
                    after[seed]["structural_delta_calls"]-before[seed]["structural_delta_calls"]}
                   for seed in sorted(before)]
    values = [row["capped_calls_delta"] for row in differences]
    loo = [(sum(values)-v)/(len(values)-1) for v in values] if len(values) > 1 else []
    return {"initial_unsolved": len(values), "wins": sum(v < 0 for v in values),
            "losses": sum(v > 0 for v in values), "ties": sum(v == 0 for v in values),
            "mean_delta_calls": mean(values) if values else None,
            "median_delta_calls": median(values) if values else None,
            "leave_one_out_range": [min(loo), max(loo)] if loo else None,
            "paired_deltas": differences}


def analyze(source, enriched, expected):
    profile = read(enriched/"plan.json")["profile"]
    if profile not in {"termination-margins", "termination-margins-smoke"}:
        raise ValueError("Select a registered margins run")
    suffix = "-smoke" if profile.endswith("-smoke") else ""
    native, trigger_only = source/("native"+suffix), source/("structural"+suffix)
    reference = compare(native, trigger_only, PR107_SOURCE)
    margins = compare(native, enriched, expected, native_source=PR107_SOURCE)
    baseline, new = reference["runs"]["structural-first"], margins["runs"]["structural-margins"]
    if baseline["heldout_termination_counts"] != new["heldout_termination_counts"]:
        raise ValueError("Margin retention changed held-out termination events")
    seeds = read(enriched/"plan.json")["seeds"]
    return {"status": "VERIFIED_STRUCTURAL_MARGIN_FEEDBACK", "baseline_source": PR107_SOURCE,
            "source_sha": expected, "staged_control_matched_seeds": len(seeds),
            "baseline": baseline, "enriched": new,
            "matched_fixed_designs": margins["matched_fixed_designs"],
            "matched_evaluated_designs": margins["matched_evaluated_designs"],
            "margin_effect": margin_effect(reference, margins, seeds, trigger_only, enriched)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Archived PR107 result directory")
    parser.add_argument("--enriched", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    reference, enriched = args.reference.resolve(), args.enriched.resolve()
    if any(output == path or path in output.parents for path in (reference, enriched)):
        parser.error("Put the margin-effect report outside the evidence directories")
    report = analyze(reference, enriched, source_commit(args.expected_sha))
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(report["status"])


if __name__ == "__main__":
    main()
