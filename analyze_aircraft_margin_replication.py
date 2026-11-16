"""Verify the paired structural-feedback replication and estimate the seed-level effect."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from random import Random

from aircraft_risk_discovery import discovery_pairs
from aircraft_risk_feedback import summarize_discovery
from aircraft_risk_protocol import TERMINATION_ARMS
from aircraft_risk_termination import STRUCTURAL_GATES, coarsen
from analyze_aircraft_structural_margins import matched_staged_prefix, staged_prefix
from analyze_aircraft_termination_factorial import loaded_calls
from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.records import DesignRecord
from run_aircraft_risk_cold_start import discovery_outcome, read, source_commit, verify_evidence

MARGIN_GP, STAGED = TERMINATION_ARMS
PRIMARY = ("retained_margins", "trigger_only")
COMPARISONS = (PRIMARY, ("retained_margins", STAGED), ("trigger_only", STAGED))


def observable_payload(record):
    """Drop view-specific provenance IDs, preserving every measured value."""
    packet = record["observation"]
    ignored = {"evidence_id", "contract_id"}
    values = {k: v for k, v in packet.items() if k not in ignored}
    values["objective"] = {k: v for k, v in packet["objective"].items() if k != "evidence_id"}
    values["constraints"] = [(head, {k: v for k, v in output.items() if k != "evidence_id"})
                             for head, output in packet["constraints"]]
    return {"candidate": record["candidate"], "observation": values, "diagnostics": record["diagnostics"]}


def check_observation_pair(trigger_only, enriched, analytical, trigger_contract):
    view = DesignRecord.from_dict(enriched)
    if "controlled_trigger" in enriched["diagnostics"] and any(
            view.observation.constraint(head).kind != "exact" for head in STRUCTURAL_GATES):
        raise ValueError("The enriched structural stop lacks a numerical margin")
    projected = coarsen(view, analytical, trigger_contract)
    if (digest(observable_payload(asdict(projected))) != digest(observable_payload(trigger_only))
            or discovery_outcome(trigger_only) != discovery_outcome(enriched)):
        raise ValueError("Candidate pair differs beyond the controlled feedback channel")


def paired_interval(rows, *, draws=20000, seed=20261116):
    deltas = [row["capped_calls_delta"] for row in rows
              if (row["arm"], row["baseline"]) == PRIMARY and not row["initial_usable"]]
    if len(deltas) < 2:
        return None
    rng = Random(seed)
    n = len(deltas)
    samples = sorted(sum(rng.choices(deltas, k=n))/n for _ in range(draws))
    return [samples[int(.025*(draws-1))], samples[int(.975*(draws-1))]]


def compare(trigger_only, enriched, expected):
    spec, new_spec = (read(path/"plan.json") for path in (trigger_only, enriched))
    profiles = (spec["profile"], new_spec["profile"])
    if profiles not in {("margin-replication-trigger"+suffix, "margin-replication-margins"+suffix)
                        for suffix in ("", "-smoke")}:
        raise ValueError("Use matched replication profiles at the same scale")
    report_a, _, traj_a = verify_evidence(trigger_only, expected)
    report_b, _, traj_b = verify_evidence(enriched, expected)
    if any(spec[field] != new_spec[field]
           for field in ("seeds", "arms", "budget", "total")):
        raise ValueError("Replication budgets or seeds differ")
    contract_a = read(trigger_only/"contract.json")
    contract_b = read(enriched/"contract.json")
    comparable = {k: v for k, v in contract_b["settings"].items() if k != "termination_feedback"}
    if (contract_a["heads"] != contract_b["heads"] or comparable.get("structural_gate_order") != list(STRUCTURAL_GATES)
            or {k: v for k, v in contract_a["settings"].items() if k != "termination_feedback"} != comparable):
        raise ValueError("Replication underlying aircraft contract differs")

    data_a, counts_a = loaded_calls(trigger_only)
    data_b, counts_b = loaded_calls(enriched)
    if counts_a != counts_b:
        raise ValueError("Feedback channels changed held-out termination outcomes")
    checks = set()
    for seed in spec["seeds"]:
        a = read(trigger_only/str(seed)/"roster.json")
        b = read(enriched/str(seed)/"roster.json")
        if a != b:
            raise ValueError(f"Replication rosters differ for seed {seed}")
        checks.update(digest(candidate) for role in ("initial", "heldout") for candidate in a[role]["designs"])
        if not matched_staged_prefix(staged_prefix(trigger_only, seed), staged_prefix(enriched, seed)):
            raise ValueError(f"Staged cold-start control differs at seed {seed}")

    overlap = data_a.keys() & data_b.keys()
    if not checks <= overlap:
        raise ValueError("Missing paired fixed observations")
    record = next(iter(data_a.values()))
    analytical = {head for head, _ in record["observation"]["constraints"]} - set(contract_a["heads"])
    for key in overlap:
        check_observation_pair(data_a[key], data_b[key], analytical, contract_a["id"])

    # The staged arm receives no structural evidence before discovery.
    first_a = [(r["seed"], r["first_usable_feasible_call"]) for r in traj_a if r["arm"] == STAGED]
    first_b = [(r["seed"], r["first_usable_feasible_call"]) for r in traj_b if r["arm"] == STAGED]
    if first_a != first_b:
        raise ValueError("Staged discovery endpoints differ")
    combined = ([{**r, "arm": "trigger_only"} for r in traj_a if r["arm"] == MARGIN_GP]
                + [{**r, "arm": "retained_margins" if r["arm"] == MARGIN_GP else STAGED} for r in traj_b])
    pairs = discovery_pairs(combined, spec["total"], COMPARISONS)
    return {"status": "VERIFIED_MARGIN_REPLICATION", "source_sha": expected,
            "calls": {"trigger_only": report_a["calls"], "retained_margins": report_b["calls"]},
            "matched_fixed_designs": len(checks), "matched_evaluated_designs": len(overlap),
            "matched_staged_seeds": len(spec["seeds"]), "heldout_termination_counts": counts_a,
            "summary": summarize_discovery(pairs), "primary_paired_bootstrap_95": paired_interval(pairs),
            "pairs": pairs}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--trigger-only", type=Path, required=True)
    cli.add_argument("--enriched", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--expected-sha", required=True)
    args = cli.parse_args()
    output = args.output.resolve()
    if any(p.resolve() == output or p.resolve() in output.parents for p in (args.trigger_only, args.enriched)):
        cli.error("Write the comparison outside the sealed evidence folders")
    report = compare(args.trigger_only, args.enriched, source_commit(args.expected_sha))
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(report["status"])


if __name__ == "__main__":
    main()
