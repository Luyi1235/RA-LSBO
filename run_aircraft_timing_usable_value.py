"""Audit archive coverage and test a usable-design acquisition prospectively."""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from scipy.stats import binomtest

from aircraft_timing_information import supports_timing
from aircraft_timing_usable_value import ARMS, choices, fit_heads, follow_up_scores
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.runner import Ledger, save
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import ModelNumericalFailure
from ra_lsbo.partial_bo.problems import candidate_pool
from ra_lsbo.partial_bo.records import DesignRecord
from run_aircraft_initial_budget import candidate_roster, nested_indices
from run_aircraft_risk_cold_start import discovery_outcome, file_hash, read, select, source_commit
from run_aircraft_timing_decision_audit import INPUT_HASHES, INPUT_RESULT, INPUT_SOURCE
from run_aircraft_timing_interval_value import (
    BenchmarkAircraft, effect, evidence_hashes, plan as archived_plan, read_ledger, structural_margins,
)


INITIAL = 8
DERATED_CURRENT = .30
PROFILES = {
    "smoke": (DERATED_CURRENT, (28802300, 28802800)),
    "headwind-derated": (DERATED_CURRENT, tuple(range(28960100, 28963300, 100))),
    "headwind-nominal": (1., tuple(range(28963300, 28964900, 100))),
}
ARCHIVE_PROFILES = {
    "archive-headwind": ("timing-decision-headwind", 1.),
    "archive-payload": ("timing-decision-payload", 1.),
    "archive-headwind-derated": ("timing-decision-headwind", DERATED_CURRENT),
}


def plan(name):
    current, seeds = PROFILES[name]
    return {"name": name, "headwind_m_s": 4., "payload_delta_kg": 0., "initial": INITIAL,
            "pool": 128, "second_pool": 64, "seeds": list(seeds), "arms": list(ARMS),
            "cell_current_limit_fraction": current, "max_calls": len(seeds)*(INITIAL + 2*len(ARMS)),
            "primary": {"profile": "headwind-derated", "reference": "usable_lookahead",
                        "control": "usable_greedy", "endpoint": "any_usable_by_two",
                        "cohort": "initially_unsolved", "test": "paired exact discordance test"}}


def with_current_limit(record, fraction):
    if fraction == 1. or record.diagnostics.get("mission_completed") is not True:
        return record
    packet = record.observation
    constraints = dict(packet.constraints)
    current = constraints["current"]
    if current.kind == "exact":
        constraints["current"] = replace(current, value=(1.+current.value)/fraction-1.)
    return DesignRecord(record.candidate, replace(packet, constraints=tuple(sorted(constraints.items()))),
                        record.diagnostics)


class UsableAircraft(BenchmarkAircraft):
    def __init__(self, spec):
        super().__init__(spec)
        self.current_fraction = spec["cell_current_limit_fraction"]
        if self.current_fraction != 1.:
            self.contract_id = digest([self.contract_id, "post_completion_current_limit_v1",
                                       self.current_fraction])

    def evaluate(self, raw):
        return with_current_limit(super().evaluate(raw), self.current_fraction)


def pools(problem, spec, seed, *, archive=False):
    if archive:
        first, panel = candidate_roster(problem, spec, seed)
    else:
        raw = candidate_pool(problem, seed+1, spec["pool"])[0]
        first, panel = [x for x in raw if passes_screen(problem, x)], []
    raw_second = candidate_pool(problem, seed+3, spec["second_pool"], map(digest, first+panel))[0]
    second = [x for x in raw_second if passes_screen(problem, x)]
    if len(first) <= INITIAL or not second:
        raise ValueError("Pool cannot support the registered two-call comparison")
    return first, second


def passes_screen(problem, candidate):
    return all(margin <= 0. for margin in structural_margins(problem, candidate).values())


def observed(record):
    if record is None:
        return {"outcome": None, "usable": None}
    return {"outcome": discovery_outcome(record.to_dict()), "usable": record.usable_feasible}


def campaign_seed(problem, seed, history, pool, second, obtain):
    outcomes = [discovery_outcome(record.to_dict()) for record in history]
    row = {"seed": seed, "initial_outcomes": outcomes}
    if any(record.usable_feasible for record in history):
        return {**row, "state": "initially_solved"}
    if not all(map(supports_timing, history)):
        return {**row, "state": "unsupported_initial_feedback"}
    used = {digest(record.candidate) for record in history}
    candidates = [x for x in pool if digest(x) not in used]
    try:
        heads, query = fit_heads(problem, history, candidates+second, seed)
        selected, before = choices(heads, query, candidates, problem.settings)
    except (ModelNumericalFailure, FloatingPointError):
        return {**row, "state": "initial_model_failure"}

    cached = {}

    def acquire(candidate, arm, step):
        key = digest(candidate)
        if key not in cached:
            cached[key] = obtain(candidate, f"{seed}/{arm}/{step}", step)
        return cached[key]

    updates = {}
    for arm, result in selected.items():
        first = candidates[result.pop("index")]
        key = digest(first)
        record = acquire(first, arm, "first")
        result["first"] = {"id": key, **observed(record)}
        result["second"] = None
        result["any_usable_by_two"] = result["first"]["usable"]
        if record is None or record.usable_feasible:
            continue
        if key not in updates:
            scores = before
            state = "unsupported_feedback"
            if supports_timing(record):
                try:
                    posterior, encoded = fit_heads(problem, history+[record], second, seed)
                    scores = follow_up_scores(posterior, encoded)
                    state = "updated"
                except ModelNumericalFailure:
                    state = "update_model_failure"
            updates[key] = scores, state
        (complete, usable), result["update"] = updates[key]
        scores = complete if arm == "interval_greedy" else usable
        index = select(second, scores)
        target = second[index]
        result["second"] = {"id": digest(target), "score": float(scores[index]),
                            **observed(acquire(target, arm, "second"))}
        result["any_usable_by_two"] = result["second"]["usable"]
    return {**row, "state": "paired", "arms": selected}


def campaign(spec, problem, obtain):
    rows = []
    for seed in spec["seeds"]:
        first, second = pools(problem, spec, seed)
        order = nested_indices(problem, first, INITIAL)
        history = [obtain(first[i], f"{seed}/initial/{j}", "initial") for j, i in enumerate(order)]
        row = campaign_seed(problem, seed, history, first, second, obtain)
        rows.append(row)
        print(spec["name"], seed, row["state"], flush=True)
    return rows


def summarize(rows):
    paired = [row["arms"] for row in rows if row["state"] == "paired"]
    coverage = {}
    for arm in ARMS:
        first = [r[arm]["first"]["outcome"] for r in paired]
        second = [r[arm]["second"]["outcome"] for r in paired if r[arm]["second"] is not None]
        results = [r[arm]["any_usable_by_two"] for r in paired]
        coverage[arm] = {"first_observed": sum(x is not None for x in first),
                         "first_outcomes": dict(Counter(x for x in first if x is not None)),
                         "second_requested": len(second), "second_observed": sum(x is not None for x in second),
                         "second_outcomes": dict(Counter(x for x in second if x is not None)),
                         "known_endpoint": sum(x is not None for x in results),
                         "hit": sum(x is True for x in results)}
    contrasts = {}
    for active, control in (("usable_lookahead", "usable_greedy"),
                            ("usable_greedy", "interval_greedy"), ("usable_lookahead", "interval_greedy")):
        results = [(r[active]["any_usable_by_two"], r[control]["any_usable_by_two"]) for r in paired]
        observed_pairs = [(a, b) for a, b in results if a is not None and b is not None]
        delta = [int(a)-int(b) for a, b in observed_pairs]
        wins, losses = delta.count(1), delta.count(-1)
        both_missed_first = [r for r in paired if r[active]["first"]["usable"] is False
                             and r[control]["first"]["usable"] is False
                             and r[active]["any_usable_by_two"] is not None
                             and r[control]["any_usable_by_two"] is not None]
        contrasts[f"{active}_vs_{control}"] = {
            "changed_first": sum(r[active]["first"]["id"] != r[control]["first"]["id"] for r in paired),
            "changed_first_both_endpoints_observed": sum(
                a is not None and b is not None and r[active]["first"]["id"] != r[control]["first"]["id"]
                for r, (a, b) in zip(paired, results)),
            "both_endpoints_observed": len(delta), "wins": wins, "losses": losses,
            "both_first_missed_second_observed": len(both_missed_first),
            "second_usable_on_shared_first_misses": {
                arm: sum(r[arm]["any_usable_by_two"] for r in both_missed_first)
                for arm in (active, control)},
            "paired_delta_observed_only": effect(delta),
            "exact_p_observed_only": (float(binomtest(wins, wins+losses).pvalue) if wins+losses else None),
        }
    return {"states": dict(Counter(row["state"] for row in rows)),
            "coverage": coverage, "contrasts": contrasts}


def offline(profile, source, analyses):
    name, fraction = ARCHIVE_PROFILES[profile]
    folder, analysis = source/name, analyses/(name+".json")
    header = read(folder/"plan.json")
    spec = archived_plan(name)
    problem = BenchmarkAircraft(spec)
    if (header != {"source_sha": INPUT_SOURCE, "spec": spec, "contract": problem.contract_id}
            or (file_hash(folder/"seal.json"), file_hash(analysis)) != INPUT_HASHES[name]
            or read(folder/"seal.json") != evidence_hashes(folder)):
        raise ValueError("PR135 sealed input identity differs")
    calls = read_ledger(folder, problem.contract_id)
    if len(calls) != read(analysis)["actual_calls"]:
        raise ValueError("PR135 call count mismatch")
    by_seed = {}
    for tag, (_, record, _) in calls.items():
        records = by_seed.setdefault(int(tag.split("/")[0]), {})
        key = digest(record.candidate)
        if key in records and records[key].to_dict() != record.to_dict():
            raise ValueError("PR135 has conflicting duplicate observations")
        records[key] = record
    rows = []
    for seed in spec["seeds"]:
        history = [with_current_limit(calls[f"{seed}/initial/{j}"][1], fraction) for j in range(INITIAL)]
        first, second = pools(problem, spec, seed, archive=True)
        if [r.candidate for r in history] != [first[i] for i in nested_indices(problem, first, INITIAL)]:
            raise ValueError("PR135 initial history does not match the reconstructed pool")
        bank = by_seed[seed]

        def lookup(candidate, tag, role):
            record = bank.get(digest(candidate))
            return None if record is None else with_current_limit(record, fraction)

        rows.append(campaign_seed(problem, seed, history, first, second, lookup))
    return {"input_result_sha": INPUT_RESULT, "profile": profile, "cell_current_limit_fraction": fraction,
            "new_evaluator_calls": 0, "rcaide_mda_calls": 0, "summary": summarize(rows), "rows": rows}


def run(spec, sha, folder):
    folder.mkdir(parents=True)
    problem = UsableAircraft(spec)
    save(folder/"plan.json", {"source_sha": sha, "spec": spec, "contract": problem.contract_id})
    ledger = Ledger(folder, spec["max_calls"])
    campaign(spec, problem, lambda candidate, tag, role: ledger.evaluate(problem, candidate, tag, role))
    save(folder/"seal.json", evidence_hashes(folder))


def verify(folder, sha):
    header = read(folder/"plan.json")
    spec = plan(header["spec"]["name"])
    problem = UsableAircraft(spec)
    if (header != {"source_sha": sha, "spec": spec, "contract": problem.contract_id}
            or read(folder/"seal.json") != evidence_hashes(folder)):
        raise ValueError("Run plan or evidence seal differs")
    calls = read_ledger(folder, problem.contract_id)
    position = 0

    def replay(candidate, tag, role):
        nonlocal position
        position += 1
        number, record, recorded_role = calls[tag]
        if (number, record.candidate, recorded_role) != (position, candidate, role):
            raise ValueError("Ledger differs from replayed decisions")
        return record

    rows = campaign(spec, problem, replay)
    if position != len(calls) or position > spec["max_calls"]:
        raise ValueError("Ledger count differs from the registered study")
    return {"status": "VERIFIED_TIMING_USABLE_VALUE", "source_sha": sha, "plan": spec,
            "actual_calls": position, "rcaide_mda_calls": 0, "summary": summarize(rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("plan", "run", "verify", "offline"))
    parser.add_argument("--profile", choices=(*PROFILES, *ARCHIVE_PROFILES))
    parser.add_argument("--expected-sha")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--analyses", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.task in ("plan", "run") and args.profile not in PROFILES:
        parser.error("Choose a registered online profile")
    if args.task == "plan":
        print(json.dumps(plan(args.profile), indent=2))
        return
    if not args.expected_sha or args.output is None:
        parser.error("--expected-sha and --output are required")
    output = args.output.resolve()
    forbidden = [Path(__file__).resolve().parent, *(p.resolve() for p in (args.input, args.analyses) if p)]
    if any(p in (output, *output.parents) for p in forbidden):
        parser.error("Write outside the code worktree and sealed inputs")
    sha = source_commit(args.expected_sha, clean=True)
    if args.task == "run":
        run(plan(args.profile), sha, output)
    elif args.task == "verify":
        if args.input is None:
            parser.error("--input is required")
        result = verify(args.input, sha)
        save(output, result)
        print(result["status"], result["plan"]["name"], result["actual_calls"])
    else:
        if args.profile not in ARCHIVE_PROFILES or args.input is None or args.analyses is None:
            parser.error("Choose an archive profile and supply --input and --analyses")
        result = offline(args.profile, args.input, args.analyses)
        save(output, {"source_sha": sha, **result})
        print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
