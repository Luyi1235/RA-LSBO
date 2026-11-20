"""Prospective one-choice confirmation for cold-start timing feedback."""

import argparse
import json
from collections import Counter
from math import isclose
from pathlib import Path

from scipy.stats import binomtest

from aircraft_timing_information import supports_timing
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.runner import Ledger
from ra_lsbo.partial_bo.problems import candidate_pool
from run_aircraft_initial_budget import nested_indices
from run_aircraft_risk_cold_start import discovery_outcome, read, select, source_commit
from run_aircraft_timing_feedback_mechanism import LENGTHS, completion_forecasts, effect
from run_aircraft_timing_interval_value import (
    BenchmarkAircraft, evidence_hashes, read_ledger, save, structural_margins,
)


ARMS = ("sign", "interval")
INITIAL = 8
PROFILES = {
    "timing-confirm-smoke": (4., 0., "seen", (28760300, 28780900)),
    "timing-confirm-headwind-seen": (4., 0., "seen", tuple(range(28940100, 28944900, 100))),
    "timing-confirm-headwind-unseen": (4., 0., "unseen", tuple(range(28944900, 28946500, 100))),
    "timing-confirm-payload-seen": (0., 24., "seen", tuple(range(28946500, 28948100, 100))),
    "timing-confirm-payload-unseen": (0., 24., "unseen", tuple(range(28948100, 28949700, 100))),
}


def plan(name):
    wind, payload, stratum, seeds = PROFILES[name]
    role = "primary" if name == "timing-confirm-headwind-seen" else (
        "smoke" if name == "timing-confirm-smoke" else "secondary")
    return {"name": name, "headwind_m_s": wind, "payload_delta_kg": payload,
            "initial": INITIAL, "raw_pool": 128, "stratum": stratum, "seeds": list(seeds),
            "length_scales": list(LENGTHS), "arms": list(ARMS),
            "max_evaluator_calls": len(seeds)*(INITIAL+len(ARMS)),
            "analysis_role": role, "endpoint": "next_design_usable_after_cold_initialization",
            "test": "two-sided exact discordant-pair binomial test"}


def campaign(spec, problem, obtain):
    rows = []
    for seed in spec["seeds"]:
        raw = candidate_pool(problem, seed+1, spec["raw_pool"])[0]
        pool = [x for x in raw if all(m <= 0. for m in structural_margins(problem, x).values())]
        if len(pool) < spec["initial"]:
            raise ValueError("The registered screened pool cannot construct a shared prefix")
        order = nested_indices(problem, pool, spec["initial"])
        history = [obtain(pool[i], f"{seed}/initial/{j}", "initial") for j, i in enumerate(order)]
        row = {"seed": seed, "initial_ids": [digest(r.candidate) for r in history],
               "initial_outcomes": [discovery_outcome(r.to_dict()) for r in history]}
        if any(r.usable_feasible for r in history):
            rows.append({**row, "state": "initially_solved"})
            continue
        if not all(map(supports_timing, history)):
            rows.append({**row, "state": "unsupported_initial_feedback"})
            continue
        seen = {digest(r.candidate["discrete"]) for r in history}
        used = set(order)
        candidates = [x for i, x in enumerate(pool) if i not in used
                      and ((digest(x["discrete"]) in seen) == (spec["stratum"] == "seen"))]
        if not candidates:
            rows.append({**row, "state": "empty_stratum"})
            continue
        predictions = completion_forecasts(problem, history, candidates, seed)
        indices = {arm: select(candidates, predictions[arm]) for arm in ARMS}
        observed = {}
        choices = {}
        for arm, index in indices.items():
            candidate = candidates[index]
            key = digest(candidate)
            if key not in observed:
                result = obtain(candidate, f"{seed}/next/{arm}", "next")
                observed[key] = {"design_id": key, "usable": result.usable_feasible,
                                 "outcome": discovery_outcome(result.to_dict())}
            choices[arm] = {**observed[key], "predicted_completion": float(predictions[arm][index])}
        rows.append({**row, "state": "paired", "candidate_count": len(candidates), "choices": choices})
    return rows


def summarize(rows):
    paired = [r["choices"] for r in rows if r["state"] == "paired"]
    differences = [int(r["interval"]["usable"])-int(r["sign"]["usable"]) for r in paired]
    wins, losses = differences.count(1), differences.count(-1)
    return {"states": dict(Counter(r["state"] for r in rows)), "paired": len(paired),
            "different_next_design": sum(r["interval"]["design_id"] != r["sign"]["design_id"] for r in paired),
            "hits": {arm: sum(r[arm]["usable"] for r in paired) for arm in ARMS},
            "interval_only": wins, "sign_only": losses, "interval_minus_sign": effect(differences),
            "discordant_exact_p_two_sided": (float(binomtest(wins, wins+losses).pvalue)
                                             if wins+losses else (1. if paired else None))}


def run(spec, sha, folder):
    source_commit(sha, clean=True)
    if Path(__file__).resolve().parent in (folder, *folder.parents):
        raise ValueError("Write evidence outside the source worktree")
    folder.mkdir(parents=True)
    problem = BenchmarkAircraft(spec)
    save(folder/"plan.json", {"source_sha": sha, "spec": spec, "contract": problem.contract_id})
    ledger = Ledger(folder, spec["max_evaluator_calls"])
    rows = campaign(spec, problem, lambda candidate, tag, role: ledger.evaluate(problem, candidate, tag, role))
    save(folder/"observed.json", rows)
    save(folder/"seal.json", evidence_hashes(folder))


def verify(folder, sha):
    source_commit(sha, clean=True)
    header = read(folder/"plan.json")
    spec = plan(header["spec"]["name"])
    problem = BenchmarkAircraft(spec)
    if (header != {"source_sha": sha, "spec": spec, "contract": problem.contract_id}
            or read(folder/"seal.json") != evidence_hashes(folder)):
        raise ValueError("Plan or evidence seal differs")
    calls = read_ledger(folder, problem.contract_id)
    if len(calls) > spec["max_evaluator_calls"]:
        raise ValueError("Evaluator budget exceeded")
    position = 0

    def replay(candidate, tag, role):
        nonlocal position
        position += 1
        number, record, recorded_role = calls[tag]
        if (number, record.candidate, recorded_role) != (position, candidate, role):
            raise ValueError("Ledger differs from the registered decision")
        return record

    rows = campaign(spec, problem, replay)
    original = read(folder/"observed.json")
    for replayed, saved in zip(rows, original):
        for arm, choice in replayed.get("choices", {}).items():
            if not isclose(choice["predicted_completion"], saved["choices"][arm]["predicted_completion"],
                           rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("Recorded probability differs from replay")
    for entries in (rows, original):
        for row in entries:
            for choice in row.get("choices", {}).values():
                choice.pop("predicted_completion")
    if position != len(calls) or rows != original:
        raise ValueError("Recorded outcomes or decisions differ from replay")
    return {"status": "VERIFIED_TIMING_FEEDBACK_CONFIRMATION", "source_sha": sha, "plan": spec,
            "actual_calls": position, "rcaide_mda_calls": 0, "summary": summarize(rows),
            "rows": read(folder/"observed.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify"), required=True)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--expected-sha")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.task in ("plan", "run") and not args.profile:
        parser.error("--profile is required")
    if args.task == "plan":
        print(json.dumps(plan(args.profile), indent=2))
        return
    if not args.expected_sha or args.output is None:
        parser.error("--expected-sha and --output are required")
    output = args.output.resolve()
    if args.task == "run":
        run(plan(args.profile), args.expected_sha, output)
    else:
        if args.input is None or args.input.resolve() in (output, *output.parents):
            parser.error("Provide --input and write outside the sealed run")
        result = verify(args.input, args.expected_sha)
        save(output, result)
        print(result["status"], result["plan"]["name"], result["actual_calls"], flush=True)


if __name__ == "__main__":
    main()
