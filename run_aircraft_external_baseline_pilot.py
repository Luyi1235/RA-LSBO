"""Verify paired first-usable-design comparisons with the existing logCEI adapter."""

import argparse
from collections import Counter
import json
from pathlib import Path

from ra_lsbo.budget import digest
from ra_lsbo.experiments.comparison.methods import ADAPTATIONS, ExternalOptimizer
from ra_lsbo.experiments.core.runner import Ledger, save
from run_aircraft_coverage_factorial import SIZES, plan as coverage_plan, play, rosters
from run_aircraft_feedback_replication import paired
from run_aircraft_risk_cold_start import read, source_commit
from run_aircraft_timing_interval_value import evidence_hashes, read_ledger
from run_aircraft_timing_usable_value import UsableAircraft


ARMS = ("interval_greedy", "interval_active", "logcei_sign")
PROFILES = {
    "smoke": ("payload", (29050100,), 18),
    "pilot-payload": ("payload", tuple(range(29050200, 29050600, 100)), 48),
    "pilot-headwind": ("headwind", tuple(range(29050600, 29051000, 100)), 48),
}


def plan(name):
    scenario, seeds, budget = PROFILES[name]
    registered = coverage_plan(scenario)
    return {
        **{key: registered[key] for key in ("headwind_m_s", "payload_delta_kg", "initial_sizes",
                                             "initial_proposals", "online_proposals", "online_size")},
        "name": name,
        "analysis_role": "adapter_pilot",
        "cell_current_limit_fraction": 1.,
        "seeds": list(seeds),
        "total_budget": budget,
        "arms": list(ARMS),
        "logcei_adapter": ADAPTATIONS["logcei"],
        "max_physical_calls": len(seeds)*(max(SIZES)+registered["online_size"]),
        "endpoints": ["first_usable_total_call_capped_at_budget_plus_one", "hit_within_budget"],
    }


def play_logcei(problem, seed, history, pool, budget, acquire):
    optimizer = ExternalOptimizer(problem, seed, "logcei", {"horizon": budget, "pool": len(pool)})
    iteration = 0

    def propose(_problem, _seed, observed, remaining, _policy):
        point, metadata = optimizer.propose(observed, remaining, iteration)
        index = remaining.index(point)
        return index, {"rule": metadata["phase"], "p_usable": metadata["joint_probability"][index]}

    def observe(point, call):
        nonlocal iteration
        record = acquire(point, call)
        optimizer.accept(record)
        iteration += 1
        return record

    return play(problem, seed, history, pool, budget, None, observe, selector=propose)


def campaign(spec, problem, obtain):
    rows = []
    for seed in spec["seeds"]:
        initial, pools = rosters(problem, spec, seed)
        pool = pools["covered"]
        history = [obtain(point, f"{seed}/initial/{i}", "initial") for i, point in enumerate(initial, 1)]
        cache = {}

        def acquire(point, tag):
            key = digest(point)
            if key not in cache:
                cache[key] = obtain(point, f"{seed}/{tag}", "online")
            return cache[key]

        by_initial = {}
        for size in SIZES:
            prefix = history[:size]
            arms = {}
            for arm in ARMS:
                def observe(point, call):
                    return acquire(point, f"{size}/{arm}/{call}")

                budget = spec["total_budget"]
                if arm == "logcei_sign":
                    arms[arm] = play_logcei(problem, seed, prefix, pool, budget, observe)
                else:
                    policy = arm.removeprefix("interval_")
                    arms[arm] = play(problem, seed, prefix, pool, budget, policy, observe)
            by_initial[str(size)] = {"cold": not any(record.usable_feasible for record in prefix),
                                     "arms": arms}
        rows.append({"seed": seed, "initial_ids": [digest(point) for point in initial],
                     "covered_pool_ids": [digest(point) for point in pool], "by_initial": by_initial})
        print(spec["name"], seed, {n: {arm: item["capped_first"] for arm, item in row["arms"].items()}
                                  for n, row in by_initial.items()}, flush=True)
    return rows


def summarize(rows):
    arms = [row["arms"] for row in rows]
    return {
        "n": len(arms),
        "hits": {arm: sum(item[arm]["first_usable"] is not None for item in arms) for arm in ARMS},
        "logical_online_calls": {arm: sum(item[arm]["online_evaluations"] for item in arms) for arm in ARMS},
        "decision_rules": {arm: dict(Counter(step["rule"] for item in arms for step in item[arm]["trace"]))
                           for arm in ARMS},
        "differences": {
            f"{left}_minus_{right}": paired([item[left]["capped_first"]-item[right]["capped_first"]
                                            for item in arms])
            for left, right in (("interval_greedy", "logcei_sign"), ("interval_active", "logcei_sign"),
                                ("interval_active", "interval_greedy"))
        },
    }


def run(spec, sha, folder):
    folder.mkdir(parents=True)
    problem = UsableAircraft(spec)
    save(folder/"plan.json", {"source_sha": sha, "spec": spec, "contract": problem.contract_id})
    ledger = Ledger(folder, spec["max_physical_calls"])
    campaign(spec, problem, lambda point, tag, role: ledger.evaluate(problem, point, tag, role))
    save(folder/"seal.json", evidence_hashes(folder))


def verify(folder, sha):
    header = read(folder/"plan.json")
    spec = plan(header["spec"]["name"])
    problem = UsableAircraft(spec)
    if (header != {"source_sha": sha, "spec": spec, "contract": problem.contract_id}
            or read(folder/"seal.json") != evidence_hashes(folder)):
        raise ValueError("Sealed experiment plan differs")
    calls = read_ledger(folder, problem.contract_id)
    position = 0

    def replay(point, tag, role):
        nonlocal position
        position += 1
        number, record, recorded_role = calls[tag]
        if (number, record.candidate, recorded_role) != (position, point, role):
            raise ValueError("Sealed call differs from the registered decision")
        return record

    rows = campaign(spec, problem, replay)
    if position != len(calls) or position > spec["max_physical_calls"]:
        raise ValueError("Ledger count differs from the registered budget")
    summary = {}
    for size in map(str, SIZES):
        groups = [row["by_initial"][size] for row in rows]
        summary[size] = {"all": summarize(groups), "cold": summarize([row for row in groups if row["cold"]])}
    return {"status": "VERIFIED_AIRCRAFT_EXTERNAL_BASELINE_PILOT", "source_sha": sha, "plan": spec,
            "physical_calls": dict(Counter(role for _, _, role in calls.values())), "rcaide_mda_calls": 0,
            "summary": summary, "rows": rows}


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
    if Path(__file__).resolve().parent in (output, *output.parents):
        parser.error("Write outside the code worktree")
    sha = source_commit(args.expected_sha, clean=True)
    if args.task == "run":
        run(plan(args.profile), sha, output)
    else:
        if not args.input or args.input.resolve() in (output, *output.parents):
            parser.error("Provide sealed --input and write outside it")
        report = verify(args.input, sha)
        save(output, report)
        print(report["status"], report["plan"]["name"], report["physical_calls"])


if __name__ == "__main__":
    main()
