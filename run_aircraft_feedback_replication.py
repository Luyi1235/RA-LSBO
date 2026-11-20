"""Replicate interval risk acquisition and separate two uses of stopped feedback."""

import argparse
from collections import Counter
from functools import partial
import json
from pathlib import Path

from scipy.stats import binomtest

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.runner import Ledger, save
from run_aircraft_coverage_factorial import SIZES, choose, plan as coverage_plan, play, rosters
from run_aircraft_risk_cold_start import discovery_outcome, read, source_commit
from run_aircraft_timing_interval_value import effect, evidence_hashes, read_ledger
from run_aircraft_timing_usable_value import UsableAircraft


# (policy, fit with interval feedback, anticipate interval feedback)
ARMS = {
    "interval_greedy": ("greedy", True, True),
    "interval_active": ("active", True, True),
    "sign_greedy": ("greedy", False, False),
    "sign_active": ("active", False, False),
    "interval_fit_sign_score": ("active", True, False),
}
PROFILES = {
    "smoke": ((28800100,), 18),
    "payload": (tuple(range(29020100, 29024900, 100)), 48),
    "nominal-smoke": ((28803700,), 18),
    "nominal-payload": (tuple(range(29030100, 29033300, 100)), 48),
    "nominal-headwind": (tuple(range(29033300, 29036500, 100)), 48),
}
NOMINAL_SCENARIOS = {"nominal-smoke": "payload", "nominal-payload": "payload",
                     "nominal-headwind": "headwind"}


def plan(name):
    seeds, budget = PROFILES[name]
    nominal = name in NOMINAL_SCENARIOS
    spec = coverage_plan(NOMINAL_SCENARIOS.get(name, "payload"))
    arm_names = tuple(arm for arm in ARMS if not nominal or arm != "interval_fit_sign_score")
    spec.pop("audit_seeds")
    spec.pop("max_audit_calls")
    spec.update(name=name, seeds=list(seeds), total_budget=budget, arms={
        arm: {"policy": ARMS[arm][0], "fit_interval": ARMS[arm][1], "score_interval": ARMS[arm][2]}
        for arm in arm_names
    }, max_optimizer_calls=len(seeds)*(max(SIZES)+spec["online_size"]),
        primary={"initial": 8 if nominal else 16, "cohort": "unsolved_at_initial",
                 "endpoint": "first_usable_total_call_capped_at_budget_plus_one",
                 "contrast": ("interval_greedy_minus_sign_greedy" if nominal
                              else "interval_active_minus_interval_greedy")})
    if nominal:
        spec["cell_current_limit_fraction"] = 1.
        spec["analysis_role"] = {"nominal-smoke": "smoke", "nominal-payload": "primary",
                                 "nominal-headwind": "secondary"}[name]
    return spec


def campaign(spec, problem, obtain):
    rows = []
    for seed in spec["seeds"]:
        initial, pools = rosters(problem, spec, seed)
        covered = pools["covered"]
        history = [obtain(point, f"{seed}/initial/{i}", "initial")
                   for i, point in enumerate(initial, 1)]
        cache = {}

        def acquire(point, tag):
            key = digest(point)
            if key not in cache:
                cache[key] = obtain(point, f"{seed}/{tag}", "online")
            return cache[key]

        row = {"seed": seed, "initial_ids": [digest(x) for x in initial],
               "initial_outcomes": [discovery_outcome(r.to_dict()) for r in history],
               "covered_pool_ids": [digest(x) for x in covered], "by_initial": {}}
        for size in spec["initial_sizes"]:
            arms = {}
            for name, (policy, fit, score) in ARMS.items():
                if name not in spec["arms"]:
                    continue
                selector = partial(choose, fit_interval=fit, score_interval=score)
                arms[name] = play(problem, seed, history[:size], covered, spec["total_budget"], policy,
                                  lambda point, i: acquire(point, f"{size}/{name}/{i}"), selector=selector)
            row["by_initial"][str(size)] = {"cold": not any(r.usable_feasible for r in history[:size]),
                                           "arms": arms}
        rows.append(row)
        print(spec["name"], seed,
              {n: [a["capped_first"] for a in r["arms"].values()] for n, r in row["by_initial"].items()},
              flush=True)
    return rows


def paired(values):
    wins = sum(x < 0 for x in values)
    losses = sum(x > 0 for x in values)
    return {**effect(values), "wins": wins, "losses": losses, "ties": len(values)-wins-losses,
            "exact_sign_p_two_sided": float(binomtest(wins, wins+losses).pvalue) if wins+losses else None}


def summarize(rows, arm_names=ARMS):
    arms = [row["arms"] for row in rows]
    times = [{name: v["capped_first"] for name, v in row.items()} for row in arms]
    differences = {
        f"{a}_minus_{b}": paired([t[a]-t[b] for t in times])
        for a, b in (("interval_active", "interval_greedy"), ("sign_active", "sign_greedy"),
                     ("interval_greedy", "sign_greedy"), ("interval_active", "sign_active"),
                     ("interval_greedy", "sign_active"), ("interval_fit_sign_score", "sign_active"),
                     ("interval_active", "interval_fit_sign_score"))
        if a in arm_names and b in arm_names
    }
    differences["feedback_by_policy_interaction"] = paired([
        (t["interval_active"]-t["interval_greedy"])-(t["sign_active"]-t["sign_greedy"]) for t in times])
    return {
        "n": len(rows), "hits": {name: sum(r[name]["first_usable"] is not None for r in arms) for name in arm_names},
        "differences": differences,
        "mean_capped_first": {name: effect([t[name] for t in times]) for name in arm_names},
        "fallbacks": {name: sum(x["rule"] == "coverage_fallback" for r in arms for x in r[name]["trace"])
                      for name in arm_names},
    }


def run(spec, sha, output):
    output.mkdir(parents=True)
    problem = UsableAircraft(spec)
    save(output/"plan.json", {"source_sha": sha, "spec": spec, "contract": problem.contract_id})
    ledger = Ledger(output, spec["max_optimizer_calls"])
    campaign(spec, problem, lambda point, tag, role: ledger.evaluate(problem, point, tag, role))
    save(output/"seal.json", evidence_hashes(output))


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
    if position != len(calls) or position > spec["max_optimizer_calls"]:
        raise ValueError("Ledger count differs from the registered budget")
    summary = {}
    arm_names = spec["arms"]
    for n in map(str, SIZES):
        groups = [r["by_initial"][n] for r in rows]
        summary[n] = {"all": summarize(groups, arm_names),
                      "cold": summarize([r for r in groups if r["cold"]], arm_names),
                      "cold_at_16": summarize([r["by_initial"][n] for r in rows
                                               if r["by_initial"]["16"]["cold"]], arm_names)}
    return {"status": "VERIFIED_AIRCRAFT_FEEDBACK_REPLICATION", "source_sha": sha,
            "plan": spec, "physical_calls": dict(Counter(role for _, _, role in calls.values())),
            "rcaide_mda_calls": 0, "summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify"), required=True)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--expected-sha")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.task in ("plan", "run") and args.profile is None:
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
        if args.input is None or args.input.resolve() in (output, *output.parents):
            parser.error("Provide sealed --input and write outside it")
        result = verify(args.input, sha)
        save(output, result)
        print(result["status"], result["plan"]["name"], result["physical_calls"])


if __name__ == "__main__":
    main()
