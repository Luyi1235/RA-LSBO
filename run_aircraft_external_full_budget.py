"""Paired cold-start discovery and full-budget feasible energy comparison."""

import argparse
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from ra_lsbo.budget import digest
from ra_lsbo.experiments.comparison.methods import ADAPTATIONS, ExternalOptimizer
from ra_lsbo.experiments.core.runner import Ledger, save
from run_aircraft_coverage_factorial import choose, rosters
from run_aircraft_external_baseline_pilot import plan as pilot_plan
from run_aircraft_feedback_replication import paired
from run_aircraft_risk_cold_start import discovery_outcome, read, source_commit
from run_aircraft_timing_interval_value import effect, evidence_hashes, read_ledger, structural_margins
from run_aircraft_timing_usable_value import UsableAircraft


POOL_ARMS = ("sign_greedy", "interval_greedy", "interval_active", "logcei_sign")
ARMS = (*POOL_ARMS, "archsbo_screened")
CONTRASTS = (
    ("interval_greedy", "sign_greedy"),
    ("interval_active", "interval_greedy"),
    ("interval_greedy", "logcei_sign"),
    ("interval_greedy", "archsbo_screened"),
)
SCENARIOS = ("payload", "headwind")
PROFILES = ("smoke", "full-pilot", "confirmation")


def plan(scenario, profile):
    spec = pilot_plan("pilot-"+scenario)
    spec.pop("max_physical_calls")
    offset = 0 if scenario == "payload" else 10_000
    if profile == "smoke":
        seeds, sizes, budget = [29060000+offset], [8, 16], 17
    elif profile == "full-pilot":
        seeds, sizes, budget = [29060100+offset], [8, 16], 48
    else:
        seeds, sizes, budget = list(range(29060200+offset, 29062600+offset, 100)), [8, 16], 48
    return {
        **spec,
        "name": f"{profile}-{scenario}", "analysis_role": profile, "scenario": scenario,
        "initial_sizes": sizes, "seeds": seeds, "total_budget": budget, "arms": list(ARMS),
        "calls_per_seed": max(sizes)+len(ARMS)*sum(budget-n for n in sizes),
        "shared_initial_prefixes": True,
        "pooled_arms": list(POOL_ARMS),
        "pool_policy": "same fixed covered roster for pooled arms; ArchSBO searches the screened design space",
        "post_feasible_policy": "all pooled arms use the same logCEI implementation with all their records",
        "logcei_adapter": ADAPTATIONS["logcei"],
        "endpoints": ["first_usable_total_call_capped_at_budget_plus_one", "hit_within_budget",
                      "terminal_best_usable_Wh", "feasibility_first_terminal_order"],
        "primary": {"scenario": "payload", "initial": 8, "cohort": "cold",
                    "contrast": "interval_greedy_minus_sign_greedy",
                    "endpoint": "first_usable_total_call_capped_at_budget_plus_one"},
    }


def passes_exact_screen(problem, candidate):
    margins = (*problem.precheck(candidate).values(), *structural_margins(problem, candidate).values())
    return all(np.isfinite(value) and value <= 0. for value in margins)


def play_pool(problem, seed, prefix, pool, arm, horizon, evaluate):
    history, remaining = list(prefix), list(pool)
    logcei = ExternalOptimizer(problem, seed, "logcei", {"horizon": horizon, "pool": len(pool)})
    trace = []
    for step in range(1, horizon+1):
        use_logcei = arm == "logcei_sign" or any(record.usable_feasible for record in history)
        if use_logcei:
            candidate, info = logcei.propose(history, remaining, logcei.next_iteration)
            entry = {"rule": info["phase"]}
        else:
            policy = "active" if arm == "interval_active" else "greedy"
            index, entry = choose(problem, seed, history, remaining, policy,
                                  fit_interval=arm != "sign_greedy")
            candidate = remaining[index]
        remaining.remove(candidate)
        record = evaluate(candidate, step)
        history.append(record)
        if use_logcei:
            logcei.accept(record)
        trace.append({"step": step, "candidate_id": digest(candidate),
                      "outcome": discovery_outcome(record.to_dict()), "usable": record.usable_feasible, **entry})
    return trace


def run(spec, seed, sha, output):
    from run_aircraft_archsbo_screened import ARCHSBO_VERSION, screened_trajectory

    if version("sb-arch-opt") != ARCHSBO_VERSION:
        raise ValueError(f"The fixed native baseline requires SBArchOpt {ARCHSBO_VERSION}")
    versions = {name: version(name) for name in ("sb-arch-opt", "pymoo", "smt", "numpy", "scipy",
                                                 "numba", "scikit-learn", "torch", "botorch")}
    problem = UsableAircraft(spec)
    initial, pools = rosters(problem, spec, seed)
    pool = pools["covered"]
    output.mkdir(parents=True)
    save(output/"plan.json", {"spec": spec, "seed": seed, "source_sha": sha, "contract": problem.contract_id,
                              "versions": versions})
    save(output/"roster.json", {"initial": initial, "covered": pool})
    ledger = Ledger(output, spec["calls_per_seed"])

    def pay(candidate, tag, role):
        if not passes_exact_screen(problem, candidate):
            raise ValueError("Every paid mission call must pass the shared exact screen")
        return ledger.evaluate(problem, candidate, tag, role)

    history = [pay(point, f"initial/{i}", "initial") for i, point in enumerate(initial, 1)]
    for size in spec["initial_sizes"]:
        prefix = history[:size]
        horizon = spec["total_budget"]-size
        for arm in ARMS:
            tag = f"{size}/{arm}"

            def payoff(candidate, step):
                return pay(candidate, f"{tag}/{step}", "online")

            if arm == "archsbo_screened":
                receipt = screened_trajectory(problem, prefix, seed, horizon, payoff)
                trace = receipt["steps"]
            else:
                trace = play_pool(problem, seed, prefix, pool, arm, horizon, payoff)
            save(output/str(size)/(arm+".json"), trace)
            print(spec["name"], seed, size, arm, len(trace), ledger.count, flush=True)
    if ledger.count != spec["calls_per_seed"]:
        raise ValueError("Mission budget was not spent as registered")
    save(output/"seal.json", evidence_hashes(output))


def trajectory_metrics(records, budget):
    usable = [(i, float(record.observation.objective.value)) for i, record in enumerate(records, 1)
              if record.usable_feasible]
    if any(not np.isfinite(value) for _, value in usable):
        raise ValueError("Non-finite usable objective")
    first = usable[0][0] if usable else None
    best = min((value for _, value in usable), default=None)
    return {"first_usable": first, "capped_first": first if first is not None else budget+1,
            "best_usable_Wh": best, "usable_count": len(usable)}


def verify(folder, sha):
    header = read(folder/"plan.json")
    spec = plan(header["spec"]["scenario"], header["spec"]["analysis_role"])
    problem = UsableAircraft(spec)
    seed = header["seed"]
    if (seed not in spec["seeds"] or header["spec"] != spec or header["contract"] != problem.contract_id
            or header["source_sha"] != sha or read(folder/"seal.json") != evidence_hashes(folder)):
        raise ValueError("Run does not match its sealed registered plan")
    initial, pools = rosters(problem, spec, seed)
    pool = pools["covered"]
    if read(folder/"roster.json") != {"initial": initial, "covered": pool}:
        raise ValueError("Sealed roster differs from its registered candidate generator")
    calls = read_ledger(folder, problem.contract_id)
    position = 0

    def settled(tag, role, point=None):
        nonlocal position
        position += 1
        number, record, recorded_role = calls[tag]
        if (number != position or recorded_role != role or (point is not None and record.candidate != point)
                or not passes_exact_screen(problem, record.candidate)):
            raise ValueError("Unexpected settled record, order or exact screen")
        return record

    history = [settled(f"initial/{i}", "initial", point) for i, point in enumerate(initial, 1)]
    pool_ids = set(map(digest, pool))
    budget = spec["total_budget"]
    rows = {}
    for size in spec["initial_sizes"]:
        prefix = history[:size]
        rows[str(size)] = {"cold": not any(record.usable_feasible for record in prefix), "arms": {}}
        for arm in ARMS:
            trace = read(folder/str(size)/(arm+".json"))
            if len(trace) != budget-size:
                raise ValueError("Incomplete budget trajectory")
            states = list(prefix)
            seen = {record.observation.design_id for record in prefix}
            for step, item in enumerate(trace, 1):
                record = settled(f"{size}/{arm}/{step}", "online")
                key = record.observation.design_id
                if ((item["step"], item["candidate_id"], item["usable"], item["outcome"])
                        != (step, key, record.usable_feasible, discovery_outcome(record.to_dict()))
                        or key in seen or (arm in POOL_ARMS and key not in pool_ids)):
                    raise ValueError("Procedure selected an unexpected candidate or outcome")
                if arm == "archsbo_screened":
                    if item["paid_calls"] != step or item["candidate"] != record.candidate:
                        raise ValueError("Native paid call does not match its settled proposal")
                else:
                    started_feasible = any(r.usable_feasible for r in states)
                    expected = "log_constrained_ei" if started_feasible else "feasibility_search"
                    if started_feasible or arm == "logcei_sign":
                        if item["rule"] != expected:
                            raise ValueError("Pooled arm failed to follow its registered LogCEI handoff")
                    elif item["rule"] not in ("active" if arm == "interval_active" else "greedy",
                                               "coverage_fallback"):
                        raise ValueError("Pooled arm used an unexpected cold-start policy")
                seen.add(key)
                states.append(record)
            metrics = trajectory_metrics(states, budget)
            if arm == "archsbo_screened":
                metrics["trained_proposals"] = sum(item["native_surrogate_trained"] for item in trace)
                metrics["fallback_proposals"] = sum(item["screen_counts"].get("fallback_selected", 0)
                                                     for item in trace)
            rows[str(size)]["arms"][arm] = metrics
    if position != len(calls) or position != spec["calls_per_seed"]:
        raise ValueError("Unaccounted mission calls in the sealed run")
    return {"seed": seed, "by_initial": rows, "paid_calls": position, "versions": header["versions"]}


def compare(rows, left, right):
    pair = [(row["arms"][left], row["arms"][right]) for row in rows]
    first = paired([a["capped_first"]-b["capped_first"] for a, b in pair])
    hits = [int(a["first_usable"] is not None)-int(b["first_usable"] is not None) for a, b in pair]
    energies = [(a["best_usable_Wh"], b["best_usable_Wh"]) for a, b in pair]
    common = [a-b for a, b in energies if a is not None and b is not None]
    def feasibility_first(value):
        return value is None, 0. if value is None else value

    order = [(feasibility_first(a), feasibility_first(b)) for a, b in energies]
    return {"first_capped_delta": first, "hit_rate_delta": effect(hits),
            "energy_Wh_delta_both_usable": paired(common),
            "feasibility_first_terminal_order": {"wins": sum(a<b for a, b in order),
                                                 "losses": sum(a>b for a, b in order),
                                                 "ties": sum(a==b for a, b in order)}}


def analyze(spec, folders, sha):
    rows = [verify(folder, sha) for folder in folders]
    expected = spec["seeds"]
    rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in rows] != expected:
        raise ValueError("Analysis requires all registered shards exactly once")
    if any(read(folder/"plan.json")["spec"] != spec for folder in folders):
        raise ValueError("Analysis cannot mix scenarios or profiles")
    if any(row["versions"] != rows[0]["versions"] for row in rows):
        raise ValueError("Formal aggregation requires one dependency set")
    summary = {}
    for size in map(str, spec["initial_sizes"]):
        paired_rows = [r["by_initial"][size] for r in rows]
        summary[size] = {
            name: {"n": len(cohort),
                   "hits": {a: sum(r["arms"][a]["first_usable"] is not None for r in cohort) for a in ARMS},
                   "contrasts": {f"{a}_minus_{b}": compare(cohort, a, b) for a, b in CONTRASTS}}
            for name, cohort in (("all", paired_rows), ("cold", [r for r in paired_rows if r["cold"]]))
        }
    return {"status": "VERIFIED_FULL_BUDGET_EXTERNAL_COMPARISON", "source_sha": sha, "spec": spec,
            "rcaide_mda_calls": 0, "paid_calls": sum(r["paid_calls"] for r in rows),
            "summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify"), required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--expected-sha")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = plan(args.scenario, args.profile)
    if args.task == "plan":
        print(json.dumps(spec, indent=2))
        return
    if not args.expected_sha or args.output is None:
        parser.error("--expected-sha and --output are required")
    output = args.output.resolve()
    if output.exists() or Path(__file__).resolve().parent in (output, *output.parents):
        parser.error("Use a new output path outside the code worktree")
    sha = source_commit(args.expected_sha, clean=True)
    if args.task == "run":
        if args.seed not in spec["seeds"]:
            parser.error("--seed must be a registered seed")
        run(spec, args.seed, sha, output)
    else:
        if args.input is None or args.input.resolve() in (output, *output.parents):
            parser.error("Provide sealed --input and write outside it")
        result = analyze(spec, [args.input/str(seed) for seed in spec["seeds"]], sha)
        save(output, result)
        print(result["status"], spec["name"], len(result["rows"]), result["paid_calls"], flush=True)


if __name__ == "__main__":
    main()
