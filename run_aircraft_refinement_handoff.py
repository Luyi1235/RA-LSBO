"""Test shared open-space refinement after budgeted cold-start discovery."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.runner import Ledger, save
from run_aircraft_external_full_budget import (
    SCENARIOS, UsableAircraft, compare, discovery_outcome, evidence_hashes, passes_exact_screen,
    plan as full_budget_plan, play_pool, read, read_ledger, rosters, trajectory_metrics,
)
from run_aircraft_risk_cold_start import source_commit


INITIAL = 8
PROFILES = ("smoke", "full-pilot", "confirmation")
FIXED = ("sign_fixed", "interval_fixed")
HANDOFF = ("sign_open", "interval_open")
NATIVE = "archsbo_screened"
ARMS = (*FIXED, *HANDOFF, NATIVE)
CONTRASTS = (
    ("interval_open", "sign_open"),
    ("interval_open", "interval_fixed"),
    ("sign_open", "sign_fixed"),
    ("interval_open", NATIVE),
)


def plan(scenario, profile):
    offset = 0 if scenario == "payload" else 10_000
    if profile == "smoke":
        seeds, budget = [29060100+offset], 12
    elif profile == "full-pilot":
        seeds, budget = [29100100+offset, 29100200+offset], 48
    else:
        seeds, budget = list(range(29100300+offset, 29102700+offset, 100)), 48
    return {
        **full_budget_plan(scenario, "full-pilot"),
        "name": f"{profile}-{scenario}", "analysis_role": profile, "seeds": seeds, "initial_sizes": [INITIAL],
        "total_budget": budget, "calls_per_seed": INITIAL+len(ARMS)*(budget-INITIAL),
        "arms": list(ARMS), "pooled_arms": list(FIXED),
        "pool_policy": "shared fixed roster during discovery; it remains fixed for the two _fixed arms",
        "post_feasible_policy": "fixed arms use logCEI; open arms both start screened ArchSBO with all evaluated records",
        "native_control": "screened ArchSBO throughout, starting from the same initial records",
        "primary": {"scenario": "headwind", "initial": INITIAL, "cohort": "cold",
                    "contrast": "interval_open_minus_sign_open", "endpoint": "feasibility_first_terminal_order"},
    }


def native_steps(native_search, problem, history, seed, count, evaluate, *, offset=0):
    result = native_search(problem, history, seed, count, lambda point, i: evaluate(point, offset+i))
    return [{**step, "step": offset+step["step"], "phase": "open"} for step in result["steps"]]


def play_arm(problem, seed, prefix, pool, arm, horizon, evaluate, native_search):
    if arm == NATIVE:
        return {"handoff_after_call": None,
                "steps": native_steps(native_search, problem, prefix, seed, horizon, evaluate)}
    discovery = "interval_greedy" if arm.startswith("interval") else "sign_greedy"
    history = list(prefix)

    def observe(point, step):
        record = evaluate(point, step)
        history.append(record)
        return record

    trace = play_pool(problem, seed, prefix, pool, discovery, horizon, observe,
                      stop_after_discovery=arm in HANDOFF)
    steps = [{**step, "phase": "fixed_refinement" if step["rule"] == "log_constrained_ei" else "discovery"}
             for step in trace]
    remaining = horizon-len(steps)
    if not remaining:
        return {"handoff_after_call": None, "steps": steps}
    if arm not in HANDOFF or not any(record.usable_feasible for record in history):
        raise RuntimeError("Open refinement requires a usable design and remaining budget")
    native = native_steps(native_search, problem, history, seed, remaining, evaluate, offset=len(steps))
    return {"handoff_after_call": len(history), "steps": steps+native}


def run(spec, seed, sha, output):
    from run_aircraft_archsbo_screened import ARCHSBO_VERSION, screened_trajectory

    if version("sb-arch-opt") != ARCHSBO_VERSION:
        raise ValueError(f"The fixed refinement backend requires SBArchOpt {ARCHSBO_VERSION}")
    import torch

    versions = {name: version(name) for name in ("sb-arch-opt", "pymoo", "smt", "numpy", "scipy",
                                                 "numba", "scikit-learn", "torch", "botorch")}
    versions["torch_runtime"] = str(torch.__version__)
    problem = UsableAircraft(spec)
    initial, pools = rosters(problem, spec, seed)
    initial, pool = initial[:INITIAL], pools["covered"]
    output.mkdir(parents=True)
    save(output/"plan.json", {"spec": spec, "seed": seed, "source_sha": sha, "contract": problem.contract_id,
                             "versions": versions})
    save(output/"roster.json", {"initial": initial, "covered": pool})
    ledger = Ledger(output, spec["calls_per_seed"])

    def pay(point, tag, role):
        if not passes_exact_screen(problem, point):
            raise ValueError("Every paid candidate must pass the shared engineering screen")
        return ledger.evaluate(problem, point, tag, role)

    history = [pay(point, f"initial/{i}", "initial") for i, point in enumerate(initial, 1)]
    for arm in ARMS:
        receipt = play_arm(problem, seed, history, pool, arm, spec["total_budget"]-INITIAL,
                           lambda point, step: pay(point, f"{arm}/{step}", "online"), screened_trajectory)
        save(output/(arm+".json"), receipt)
        print(spec["name"], seed, arm, receipt["handoff_after_call"], ledger.count, flush=True)
    if ledger.count != spec["calls_per_seed"]:
        raise ValueError("The planned per-arm budget was not spent")
    save(output/"seal.json", evidence_hashes(output))


def verify(folder, spec, seed, sha):
    header = read(folder/"plan.json")
    problem = UsableAircraft(spec)
    if (header["spec"] != spec or header["seed"] != seed or header["source_sha"] != sha
            or header["contract"] != problem.contract_id or read(folder/"seal.json") != evidence_hashes(folder)):
        raise ValueError("The run does not match the registered sealed plan")
    initial, pools = rosters(problem, spec, seed)
    roster = {"initial": initial[:INITIAL], "covered": pools["covered"]}
    if read(folder/"roster.json") != roster:
        raise ValueError("The recorded roster differs from the registered generator")
    pool_ids = set(map(digest, roster["covered"]))
    ledger = read_ledger(folder, problem.contract_id)
    position = 0

    def settled(tag, role):
        nonlocal position
        position += 1
        number, record, recorded_role = ledger[tag]
        if (number, recorded_role) != (position, role) or not passes_exact_screen(problem, record.candidate):
            raise ValueError("A ledger entry failed the order or engineering screen check")
        return record

    prefix = [settled(f"initial/{i}", "initial") for i in range(1, INITIAL+1)]
    if [record.candidate for record in prefix] != roster["initial"]:
        raise ValueError("The initial paid evaluations differ from the shared initial roster")
    arms, discovery = {}, {}
    for arm in ARMS:
        receipt = read(folder/(arm+".json"))
        trace = receipt["steps"]
        if len(trace) != spec["total_budget"]-INITIAL:
            raise ValueError("An arm did not spend its registered logical budget")
        history = list(prefix)
        seen = {record.observation.design_id for record in prefix}
        handoff = None
        discovery[arm] = []
        for step, item in enumerate(trace, 1):
            record = settled(f"{arm}/{step}", "online")
            key = record.observation.design_id
            has_usable = any(r.usable_feasible for r in history)
            phase = "open" if arm == NATIVE else "discovery"
            if has_usable and arm in HANDOFF:
                if handoff is None:
                    handoff = len(history)
                phase = "open"
            elif has_usable and arm in FIXED:
                phase = "fixed_refinement"
            expected = (step, key, phase, record.usable_feasible, discovery_outcome(record.to_dict()))
            if tuple(item[name] for name in ("step", "candidate_id", "phase", "usable", "outcome")) != expected or key in seen:
                raise ValueError("A proposal differs from its sealed observation or phase")
            if phase == "open":
                native_step = step if arm == NATIVE else INITIAL+step-handoff
                if item["paid_calls"] != native_step or item["candidate"] != record.candidate:
                    raise ValueError("The native call differs from its paid receipt")
            else:
                rules = ("log_constrained_ei",) if has_usable else ("greedy", "coverage_fallback")
                if key not in pool_ids or item["rule"] not in rules:
                    raise ValueError("A fixed-roster arm violated its registered acquisition rule")
                if phase == "discovery":
                    discovery[arm].append(key)
            history.append(record)
            seen.add(key)
        if receipt["handoff_after_call"] != handoff:
            raise ValueError("An arm switched outside its registered handoff")
        arms[arm] = {**trajectory_metrics(history, spec["total_budget"]),
                     "handoff_after_call": handoff,
                     "open_paid_calls": sum(item["phase"] == "open" for item in trace),
                     "open_outside_pool": sum(item["phase"] == "open" and item["candidate_id"] not in pool_ids
                                              for item in trace)}
    if any(discovery[a] != discovery[b] for a, b in zip(FIXED, HANDOFF)):
        raise ValueError("Changing the refinement backend altered the pre-refinement sequence")
    if position != len(ledger) or position != spec["calls_per_seed"]:
        raise ValueError("Unaccounted paid evaluations")
    return {"seed": seed, "cold": not any(record.usable_feasible for record in prefix), "arms": arms,
            "paid_calls": position, "versions": header["versions"]}


def analyze(root, spec, sha):
    rows = [verify(root/str(seed), spec, seed, sha) for seed in spec["seeds"]]
    if any(row["versions"] != rows[0]["versions"] for row in rows):
        raise ValueError("One runtime is required within the registered cohort")
    summary = {}
    for name, group in (("all", rows), ("cold", [row for row in rows if row["cold"]])):
        summary[name] = {"n": len(group),
                         "hits": {arm: sum(row["arms"][arm]["first_usable"] is not None for row in group)
                                  for arm in ARMS},
                         "contrasts": {f"{a}_minus_{b}": compare(group, a, b) for a, b in CONTRASTS}}
    return {"status": "VERIFIED_AIRCRAFT_REFINEMENT_HANDOFF", "spec": spec, "source_sha": sha,
            "paid_calls": sum(row["paid_calls"] for row in rows), "rcaide_mda_calls": 0,
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
        parser.error("Use a new output outside the code worktree")
    sha = source_commit(args.expected_sha, clean=True)
    if args.task == "run":
        if args.seed not in spec["seeds"]:
            parser.error("--seed must be listed in the registered plan")
        run(spec, args.seed, sha, output)
    else:
        if args.input is None or args.input.resolve() in (output, *output.parents):
            parser.error("Provide --input and write outside the sealed runs")
        result = analyze(args.input, spec, sha)
        save(output, result)
        print(result["status"], spec["name"], len(result["rows"]), result["paid_calls"], flush=True)


if __name__ == "__main__":
    main()
