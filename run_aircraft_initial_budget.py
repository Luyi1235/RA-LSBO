"""Equal-total-budget comparison of initial design allocation and risk acquisition."""

import argparse
from collections import Counter
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from aircraft_risk_discovery import fit_stage_heads, rollout_components, shortlist, stage_forecast
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.runner import Ledger, append, lines, save
from ra_lsbo.experiments.discovery_stages.policies import nearest_distance
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import ModelNumericalFailure
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.partial_bo.problems import candidate_pool
from run_aircraft_ra_lsbo_comparison import (
    BASE, BenchmarkAircraft, FullBudgetLSBO, evidence_hashes, paired_interval, read_ledger,
)
from run_aircraft_risk_cold_start import discovery_outcome, read, select, source_commit


METHODS = ("lsbo", "greedy", "lookahead")
TIMING_METHODS = ("timing_sign", "timing_interval")
TIMING_PROFILES = {
    "timing-budget-smoke": (4., 0., 8, (4, 6), (28740000,), 64, 8),
    "timing-budget-headwind": (4., 0., 48, (8, 16, 32),
                             tuple(range(28740100, 28742500, 100)), 128, 24),
    "timing-budget-payload": (0., 24., 48, (8, 16, 32),
                             tuple(range(28740100, 28742500, 100)), 128, 24),
}
PROFILES = {
    "smoke": (4., 0., 10, (4, 6), (28470100, 28475000), 64, 8),
    **{f"pilot-{scenario}-{budget}": (
        wind, payload, budget, levels, tuple(range(28500000, 28500800, 100)), 256, 32,
    ) for (scenario, wind, payload), (budget, levels) in product(
        (("headwind", 4., 0.), ("payload", 0., 24.)),
        ((48, (4, 8, 16, 32)), (96, (4, 8, 16, 32, 64))),
    )},
    **TIMING_PROFILES,
}


def plan(name):
    wind, payload, budget, levels, seeds, pool, panel = PROFILES[name]
    timing = name in TIMING_PROFILES
    methods = ("lsbo", *TIMING_METHODS) if timing else METHODS
    arms = [{"name": f"{method}-{size}", "method": method, "initial": size}
            for method, size in product(methods, levels)]
    calls_per_seed = budget + panel + sum(budget - a["initial"] for a in arms)
    spec = {"name": name, "headwind_m_s": wind, "payload_delta_kg": payload,
            "total_budget": budget, "pool": pool, "heldout": panel, "arms": arms,
            "checkpoints": [budget // 4, budget // 2, budget], "native_gp_steps": 45,
            "seeds": list(seeds), "max_calls": calls_per_seed * len(seeds)}
    if timing:
        spec["structural_screen"] = True
        spec["primary"] = {"reference": f"timing_interval-{levels[0]}",
                           "control": f"timing_sign-{levels[0]}",
                           "metric": "capped_first", "undiscovered_score": budget+1}
    return spec


def candidate_roster(problem, spec, seed):
    pool = candidate_pool(problem, seed + 1, spec["pool"])[0]
    panel = candidate_pool(problem, seed + 2, spec["heldout"], map(digest, pool))[0]
    if spec.get("structural_screen"):
        from run_aircraft_timing_online import structural_margins

        pool = [candidate for candidate in pool
                if all(margin <= 0. for margin in structural_margins(problem, candidate).values())]
        if len(pool) < spec["total_budget"]:
            raise ValueError(f"Seed {seed} cannot supply the registered screened budget")
    return pool, panel


def coverage_index(problem, pool, previous):
    counts = Counter(digest(candidate["discrete"]) for candidate in previous)

    def rank(candidate):
        key = digest(candidate["discrete"])
        distance = nearest_distance(problem, candidate, previous)
        return -counts[key], 0. if distance is None else distance

    # Pool order resolves ties using the common randomized roster.
    return max(range(len(pool)), key=lambda i: rank(pool[i]))


def nested_indices(problem, pool, budget):
    remaining = list(range(len(pool)))
    selected = []
    for _ in range(budget):
        offset = coverage_index(problem, [pool[i] for i in remaining],
                                [pool[i] for i in selected])
        selected.append(remaining.pop(offset))
    return selected


def fit(problem, history, seed):
    features = PrefixFeatures(problem, BASE, history, seed, latent=False)
    designs = [row.candidate for row in history]
    outcomes = [discovery_outcome(row.to_dict()) for row in history]
    heads = fit_stage_heads(history, features.transform(designs), outcomes, seed,
                            len(problem.space.continuous))
    return features, heads


def risk_index(problem, history, remaining, seed, method, calls_left):
    if method in TIMING_METHODS:
        from run_aircraft_timing_online import choose

        index, reason, probability = choose(problem, history, remaining, seed, method)
        return index, {"mode": method if reason == "greedy" else "coverage_fallback",
                       "reason": reason, "completion_probability": probability}
    try:
        features, heads = fit(problem, history, seed)
        query = features.transform(remaining)
        p, moments, z = stage_forecast(heads, query)
    except ModelNumericalFailure as error:
        index = coverage_index(problem, remaining, [row.candidate for row in history])
        return index, {"mode": "coverage_fallback", "head": error.head}

    usable = p["usable_factored"].tolist()
    greedy = select(remaining, usable)
    if method == "greedy" or calls_left == 1:
        return greedy, {"mode": "greedy", "greedy": greedy, "usable": usable}

    indices = shortlist(p, [digest(candidate) for candidate in remaining])
    components = rollout_components(heads, query, p, moments, z, indices)
    scores = [-1.] * len(remaining)
    for index, abort, violation, _ in components:
        scores[index] = min(1., usable[index] + abort + violation)
    return select(remaining, scores), {"mode": "lookahead", "greedy": greedy,
                                       "usable": usable, "components": components}


def trace(problem, spec, seed, records, panel):
    forecasts = []
    for count in spec["checkpoints"]:
        try:
            features, heads = fit(problem, records[:count], seed)
            p = stage_forecast(heads, features.transform(panel))[0]
            forecasts.append({"calls": count, "usable": p["usable_factored"].tolist(),
                              "completion": p["mission"].tolist()})
        except ModelNumericalFailure as error:
            forecasts.append({"calls": count, "fit_failure": error.head})
    return forecasts


def run(spec, sha, output):
    import torch

    source_commit(sha, clean=True)
    root = Path(__file__).resolve().parent
    if root == output or root in output.parents:
        raise ValueError("Use an evidence directory outside the source worktree")
    torch.set_num_threads(1)
    problem = BenchmarkAircraft(spec)
    rosters = {seed: candidate_roster(problem, spec, seed) for seed in spec["seeds"]}
    output.mkdir(parents=True)
    budget = spec["total_budget"]
    save(output / "plan.json", {"source_sha": sha, "experiment": spec,
                                "contract": problem.contract_id})
    ledger = Ledger(output, spec["max_calls"])
    for seed in spec["seeds"]:
        unit = output / str(seed)
        pool, panel = rosters[seed]
        order = nested_indices(problem, pool, budget)
        save(unit / "roster.json", {"pool": pool, "initial_order": order, "heldout": panel})
        shared = [ledger.evaluate(problem, pool[i], f"{seed}/initial/{j}", "initial")
                  for j, i in enumerate(order)]
        for arm in spec["arms"]:
            folder = unit / arm["name"]
            folder.mkdir()
            torch.manual_seed(seed)
            size, method = arm["initial"], arm["method"]
            history = shared[:size]
            native, native_start = None, None
            used = {row.observation.design_id for row in history}
            for step in range(size, budget):
                remaining = [candidate for candidate in pool if digest(candidate) not in used]
                if method == "lsbo" or native is not None or any(row.usable_feasible for row in history):
                    if native is None:
                        native_start = step
                        native = FullBudgetLSBO(problem, BASE, seed,
                                                {**spec, "horizon": budget-step}, folder / "native")
                    candidate, _ = native.propose(history, remaining, step-native_start)
                    detail = {"mode": "lsbo"}
                else:
                    index, detail = risk_index(problem, history, remaining, seed, method, budget-step)
                    candidate = remaining[index]
                key = digest(candidate)
                if key in used:
                    raise ValueError("Repeated candidate within policy budget")
                append(folder / "proposals.jsonl", {"call": step+1, "candidate": candidate,
                       "history": digest([row.to_dict() for row in history]), **detail})
                record = ledger.evaluate(problem, candidate, f"{seed}/{arm['name']}/{step}", "online")
                if native is not None:
                    native.accept(record)
                    if native.stop_reason:
                        raise RuntimeError("Full-budget LSBO stopped early")
                history.append(record)
                used.add(key)
            save(folder / "forecasts.json", trace(problem, spec, seed, history, panel))
        save(unit / "coverage-forecasts.json", trace(problem, spec, seed, shared, panel))
        for j, candidate in enumerate(panel):
            ledger.evaluate(problem, candidate, f"{seed}/heldout/{j}", "heldout")
        print(f"{seed}: {ledger.count} calls", flush=True)
    save(output / "seal.json", evidence_hashes(output))


def brier_by_configuration(probabilities, panel, labels):
    groups = {}
    if len(probabilities) != len(panel):
        raise ValueError("Held-out forecasts do not match the panel")
    for p, row, label in zip(probabilities, panel, labels):
        if not 0. <= p <= 1.:
            raise ValueError("Invalid held-out probability")
        if label is not None:
            groups.setdefault(digest(row.candidate["discrete"]), []).append((p-int(label))**2)
    return mean(map(mean, groups.values())) if groups else None


def summarize(history, forecasts, panel, seed, arm, budget):
    first = next((i for i, row in enumerate(history, 1) if row.usable_feasible), None)
    energy = min((row.observation.objective.value for row in history if row.usable_feasible),
                 default=None)
    outcomes = [discovery_outcome(row.to_dict()) for row in panel]
    scores = []
    for forecast in forecasts:
        score = {"calls": forecast["calls"]}
        if "fit_failure" in forecast:
            score["fit_failure"] = forecast["fit_failure"]
        else:
            score["usable_brier"] = brier_by_configuration(
                forecast["usable"], panel, [None if x in ("unknown", "complete_unknown")
                                             else x == "usable" for x in outcomes])
            score["completion_brier"] = brier_by_configuration(
                forecast["completion"], panel, [None if x == "unknown" else x != "abort"
                                                for x in outcomes])
        scores.append(score)
    return {"seed": seed, "arm": arm, "first_total": first, "capped_first": first or budget+1,
            "best_energy": energy, "heldout": scores}


def comparison(rows, reference, control, budget):
    left = {r["seed"]: r for r in rows if r["arm"] == reference}
    right = {r["seed"]: r for r in rows if r["arm"] == control}
    pairs = [(left[key], right[key]) for key in left]
    diffs = [a["capped_first"] - b["capped_first"] for a, b in pairs]
    energy = [a["best_energy"] - b["best_energy"] for a, b in pairs
              if a["best_energy"] is not None and b["best_energy"] is not None]
    wins = Counter()
    for a, b in pairs:
        x, y = a["best_energy"], b["best_energy"]
        if x is None and y is None:
            wins["tie"] += 1
        elif x is None or y is None:
            wins["win" if y is None else "loss"] += 1
        else:
            wins["tie" if math.isclose(x, y, rel_tol=1e-10) else
                 "win" if x < y else "loss"] += 1
    calibration = []
    for i, point in enumerate(pairs[0][0]["heldout"]):
        for target in ("completion_brier", "usable_brier"):
            differences = []
            for a, b in pairs:
                x, y = (row["heldout"][i].get(target) for row in (a, b))
                if x is not None and y is not None:
                    differences.append(x-y)
            calibration.append({"calls": point["calls"], "target": target,
                                "n": len(differences),
                                "delta": mean(differences) if differences else None,
                                "paired_95": paired_interval(differences)})
    return {"reference": reference, "control": control, "paired_n": len(diffs),
            "capped_first_delta": mean(diffs), "paired_95": paired_interval(diffs),
            "discovered": [sum(row["capped_first"] <= budget for row in arm.values())
                           for arm in (left, right)],
            "energy_delta_both_feasible": mean(energy) if energy else None,
            "energy_pair_n": len(energy), "terminal_wins_losses_ties": dict(wins),
            "heldout_comparisons": calibration}


def cold_comparison(rows, reference, control, initial, budget):
    relevant = [row for row in rows if row["arm"] in (reference, control)]
    cold = [row for row in relevant if row["first_total"] is None or row["first_total"] > initial]
    if len(cold) % 2 or any(count != 2 for count in Counter(r["seed"] for r in cold).values()):
        raise ValueError("Matched policies must share the same initialization")
    if not cold:
        return {"reference": reference, "control": control, "initial": initial, "paired_n": 0}
    return {"initial": initial, **comparison(cold, reference, control, budget)}


def verify(folder, sha):
    source_commit(sha)
    if read(folder / "seal.json") != evidence_hashes(folder):
        raise ValueError("Archived evidence differs from its seal")
    header = read(folder / "plan.json")
    spec = plan(header["experiment"]["name"])
    if header["source_sha"] != sha or header["experiment"] != spec:
        raise ValueError("Archived protocol identity differs")
    problem = BenchmarkAircraft(spec)
    if header["contract"] != problem.contract_id:
        raise ValueError("Evaluator contract differs")
    calls = read_ledger(folder, header["contract"])
    used = set()

    def consume(tag, candidate, role):
        _, row, actual_role = calls[tag]
        if tag in used or row.candidate != candidate or actual_role != role:
            raise ValueError("Ledger differs from the archived decisions")
        used.add(tag)
        return row

    budget = spec["total_budget"]
    rows, modes = [], {}
    timing = spec.get("structural_screen", False)
    for seed in spec["seeds"]:
        unit = folder / str(seed)
        roster = read(unit / "roster.json")
        pool, order, labels = (roster[key] for key in ("pool", "initial_order", "heldout"))
        registered = candidate_roster(problem, spec, seed) if timing else None
        if ((registered != (pool, labels) if timing else len(pool) != spec["pool"])
                or len(labels) != spec["heldout"]
                or len(set(map(digest, pool+labels))) != len(pool)+len(labels)
                or order != nested_indices(problem, pool, budget)):
            raise ValueError("Invalid common candidate or initialization roster")
        shared = [consume(f"{seed}/initial/{j}", pool[i], "initial") for j, i in enumerate(order)]
        panel = [consume(f"{seed}/heldout/{j}", candidate, "heldout")
                 for j, candidate in enumerate(labels)]
        rows.append(summarize(shared, read(unit / "coverage-forecasts.json"),
                              panel, seed, "coverage", budget))
        for arm in spec["arms"]:
            name, size = arm["name"], arm["initial"]
            folder_arm = unit / name
            receipts = list(lines(folder_arm / "proposals.jsonl"))
            if len(receipts) != budget-size:
                raise ValueError("Policies must exhaust the same total evaluation budget")
            history = shared[:size]
            for step, receipt in enumerate(receipts, size):
                candidate = receipt["candidate"]
                if (receipt["call"] != step+1 or receipt["history"] != digest([r.to_dict() for r in history])
                        or any(candidate == row.candidate for row in history)):
                    raise ValueError("Invalid proposal chronology")
                mode = receipt["mode"]
                native = arm["method"] == "lsbo" or any(r.usable_feasible for r in history)
                expected = "lsbo" if native else (
                    "greedy" if step == budget-1 and arm["method"] == "lookahead" else arm["method"])
                if mode != expected and (native or mode != "coverage_fallback"):
                    raise ValueError("Invalid acquisition or handoff phase")
                if not native:
                    remaining = [raw for raw in pool if all(raw != r.candidate for r in history)]
                    if timing:
                        selected, detail = risk_index(problem, history, remaining, seed,
                                                      arm["method"], budget-step)
                        p, q = detail["completion_probability"], receipt.get("completion_probability")
                        matching = (p is q if p is None or q is None else
                                    math.isclose(p, q, rel_tol=1e-8, abs_tol=1e-10))
                        if (mode != detail["mode"] or receipt.get("reason") != detail["reason"]
                                or not matching):
                            raise ValueError("Timing acquisition differs from its replay")
                    elif mode == "coverage_fallback":
                        selected = coverage_index(problem, remaining, [r.candidate for r in history])
                    else:
                        scores = receipt["usable"]
                        greedy = select(remaining, scores)
                        if receipt["greedy"] != greedy:
                            raise ValueError("Greedy control differs from its forecast")
                        if mode == "lookahead":
                            scores = [-1.] * len(remaining)
                            for i, abort, violation, _ in receipt["components"]:
                                scores[i] = min(1., receipt["usable"][i]+abort+violation)
                        selected = select(remaining, scores)
                    if candidate != remaining[selected]:
                        raise ValueError("Proposal differs from its archived acquisition")
                key = "lookahead_changed_choice" if mode == "lookahead" and selected != greedy else mode
                modes.setdefault(name, Counter())[key] += 1
                history.append(consume(f"{seed}/{name}/{step}", candidate, "online"))
            forecasts = read(folder_arm / "forecasts.json")
            rows.append(summarize(history, forecasts, panel, seed, name, budget))
    if set(calls) != used or len(used) != spec["max_calls"]:
        raise ValueError("Unexpected or missing charged calls")
    arms = [arm["name"] for arm in spec["arms"]]
    pairs = [(name, "coverage") for name in arms]
    sizes = [arm["initial"] for arm in spec["arms"] if arm["method"] == "lsbo"]
    active, controls = (("timing_interval", ("timing_sign", "lsbo")) if timing else
                        ("lookahead", ("greedy", "lsbo")))
    prefixes = ("lsbo", *TIMING_METHODS) if timing else METHODS
    pairs += [(f"{active}-{n}", f"{m}-{n}") for n, m in product(sizes, controls)]
    pairs += [(f"{m}-{a}", f"{m}-{b}") for m in prefixes for a in sizes for b in sizes if a < b]
    pairs += [(f"{active}-{a}", f"lsbo-{b}") for a in sizes for b in sizes if a < b]
    if timing:
        pairs += [(f"timing_sign-{n}", f"lsbo-{n}") for n in sizes]
        cold = [cold_comparison(rows, f"{active}-{n}", f"{control}-{n}", n, budget)
                for n, control in product(sizes, controls)]
    return {"status": "VERIFIED_INITIAL_BUDGET", "profile": spec["name"],
            "source_sha": sha, "actual_calls": len(used), "rcaide_mda_calls": 0,
            "total_per_policy": budget, "decision_counts": modes, "per_seed": rows,
            "comparisons": [comparison(rows, a, b, budget) for a, b in pairs],
            **({"primary": spec["primary"], "initially_unsolved_comparisons": cold} if timing else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify"), required=True)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--expected-sha")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.task in ("plan", "run") and args.profile is None:
        parser.error("plan and run require --profile")
    if args.task != "plan" and (not args.expected_sha or args.output is None):
        parser.error("run and verify require --expected-sha and --output")
    if args.task == "plan":
        print(json.dumps(plan(args.profile), indent=2))
    elif args.task == "run":
        run(plan(args.profile), args.expected_sha, args.output.resolve())
    else:
        if args.input is None:
            parser.error("verify requires --input")
        if args.input.resolve() in args.output.resolve().parents:
            parser.error("write analysis outside the sealed evidence directory")
        result = verify(args.input, args.expected_sha)
        save(args.output, result)
        print(result["status"], result["actual_calls"])


if __name__ == "__main__":
    main()
