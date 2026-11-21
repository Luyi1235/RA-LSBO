"""Prospective factorial comparison of candidate coverage and risk acquisition."""

import argparse
from collections import Counter
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from aircraft_timing_information import PROBABILITY_FLOOR, score_information, supports_timing
from aircraft_timing_interval_value import deficit_edges
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.core.runner import Ledger, save
from ra_lsbo.experiments.discovery_stages.policies import active_distance, coverage_pick
from ra_lsbo.partial_bo.models import EvidenceGP, PrefixFeatures
from ra_lsbo.partial_bo.problems import candidate_pool
from ra_lsbo.partial_refinement.posterior import group_divisors
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings
from run_aircraft_initial_budget import nested_indices
from run_aircraft_risk_cold_start import discovery_outcome, read, select, source_commit
from run_aircraft_risk_timing_replay import MARGIN_SCALE, timing_feedback
from run_aircraft_timing_feedback_mechanism import LENGTHS
from run_aircraft_timing_interval_value import effect, evidence_hashes, read_ledger
from run_aircraft_timing_usable_value import DERATED_CURRENT, UsableAircraft, passes_screen


SIZES = (8, 16)
GENERATORS = ("standard", "covered")
POLICIES = ("greedy", "active")
ARMS = tuple(product(GENERATORS, POLICIES))
PROFILES = {
    "smoke": (0., 24., (28800100,), 18, 1),
    "payload": (0., 24., tuple(range(29010100, 29011700, 100)), 48, 6),
    "headwind": (4., 0., tuple(range(29011700, 29012900, 100)), 48, 4),
}


def plan(name):
    wind, payload, seeds, budget, audits = PROFILES[name]
    online_size = 48
    return {
        "name": name, "headwind_m_s": wind, "payload_delta_kg": payload,
        "cell_current_limit_fraction": DERATED_CURRENT,
        "seeds": list(seeds), "initial_sizes": list(SIZES), "total_budget": budget,
        "initial_proposals": 128, "online_proposals": 256, "online_size": online_size,
        "arms": [f"{g}_{p}" for g, p in ARMS], "audit_seeds": list(seeds[:audits]),
        "kernel_lengths": list(LENGTHS), "active_probability_floor": PROBABILITY_FLOOR,
        "primary": {"profile": "payload", "initial": 8, "cohort": "unsolved_at_initial",
                    "endpoint": "first_usable_total_call_capped_at_budget_plus_one",
                    "contrast": "covered_active_minus_covered_greedy"},
        "max_optimizer_calls": len(seeds)*(max(SIZES)+len(ARMS)*sum(budget-n for n in SIZES)),
        "max_audit_calls": audits*len(GENERATORS)*online_size,
    }


def covered_roster(problem, screened, size):
    """Round-robin over available configurations, using within-branch maximin."""
    groups = {}
    for candidate in screened:
        groups.setdefault(digest(candidate["discrete"]), []).append(candidate)
    remaining = [group for group in groups.values()]
    nearest = [[float("inf")]*len(group) for group in remaining]
    chosen = []
    while len(chosen) < size:
        for group, distances in zip(remaining, nearest):
            if group:
                index = max(range(len(group)), key=distances.__getitem__)
                point = group.pop(index)
                distances.pop(index)
                chosen.append(point)
                for i, candidate in enumerate(group):
                    distances[i] = min(distances[i], active_distance(problem, candidate, point))
                if len(chosen) == size:
                    return chosen
        if not any(remaining):
            raise ValueError("Screened pool smaller than the registered roster")
    return chosen


def rosters(problem, spec, seed):
    raw_initial = candidate_pool(problem, seed+1, spec["initial_proposals"])[0]
    eligible_initial = [x for x in raw_initial if passes_screen(problem, x)]
    if len(eligible_initial) < max(SIZES):
        raise ValueError("Screened initialization smaller than the registered prefix")
    initial = [eligible_initial[i] for i in nested_indices(problem, eligible_initial, max(SIZES))]
    # Exclude all common initialization designs, even the unused ones at n=8.
    raw = candidate_pool(problem, seed+2, spec["online_proposals"], map(digest, initial))[0]
    eligible = [x for x in raw if passes_screen(problem, x)]
    size = spec["online_size"]
    if len(eligible) < size:
        raise ValueError("Screened proposals cannot supply both equal-size rosters")
    return initial, {"standard": eligible[:size], "covered": covered_roster(problem, eligible, size)}


class FixedTimingHead:
    scale = MARGIN_SCALE
    model_class = EvidenceGP
    metadata_dimensions = 0

    def __init__(self, problem, inputs, history, seed, *, use_interval=True):
        native_divisor = group_divisors(inputs.shape[1]-self.metadata_dimensions,
                                       len(problem.space.continuous), *LENGTHS)
        self.divisor = np.r_[native_divisor, np.ones(self.metadata_dimensions)]
        evidence = [timing_feedback(record.to_dict(), problem.settings, use_interval=use_interval,
                                    allow_hidden_reserve_stop=True) for record in history]
        normalized = [replace(o, lower=None if o.lower is None else o.lower/self.scale,
                              upper=None if o.upper is None else o.upper/self.scale) for o in evidence]
        self.model = self.model_class(inputs/self.divisor, normalized, seed=seed,
                                      settings=MixedGPSettings(interval_noise=0., stable_tail_moments=True))

    def predict(self, query):
        result = self.model.predict(query/self.divisor)
        return {"mean": self.scale*result["mean"], "variance": self.scale**2*result["variance"]}


def choose(problem, seed, history, remaining, policy, *, fit_interval=True, score_interval=True):
    if all(map(supports_timing, history)):
        try:
            features = PrefixFeatures(problem, BASE, history, seed, latent=False)
            inputs = features.transform([record.candidate for record in history])
            query = features.transform(remaining)
            head = FixedTimingHead(problem, inputs, history, seed, use_interval=fit_interval)
            if policy == "active":
                edges = [deficit_edges(x, problem.settings) for x in remaining] if score_interval else None
                p, _, eligible, gains = score_information(head, query, remaining, edges)
                shortlisted = [remaining[i] for i in eligible]
                offset = select(shortlisted, gains)
                index = int(eligible[offset])
                return index, {"rule": policy, "p_complete": float(p[index]),
                               "expected_brier_reduction": float(gains[offset])}
            pred = head.predict(query)
            p = ndtr(-pred["mean"]/np.sqrt(np.maximum(pred["variance"], 1e-14)))
            index = select(remaining, p)
            return index, {"rule": policy, "p_complete": float(p[index])}
        except (np.linalg.LinAlgError, FloatingPointError, RuntimeError):
            reason = "numerical_failure"
    else:
        reason = "unsupported_timing_feedback"
    index, _ = coverage_pick(problem, remaining, history, seed, len(history))
    return index, {"rule": "coverage_fallback", "reason": reason}


def play(problem, seed, history, pool, budget, policy, acquire, *, selector=None):
    if selector is None:
        selector = choose
    history = list(history)
    remaining = list(pool)
    first = next((i for i, r in enumerate(history, 1) if r.usable_feasible), None)
    trace = []
    while first is None and len(history) < budget and remaining:
        index, decision = selector(problem, seed, history, remaining, policy)
        point = remaining.pop(index)
        record = acquire(point, len(history)+1)
        history.append(record)
        trace.append({"call": len(history), "id": digest(point), "usable": record.usable_feasible,
                      "outcome": discovery_outcome(record.to_dict()), **decision})
        if record.usable_feasible:
            first = len(history)
    return {"first_usable": first, "capped_first": first if first is not None else budget+1,
            "online_evaluations": len(trace), "trace": trace}


def campaign(spec, problem, obtain):
    rows = []
    for seed in spec["seeds"]:
        initialization, pools = rosters(problem, spec, seed)
        history = [obtain(point, f"{seed}/initial/{i}", "initial")
                   for i, point in enumerate(initialization, 1)]
        observed = {}

        def acquire(point, tag, role):
            identifier = digest(point)
            if identifier not in observed:
                observed[identifier] = obtain(point, f"{seed}/{tag}", role)
            return observed[identifier]

        row = {"seed": seed, "initial_ids": [digest(x) for x in initialization],
               "initial_outcomes": [discovery_outcome(x.to_dict()) for x in history],
               "pool_ids": {key: [digest(x) for x in value] for key, value in pools.items()},
               "configuration_counts": {key: dict(Counter(digest(x["discrete"]) for x in value))
                                        for key, value in pools.items()},
               "by_initial": {}}
        for size in spec["initial_sizes"]:
            arms = {}
            for generator, policy in ARMS:
                key = f"{generator}_{policy}"
                arms[key] = play(problem, seed, history[:size], pools[generator], spec["total_budget"],
                                 policy, lambda point, i: acquire(point, f"{size}/{key}/{i}", "online"))
            row["by_initial"][str(size)] = {"cold": not any(r.usable_feasible for r in history[:size]),
                                           "arms": arms}
        if seed in spec["audit_seeds"]:
            # Both full rosters are labeled only after every arm has finished.
            row["pool_audit"] = {}
            for generator, pool in pools.items():
                records = [acquire(point, f"audit/{generator}/{i}", "audit")
                           for i, point in enumerate(pool, 1)]
                row["pool_audit"][generator] = {
                    "outcomes": dict(Counter(discovery_outcome(r.to_dict()) for r in records)),
                    "usable_count": sum(r.usable_feasible for r in records),
                }
        rows.append(row)
        print(spec["name"], seed,
              {n: [a["capped_first"] for a in s["arms"].values()] for n, s in row["by_initial"].items()},
              flush=True)
    return rows


def contrasts(rows):
    names = [f"{g}_{p}" for g, p in ARMS]
    results = [row["arms"] for row in rows]
    paired = [
        [r[name]["capped_first"] for name in names] for r in results
    ]
    return {
        "n": len(rows),
        "hits": {name: sum(r[name]["first_usable"] is not None for r in results) for name in names},
        "mean_capped_first": {name: effect([r[name]["capped_first"] for r in results]) for name in names},
        "delta_calls": {
            "active_under_standard": effect([sa-sg for sg, sa, cg, ca in paired]),
            "active_under_covered": effect([ca-cg for sg, sa, cg, ca in paired]),
            "coverage_under_greedy": effect([cg-sg for sg, sa, cg, ca in paired]),
            "coverage_under_active": effect([ca-sa for sg, sa, cg, ca in paired]),
            "interaction": effect([(ca-cg)-(sa-sg) for sg, sa, cg, ca in paired]),
        },
        "coverage_fallbacks": {
            name: sum(s["rule"] == "coverage_fallback" for r in results for s in r[name]["trace"])
            for name in names},
    }


def summarize(rows):
    result = {"by_initial": {}, "audit": {}}
    for size in map(str, SIZES):
        all_rows = [row["by_initial"][size] for row in rows]
        common = [row["by_initial"][size] for row in rows if row["by_initial"][str(max(SIZES))]["cold"]]
        cold = [row for row in all_rows if row["cold"]]
        result["by_initial"][size] = {"all": contrasts(all_rows), "cold": contrasts(cold),
                                      "cold_at_16": contrasts(common)}
        audited = [row["pool_audit"] for row in rows
                   if "pool_audit" in row and row["by_initial"][size]["cold"]]
        pairs = [(r["standard"]["usable_count"], r["covered"]["usable_count"]) for r in audited]
        result["audit"][size] = {"n_cold": len(pairs),
                                 "standard_has_usable": sum(a > 0 for a, b in pairs),
                                 "covered_has_usable": sum(b > 0 for a, b in pairs),
                                 "covered_minus_standard_usable_count": effect([b-a for a, b in pairs])}
    return result


def run(spec, sha, output):
    output.mkdir(parents=True)
    problem = UsableAircraft(spec)
    save(output/"plan.json", {"source_sha": sha, "spec": spec, "contract": problem.contract_id})
    ledger = Ledger(output, spec["max_optimizer_calls"]+spec["max_audit_calls"])
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
            raise ValueError("Physical call differs from the registered decision")
        return record

    rows = campaign(spec, problem, replay)
    count = Counter(role for _, _, role in calls.values())
    if (position != len(calls) or count["audit"] > spec["max_audit_calls"]
            or count["initial"]+count["online"] > spec["max_optimizer_calls"]):
        raise ValueError("Ledger count differs from the registered budget")
    return {"status": "VERIFIED_AIRCRAFT_COVERAGE_FACTORIAL", "source_sha": sha,
            "plan": spec, "physical_calls": dict(count), "rcaide_mda_calls": 0,
            "summary": summarize(rows), "rows": rows}


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
        report = verify(args.input, sha)
        save(output, report)
        print(report["status"], report["plan"]["name"], report["physical_calls"])


if __name__ == "__main__":
    main()
