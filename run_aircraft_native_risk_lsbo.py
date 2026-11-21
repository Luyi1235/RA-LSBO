"""Compare early-termination feedback within LSBO, with ArchSBO as an external control."""

import argparse
from functools import partial
import importlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr

from aircraft_timing_information import score_information, supports_timing
from aircraft_timing_interval_value import deficit_edges
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.core.runner import Ledger, save
from run_aircraft_coverage_factorial import FixedTimingHead
from run_aircraft_external_full_budget import (
    SCENARIOS, UsableAircraft, compare, discovery_outcome, evidence_hashes, passes_exact_screen,
    plan as full_budget_plan, read, read_ledger, rosters, structural_margins, trajectory_metrics,
)
from run_aircraft_ra_lsbo_comparison import FullBudgetLSBO
from run_aircraft_risk_cold_start import select, source_commit


INITIAL = 8
CANDIDATES = 64
ORIGINAL = "lsbo_native"
EXTERNAL = "archsbo_screened"
ARMS = (ORIGINAL, "ra_sign_greedy", "ra_interval_greedy", "ra_interval_active", EXTERNAL)
RISK_ARMS = ARMS[1:-1]
CONTRASTS = (("ra_interval_greedy", "ra_sign_greedy"), ("ra_interval_active", "ra_interval_greedy"),
             ("ra_interval_greedy", ORIGINAL), ("ra_interval_greedy", EXTERNAL))
PROFILES = ("smoke", "full-pilot")


def plan(scenario, profile):
    offset = 0 if scenario == "payload" else 10_000
    seed, budget = (29060100+offset, 12) if profile == "smoke" else (29200100+offset, 48)
    base = full_budget_plan(scenario, "full-pilot")
    del base["pooled_arms"], base["logcei_adapter"]
    return {**base, "name": f"{profile}-{scenario}", "analysis_role": profile, "seeds": [seed],
            "initial_sizes": [INITIAL], "total_budget": budget, "horizon": budget-INITIAL,
            "calls_per_seed": INITIAL+len(ARMS)*(budget-INITIAL), "arms": list(ARMS),
            "native_gp_steps": 45, "risk_candidate_count": CANDIDATES,
            "pool_policy": "LSBO grammar-decoded plus native mixed proposals in the risk-aware cold stage",
            "post_feasible_policy": "all LSBO arms continue through their own native LSBO engine",
            "primary": {"scenario": "headwind", "initial": INITIAL, "cohort": "cold",
                        "contrast": "ra_interval_greedy_minus_ra_sign_greedy",
                        "endpoint": "first_usable_total_call_capped_at_budget_plus_one"}}


def engineering_screen(problem, point):
    return {**problem.precheck(point), **structural_margins(problem, point)}


class RiskAwareLSBO(FullBudgetLSBO):
    def __init__(self, *args, arm, **kwargs):
        super().__init__(*args, **kwargs)
        self.arm = arm

    def _propose(self, history):
        if any(record.usable_feasible for record in history):
            return self.engine.propose()
        return self.engine.propose(selection=partial(self._cold_proposal, history=history))

    def _cold_proposal(self, native, phase, *, history):
        trusted = [record for record in history if supports_timing(record)]
        if not trusted:
            candidate, info = native.propose(phase)
            return candidate, {**info, "risk_fallback": "no_supported_completion_feedback"}
        rep = native.representation
        if rep.version == 0 or len(native.observations)-native.last_representation_update >= native.settings.update_every:
            epochs = native.settings.pretrain_epochs if rep.version == 0 else native.settings.update_epochs
            rep.fit(native.observations, epochs)
            native.last_representation_update = len(native.observations)
            torch.save(rep.checkpoint(), native.output/"representation.pt")
        pool, decoded = self._native_cold_pool(native, phase)
        if not pool:
            candidate, info = native.propose(phase)
            return candidate, {**info, "risk_fallback": "no_screened_cold_candidates"}
        to_dict = [candidate.to_dict() for candidate in pool]
        inputs = rep.features([self.problem.candidate_class(**record.candidate) for record in trusted]).numpy()
        query = rep.features(pool).numpy()
        use_interval = self.arm != "ra_sign_greedy"
        try:
            head = FixedTimingHead(self.problem, inputs, trusted, self.seed, use_interval=use_interval)
            if self.arm == "ra_interval_active":
                edges = [deficit_edges(point, self.problem.settings) for point in to_dict]
                p, _, eligible, gains = score_information(head, query, to_dict, edges)
                pick = select([to_dict[i] for i in eligible], gains)
                index, gain = int(eligible[pick]), float(gains[pick])
            else:
                moments = head.predict(query)
                p = ndtr(-moments["mean"]/np.sqrt(np.maximum(moments["variance"], 1e-14)))
                if not np.isfinite(p).all():
                    raise FloatingPointError("Invalid LSBO completion posterior")
                index, gain = select(to_dict, p), None
        except (np.linalg.LinAlgError, FloatingPointError, RuntimeError):
            candidate, info = native.propose(phase)
            return candidate, {**info, "risk_fallback": "invalid_completion_posterior"}
        return pool[index], {"source": "lsbo_latent_completion_risk", "feedback": "interval" if use_interval else "sign",
                             "policy": "information" if self.arm == "ra_interval_active" else "greedy",
                             "p_complete": float(p[index]), "expected_brier_reduction": gain,
                             "candidate_count": len(pool), "candidate_list_sha256": digest(to_dict),
                             "latent_decoded_candidates": decoded,
                             "representation_version": rep.version, "timing_observations": len(trusted)}

    def _native_cold_pool(self, native, phase):
        factory = importlib.import_module(type(native).__module__).mixed_candidate_pool
        rep, rng, space = native.representation, native.rng, native.space
        target = native.coverage_target(phase)
        limit = self.profile["risk_candidate_count"]
        latents = np.vstack([rep.latent(row.candidate.discrete).numpy() for row in native.observations])
        low, high = latents.min(axis=0)-.5, latents.max(axis=0)+.5
        candidates, keys = [], set()
        for _ in range(limit):
            if len(candidates) >= limit//2:
                break
            discrete = rep.decode(rng.uniform(low, high), rng, stochastic=True)
            if target and any(discrete[name] != value for name, value in target.items()):
                continue
            point = space.sample(rng, discrete)
            key = space.key(point)
            if key not in keys and native.accepted(point):
                candidates.append(point)
                keys.add(key)
        return candidates+factory(native, limit-len(candidates), target, keys), len(candidates)


def run(spec, seed, sha, output):
    from run_aircraft_archsbo_screened import ARCHSBO_VERSION, screened_trajectory

    if version("sb-arch-opt") != ARCHSBO_VERSION:
        raise ValueError("Unexpected screened ArchSBO version")
    torch.set_num_threads(1)
    problem = UsableAircraft(spec)
    initial = rosters(problem, spec, seed)[0][:INITIAL]
    output.mkdir(parents=True)
    versions = {name: version(name) for name in ("botorch", "gpytorch", "numpy", "scipy", "sb-arch-opt",
                                                 "pymoo", "smt", "scikit-learn", "numba")}
    versions["torch_runtime"] = str(torch.__version__)
    save(output/"plan.json", {"spec": spec, "seed": seed, "source_sha": sha, "contract": problem.contract_id,
                             "versions": versions})
    ledger = Ledger(output, spec["calls_per_seed"])

    def pay(point, tag, role):
        if not passes_exact_screen(problem, point):
            raise ValueError("The candidate failed the common engineering screen")
        return ledger.evaluate(problem, point, tag, role)

    prefix = [pay(point, f"initial/{i}", "initial") for i, point in enumerate(initial, 1)]
    for arm in ARMS:
        def acquire(point, step):
            return pay(point, f"{arm}/{step}", "online")
        if arm == EXTERNAL:
            trace = screened_trajectory(problem, prefix, seed, spec["horizon"], acquire)["steps"]
        else:
            torch.manual_seed(seed)
            factory = FullBudgetLSBO if arm == ORIGINAL else partial(RiskAwareLSBO, arm=arm)
            engine = factory(problem, BASE, seed, spec, output/arm, precheck=partial(engineering_screen, problem))
            history, trace = list(prefix), []
            for step in range(1, spec["horizon"]+1):
                point, info = engine.propose(history, [], step-1)
                record = acquire(point, step)
                engine.accept(record)
                trace.append({"step": step, "candidate_id": digest(point),
                              "usable": record.usable_feasible, "outcome": discovery_outcome(record.to_dict()),
                              "proposal": info["original_proposal"], "lsbo_phase": info["original_phase"]})
                history.append(record)
        save(output/(arm+".json"), trace)
        print(spec["name"], seed, arm, len(trace), ledger.count, flush=True)
    if ledger.count != spec["calls_per_seed"]:
        raise ValueError("Evaluation budget differs from the registered plan")
    save(output/"seal.json", evidence_hashes(output))


def verify(folder, spec, sha):
    seed = spec["seeds"][0]
    problem = UsableAircraft(spec)
    header = read(folder/"plan.json")
    if (header["spec"] != spec or header["seed"] != seed or header["source_sha"] != sha
            or header["contract"] != problem.contract_id or read(folder/"seal.json") != evidence_hashes(folder)):
        raise ValueError("The run does not match the registered sealed plan")
    calls = read_ledger(folder, problem.contract_id)
    if len(calls) != spec["calls_per_seed"]:
        raise ValueError("Unexpected physical evaluation count")
    position = 0

    def settled(tag, role):
        nonlocal position
        position += 1
        number, record, actual_role = calls[tag]
        if (number, actual_role) != (position, role) or not passes_exact_screen(problem, record.candidate):
            raise ValueError("A paid call failed ledger order or engineering checks")
        return record

    prefix = [settled(f"initial/{i}", "initial") for i in range(1, INITIAL+1)]
    if [record.candidate for record in prefix] != rosters(problem, spec, seed)[0][:INITIAL]:
        raise ValueError("Unexpected shared initialization")
    arms, candidate_lists = {}, {}
    for arm in ARMS:
        history = list(prefix)
        seen = {record.observation.design_id for record in prefix}
        trace = read(folder/(arm+".json"))
        if len(trace) != spec["horizon"]:
            raise ValueError("An arm did not spend its registered budget")
        risk_calls, fallbacks = 0, 0
        for i, step in enumerate(trace, 1):
            record = settled(f"{arm}/{i}", "online")
            key = record.observation.design_id
            if key in seen or (
                step["step"], step["candidate_id"], step["outcome"], step["usable"]
            ) != (i, key, discovery_outcome(record.to_dict()), record.usable_feasible):
                raise ValueError("A proposal does not match its settled observation")
            if arm == EXTERNAL:
                if (step["paid_calls"], step["candidate"]) != (i, record.candidate):
                    raise ValueError("ArchSBO paid proposal does not match its ledger")
            else:
                proposal = step["proposal"]
                if proposal["source"] == "lsbo_latent_completion_risk":
                    if arm not in RISK_ARMS or any(old.usable_feasible for old in history):
                        raise ValueError("A cold completion policy used an unregistered phase")
                    feedback = "sign" if arm == "ra_sign_greedy" else "interval"
                    policy = "information" if arm == "ra_interval_active" else "greedy"
                    if (proposal["feedback"], proposal["policy"]) != (feedback, policy):
                        raise ValueError("The receipt differs from the registered cold-start policy")
                    prefix_key = digest([old.observation.design_id for old in history])
                    list_key = proposal["candidate_list_sha256"]
                    if candidate_lists.setdefault(prefix_key, list_key) != list_key:
                        raise ValueError("Feedback ablations generated different lists for the same history")
                    risk_calls += 1
                fallbacks += int("risk_fallback" in proposal)
            seen.add(key)
            history.append(record)
        arms[arm] = {**trajectory_metrics(history, spec["total_budget"]),
                     "cold_risk_calls": risk_calls, "cold_risk_fallbacks": fallbacks}
    row = {"arms": arms, "cold": not any(record.usable_feasible for record in prefix)}
    return {"status": "VERIFIED_NATIVE_RISK_LSBO", "source_sha": sha, "spec": spec, "seed": seed,
            "paid_calls": position, "rcaide_mda_calls": 0, **row, "versions": header["versions"],
            "contrasts": {f"{a}_minus_{b}": compare([row], a, b) for a, b in CONTRASTS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plan", "run", "verify"), required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
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
        run(spec, spec["seeds"][0], sha, output)
    else:
        if args.input is None or args.input.resolve() in (output, *output.parents):
            parser.error("Provide --input and write outside the sealed run")
        report = verify(args.input, spec, sha)
        save(output, report)
        print(report["status"], spec["name"], report["paid_calls"], flush=True)


if __name__ == "__main__":
    main()
