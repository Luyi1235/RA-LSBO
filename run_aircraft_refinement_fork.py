"""Shared-history ablation of energy censoring during native LSBO refinement."""

from functools import partial
from importlib.metadata import version

import numpy as np
import torch

from aircraft_censored_refinement import CensoredEnergyRefinement, fit_energy_update
from aircraft_timing_energy_value import energy_evidence
from aircraft_timing_information import supports_timing
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.core.runner import Ledger, save
from run_aircraft_native_risk_lsbo import (INITIAL, RiskAwareLSBO, UsableAircraft, discovery_outcome,
                                         engineering_screen, evidence_hashes, passes_exact_screen, read,
                                         read_ledger, rosters, trajectory_metrics)


PROFILE = "energy-refinement-fork"
CONTROL = "ra_refine_interval"
TREATMENT = "ra_refine_energy_bounds"
ARMS = (CONTROL, TREATMENT)
FORK_FEASIBLE = 3


class ForkLSBO(RiskAwareLSBO):
    use_energy_bounds = False

    def _refinement_energy(self, surrogate, inputs, trusted, head, state):
        if not self.use_energy_bounds:
            return surrogate
        try:
            update = fit_energy_update(trusted, inputs, head.divisor, self.seed)
        except (np.linalg.LinAlgError, FloatingPointError, RuntimeError):
            state["energy_update"] = "numerical_fallback"
            return surrogate
        if update is None:
            state["energy_update"] = "no_mixed_energy_evidence"
            return surrogate
        prior, updated, count = update
        state.update(energy_update="incremental_probability", energy_lower_bounds=count)
        return CensoredEnergyRefinement(surrogate, prior, updated)


def reached_fork(history):
    return sum(record.usable_feasible for record in history) >= FORK_FEASIBLE


def receipt(point, record, info, evaluation):
    return {"evaluation": evaluation, "candidate_id": digest(point),
            "outcome": discovery_outcome(record.to_dict()),
            "proposal": info["original_proposal"]}


def run_fork(spec, seed, sha, output):
    torch.set_num_threads(1)
    problem = UsableAircraft(spec)
    initial = rosters(problem, spec, seed)[0][:INITIAL]
    output.mkdir(parents=True)
    versions = {name: version(name) for name in ("botorch", "gpytorch", "numpy", "scipy", "torch")}
    save(output/"plan.json", {"spec": spec, "seed": seed, "source_sha": sha, "contract": problem.contract_id,
                             "versions": versions})
    ledger = Ledger(output, spec["calls_per_seed"])

    def pay(point, tag, role):
        if not passes_exact_screen(problem, point):
            raise ValueError("The candidate failed the common engineering screen")
        return ledger.evaluate(problem, point, tag, role)

    def engine(path):
        torch.manual_seed(seed)
        return ForkLSBO(problem, BASE, seed, spec, path, arm=CONTROL,
                        precheck=partial(engineering_screen, problem))

    prefix = [pay(point, f"initial/{i}", "initial") for i, point in enumerate(initial, 1)]
    control = engine(output/CONTROL)
    shared, shared_trace = list(prefix), []
    budget = spec["total_budget"]
    while len(shared) < budget and not reached_fork(shared):
        point, info = control.propose(shared, [], len(shared)-INITIAL)
        index = len(shared)+1
        record = pay(point, f"shared/{index}", "shared")
        control.accept(record)
        shared_trace.append(receipt(point, record, info, index))
        shared.append(record)

    save(output/"shared.json", shared_trace)
    for arm in ARMS:
        history, trace = list(shared), []
        if len(history) < budget:
            branch = control
            if arm == TREATMENT:
                branch = engine(output/TREATMENT)
                # Recreate the exact optimizer and RNG state from paid observations.
                replay = list(prefix)
                for recorded, expected in zip(shared[INITIAL:], shared_trace):
                    point, info = branch.propose(replay, [], len(replay)-INITIAL)
                    if receipt(point, recorded, info, len(replay)+1) != expected or point != recorded.candidate:
                        raise ValueError("Native proposals diverged during the shared-prefix replay")
                    branch.accept(recorded)
                    replay.append(recorded)
                branch.use_energy_bounds = True
            while len(history) < budget:
                point, info = branch.propose(history, [], len(history)-INITIAL)
                index = len(history)+1
                record = pay(point, f"{arm}/{index}", "online")
                branch.accept(record)
                trace.append(receipt(point, record, info, index))
                history.append(record)
        save(output/(arm+".json"), trace)
    if ledger.count != 2*budget-len(shared):
        raise ValueError("Incorrect physical evaluation count after the shared fork")
    save(output/"seal.json", evidence_hashes(output))


def verify_fork(folder, spec, sha):
    seed = spec["seeds"][0]
    budget = spec["total_budget"]
    problem = UsableAircraft(spec)
    header = read(folder/"plan.json")
    if (header["spec"], header["seed"], header["source_sha"], header["contract"]) != (
            spec, seed, sha, problem.contract_id) or read(folder/"seal.json") != evidence_hashes(folder):
        raise ValueError("The sealed fork identity differs from the plan")
    ledger = read_ledger(folder, problem.contract_id)
    position = 0

    def settled(tag, role):
        nonlocal position
        position += 1
        number, record, actual_role = ledger[tag]
        if (number, actual_role) != (position, role) or not passes_exact_screen(problem, record.candidate):
            raise ValueError("Invalid ledger order or engineering screen")
        return record

    def check_step(step, tag, role, history):
        index = len(history)+1
        record = settled(f"{tag}/{index}", role)
        key = record.observation.design_id
        outcome = discovery_outcome(record.to_dict())
        if ((step["evaluation"], step["candidate_id"], step["outcome"]) != (index, key, outcome)
                or any(old.observation.design_id == key for old in history)):
            raise ValueError("Invalid proposal identity in the fork trajectory")
        return record

    initial = [settled(f"initial/{i}", "initial") for i in range(1, INITIAL+1)]
    if [record.candidate for record in initial] != rosters(problem, spec, seed)[0][:INITIAL]:
        raise ValueError("Unexpected initialization")
    shared = list(initial)
    for step in read(folder/"shared.json"):
        if reached_fork(shared) or len(shared) >= budget:
            raise ValueError("The shared prefix extended past the registered branch point")
        shared.append(check_step(step, "shared", "shared", shared))
    if len(shared) < budget and not reached_fork(shared):
        raise ValueError("The shared prefix stopped before the registered branch point")

    results = {}
    for arm in ARMS:
        history = list(shared)
        applied = fallbacks = refinement_calls = 0
        for step in read(folder/(arm+".json")):
            record = check_step(step, arm, "online", history)
            risk = step["proposal"].get("refinement_risk", {})
            if risk.get("mode") == "risk_weighted_native_search":
                if risk.get("feedback") != "interval":
                    raise ValueError("The fork used a different completion model")
                refinement_calls += 1
            mode = risk.get("energy_update")
            if mode is not None and (arm != TREATMENT or risk.get("mode") != "risk_weighted_native_search"):
                raise ValueError("The energy update was applied outside its registered arm")
            if mode == "incremental_probability":
                bounds = sum(ev is not None and ev[0].kind == "interval"
                             for ev in (energy_evidence(old) for old in history if supports_timing(old)))
                if not bounds or risk["energy_lower_bounds"] != bounds:
                    raise ValueError("Incorrect energy evidence receipt")
                applied += 1
            elif mode in {"no_mixed_energy_evidence", "numerical_fallback"}:
                fallbacks += 1
            elif mode is not None or (arm == TREATMENT and risk.get("mode") == "risk_weighted_native_search"):
                raise ValueError("Unregistered refinement update receipt")
            history.append(record)
        if len(history) != budget:
            raise ValueError("A fork arm did not spend its registered budget")
        target = next((i for i, record in enumerate(history, 1) if record.usable_feasible
                       and record.observation.objective.value <= spec["energy_target_Wh"]), None)
        results[arm] = {**trajectory_metrics(history, budget),
                        "refinement_risk_calls": refinement_calls, "first_energy_target": target,
                        "energy_update_calls": applied, "energy_update_fallbacks": fallbacks,
                        "quality_by_budget": {str(b): trajectory_metrics(history[:b], b)["best_usable_Wh"]
                                              for b in spec["quality_budgets"]}}
    if position != len(ledger) or position != 2*budget-len(shared):
        raise ValueError("Incorrect or unaccounted paid evaluations")
    return {"seed": seed, "cold": not any(row.usable_feasible for row in initial),
            "fork_evaluation": len(shared) if len(shared) < budget else None,
            "shared_target_reached": any(row.usable_feasible and
                                        row.observation.objective.value <= spec["energy_target_Wh"] for row in shared),
            "shared_history_sha256": digest([row.to_dict() for row in shared]),
            "arms": results, "paid_calls": position, "versions": header["versions"]}
