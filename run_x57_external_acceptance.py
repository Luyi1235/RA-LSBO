"""Qualify X-57 external ask/tell adapters and replay a frozen RA-LSBO prefix."""

import argparse
from dataclasses import asdict
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from ra_lsbo.budget import digest
from ra_lsbo import x57_cold_risk
from ra_lsbo.x57_cold_campaign import ObservationAdapter, update_native
from ra_lsbo.x57_cold_sampling import LAYOUTS, active_names, violations
from ra_lsbo.x57_cold_worker import Dispatcher, dump
from ra_lsbo.x57_external import ARMS, TOTAL_BUDGET, X57External, record_from_row
from ra_lsbo.x57_native_slot import configure_native_search
from run_x57_distance_diagnostic import ROOT, THREAD_VARS, read_events
from run_x57_engineering_energy_pilot import POLICY, code_sha, fit_head
from run_x57_formal_prefix import ARCHIVE, prepare
from run_x57_initialization_diagnostic import verified_hash

SEED = 20261127
INITIAL_HASH = "c067e55986727fe945c43f211cb317823124bd13aeb8d9eb552f2df8a2e1c7dd"
LEDGER_HASH = "990caac32365893a8100e3cc10660cd554e1e7a4391b4b467a74a570c77314df"
OUTCOMES = ("engineering_energy_stop", "numerical_nonconvergence",
            "completed_constraint_violation", "completed_usable")


def archived_prefix(archive):
    path = archive / f"seed-{SEED}" / "shared_initial.json"
    verified_hash(path, INITIAL_HASH)
    verified_hash(archive / "calls.jsonl", LEDGER_HASH)
    ledger = read_events(archive / "calls.jsonl")
    rows = [r for r in ledger if r["event"] == "settled" and r["seed"] == SEED]
    initial = json.loads(path.read_bytes())
    if (len(initial) != 8 or [r["candidate_id"] for r in initial] != [r["candidate_id"] for r in rows[:8]]
            or [r["feedback"] for r in initial] != [r["feedback"] for r in rows[:8]]
            or sum(r["feedback"]["usable"] for r in initial) != 1
            or sum(r["feedback"]["objective_kind"] == "exact" for r in initial) != 2):
        raise ValueError("Archive does not contain the sealed one-usable, two-target prefix")
    return initial, rows


def projection(problem, row):
    record = record_from_row(problem, row)
    obs = record.observation
    active = active_names(problem.space, problem.candidate_class(**record.candidate))
    if active != row["active_variables"]:
        raise ValueError("Candidate activation changed across evaluator and optimizer")
    return {"candidate_id": row["candidate_id"], "outcome": row["feedback"]["outcome"],
            "charged": row["charged"], "completion_label": obs.completion_label(),
            "usable": record.usable_feasible, "objective": asdict(obs.objective),
            "active_variables": active, "precheck": row["precheck"],
            "constraints": {name: {**asdict(value), "satisfied": value.constraint_satisfaction()}
                            for name, value in obs.constraints}}


def qualify(problem, initial, output, arm):
    output.mkdir()
    adapter = X57External(problem, arm, SEED, initial)
    worker = Dispatcher(problem, output, backend="mock", worker_root=ROOT, energy_stop=True,
                        mock_profile="formal-prefix", max_calls=len(OUTCOMES))
    activation = {}
    for layout in LAYOUTS:
        row = next(r for r in initial if r["candidate"]["discrete"]["prop_layout"] == layout)
        point = projection(problem, row)
        if ("propeller_count" in point["active_variables"]) != (layout == "leading_edge_distributed"):
            raise ValueError("Propeller-count activation disagrees with the frozen architecture")
        activation[layout] = point["active_variables"]
    rng = np.random.default_rng(SEED)
    for _ in range(1024):
        invalid = problem.space.sample(rng)
        if violations(problem.problem.precheck(invalid)):
            break
    else:
        raise ValueError("No negative precheck control was generated")
    try:
        worker.evaluate(invalid, seed=SEED, arm=arm, evaluation=0, online=0,
                        phase="negative_precheck_control", proposal={})
    except ValueError:
        pass
    else:
        raise ValueError("Precheck admitted a known invalid design")
    steps = []
    for online, expected in enumerate(OUTCOMES, 1):
        candidate, metadata = adapter.ask()
        row = worker.evaluate(problem.candidate_class(**candidate), seed=SEED, arm=arm,
                              evaluation=len(initial) + online, online=online, phase="mock_online",
                              proposal=metadata)
        state = projection(problem, row)
        if state["outcome"] != expected:
            raise ValueError(f"{arm} expected {expected}, received {state['outcome']}")
        adapter.tell(row)
        steps.append({**state, "proposal": metadata})
    checkpoint_file = output / "checkpoint.json"
    dump(checkpoint_file, adapter.checkpoint())
    resumed = X57External.resume(problem, json.loads(checkpoint_file.read_bytes()))
    following, _ = adapter.ask()
    resumed_following, _ = resumed.ask()
    if resumed_following != following:
        raise ValueError("Resumed and continuous external trajectories diverged")
    events = read_events(worker.journal)
    starts = [r for r in events if r["event"] == "worker_started"]
    settled = [r for r in events if r["event"] == "settled"]
    rejected = [r for r in events if r["event"] == "precheck_rejected"]
    if (len(starts) != len(OUTCOMES) or any(r["backend"] != "mock" or not r["charged"] for r in starts)
            or [r["candidate_id"] for r in starts] != [r["candidate_id"] for r in settled]
            or len(rejected) != 1 or rejected[0]["charged"] is not False
            or steps[0]["constraints"]["usable_energy"]["satisfied"] is not False
            or any(s["completion_label"] is not None or s["objective"]["kind"] != "missing" for s in steps[:2])
            or [s["usable"] for s in steps[2:]] != [False, True]):
        raise ValueError("Worker feedback or charging did not survive the ask/tell contract")
    return {"total_budget": adapter.total_budget, "shared_prefix_charged_again": 0,
            "mock_worker_starts": len(starts), "rcaide_starts": 0, "steps": steps,
            "layout_activity": activation, "uncharged_precheck_rejection": rejected[0]["reasons"],
            "resume_next_candidate_id": problem.space.key(problem.candidate_class(**following))}


def ra_replay(problem, initial, all_rows, output):
    native_module = importlib.import_module("lsbo.optimizer")
    truth = {}
    for row in all_rows:
        truth.setdefault(row["candidate_id"], []).append({
            "arm": row["arm"], "evaluation": row["evaluation"], "feedback": row["feedback"]})
    incumbent = next(r for r in initial if r["feedback"]["usable"])
    anchors = [problem.candidate_class(**incumbent["candidate"])]
    ordinary = x57_cold_risk.latent_pool
    cases = {}
    for arm in ("ra_lsbo_energy_sign", "ra_lsbo_energy_interval"):
        selected = {}
        for mode in ("frozen", "usable_neighborhood"):
            folder = output / f"{arm}-{mode}"
            folder.mkdir()
            torch.manual_seed(SEED)
            native = native_module.Optimizer(ObservationAdapter(problem.problem, native_module.Evaluation),
                                             native_module.Settings(seed=SEED, initial=8, budget=20), folder)
            configure_native_search(native, reserve=True, joint_final=True, force_weighted=True, audit=True)
            for row in initial:
                update_native(native, problem.candidate_class(**row["candidate"]), row, "initial", row["proposal"])
            audit = {}

            def pool(optimizer, phase, generator, limit):
                if mode == "frozen":
                    candidates = ordinary(optimizer, phase, generator, limit)
                else:
                    space = optimizer.space
                    anchor, = anchors
                    center, mask = space.unit(anchor), space.mask(anchor.discrete)
                    radius = optimizer.settings.initial_local_radius
                    candidates, seen = [], set()
                    for _ in range(30 * limit):
                        vector = np.clip(center + optimizer.rng.uniform(-radius, radius, len(center)) * mask, 0., 1.)
                        point = space.from_unit(vector, anchor.discrete)
                        key = space.key(point)
                        if key not in seen and optimizer.accepted(point):
                            candidates.append(point)
                            seen.add(key)
                            if len(candidates) == limit:
                                break
                    audit.update(radius_in_unit_coordinates=radius, anchor_candidate_id=incumbent["candidate_id"])
                if mode == "usable_neighborhood" and len(candidates) != limit:
                    raise ValueError("The incumbent neighborhood could not supply a matching-size candidate pool")
                audit["pool_sha256"] = digest([point.to_dict() for point in candidates])
                return candidates

            def fit(rep, bounds):
                head, _ = fit_head(rep, [o.candidate for o in native.observations], initial, SEED, 45,
                                   arm == "ra_lsbo_energy_interval", bounds)
                return head

            with patch.object(x57_cold_risk, "latent_pool", pool), torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED + 1)
                candidate, phase, info = x57_cold_risk.propose(
                    native, initial, POLICY[arm], native_module.mixed_candidate_pool, 512, fit_energy=fit)
            candidate_id = native.space.key(candidate)
            if mode == "frozen":
                recorded = next(r for r in all_rows if r["arm"] == arm and r["online"] == 1)
                if recorded["candidate_id"] != candidate_id:
                    raise ValueError(f"Frozen RA replay failed to reproduce archived {arm} proposal")
            selected[mode] = {**audit, "candidate": candidate.to_dict(), "candidate_id": candidate_id,
                              "phase": phase, "proposal": info, "archived_truth": truth.get(candidate_id)}
        cases[arm] = selected
    return {"seed": SEED, "archive_sha": ARCHIVE, "initial_count": 8, "initial_exact": 2,
            "initial_usable": 1, "initial_sha256": INITIAL_HASH,
            "observed_followup_rule": "Candidate hashes must match observed truth in this archived seed.",
            "arms": cases, "new_rcaide_calls": 0}


def native_checkpoints(problem, initial, directory):
    from ra_lsbo.x57_formal_native import ARMS as NATIVE_ARMS, CHECKPOINTS, FormalNative

    report = {}
    steps = CHECKPOINTS[len(initial)][1] - len(initial)
    for arm in NATIVE_ARMS:
        output = directory / arm
        adapter = FormalNative(problem, arm, SEED, initial, output)
        worker = Dispatcher(problem, output, backend="mock", worker_root=ROOT,
                            mock_profile="formal-prefix", max_calls=steps, energy_stop=True)
        for online in range(1, steps + 1):
            candidate, phase, proposal = adapter.ask()
            row = worker.evaluate(candidate, seed=SEED, arm=arm, evaluation=len(initial) + online,
                                  online=online, phase=phase, proposal=proposal)
            if adapter.tell(row):
                worker.event("native_missing_objective_stop", call_id=row["call_id"], charged=False)
        path = output / "checkpoint.json"
        dump(path, adapter.checkpoint())
        resumed = FormalNative.resume(problem, json.loads(path.read_bytes()), output / "replay")
        expected, phase, _ = adapter.ask()
        candidate, resumed_phase, _ = resumed.ask()
        if expected != candidate or phase != resumed_phase or worker.count != steps:
            raise ValueError(f"Resume diverged at the frozen checkpoint for {arm}")
        report[arm] = {"budget": adapter.settings.budget, "boundary": len(initial) + steps,
                       "mock_starts": worker.count, "resume_candidate_id": problem.space.key(candidate),
                       "native_stop_positions": resumed.native_stop_positions}
    dump(directory / "native_checkpoints.json", {"code_sha": code_sha(), "new_rcaide_calls": 0, "arms": report})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--qualification-plan", type=Path, required=True)
    parser.add_argument("--baseline-deps", type=Path, action="append", default=[])
    parser.add_argument("--component", choices=("adapters", "ra-replay", "native-checkpoints", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in THREAD_VARS:
        os.environ[name] = "1"
    args.workers, args.execute_rcaide = 12, False
    args.output = args.output.resolve()
    _, problems, _ = prepare(args)
    problem = problems[45]
    initial, rows = archived_prefix(args.archive)
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.extend(str(path.resolve()) for path in args.baseline_deps)
    if args.component == "native-checkpoints":
        native_checkpoints(problem, initial, args.output)
        return
    if args.component in ("ra-replay", "all"):
        experiment = ra_replay(problem, initial, rows, args.output)
        dump(args.output / "ra_replay.json", {**experiment, "code_sha": code_sha()})
    if args.component == "ra-replay":
        return
    audit = {}
    for arm in ARMS:
        audit[arm] = qualify(problem, initial, args.output / arm, arm)
    compared = [audit[arm]["steps"] for arm in ARMS]
    fields = ("outcome", "charged", "completion_label", "usable")
    if [tuple(s[k] for k in fields) for s in compared[0]] != [tuple(s[k] for k in fields) for s in compared[1]]:
        raise ValueError("External adapters diverged on the shared worker contract")
    dump(args.output / "acceptance.json", {"status": "complete", "code_sha": code_sha(),
         "archive_sha": ARCHIVE, "python": sys.version, "version": {name: version(name) for name in (
             "sb-arch-opt", "pymoo", "smt", "numba", "numpy", "scipy", "scikit-learn")},
         "shared_archive_projection": [projection(problem, r) for r in initial],
         "adapters": audit, "new_rcaide_calls": 0,
         "archsbo_contract": "Native hidden-constraint viability filters incomplete output rows before "
                             "the objective and numerical constraint GPs; known sign remains in tell and the ledger.",
         "logcei_contract": "Objective GP consumes exact objectives. An unknown completion with a verified "
                            "energy sign contributes to the energy classifier only."})
    print(f"External adapters qualified | RCAIDE starts=0 | output={args.output}", flush=True)


if __name__ == "__main__":
    main()
