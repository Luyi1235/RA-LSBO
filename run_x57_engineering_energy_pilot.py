"""Replay the X-57 energy constraint and run a paired 45 nmi online pilot."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import subprocess
from threading import Event, Lock
import time

from run_x57_distance_diagnostic import (
    CONFIG, QUALIFICATION_COMMIT, QUALIFICATION_LEDGER_SHA256, QUALIFICATION_PLAN_SHA256,
    ROOT, THREAD_VARS, TIMEOUT, elapsed_concurrency, prefix_energy, read_events, run_wave,
)
from run_x57_initialization_diagnostic import verified_hash
from run_x57_paired_distance_feedback import (
    DISTANCE_CONFIGS, INITIALIZATION_COMMIT, INITIALIZATION_LEDGER_SHA256,
)

import numpy as np
from scipy.special import ndtr
import torch

from ra_lsbo.budget import digest
from ra_lsbo.x57_cold_campaign import ObservationAdapter, identity, update_native
from ra_lsbo.x57_cold_risk import propose
from ra_lsbo.x57_cold_sampling import LAYOUT, LAYOUTS, freeze_roster, violations
from ra_lsbo.x57_cold_worker import Dispatcher, dump, sha256
from ra_lsbo.x57_energy_surrogate import (
    CONSTRAINT, SETTINGS, EnergyAcquisition, EnergyPosterior, energy_training_row,
)
from ra_lsbo.x57_live import X57LiveProblem, evaluator_files

BASE = "9cb135a238bd4de2b2df7945f8c2432d231af6c0"
PAIRED_COMMIT = "1aea06158e6e4ac8932b1c1269d81e59902214f1"
PAIRED_HASH = "8863e3ddcdf74ec854239a9617f06fdf16100d194539036a7ce167b565f26b6f"
PILOT_SEED, POOL_SEED = 20261123, 20261124
INITIAL, ONLINE, WORKERS = 8, 12, 12
POLICY = {"lsbo": "lsbo_full_budget", "ra_lsbo_energy_sign": "structured_sign",
          "ra_lsbo_energy_interval": "structured_sign"}
ARMS = tuple(POLICY)


def verify_inputs(args):
    files = {
        "qualification": (args.qualification_ledger, QUALIFICATION_LEDGER_SHA256, QUALIFICATION_COMMIT),
        "paired": (args.paired_archive / "calls.jsonl", PAIRED_HASH, PAIRED_COMMIT),
        "initialization_40": (args.initialization_ledger, INITIALIZATION_LEDGER_SHA256, INITIALIZATION_COMMIT),
    }
    sources = {name: {"sha256": verified_hash(path, checksum), "commit": commit}
               for name, (path, checksum, commit) in files.items()}
    ledgers = {name: read_events(path) for name, (path, _, _) in files.items()}
    verified_hash(args.qualification_plan, QUALIFICATION_PLAN_SHA256)
    qualification = json.loads(args.qualification_plan.read_bytes())
    base = args.qualification_plan.parent / "physical-base"
    worker = qualification["worker"]["worker_sha256"]
    if evaluator_files(base) != worker:
        raise ValueError("Evaluator differs from the frozen qualification manifest")
    configs = {55: CONFIG, **{n: args.paired_archive / "configs" / f"x57_qualification_{n}_nmi.json"
                             for n in DISTANCE_CONFIGS}}
    for distance, checksum in DISTANCE_CONFIGS.items():
        verified_hash(configs[distance], checksum)
    expected55 = dict(json.loads(configs[50].read_bytes()))
    expected55["mission"]["cruise_distance_nmi"] = 55.
    if json.loads(CONFIG.read_bytes()) != expected55:
        raise ValueError("50 and 55 nmi configs differ beyond the mission distance")
    excluded = {row["candidate_id"] for ledger in ledgers.values() for row in ledger
                if row["event"] == "worker_started"}
    return base, worker, qualification, configs, sources, ledgers, excluded


def points(problem, seed, excluded, count):
    roster, audit = freeze_roster(problem.problem, seed, exclude_keys=excluded)
    records = [{"position": i, "task_id": f"{i:02d}", "candidate": point.to_dict(),
                "candidate_id": problem.space.key(point), "layout": point.discrete[LAYOUT],
                "precheck": problem.problem.precheck(point)} for i, point in enumerate(roster[:count], 1)]
    keys = {row["candidate_id"] for row in records}
    if (len(keys) != count or keys & excluded
            or [row["layout"] for row in records] != list(LAYOUTS) * (count // 2)
            or any(violations(row["precheck"]) for row in records)):
        raise ValueError("Invalid frozen X-57 candidate roster")
    return roster[:count], records, audit


def replay_rows(ledger, commit, distance):
    def row_key(row):
        return row.get("task_id", ""), row["call_id"]

    def relevant(row):
        return distance == 55 or row["proposal"]["distance_nmi"] == distance

    starts = Counter(row_key(r) for r in ledger if r["event"] == "worker_started" and relevant(r))
    settlements = [r for r in ledger if r["event"] == "settled" and relevant(r)]
    if starts != Counter(row_key(r) for r in settlements):
        raise ValueError(f"Every historical start must have exactly one {distance} nmi settlement")
    for row in settlements:
        if starts[row_key(row)] != 1 or row["charged"] is not True:
            raise ValueError(f"Historical call charge mismatch: {row_key(row)}")
        yield {**row, "source_id": ":".join((commit, *filter(None, row_key(row))))}


def fit_head(rep, candidates, rows, seed, distance, use_interval, bounds=None):
    projected = [energy_training_row(row, row["source_id"], distance, prefix_energy, use_interval)
                 for row in rows]
    evidence, prefixes = zip(*projected)
    return EnergyPosterior(rep, candidates, evidence, seed, bounds), prefixes


def offline(args, base, configs, sources, ledgers, excluded):
    start = time.perf_counter()
    pool_problem = X57LiveProblem(base, configs[50], args.output / "offline")
    pool, roster, _ = points(pool_problem, POOL_SEED, excluded, 32)
    native = importlib.import_module("lsbo.optimizer")
    records = importlib.import_module("lsbo.records")
    report = {"code_sha": code_sha(), "sources": sources, "candidate_pool_seed": POOL_SEED,
              "pool_sha256": digest(roster), "pool": roster, "new_worker_calls": 0, "distances": {}}
    for distance, source in ((50, "paired"), (55, "qualification")):
        problem = X57LiveProblem(base, configs[distance], args.output / "offline")
        rows = list(replay_rows(ledgers[source], sources[source]["commit"], distance))
        candidates = [problem.candidate_class(**row["candidate"]) for row in rows]
        if any(problem.space.key(c) != r["candidate_id"] for c, r in zip(candidates, rows)):
            raise ValueError("Candidate identity mismatch during offline replay")
        if any(problem.problem.precheck(c) != r["precheck"] for c, r in zip(pool, roster)):
            raise ValueError("Frozen candidate pool precheck differs between distances")
        adapter = ObservationAdapter(problem.problem, native.Evaluation)
        observations = []
        for i, (row, point) in enumerate(zip(rows, candidates), 1):
            adapter.pending = row
            value = adapter.evaluate(point)
            if (value.solver_success != (row["feedback"]["completion"] is True)
                    or value.objective != row["feedback"]["objective_Wh"]):
                raise ValueError("Native replay alters completion or objective observations")
            observations.append(records.Observation(i, point, value, "archived"))
        rep = importlib.import_module("lsbo.representation").Representation(problem.space, seed=POOL_SEED)
        settings = native.Settings(seed=POOL_SEED)
        rep.fit(observations, settings.pretrain_epochs)
        surrogate = importlib.import_module("lsbo.surrogate").Surrogate(
            observations, rep, problem.required_constraints, POOL_SEED, steps=settings.gp_steps,
            use_prior=settings.use_prior, analytical_constraints=problem.analytical_constraints,
            solver_domain_metric=(getattr(problem.problem, "solver_domain_metric", None)
                                  if settings.use_solver_domain else None))
        features = rep.features(pool)
        bounds = surrogate.offset, surrogate.scale
        replays, fits, projections = {}, {}, {}
        for arm, interval in (("sign", False), ("interval", True)):
            head, prefixes = fit_head(rep, candidates, rows, POOL_SEED, distance, interval, bounds)
            acquisition = EnergyAcquisition(surrogate, head)
            probe = features.clone().requires_grad_(True)
            log_p = head.log_probability(probe)
            gradient, = torch.autograd.grad(log_p.sum(), probe)
            prediction = head.model.predict(((features - bounds[0]) / bounds[1]).numpy())
            probability = log_p.detach().exp().numpy()
            np.testing.assert_allclose(probability, ndtr(-prediction["mean"] /
                                                       np.maximum(prediction["variance"], 1e-14)**.5),
                                       rtol=1e-10, atol=1e-12)
            assert torch.isfinite(gradient).all()
            scores = acquisition.score(features)
            assert np.isfinite(scores).all()
            engineering = acquisition.engineering_probability(features)
            order = sorted(range(len(pool)), key=lambda i: (-scores[i], roster[i]["candidate_id"]))
            replays[arm] = [{"p_usable_energy": float(probability[i]),
                             "p_engineering_feasible": float(engineering[i]),
                             "log_acquisition": float(scores[i]), "rank": order.index(i) + 1}
                            for i in range(len(pool))]
            fits[arm] = head.audit()
            projections[arm] = [(asdict(value), value.constraint_satisfaction(), prefix)
                                for value, prefix in zip(head.model.observations, prefixes)]
        aligned = []
        for row, sign, interval in zip(rows, projections["sign"], projections["interval"]):
            sign_obs, sign_satisfied, prefix = sign
            interval_obs, interval_satisfied, other_prefix = interval
            assert sign_satisfied == interval_satisfied and prefix == other_prefix
            if sign_obs["kind"] == "interval":
                assert sign_obs["lower"] == 0 < interval_obs["lower"]
                # Endpoint inclusion expresses the same proven violation; EP uses only bounds.
                assert {**sign_obs, "lower": interval_obs["lower"], "lower_inclusive": True} == interval_obs
            else:
                assert sign_obs == interval_obs
            common = {k: row["feedback"][k] for k in ("completion", "outcome", "objective_kind", "objective_Wh")}
            aligned.append({"source_id": row["source_id"], "candidate_id": row["candidate_id"],
                            "terminal_energy_lower_Wh": prefix, "constraint_satisfied": sign_satisfied,
                            **{arm: {**common, "historical_worker_charges": int(row["charged"]),
                                     "usable_energy": value}
                               for arm, value in (("sign", sign_obs), ("interval", interval_obs))}})
        rankings = [{"candidate_id": r["candidate_id"], "position": r["position"], "layout": r["layout"],
                     **{arm: replays[arm][i] for arm in replays}} for i, r in enumerate(roster)]
        max_delta = max(abs(r["interval"]["p_usable_energy"] - r["sign"]["p_usable_energy"]) for r in rankings)
        if fits["interval"]["diagnostics"]["n_interval"] == 0:
            raise ValueError(f"{distance} nmi replay did not consume interval evidence")
        report["distances"][distance] = {
            "config_sha256": sha256(configs[distance]), "fits": fits, "historical_charges": len(rows),
            "rank_changes": sum(r["sign"]["rank"] != r["interval"]["rank"] for r in rankings),
            "max_abs_p_usable_energy_change": max_delta, "rankings": rankings, "alignment": aligned,
        }
        print(f"Offline {distance} nmi: {len(rows)} paired rows; max probability change {max_delta:.6g}", flush=True)
    report["wall_seconds"] = time.perf_counter() - start
    dump(args.output / "offline_report.json", report)
    return report


def code_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def source_row(row, journal_name):
    return {**row, "source_id": f"{journal_name}:{row['call_id']}"}


def pilot(args, base, worker, qualification, config, sources, excluded):
    started = time.perf_counter()
    output = args.output
    sha = code_sha()
    backend = "rcaide" if args.execute_rcaide else "mock"
    if args.execute_rcaide:
        verified_hash(args.offline_report, args.offline_report_sha256)
        proof = json.loads(args.offline_report.read_bytes())
        if (proof["code_sha"] != sha or set(proof["distances"]) != {"50", "55"}
                or proof["sources"] != sources or proof["new_worker_calls"] != 0):
            raise ValueError("Offline replay must pass at the same code and input identity")
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                   cwd=ROOT, text=True).strip():
            raise ValueError("RCAIDE pilot requires committed code")
        subprocess.run(["git", "merge-base", "--is-ancestor", BASE, sha], cwd=ROOT, check=True)
    (output / "config.json").write_bytes(config.read_bytes())
    problem = X57LiveProblem(base, output / "config.json", output / "workers")
    environment = identity(problem)
    if args.execute_rcaide:
        reference = qualification["identity"]
        if (any(environment[k] != reference[k] for k in ("python_version", "dependencies", "lsbo_sha256"))
                or environment["rcaide_module"]["sha256"] != reference["rcaide_module"]["sha256"]
                or any(os.environ[v] != "1" for v in THREAD_VARS)):
            raise ValueError("RCAIDE worker environment differs from the qualification")
    if problem.config["mission"]["cruise_distance_nmi"] != 45 or problem.config["runtime"]["timeout_seconds"] != TIMEOUT:
        raise ValueError("Pilot requires the archived 45 nmi configuration and timeout")
    roster, tasks, _ = points(problem, args.seed, excluded, INITIAL)
    native_module = importlib.import_module("lsbo.optimizer")
    knobs = ({} if args.execute_rcaide else
             dict(pretrain_epochs=3, update_epochs=2, gp_steps=3, branches=3, points_per_branch=3,
                  restarts=1, raw_samples=8, local_branches=1, proposal_pool_size=24,
                  exploration_probability=0.))
    settings = native_module.Settings(seed=args.seed, initial=INITIAL, budget=INITIAL + args.online, **knobs)
    limit = INITIAL + len(ARMS) * args.online
    risk_pool_size = 512 if args.execute_rcaide else 32
    plan = {"code_sha": sha, "base_sha": BASE, "backend": backend, "seed": args.seed, "distance_nmi": 45,
            "identity": environment, "sources": sources, "historical_candidate_exclusions": sorted(excluded),
            "qualification_plan_sha256": QUALIFICATION_PLAN_SHA256,
            "offline_report_sha256": args.offline_report_sha256, "config_sha256": sha256(config),
            "worker_manifest_sha256": digest(worker), "thread_settings": {v: os.environ[v] for v in THREAD_VARS},
            "roster_sha256": digest(tasks), "tasks": tasks, "initial": INITIAL, "online_per_arm": args.online,
            "max_worker_calls": limit, "worker_concurrency_limit": args.workers, "timeout_seconds": TIMEOUT,
            "settings": asdict(settings), "risk_pool_size": risk_pool_size, "energy_gp": asdict(SETTINGS),
            "native_missing_objective_policy": "log and continue to the common budget",
            "protocol": "matched_engineering_energy_feedback_v3",
            "performance_gate": {"usable_designs_min": 1, "exact_objectives_min": 3},
            "arms": ARMS, "completion_policies": POLICY}
    plan_path = output / "pilot_plan.json"
    dump(plan_path, plan)
    print(f"Frozen 45 nmi roster {plan['roster_sha256']} at {output}", flush=True)

    def dispatcher(folder, max_calls):
        folder.mkdir(parents=True, exist_ok=True)
        return Dispatcher(problem, folder, backend=backend, worker_root=base if backend == "rcaide" else ROOT,
                          timeout=TIMEOUT, expected_worker=worker, max_calls=max_calls,
                          mock_profile="engineering-pilot")

    init_root = output / "initial"
    def launch(task):
        point = roster[task["position"] - 1]
        journal = dispatcher(init_root / task["task_id"], 1)
        row = journal.evaluate(point, seed=args.seed, arm="shared_initial", evaluation=task["position"],
                               online=0, phase="initial", proposal={"roster_sha256": plan["roster_sha256"]})
        return {"status": row["feedback"]["outcome"], "row": source_row(row, f"initial/{task['task_id']}")}

    initial_results = run_wave(tasks, args.workers, launch)
    initial_rows = [initial_results[t["task_id"]]["row"] for t in tasks
                    if "row" in initial_results.get(t["task_id"], {})]
    dump(output / "shared_initial.json", initial_rows)
    stop = Event()
    optimizer_lock = Lock()  # The native optimizer and VAE use process-wide Torch RNG state.

    def run_arm(arm):
        folder = output / arm
        journal = dispatcher(folder, args.online)
        is_ra = arm != "lsbo"
        use_interval = arm == "ra_lsbo_energy_interval"
        history = list(initial_rows)
        status = "complete"
        with optimizer_lock:
            adapter = ObservationAdapter(problem.problem, native_module.Evaluation, native_feedback=not is_ra)
            optimizer = native_module.Optimizer(adapter, settings, folder)
            for point, row in zip(roster, initial_rows):
                if update_native(optimizer, point, row, "initial", row["proposal"]):
                    journal.event("native_stop_during_shared_replay", source_id=row["source_id"], charged=False)

        def fit(rep, bounds):
            return fit_head(rep, [o.candidate for o in optimizer.observations], history,
                            args.seed, 45, use_interval, bounds)[0]

        for i in range(1, args.online + 1):
            if status != "complete" or stop.is_set():
                break
            try:
                with optimizer_lock, torch.random.fork_rng(devices=[]):
                    torch.manual_seed((args.seed + i) % (1 << 63))
                    point, phase, metadata = propose(
                        optimizer, history, POLICY[arm],
                        native_module.mixed_candidate_pool, risk_pool_size, fit_energy=fit if is_ra else None)
            except Exception as error:
                journal.event("proposal_error", online=i, charged=False, message=str(error),
                              error_type=type(error).__name__)
                status = "proposal_error"
                break
            if stop.is_set():
                break
            try:
                row = source_row(journal.evaluate(point, seed=args.seed, arm=arm, evaluation=INITIAL + i,
                                                  online=i, phase=phase, proposal=metadata), arm)
            except Exception as error:
                journal.event("dispatch_error", online=i, message=str(error), error_type=type(error).__name__)
                stop.set()
                status = "infrastructure_error"
                break
            history.append(row)
            if row["feedback"]["outcome"] == "infrastructure_error":
                stop.set()
                status = "infrastructure_error"
                break
            with optimizer_lock:
                if update_native(optimizer, point, row, phase, metadata):
                    journal.event("native_missing_objective_stop", call_id=row["call_id"], charged=False)
        if status == "complete" and len(history) < INITIAL + args.online:
            status = "infrastructure_stop"
        with optimizer_lock:
            optimizer.save_summary(status)
        projections = [(row, *energy_training_row(row, row["source_id"], 45, prefix_energy, use_interval))
                       for row in history]
        fits = [(r["evaluation"], r["proposal"]["engineering_energy_fit"]) for r in history[INITIAL:]
                if "engineering_energy_fit" in r["proposal"]]
        fitted_ids = sorted({o["evidence_id"] for _, fit_audit in fits
                             for o in fit_audit["partial_observations"]})
        usable = [r for r in history if r["feedback"]["usable"]]
        result = {"status": status, "first_usable_call": usable[0]["evaluation"] if usable else None,
                  "first_usable_online": next((r["online"] for r in usable if r["online"] > 0), None),
                  "best_usable_Wh": min((r["feedback"]["objective_Wh"] for r in usable), default=None),
                  "usable": len(usable), "online_records": len(history) - INITIAL,
                  "available_partial_feedback": sum(value.kind == "interval" for _, value, _ in projections),
                  "partial_feedback_fitted": len(fitted_ids), "fitted_partial_ids": fitted_ids,
                  "energy_acquisition_fits": len(fits), "history": [
                      {"source_id": r["source_id"], "candidate_id": r["candidate_id"], "evaluation": r["evaluation"],
                       "feedback": r["feedback"], "terminal_energy_lower_Wh": prefix,
                       "energy_evidence": asdict(value), "charged": r["charged"] and r["arm"] == arm}
                      for r, value, prefix in projections]}
        dump(folder / "arm_report.json", result)
        return result

    arms = {}
    initial_ok = len(initial_rows) == INITIAL and all(r["status"] != "infrastructure_error"
                                                      for r in initial_results.values())
    if initial_ok:
        with ThreadPoolExecutor(max_workers=min(len(ARMS), args.workers)) as pool:
            futures = {arm: pool.submit(run_arm, arm) for arm in ARMS}
            for arm, future in futures.items():
                try:
                    arms[arm] = future.result()
                except Exception as error:
                    stop.set()
                    arms[arm] = {"status": "error", "message": f"{type(error).__name__}: {error}"}
    journals = [f"initial/{t['task_id']}" for t in tasks] + list(ARMS)
    events, counts, worker_times = [], Counter(), []
    for name in journals:
        path = output / name / "calls.jsonl"
        rows = read_events(path) if path.is_file() else []
        counts[name] = sum(r["event"] == "worker_started" for r in rows)
        ends = {r["call_id"]: r for r in rows if r["event"] in ("settled", "interrupted")}
        for r in rows:
            events.append({"journal": name, **r})
            if r["event"] == "worker_started" and r["call_id"] in ends:
                worker_times.append({"pid": r["pid"], "started_at_utc": r["started_at_utc"],
                                     "finished_at_utc": ends[r["call_id"]]["finished_at_utc"]})
    with (output / "calls.jsonl").open("x", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
    charges = sum(counts.values())
    peak = elapsed_concurrency(worker_times)
    if charges > limit or peak > args.workers:
        raise RuntimeError("Pilot overran the authorized call or concurrency budget")
    status = ("complete" if initial_ok and charges == limit and set(arms) == set(ARMS)
              and all(r["status"] == "complete" and counts[name] == args.online
                      for name, r in arms.items()) else "incomplete")
    report = {"status": status, "code_sha": sha, "backend": backend, "seed": args.seed,
              "plan_sha256": sha256(plan_path), "roster_sha256": plan["roster_sha256"],
              "rcaide_calls": charges if args.execute_rcaide else 0, "worker_starts": charges,
              "shared_initial_starts": sum(counts[name] for name in journals[:INITIAL]),
              "shared_initial_usable": sum(row["feedback"]["usable"] for row in initial_rows),
              "initial_results": {key: value["status"] for key, value in initial_results.items()},
              "online_starts": {name: counts[name] for name in ARMS}, "peak_workers": peak,
              "worker_times": worker_times, "arms": arms, "wall_seconds": time.perf_counter() - started}
    dump(output / "pilot_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("offline", "pilot"))
    for key in ("qualification-plan", "qualification-ledger", "paired-archive", "initialization-ledger", "output"):
        parser.add_argument(f"--{key}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--online", type=int, default=ONLINE)
    parser.add_argument("--execute-rcaide", action="store_true")
    parser.add_argument("--offline-report", type=Path)
    parser.add_argument("--offline-report-sha256")
    args = parser.parse_args()
    if not (0 <= args.seed < (1 << 32) - ONLINE and 1 <= args.workers <= WORKERS
            and 1 <= args.online <= ONLINE):
        parser.error("Require a valid seed, 1–12 workers and 1–12 online evaluations per arm")
    if args.execute_rcaide and (args.mode != "pilot" or not args.offline_report or not args.offline_report_sha256):
        parser.error("--execute-rcaide requires pilot mode and an offline report with its SHA256")
    torch.set_num_threads(1)
    base, worker, qualification, configs, sources, ledgers, excluded = verify_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "offline":
        offline(args, base, configs, sources, ledgers, excluded)
        return 0
    report = pilot(args, base, worker, qualification, configs[45], sources, excluded)
    print(args.output / "pilot_report.json", flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
