"""Run or resume the five X-57 arms with shared, frozen initialization."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stdout
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock

from ra_lsbo.budget import digest
from ra_lsbo.x57_cold_sampling import LAYOUTS, violations
from ra_lsbo.x57_cold_worker import Dispatcher, dump, sha256
from ra_lsbo.x57_progress import CampaignProgress
from ra_lsbo.x57_energy_stop import prefix_energy, verified_stop
from ra_lsbo.x57_energy_surrogate import energy_training_row
from ra_lsbo.x57_live import X57LiveProblem, evaluator_files
from run_x57_cold_start_debug import verify_environment
from run_x57_distance_diagnostic import (
    QUALIFICATION_PLAN_SHA256, ROOT, THREAD_VARS, TIMEOUT, elapsed_concurrency, read_events,
)
from run_x57_engineering_energy_pilot import code_sha, points
from run_x57_initialization_diagnostic import verified_hash


CODE_REFERENCE = "508398032f98b389be0701b8004aa5279c00b485"
LEGACY_RESUME_CODE = "0a3c1ea7db2ab093ada3b2cd4ae24b6d70c14377"
RESULT_REFERENCE = "9c5ac77c5ecc3987787292981dc679045441972c"
RESULT_DIR = "x57-formal-prefix-qualification"
PREFIX_REFERENCE = "bcc54e6ecfda8a68fdcf93238d13b8a07d023daa"
PREFIX_HASHES = {
    "run_plan.json": "1160101863a748cc0eda4283cd8c8cf1b269bcf432f441493114f9ada5b8531a",
    "calls.jsonl": "da57219a60b112165fce52ed2a048f2710872ee060541be24a084e5967022f1b",
}
INPUTS = {
    f"{RESULT_DIR}/preregistered_47_5.json": "a984e0d42adbb9752c2da7fbc3a5e8e3805b9ef25ca361d9b02b6c094f576437",
    f"{RESULT_DIR}/physical/configs/distance_47.5.json": "9902197262e862b13a3b3ac4a75c6f2c7892c91d8a69d6ce1af8aa884f6e6c3d",
    f"{RESULT_DIR}/physical/report.json": "74092569b8a83e915e1cfefd25c0253506e1e51fb6ba980fbd33150fef5b64b1",
    f"{RESULT_DIR}/physical/calls.jsonl": "3b3996b794d0584f1fe76cf6bc563dccc169ba94649e196fc383d46ecaac87a8",
    "x57-cold-start-debug-fccf94c6/calls.jsonl": "990caac32365893a8100e3cc10660cd554e1e7a4391b4b467a74a570c77314df",
}
SEEDS = (307261, 356821, 412849, 470299, 528899, 590089)
COLD_SEEDS = (643118, 705241, 767363, 829486, 891608, 953731)
NATIVE = ("lsbo_full_budget", "ra_lsbo_energy_sign", "ra_lsbo_energy_interval")
EXTERNAL = ("archsbo_native", "logcei_screened_online")
RECOVERY_ARMS = ("lsbo_full_budget", "ra_lsbo_energy_interval", "ra_lsbo_energy_interval_recovery_pool")
RECOVERY_RESULT = "e5fcec2710f1a8f3b5326974a76fff23f67ed917"
RECOVERY_HASHES = {"run_plan.json": "6cf4dfe88c249a238d99ea61f905702d0d703bebe9fd4e1040d5dcf0346eea55",
                   "calls.jsonl": "22e5658d4c4dffaea7b1c18e6ba32bdd501d187315333a5ebf8197051fd4911c"}
RECOVERY_SEEDS = tuple(range(2026112801, 2026112807))
COLD_EXTERNAL = ("logcei_screened_online", "scbo_screened_online")
ARMS = NATIVE + EXTERNAL
ARCHIVED_INITIAL = 8
INITIAL = 64
COLD_CHECKPOINT = 16
COLD_SCHEMA = "x57_five_arm_cold_n8"
HORIZON = 320
TARGET_WH = 47500
THREADS = (*THREAD_VARS, "NUMBA_NUM_THREADS")
BASELINE_VERSIONS = {"sb-arch-opt": "1.7.1", "smt": "2.14.1", "pymoo": "0.6.1.5",
                     "torch": "2.7.1", "scikit-learn": "1.6.1"}
COLD_BASELINE_VERSIONS = {"numpy": "2.2.6", "scipy": "1.15.3",
                          "torch": "2.7.1", "scikit-learn": "1.6.1"}


def load(path):
    return json.loads(path.read_bytes())


def load_resume_plan(path, plan):
    frozen = load(path)
    # The legacy executor used Python equality before normalizing JSON arrays.
    expected_code = frozen["code_sha"]
    if expected_code not in (plan["code_sha"], LEGACY_RESUME_CODE) or (
        digest({**plan, "code_sha": expected_code}) != digest(frozen)
    ):
        raise ValueError("Resume requires the original run plan and source identities")
    return frozen


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def check_threads():
    if any(os.environ.get(name) != "1" for name in THREADS):
        raise ValueError(f"Before starting Python, set {', '.join(THREADS)} to 1")


def external_service(base, config, work, arm, *, cold=False, mock=False):
    from ra_lsbo.x57_external import X57External, native_kpls_minimum

    check_threads()
    expected = COLD_BASELINE_VERSIONS if cold else BASELINE_VERSIONS
    versions = {name: version(name) for name in expected}
    if not mock and (versions != expected or sys.version_info[:2] != (3, 11)):
        raise ValueError(f"Unexpected external baseline environment: {versions}")
    output = sys.stdout
    with redirect_stdout(sys.stderr):
        problem = X57LiveProblem(base, config, work)
        trajectories = {}
        for line in sys.stdin:
            request = json.loads(line)
            op = request["op"]
            if op == "stop":
                break
            try:
                seed = request.get("seed")
                if op == "identity":
                    value = {"python": sys.version, "exe": sys.executable, "packages": versions,
                             "contract_id": problem.contract_id}
                    if arm == "archsbo_native":
                        value["kpls_minimum_complete_rows"] = native_kpls_minimum(len(problem.space.variables))
                elif op == "init":
                    trajectories[seed] = X57External(problem, arm, seed, request["initial"], HORIZON)
                    value = None
                elif op == "drop":
                    del trajectories[seed]
                    value = None
                elif op == "ask":
                    value = trajectories[seed].ask()
                elif op == "qualify":
                    adapter = X57External(problem, arm, seed, request["initial"], HORIZON)
                    complete = adapter.native_complete_rows()
                    minimum = native_kpls_minimum(len(problem.space.variables))
                    value = {"complete_training_rows": complete, "minimum_complete_rows": minimum,
                             "qualified": False}
                    if minimum is None or complete >= minimum:
                        try:
                            candidate, _ = adapter.ask()
                        except Exception as error:
                            value["error"] = f"{type(error).__name__}: {error}"
                        else:
                            point = problem.candidate_class(**candidate)
                            value.update(qualified=True, first_proposal_id=problem.space.key(point))
                    else:
                        value["error"] = "Insufficient complete rows for native ArchSBO KPLS"
                elif op == "tell":
                    trajectories[seed].tell(request["row"])
                    value = False
                elif op == "checkpoint":
                    value = trajectories[seed].checkpoint()
                else:
                    raise ValueError(f"Unknown external operation {op}")
                response = json.dumps({"value": value}, allow_nan=False)
            except Exception as exc:
                response = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            print(response, file=output, flush=True)


class ExternalService:
    def __init__(self, python, base, config, output, arm, stack, *, cold=False, mock=False):
        log = stack.enter_context((output / f"{arm}.optimizer.log").open("a", encoding="utf-8"))
        flags = [flag for flag, enabled in (("--cold-baselines", cold), ("--mock", mock)) if enabled]
        self.process = subprocess.Popen(
            [str(python), "-B", "-u", str(Path(__file__).resolve()), "--external-service",
             str(base), str(config), str(output / "unused"), arm, *flags],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            text=True, encoding="utf-8",
        )
        stack.callback(self.close)
        self.lock = Lock()
        self.identity = self.rpc("identity")

    def rpc(self, op, **fields):
        with self.lock:
            self.process.stdin.write(json.dumps({"op": op, **fields}, allow_nan=False) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("External optimizer exited. See optimizer log.")
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(response["error"])
            return response["value"]

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


def archived_initial(path, problem, *, seeds=SEEDS, hashes=PREFIX_HASHES, reference=PREFIX_REFERENCE):
    from ra_lsbo.x57_external import record_from_row

    for filename, expected in hashes.items():
        verified_hash(path / filename, expected)
    plan = load(path / "run_plan.json")
    if (plan["backend"] != "rcaide" or plan["contract_id"] != problem.contract_id
            or plan["seeds"] != list(seeds) or plan["initial_count"] != ARCHIVED_INITIAL
            or plan["total_budget_per_arm"] != HORIZON):
        raise ValueError("The archived initialization belongs to a different evaluation protocol")
    events = [e for e in read_events(path / "calls.jsonl") if e["arm"] == "shared"]
    ledger = [e for e in events if e["event"] == "settled"]
    started = [e for e in events if e["event"] == "worker_started"]
    if (len(started) != len(ledger)
            or any(e["backend"] != "rcaide" or e["charged"] is not True for e in started)
            or [(e["task_id"], e["candidate_id"]) for e in started]
            != [(e["task_id"], e["candidate_id"]) for e in ledger]):
        raise ValueError("Archived shared settlements do not match charged RCAIDE starts")
    reused = {}
    for seed in seeds:
        rows = [r for r in ledger if r["seed"] == seed]
        expected = plan["rosters"][str(seed)]["records"]
        if (len(rows) != ARCHIVED_INITIAL or
                any((r["evaluation"], r["candidate_id"], r["candidate"], r["precheck"], r["online"],
                     r["phase"], r["charged"])
                    != (p["position"], p["candidate_id"], p["candidate"], p["precheck"], 0,
                        "formal_shared_initial", True) for r, p in zip(rows, expected))):
            raise ValueError(f"Archived shared ledger and roster disagree for seed {seed}")
        for row in rows:
            record = record_from_row(problem, row)
            if record.observation.status == "infrastructure_error":
                raise ValueError(f"Unusable infrastructure settlement in archived seed {seed}")
        reused[seed] = [{**{k: v for k, v in r.items() if k != "event"},
                         "source_id": f"{reference}:{r['task_id']}"} for r in rows]
    return plan, reused


def extended_rosters(problem, seeds, historical_ids, archived_plan=None):
    seen = set(historical_ids)
    rosters = {}
    for seed in seeds:
        if archived_plan is None:
            rosters[str(seed)] = {"records": [], "sampling": []}
        else:
            old = archived_plan["rosters"][str(seed)]
            keys = {r["candidate_id"] for r in old["records"]}
            if len(keys) != ARCHIVED_INITIAL or seen & keys:
                raise ValueError("Archived shared prefix overlaps another frozen sample")
            seen.update(keys)
            rosters[str(seed)] = {"records": list(old["records"]),
                                  "sampling": [{"source": PREFIX_REFERENCE, "audit": old["sampling"]}]}
    for seed in seeds:
        roster = rosters[str(seed)]
        for block in range(1, 3):
            sampler_seed = seed + block * 1000000
            take = min(32, INITIAL - len(roster["records"]))
            _, rows, audit = points(problem, sampler_seed, seen, take)
            start = len(roster["records"])
            roster["records"].extend({**row, "position": start + i, "task_id": f"{start + i:03d}"}
                                     for i, row in enumerate(rows, 1))
            seen.update(r["candidate_id"] for r in rows)
            roster["sampling"].append({"seed": sampler_seed, "count": take, "audit": audit})
    return rosters


def freeze(args, output):
    verified_hash(args.qualification_plan, QUALIFICATION_PLAN_SHA256)
    qualification = load(args.qualification_plan)
    base = args.qualification_plan.parent / "physical-base"
    paths = {name: args.archive / name for name in INPUTS}
    for name, path in paths.items():
        verified_hash(path, INPUTS[name])
    prereg = load(paths[f"{RESULT_DIR}/preregistered_47_5.json"])
    confirmation = load(paths[f"{RESULT_DIR}/physical/report.json"])["confirmation"]
    usable = [r for r in confirmation if r.get("usable")]
    best = min(r["objective_Wh"] for r in usable)
    if {r["layout"] for r in usable} != set(LAYOUTS) or not TARGET_WH - 500 < best <= TARGET_WH:
        raise ValueError("The registered confirmation reference changed")
    config = output / "distance_47.5.json"
    frozen_config = paths[f"{RESULT_DIR}/physical/configs/distance_47.5.json"]
    if not config.exists():
        with config.open("xb") as stream:
            stream.write(frozen_config.read_bytes())
    verified_hash(config, prereg["config_sha256"])
    worker = qualification["worker"]["worker_sha256"]
    overlay = sha256(ROOT / "run_x57_energy_stop_worker.py")
    problem = X57LiveProblem(base, config, output / "unused")
    if (worker != evaluator_files(base) or digest(worker) != prereg["worker_manifest_sha256"]
            or overlay != prereg["worker_overlay_sha256"]
            or problem.config["mission"]["cruise_distance_nmi"] != 47.5):
        raise ValueError("The registered physical contract changed")
    environment = verify_environment(problem, qualification) if args.execute_rcaide else {}

    historical_ids = {row["candidate_id"] for name, path in paths.items() if name.endswith("/calls.jsonl")
                      for row in read_events(path) if row["event"] == "worker_started"}
    confirmation_ids = {row["candidate_id"] for row in confirmation}
    if len(confirmation_ids) != 4 or not confirmation_ids <= historical_ids:
        raise ValueError("Expected four charged confirmation candidates in the exclusions")
    fork = getattr(args, "risk_screen_fork", False)
    long_pair = getattr(args, "long_pair", False)
    matched = getattr(args, "matched_external", False) or fork or long_pair
    if matched:
        seeds, arms, initial_count = RECOVERY_SEEDS, COLD_EXTERNAL, ARCHIVED_INITIAL
        archived, reused = archived_initial(args.initial_replay, problem, seeds=seeds,
                                          hashes=RECOVERY_HASHES, reference=RECOVERY_RESULT)
        rosters = archived["rosters"]
        if long_pair:
            arms = ("lsbo_full_budget", "ra_lsbo_energy_interval")
        elif fork:
            arms = ("ra_lsbo_energy_interval",)
    elif args.cold_baselines:
        seeds = COLD_SEEDS if args.execute_rcaide else COLD_SEEDS[:1]
        initial_count, arms, reused = ARCHIVED_INITIAL, NATIVE + COLD_EXTERNAL, {}
        if getattr(args, "recovery_pool_pilot", False):
            seeds = RECOVERY_SEEDS if args.execute_rcaide else RECOVERY_SEEDS[:1]
            arms = RECOVERY_ARMS
        rosters, seen = {}, set(historical_ids)
        for seed in seeds:
            _, records, audit = points(problem, seed, seen, initial_count)
            rosters[str(seed)] = {"records": records, "sampling": audit}
            seen.update(r["candidate_id"] for r in records)
    else:
        seeds = SEEDS if args.execute_rcaide else SEEDS[:1]
        initial_count, arms = INITIAL, ARMS
        archived_plan, reused = archived_initial(args.initial_replay, problem) if args.execute_rcaide else (None, {})
        rosters = extended_rosters(problem, seeds, historical_ids, archived_plan)
    plan = {
        "schema": COLD_SCHEMA if args.cold_baselines else "x57_five_arm_native_archsbo_n64",
        "code_sha": code_sha(), "code_base": CODE_REFERENCE,
        "result_reference": RESULT_REFERENCE, "input_sha256": INPUTS,
        "qualification_plan_sha256": QUALIFICATION_PLAN_SHA256, "config_sha256": sha256(config),
        "worker_manifest_sha256": digest(worker), "worker_overlay_sha256": overlay,
        "environment": environment, "contract_id": problem.contract_id, "distance_nmi": 47.5,
        "quality_target_Wh": TARGET_WH, "target_basis": "Ceil best prior confirmation to 500 Wh",
        "best_prior_confirmation_Wh": best, "45_nmi_quality_target_Wh": 46000,
        "seeds": list(seeds), "seed_selection": "Fixed before evaluation, independent of prior seeds",
        "arms": list(arms), "initial_count": initial_count, "total_budget_per_arm": HORIZON,
        "first_checkpoint": initial_count,
        "first_checkpoint_new_call_cap": len(seeds) * initial_count - sum(map(len, reused.values())),
        "reused_shared_prefix": {"commit": PREFIX_REFERENCE, "hashes": PREFIX_HASHES,
                                 "per_seed": ARCHIVED_INITIAL} if reused else None,
        "max_workers": args.workers, "timeout_seconds": TIMEOUT, "thread_settings": dict.fromkeys(THREADS, "1"),
        "backend": "rcaide" if args.execute_rcaide else "mock",
        "confirmation_exclusions": sorted(confirmation_ids), "initial_exclusions": sorted(historical_ids),
        "rosters": rosters,
    }
    if matched:
        for key in ("config_sha256", "quality_target_Wh", "total_budget_per_arm", "qualification_plan_sha256",
                    "worker_manifest_sha256", "worker_overlay_sha256", "confirmation_exclusions",
                    "initial_exclusions", "thread_settings", "environment"):
            if digest(plan[key]) != digest(archived[key]):
                raise ValueError(f"External comparison differs from the archived campaign: {key}")
        plan.update(schema="x57_recovery_matched_external_v1",
                    reused_shared_prefix={"commit": RECOVERY_RESULT, "hashes": RECOVERY_HASHES,
                                          "per_seed": ARCHIVED_INITIAL})
        if long_pair:
            plan.update(schema="x57_long_pair_v1", common_prefix=32, checkpoints=[32, 64, 128, 320])
        elif fork:
            plan.update(schema="x57_posterior_screen_fork_v1", common_prefix=16, screened_probability=0.5)
    elif getattr(args, "recovery_pool_pilot", False):
        plan.update(schema="x57_recovery_pool_pilot_v1", shared_max_workers=min(4, args.workers))
    return plan, problem, worker, reused


def group(initial):
    count = sum(row["feedback"]["usable"] for row in initial)
    return "0" if count == 0 else "1-2" if count <= 2 else "3+"


class Campaign:
    def __init__(self, args, plan, problem, worker, reused):
        self.args, self.plan, self.problem, self.worker = args, plan, problem, worker
        self.output = args.output
        self.halt = Event()
        self.start_lock = Lock()
        self.admitted = 0
        self.reused = reused
        self.reused_count = sum(map(len, reused.values()))
        self.initial_count, self.arms = plan["initial_count"], plan["arms"]
        shared_total = len(plan["seeds"]) * self.initial_count
        total = shared_total + len(plan["seeds"]) * len(self.arms) * (args.checkpoint - self.initial_count)
        self.cap = total - self.reused_count
        self.prefix = {}
        self.reference_online = {}
        self.matched_feedback = {}
        self.prefix_end = plan.get("common_prefix", 0)
        if self.prefix_end:
            for event in read_events(args.initial_replay / "calls.jsonl"):
                if (event["event"] == "settled" and event["arm"] in self.arms
                        and self.initial_count < event["evaluation"] <= self.prefix_end):
                    self.prefix[event["task_id"]] = event
                elif (event["event"] == "settled" and event["arm"] in self.arms and event["evaluation"] > self.prefix_end):
                    self.reference_online[event["task_id"]] = event
            if len(self.prefix) != len(plan["seeds"]) * len(self.arms) * (self.prefix_end - self.initial_count):
                raise ValueError("Incomplete archived online prefix")
            self.cap -= len(self.prefix)
        self.progress = CampaignProgress(total, shared_total, args.workers)
        self.initial = {}
        self.qualification = {}

    def evaluate(self, job, point, *, seed, arm, evaluation, online, phase, proposal):
        key = self.problem.space.key(point)
        candidate = self.problem.canonical(point.to_dict())
        if (violations(self.problem.problem.precheck(point))
                or candidate != point.to_dict()
                or (online and key in self.plan["confirmation_exclusions"])):
            raise ValueError(f"Candidate failed dispatch identity or eligibility checks: {job}")
        if self.prefix and online and evaluation <= self.prefix_end:
            r = self.prefix[job]
            if (r["candidate_id"], r["candidate"], r["phase"], r["charged"]) != (key, candidate, phase, True):
                raise RuntimeError(f"Archived prefix proposal differs at {job}")
            self.progress.settle(job, r["feedback"], shared=False, replayed=True)
            return {**{k: v for k, v in r.items() if k not in ("task_id", "event")}, "source_id": job}
        out = self.output / job
        ref = self.reference_online.get(job)
        if (ref and not out.exists() and (key, candidate, phase) ==
                (ref["candidate_id"], ref["candidate"], ref["phase"])):
            self.matched_feedback[job] = {"candidate_id": key, "proposal": proposal, "source": RECOVERY_RESULT}
            self.progress.settle(job, ref["feedback"], shared=False, replayed=True)
            return {**{k: v for k, v in ref.items() if k not in ("event", "task_id")},
                    "source_id": job, "proposal": proposal}
        expected = ("dispatch_intent", "worker_started", "settled")
        replayed = out.exists()
        if replayed:
            ledger = out / "calls.jsonl"
            events = read_events(ledger) if ledger.is_file() else []
            if (tuple(e["event"] for e in events) != expected or
                    any((e["seed"], e["arm"], e["evaluation"], e["online"], e["phase"], e["candidate_id"])
                        != (seed, arm, evaluation, online, phase, key) for e in events)):
                raise RuntimeError(f"Cannot replay started or inconsistent call: {job}")
            row = events[-1]
        else:
            with self.start_lock:
                if self.halt.is_set():
                    return None
                if self.admitted >= self.cap:
                    raise RuntimeError("Checkpoint call cap reached")
                self.admitted += 1
            out.mkdir(parents=True, exist_ok=False)
            dispatcher = Dispatcher(
                self.problem, out, backend=self.plan["backend"],
                worker_root=self.problem.base if self.args.execute_rcaide else ROOT,
                expected_worker=self.worker, max_calls=1, timeout=TIMEOUT, energy_stop=True,
                mock_profile="formal-prefix",
            )
            with self.progress.activity("evaluating"):
                row = dispatcher.evaluate(point, seed=seed, arm=arm, evaluation=evaluation,
                                          online=online, phase=phase, proposal=proposal)
        if not row["charged"] or row["feedback"]["outcome"] == "infrastructure_error":
            raise RuntimeError(f"Infrastructure failure or uncharged settlement: {job}")
        row = {k: v for k, v in row.items() if k != "event"}
        row["source_id"] = job
        self.progress.settle(job, row["feedback"], shared=arm == "shared", replayed=replayed)
        return row

    def dispatch_initial(self, seed, record):
        point = self.problem.candidate_class(**record["candidate"])
        if (self.problem.space.key(point) != record["candidate_id"]
                or self.problem.problem.precheck(point) != record["precheck"]):
            raise ValueError("Frozen initial candidate or precheck changed")
        evaluation = record["position"]
        job = f"seed-{seed}/shared/{evaluation:03d}"
        reused = self.reused.get(seed, [])
        if evaluation <= len(reused):
            row = reused[evaluation - 1]
            self.progress.settle(job, row["feedback"], shared=True, replayed=True)
            return row
        return self.evaluate(job, point, seed=seed, arm="shared", evaluation=evaluation,
                             online=0, phase="formal_shared_initial", proposal={"frozen_task": job})

    def trajectory(self, seed, arm, services):
        output = self.output / f"seed-{seed}" / arm
        output.mkdir(parents=True, exist_ok=True)
        initial = self.initial[seed]
        if arm in NATIVE or arm == RECOVERY_ARMS[-1]:
            from tempfile import TemporaryDirectory
            from ra_lsbo.x57_formal_native import FormalNative

            with TemporaryDirectory(dir=self.output, prefix="optimizer-") as temp:
                adapter = FormalNative(self.problem, arm, seed, initial, Path(temp) / "native")

                if self.plan.get("schema") == "x57_posterior_screen_fork_v1":
                    from ra_lsbo.x57_native_slot import configure_native_search
                    configure_native_search(adapter.optimizer, reserve=True, joint_final=True,
                                            force_weighted=True, screen_after=16)

                def ask():
                    point, phase, info = adapter.ask()
                    return point.to_dict(), phase, info

                return self.advance(seed, arm, output, ask, adapter.tell, adapter.checkpoint)
        service = services[arm]

        def ask():
            candidate, info = service.rpc("ask", seed=seed)
            return candidate, arm, info

        service.rpc("init", seed=seed, initial=initial)
        try:
            return self.advance(seed, arm, output, ask, lambda row: service.rpc("tell", seed=seed, row=row),
                                lambda: service.rpc("checkpoint", seed=seed))
        finally:
            service.rpc("drop", seed=seed)

    def advance(self, seed, arm, output, ask, tell, checkpoint):
        previews = output / "next_proposal.json"
        previous = load(previews) if previews.is_file() else None
        replay_verified = False
        rows = []
        plan_hash = sha256(self.output / "run_plan.json")
        checkpoint_path = output / "checkpoint.json"
        previous_checkpoint = load(checkpoint_path) if checkpoint_path.is_file() else None
        if previous_checkpoint and previous_checkpoint["plan_sha256"] != plan_hash:
            raise RuntimeError(f"Checkpoint has a different plan: {output}")
        if previous_checkpoint and previous_checkpoint["evaluation"] > self.args.checkpoint:
            raise RuntimeError(f"Cannot rewind an already settled checkpoint: {output}")

        def persist():
            save(checkpoint_path, {"plan_sha256": plan_hash, "evaluation": self.initial_count + len(rows),
                                   "optimizer": checkpoint()})

        for evaluation in range(self.initial_count + 1, self.args.checkpoint + 1):
            job = f"seed-{seed}/{arm}/{evaluation:03d}"
            if self.halt.is_set() and not (self.output / job).exists():
                break
            with self.progress.activity("selecting"):
                raw, phase, info = ask()
            point = self.problem.candidate_class(**raw)
            if arm == "archsbo_native" and evaluation == self.initial_count + 1:
                if self.problem.space.key(point) != self.qualification[str(seed)]["first_proposal_id"]:
                    raise RuntimeError(f"Native ArchSBO first proposal differs from qualification: {job}")
            if previous and previous["evaluation"] == evaluation:
                if previous["candidate_id"] != self.problem.space.key(point) or previous["phase"] != phase:
                    raise RuntimeError(f"Next proposal changed upon resume: {job}")
                replay_verified = True
            row = self.evaluate(job, point, seed=seed, arm=arm, evaluation=evaluation,
                                online=evaluation - self.initial_count, phase=phase, proposal=info)
            if row is None:
                break
            if arm == "archsbo_native" and row["proposal"] != info:
                raise RuntimeError(f"Native ArchSBO proposal audit changed upon replay: {job}")
            tell(row)
            rows.append(row)
            if previous_checkpoint and evaluation == previous_checkpoint["evaluation"]:
                if checkpoint() != previous_checkpoint["optimizer"]:
                    raise RuntimeError(f"Settlement replay changed optimizer state: {job}")
            if not previous_checkpoint or evaluation >= previous_checkpoint["evaluation"]:
                persist()
        if not self.args.execute_rcaide and self.args.checkpoint < HORIZON and not self.halt.is_set():
            with self.progress.activity("selecting"):
                raw, phase, _ = ask()
            next_key = self.problem.space.key(self.problem.candidate_class(**raw))
            save(previews, {"evaluation": self.args.checkpoint + 1, "candidate_id": next_key, "phase": phase})
        return {"rows": rows, "replay_verified": replay_verified}

    def qualify_native_archsbo(self, service):
        path = self.output / "native_archsbo_gate.json"
        input_id = digest(self.initial)
        if path.is_file():
            previous = load(path)
            if previous["initial_sha256"] != input_id:
                raise ValueError("The shared initialization differs from the qualification gate")
            return previous["seeds"]
        seeds = {}
        for seed, initial in self.initial.items():
            try:
                seeds[str(seed)] = service.rpc("qualify", seed=seed, initial=initial)
            except RuntimeError as error:
                seeds[str(seed)] = {"qualified": False, "error": str(error)}
            print(f"Native ArchSBO qualification {seed}: {seeds[str(seed)]}", flush=True)
        save(path, {"initial_sha256": input_id, "seeds": seeds})
        return seeds

    def run(self, services):
        errors = {}

        def wave(items, launch, workers=None):
            def guarded(item):
                if self.halt.is_set():
                    return None
                try:
                    return launch(*item)
                except Exception:
                    self.halt.set()
                    raise

            results = {}
            with ThreadPoolExecutor(max_workers=workers or self.args.workers) as pool:
                futures = {pool.submit(guarded, item): key for key, item in items}
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        results[key] = future.result()
                    except Exception as exc:
                        errors[str(key)] = f"{type(exc).__name__}: {exc}"
                        print(f"Suspended {key}: {errors[str(key)]}", file=sys.stderr, flush=True)
            return results

        items = [((seed, row["position"]), (seed, row)) for seed in self.plan["seeds"]
                 for row in self.plan["rosters"][str(seed)]["records"]]
        self.progress.phase("shared initialization")
        shared = wave(items, self.dispatch_initial, self.plan.get("shared_max_workers"))
        for seed in self.plan["seeds"]:
            rows = [shared.get((seed, i)) for i in range(1, self.initial_count + 1)]
            if all(rows):
                self.initial[seed] = rows
        print(f"Shared prefixes complete: {len(self.initial)}; "
              f"groups: {dict(Counter(group(rows) for rows in self.initial.values()))}", flush=True)
        if errors:
            return errors, {}
        if "archsbo_native" in services:
            self.progress.phase("ArchSBO qualification")
            self.qualification = self.qualify_native_archsbo(services["archsbo_native"])
            errors.update({f"{seed}/archsbo_native": row["error"] for seed, row in self.qualification.items()
                           if not row["qualified"]})
        if errors or self.args.checkpoint == self.initial_count:
            return errors, {}
        items = [((seed, arm), (seed, arm, services))
                 for seed in sorted(self.initial, key=lambda seed: group(self.initial[seed])) for arm in self.arms]
        self.progress.phase("online search")
        trajectories = wave(items, self.trajectory)
        return errors, trajectories


def history_stats(initial, online, arm):
    series = []
    best = None
    first = quality = None
    partial = set()
    proven = set()
    for row in initial + online:
        feedback = row["feedback"]
        evidence_id = f"{row['evaluation']}:{row['candidate_id']}"
        exact = feedback["objective_Wh"] if feedback["objective_kind"] == "exact" else None
        if feedback["usable"]:
            best = exact if best is None else min(best, exact)
            first = first or row["evaluation"]
            if exact <= TARGET_WH:
                quality = quality or row["evaluation"]
        proposal = row["proposal"]
        for fitted in proposal.get("engineering_energy_fit", {}).get("partial_observations", []):
            if fitted["evidence_id"] not in proven:
                raise ValueError("A fit references partial feedback absent from its past")
            partial.add(fitted["evidence_id"])
        evidence, _ = energy_training_row(row, evidence_id, 47.5, prefix_energy, True)
        if evidence.kind == "interval":
            proven.add(evidence_id)
        series.append({"evaluation": row["evaluation"], "candidate_id": row["candidate_id"],
                       "outcome": feedback["outcome"], "completion": feedback["completion"],
                       "objective_kind": feedback["objective_kind"], "objective_Wh": exact,
                       "usable": feedback["usable"], "best_usable_Wh": best,
                       "verified_energy_lower_bound": evidence.kind == "interval",
                       "verified_engineering_stop": verified_stop(row["worker_result"])})
    initial_best = series[len(initial) - 1]["best_usable_Wh"]
    online_series = series[len(initial):]
    return {"first_usable_call": first, "first_quality_call": quality, "best_usable_Wh": best,
            "best_usable_at_initialization_Wh": initial_best,
            "gain_over_initial_Wh": None if initial_best is None else initial_best - best,
            "first_online_usable_call": next((r["evaluation"] for r in online_series if r["usable"]), None),
            "first_improvement_over_initial_call": next(
                (r["evaluation"] for r in online_series if initial_best is not None and r["usable"]
                 and r["objective_Wh"] < initial_best), None),
            "unique_partial_rows_in_engineering_fit": len(partial), "partial_fit_ids": sorted(partial),
            "verified_energy_lower_bound_rows": len(proven), "outcomes": dict(Counter(r["outcome"] for r in series)),
            "trajectory": series, "arm": arm}


def report(campaign, errors, trajectories, services, execution_code_sha):
    output = campaign.output
    windows = []
    ledger = output / "calls.jsonl"
    events = []
    initial_count, arms = campaign.initial_count, campaign.arms
    keys = [(seed, "shared", i) for seed in campaign.plan["seeds"] for i in range(1, initial_count + 1)]
    keys.extend((seed, arm, i) for seed in campaign.plan["seeds"] for arm in arms
                for i in range(initial_count + 1, campaign.args.checkpoint + 1))
    for seed, arm, evaluation in keys:
        job = f"seed-{seed}/{arm}/{evaluation:03d}"
        path = output / job / "calls.jsonl"
        if not path.is_file():
            continue
        entries = read_events(path)
        events.extend({**e, "task_id": job} for e in entries)
        for entry in entries:
            if entry["event"] == "worker_started":
                stop = next((e.get("finished_at_utc") for e in reversed(entries) if e.get("finished_at_utc")), None)
                windows.append({"pid": entry["pid"], "started_at_utc": entry["started_at_utc"],
                                "finished_at_utc": stop, "task_id": job, "arm": arm})
    temporary = ledger.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for entry in events:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
    temporary.replace(ledger)
    peak = elapsed_concurrency([w for w in windows if w["finished_at_utc"]])
    if len(windows) > campaign.cap or peak > campaign.args.workers:
        raise RuntimeError("The durable ledger exceeded the configured limit")
    summaries = {}
    initial_groups = {}
    replay = {}
    for seed in campaign.plan["seeds"]:
        initial = campaign.initial.get(seed)
        if not initial:
            continue
        initial_groups[str(seed)] = group(initial)
        summaries[str(seed)] = {"shared": history_stats(initial, [], "shared")}
        for arm in arms:
            result = trajectories.get((seed, arm))
            if result:
                summaries[str(seed)][arm] = history_stats(initial, result["rows"], arm)
                replay[f"{seed}/{arm}"] = result["replay_verified"]
    complete = (not errors and len(windows) + len(campaign.matched_feedback) == campaign.cap and
                all(len(trajectories.get((seed, arm), {}).get("rows", [])) == campaign.args.checkpoint - initial_count
                    for seed in campaign.plan["seeds"] for arm in arms))
    groups = {}
    for label in ("0", "1-2", "3+"):
        members = [seed for seed in initial_groups if initial_groups[seed] == label]
        by_arm = {}
        for arm in arms:
            settled = [summaries[s][arm] for s in members if arm in summaries[s] and
                       len(summaries[s][arm]["trajectory"]) == campaign.args.checkpoint]
            by_arm[arm] = {
                "checkpoint_settled": len(settled),
                "first_usable_calls": [r["first_usable_call"] for r in settled],
                "first_quality_calls": [r["first_quality_call"] for r in settled],
                "no_usable_fraction": (sum(r["first_usable_call"] is None for r in settled) / len(settled)
                                       if settled else None),
            }
        groups[label] = {"seeds": members, "arms": by_arm}
    result = {
        "status": "complete" if complete else "incomplete", "errors": errors, "backend": campaign.plan["backend"],
        "execution_code_sha": execution_code_sha,
        "plan_sha256": sha256(output / "run_plan.json"), "checkpoint": campaign.args.checkpoint,
        "total_budget_per_arm": HORIZON, "quality_target_Wh": TARGET_WH,
        "actual_rcaide_calls": len(windows) if campaign.args.execute_rcaide else 0,
        "reused_shared_calls": campaign.reused_count,
        "worker_starts": len(windows), "peak_workers": peak, "worker_times": windows,
        "external_environments": {arm: service.identity for arm, service in services.items()},
        "initial_groups": initial_groups, "group_metrics": groups, "per_seed": summaries,
        "next_proposal_replay_verified": replay,
    }
    if campaign.prefix:
        result["reused_online_prefix_calls"] = len(campaign.prefix)
        result["reused_matching_late_feedback"] = campaign.matched_feedback
    if "archsbo_native" in arms:
        result["native_archsbo_qualification"] = campaign.qualification
    save(output / "report.json", result)
    print(f"{result['status']} | starts={len(windows)} | peak={peak} | output={output}", flush=True)
    return 0 if complete else 1


def run(args):
    check_threads()
    cold = args.cold_baselines
    initial_count = ARCHIVED_INITIAL if cold else INITIAL
    if args.checkpoint < initial_count:
        raise ValueError(f"Checkpoint must include the {initial_count} shared initial evaluations")
    if args.execute_rcaide and not cold and (args.initial_replay is None or
                                           (not args.resume and args.checkpoint != INITIAL)):
        raise ValueError("Start the physical campaign with the frozen prefix archive at checkpoint 64")
    if args.execute_rcaide:
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT)
        if status:
            raise ValueError("Physical campaign requires committed source")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with ExitStack() as stack:
        if args.execute_rcaide or os.name == "nt":
            import msvcrt

            lockfile = stack.enter_context((args.qualification_plan.parent / ".x57-five-arm.lock").open("a+b"))
            lockfile.seek(0)
            msvcrt.locking(lockfile.fileno(), msvcrt.LK_NBLCK, 1)
        print("[prepare] Verifying the worker and building the frozen roster", flush=True)
        plan, problem, worker, reused = freeze(args, args.output)
        external_arms = () if (getattr(args, "recovery_pool_pilot", False) or getattr(args, "risk_screen_fork", False) or getattr(args, "long_pair", False)) else (COLD_EXTERNAL if cold else EXTERNAL)
        services = {arm: ExternalService(args.external_python, problem.base, problem.config_path,
                                         args.output, arm, stack, cold=cold, mock=cold and not args.execute_rcaide)
                    for arm in external_arms}
        if not cold:
            minimum = services["archsbo_native"].identity["kpls_minimum_complete_rows"]
            if args.execute_rcaide and minimum is not None and INITIAL < minimum:
                raise ValueError(f"The {INITIAL}-point DOE cannot supply {minimum} complete rows "
                                 "for native ArchSBO KPLS. No worker has been started.")
        plan["external_environments"] = {arm: service.identity for arm, service in services.items()}
        execution_code_sha = plan["code_sha"]
        path = args.output / "run_plan.json"
        if args.resume:
            plan = load_resume_plan(path, plan)
            previous_report = args.output / "report.json"
            if previous_report.is_file() and load(previous_report)["checkpoint"] > args.checkpoint:
                raise ValueError("Resume cannot rewind a reported checkpoint")
        else:
            dump(path, plan)
            print(f"Frozen plan {sha256(path)} at {path}", flush=True)
        campaign = Campaign(args, plan, problem, worker, reused)
        with campaign.progress:
            errors, trajectories = campaign.run(services)
            campaign.progress.phase("writing report")
            return report(campaign, errors, trajectories, services, execution_code_sha)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--external-service":
        external_service(*(Path(s) for s in sys.argv[2:5]), sys.argv[5],
                         cold="--cold-baselines" in sys.argv[6:], mock="--mock" in sys.argv[6:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--initial-replay", type=lambda p: Path(p).resolve())
    parser.add_argument("--qualification-plan", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--external-python", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--output", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--checkpoint", type=int, choices=range(ARCHIVED_INITIAL, HORIZON + 1))
    parser.add_argument("--cold-baselines", action="store_true")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--recovery-pool-pilot", action="store_true")
    profile.add_argument("--risk-screen-fork", action="store_true")
    profile.add_argument("--long-pair", action="store_true")
    profile.add_argument("--matched-external", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-rcaide", action="store_true")
    args = parser.parse_args()
    if (args.matched_external or args.risk_screen_fork or args.long_pair) and (args.initial_replay is None or not args.execute_rcaide):
        parser.error("--matched-external requires the archived --initial-replay and --execute-rcaide")
    if args.risk_screen_fork and args.checkpoint not in (16, 32):
        parser.error("The risk-screen fork supports checkpoints 16 and 32")
    if args.long_pair and args.checkpoint not in (32, 64, 128, 320):
        parser.error("The long paired comparison uses checkpoints 32, 64, 128, 320")
    if args.recovery_pool_pilot or args.matched_external or args.risk_screen_fork or args.long_pair:
        args.cold_baselines = True
    if args.checkpoint is None:
        args.checkpoint = COLD_CHECKPOINT if args.cold_baselines else INITIAL
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
