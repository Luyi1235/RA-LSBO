"""Run or resume the five X-57 arms with shared, frozen initialization."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stdout
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock

from ra_lsbo.budget import digest
from ra_lsbo.x57_cold_sampling import LAYOUTS, violations
from ra_lsbo.x57_cold_worker import Dispatcher, dump, sha256
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
RESULT_REFERENCE = "9c5ac77c5ecc3987787292981dc679045441972c"
RESULT_DIR = "x57-formal-prefix-qualification"
INPUTS = {
    f"{RESULT_DIR}/preregistered_47_5.json": "a984e0d42adbb9752c2da7fbc3a5e8e3805b9ef25ca361d9b02b6c094f576437",
    f"{RESULT_DIR}/physical/configs/distance_47.5.json": "9902197262e862b13a3b3ac4a75c6f2c7892c91d8a69d6ce1af8aa884f6e6c3d",
    f"{RESULT_DIR}/physical/report.json": "74092569b8a83e915e1cfefd25c0253506e1e51fb6ba980fbd33150fef5b64b1",
    f"{RESULT_DIR}/physical/calls.jsonl": "3b3996b794d0584f1fe76cf6bc563dccc169ba94649e196fc383d46ecaac87a8",
    "x57-cold-start-debug-fccf94c6/calls.jsonl": "990caac32365893a8100e3cc10660cd554e1e7a4391b4b467a74a570c77314df",
}
SEEDS = (307261, 356821, 412849, 470299, 528899, 590089)
NATIVE = ("lsbo_full_budget", "ra_lsbo_energy_sign", "ra_lsbo_energy_interval")
EXTERNAL = ("archsbo_screened", "logcei_screened_online")
ARMS = NATIVE + EXTERNAL
INITIAL = 8
HORIZON = 320
TARGET_WH = 47500
THREADS = (*THREAD_VARS, "NUMBA_NUM_THREADS")
BASELINE_VERSIONS = {"sb-arch-opt": "1.7.1", "smt": "2.14.1", "pymoo": "0.6.1.5",
                     "torch": "2.7.1", "scikit-learn": "1.6.1"}


def load(path):
    return json.loads(path.read_bytes())


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def check_threads():
    import os

    if any(os.environ.get(name) != "1" for name in THREADS):
        raise ValueError(f"Before starting Python, set {', '.join(THREADS)} to 1")


def external_service(base, config, work, arm):
    from ra_lsbo.x57_external import X57External

    check_threads()
    versions = {name: version(name) for name in BASELINE_VERSIONS}
    if versions != BASELINE_VERSIONS or sys.version_info[:2] != (3, 11):
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
                elif op == "init":
                    trajectories[seed] = X57External(problem, arm, seed, request["initial"], HORIZON)
                    value = None
                elif op == "drop":
                    del trajectories[seed]
                    value = None
                elif op == "ask":
                    value = trajectories[seed].ask()
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
    def __init__(self, python, base, config, output, arm, stack):
        log = stack.enter_context((output / f"{arm}.optimizer.log").open("a", encoding="utf-8"))
        self.process = subprocess.Popen(
            [str(python), "-B", "-u", str(Path(__file__).resolve()), "--external-service",
             str(base), str(config), str(output / "unused"), arm],
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
    seeds = SEEDS if args.execute_rcaide else SEEDS[:1]
    seen = set(historical_ids)
    rosters = {}
    for seed in seeds:
        _, records, audit = points(problem, seed, seen, INITIAL)
        rosters[str(seed)] = {"records": records, "sampling": audit}
        seen.update(r["candidate_id"] for r in records)
    plan = {
        "schema": "x57_five_arm_v1", "code_sha": code_sha(), "code_base": CODE_REFERENCE,
        "result_reference": RESULT_REFERENCE, "input_sha256": INPUTS,
        "qualification_plan_sha256": QUALIFICATION_PLAN_SHA256, "config_sha256": sha256(config),
        "worker_manifest_sha256": digest(worker), "worker_overlay_sha256": overlay,
        "environment": environment, "contract_id": problem.contract_id, "distance_nmi": 47.5,
        "quality_target_Wh": TARGET_WH, "target_basis": "Ceil best prior confirmation to 500 Wh",
        "best_prior_confirmation_Wh": best, "45_nmi_quality_target_Wh": 46000,
        "seeds": list(seeds), "seed_selection": "Fixed before evaluation, independent of prior seeds",
        "arms": list(ARMS), "initial_count": INITIAL, "total_budget_per_arm": HORIZON,
        "first_checkpoint": 16, "first_checkpoint_call_cap": len(seeds) * (INITIAL + len(ARMS) * 8),
        "max_workers": args.workers, "timeout_seconds": TIMEOUT, "thread_settings": dict.fromkeys(THREADS, "1"),
        "backend": "rcaide" if args.execute_rcaide else "mock",
        "confirmation_exclusions": sorted(confirmation_ids), "initial_exclusions": sorted(historical_ids),
        "rosters": rosters,
    }
    return plan, problem, worker


def group(initial):
    count = sum(row["feedback"]["usable"] for row in initial)
    return "0" if count == 0 else "1-2" if count <= 2 else "3+"


class Campaign:
    def __init__(self, args, plan, problem, worker):
        self.args, self.plan, self.problem, self.worker = args, plan, problem, worker
        self.output = args.output
        self.halt = Event()
        self.start_lock = Lock()
        self.admitted = 0
        self.cap = len(plan["seeds"]) * (INITIAL + len(ARMS) * (args.checkpoint - INITIAL))
        self.initial = {}

    def evaluate(self, job, point, *, seed, arm, evaluation, online, phase, proposal):
        key = self.problem.space.key(point)
        candidate = self.problem.canonical(point.to_dict())
        if (violations(self.problem.problem.precheck(point))
                or candidate != point.to_dict()
                or (online and key in self.plan["confirmation_exclusions"])):
            raise ValueError(f"Candidate failed dispatch identity or eligibility checks: {job}")
        out = self.output / job
        expected = ("dispatch_intent", "worker_started", "settled")
        if out.exists():
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
            row = dispatcher.evaluate(point, seed=seed, arm=arm, evaluation=evaluation,
                                      online=online, phase=phase, proposal=proposal)
        if not row["charged"] or row["feedback"]["outcome"] == "infrastructure_error":
            raise RuntimeError(f"Infrastructure failure or uncharged settlement: {job}")
        row = {k: v for k, v in row.items() if k != "event"}
        row["source_id"] = job
        return row

    def dispatch_initial(self, seed, record):
        point = self.problem.candidate_class(**record["candidate"])
        if (self.problem.space.key(point) != record["candidate_id"]
                or self.problem.problem.precheck(point) != record["precheck"]):
            raise ValueError("Frozen initial candidate or precheck changed")
        evaluation = record["position"]
        job = f"seed-{seed}/shared/{evaluation:03d}"
        return self.evaluate(job, point, seed=seed, arm="shared", evaluation=evaluation,
                             online=0, phase="formal_shared_initial", proposal={"frozen_task": job})

    def trajectory(self, seed, arm, services):
        output = self.output / f"seed-{seed}" / arm
        output.mkdir(parents=True, exist_ok=True)
        initial = self.initial[seed]
        if arm in NATIVE:
            from tempfile import TemporaryDirectory
            from ra_lsbo.x57_formal_native import FormalNative

            with TemporaryDirectory(dir=self.output, prefix="optimizer-") as temp:
                adapter = FormalNative(self.problem, arm, seed, initial, Path(temp) / "native")

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
            save(checkpoint_path, {"plan_sha256": plan_hash, "evaluation": INITIAL + len(rows),
                                   "optimizer": checkpoint()})

        for evaluation in range(INITIAL + 1, self.args.checkpoint + 1):
            job = f"seed-{seed}/{arm}/{evaluation:03d}"
            if self.halt.is_set() and not (self.output / job).exists():
                break
            raw, phase, info = ask()
            point = self.problem.candidate_class(**raw)
            if previous and previous["evaluation"] == evaluation:
                if previous["candidate_id"] != self.problem.space.key(point) or previous["phase"] != phase:
                    raise RuntimeError(f"Next proposal changed upon resume: {job}")
                replay_verified = True
            row = self.evaluate(job, point, seed=seed, arm=arm, evaluation=evaluation,
                                online=evaluation - INITIAL, phase=phase, proposal=info)
            if row is None:
                break
            tell(row)
            rows.append(row)
            if previous_checkpoint and evaluation == previous_checkpoint["evaluation"]:
                if checkpoint() != previous_checkpoint["optimizer"]:
                    raise RuntimeError(f"Settlement replay changed optimizer state: {job}")
            if not previous_checkpoint or evaluation >= previous_checkpoint["evaluation"]:
                persist()
        if not self.args.execute_rcaide and self.args.checkpoint < HORIZON and not self.halt.is_set():
            raw, phase, _ = ask()
            next_key = self.problem.space.key(self.problem.candidate_class(**raw))
            save(previews, {"evaluation": self.args.checkpoint + 1, "candidate_id": next_key, "phase": phase})
        return {"rows": rows, "replay_verified": replay_verified}

    def run(self, services):
        errors = {}

        def wave(items, launch):
            def guarded(item):
                if self.halt.is_set():
                    return None
                try:
                    return launch(*item)
                except Exception:
                    self.halt.set()
                    raise

            results = {}
            with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
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
        shared = wave(items, self.dispatch_initial)
        for seed in self.plan["seeds"]:
            rows = [shared.get((seed, i)) for i in range(1, INITIAL + 1)]
            if all(rows):
                self.initial[seed] = rows
        print(f"Shared prefixes complete: {len(self.initial)}; "
              f"groups: {dict(Counter(group(rows) for rows in self.initial.values()))}", flush=True)
        items = [((seed, arm), (seed, arm, services))
                 for seed in sorted(self.initial, key=lambda seed: group(self.initial[seed])) for arm in ARMS]
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
    return {"first_usable_call": first, "first_quality_call": quality, "best_usable_Wh": best,
            "unique_partial_rows_in_engineering_fit": len(partial), "partial_fit_ids": sorted(partial),
            "verified_energy_lower_bound_rows": len(proven), "outcomes": dict(Counter(r["outcome"] for r in series)),
            "trajectory": series, "arm": arm}


def report(campaign, errors, trajectories, services):
    output = campaign.output
    windows = []
    ledger = output / "calls.jsonl"
    events = []
    keys = [(seed, "shared", i) for seed in campaign.plan["seeds"] for i in range(1, INITIAL + 1)]
    keys.extend((seed, arm, i) for seed in campaign.plan["seeds"] for arm in ARMS
                for i in range(INITIAL + 1, campaign.args.checkpoint + 1))
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
        for arm in ARMS:
            result = trajectories.get((seed, arm))
            if result:
                summaries[str(seed)][arm] = history_stats(initial, result["rows"], arm)
                replay[f"{seed}/{arm}"] = result["replay_verified"]
    complete = (not errors and len(windows) == campaign.cap and
                all(len(trajectories.get((seed, arm), {}).get("rows", [])) == campaign.args.checkpoint - INITIAL
                    for seed in campaign.plan["seeds"] for arm in ARMS))
    groups = {}
    for label in ("0", "1-2", "3+"):
        members = [seed for seed in initial_groups if initial_groups[seed] == label]
        by_arm = {}
        for arm in ARMS:
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
        "plan_sha256": sha256(output / "run_plan.json"), "checkpoint": campaign.args.checkpoint,
        "total_budget_per_arm": HORIZON, "quality_target_Wh": TARGET_WH,
        "actual_rcaide_calls": len(windows) if campaign.args.execute_rcaide else 0,
        "worker_starts": len(windows), "peak_workers": peak, "worker_times": windows,
        "external_environments": {arm: service.identity for arm, service in services.items()},
        "initial_groups": initial_groups, "group_metrics": groups, "per_seed": summaries,
        "next_proposal_replay_verified": replay,
    }
    save(output / "report.json", result)
    print(f"{result['status']} | starts={len(windows)} | peak={peak} | output={output}", flush=True)
    return 0 if complete else 1


def run(args):
    check_threads()
    if args.execute_rcaide and not args.resume and args.checkpoint != 16:
        raise ValueError("The initial physical campaign is authorized through checkpoint 16")
    if args.execute_rcaide:
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT)
        if status:
            raise ValueError("Physical campaign requires committed source")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with ExitStack() as stack:
        import msvcrt
        lockfile = stack.enter_context((args.qualification_plan.parent / ".x57-five-arm.lock").open("a+b"))
        lockfile.seek(0)
        msvcrt.locking(lockfile.fileno(), msvcrt.LK_NBLCK, 1)
        plan, problem, worker = freeze(args, args.output)
        services = {arm: ExternalService(args.external_python, problem.base, problem.config_path,
                                         args.output, arm, stack) for arm in EXTERNAL}
        plan["external_environments"] = {arm: service.identity for arm, service in services.items()}
        path = args.output / "run_plan.json"
        if args.resume:
            if plan != load(path):
                raise ValueError("Resume requires the original run plan and source identities")
        else:
            dump(path, plan)
            print(f"Frozen plan {sha256(path)} at {path}", flush=True)
        campaign = Campaign(args, plan, problem, worker)
        return report(campaign, *campaign.run(services), services)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--external-service":
        external_service(*(Path(s) for s in sys.argv[2:5]), sys.argv[5])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--qualification-plan", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--external-python", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--output", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--checkpoint", type=int, choices=range(INITIAL + 1, HORIZON + 1), default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-rcaide", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
