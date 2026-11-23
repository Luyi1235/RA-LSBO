"""Isolated worker protocol and durable dispatch/settlement records."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .x57_cold_feedback import normalize
from .x57_cold_sampling import active_names, violations
from .x57_live import evaluator_files


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def restore_assets(base, original, destination):
    """Materialize a byte-verified copy of the historical worker asset tree."""
    manifest = json.loads((base / "problems/x57/model_provenance.json").read_text(encoding="utf-8"))
    model = Path("problems/x57/model")
    assets = {}
    for name, expected in manifest["files_sha256"].items():
        relative = model / name
        checkout = base / relative
        data = checkout.read_bytes()
        if sha256(checkout) != expected:
            source = original / relative
            replacement = source.read_bytes()
            if (replacement.replace(b"\r\n", b"\n") != data.replace(b"\r\n", b"\n")
                    or sha256(source) != expected):
                raise ValueError(f"Cannot restore frozen asset {relative}")
            assets[relative] = source
    shutil.copytree(base, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative, source in assets.items():
        shutil.copyfile(source, destination / relative)
    for name, expected in manifest["files_sha256"].items():
        if sha256(destination / model / name) != expected:
            raise RuntimeError(f"Restored model identity mismatch: {name}")
    return {"asset_count": len(manifest["files_sha256"]),
            "restored": {p.as_posix(): sha256(source) for p, source in assets.items()},
            "worker_sha256": evaluator_files(destination)}


class Dispatcher:
    def __init__(self, problem, output, *, backend, worker_root, mock_profile="mixed",
                 timeout=1800., expected_worker=None, max_calls=None, energy_stop=False):
        if backend not in ("mock", "rcaide"):
            raise ValueError("Unknown worker backend")
        self.problem = problem
        self.output = output
        self.backend = backend
        self.worker_root = worker_root
        self.mock_profile = mock_profile
        self.timeout = timeout
        self.expected_worker = expected_worker
        self.max_calls = max_calls
        self.energy_stop = energy_stop
        self.count = 0
        self.sequence = 0
        self.elapsed = 0.
        self.journal = output / "calls.jsonl"
        self.journal.touch(exist_ok=False)

    def event(self, event, **fields):
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, **fields}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def evaluate(self, candidate, *, seed, arm, evaluation, online, phase, proposal):
        started = time.perf_counter()
        self.problem.verify_frozen()
        point = self.problem.canonical(candidate.to_dict())
        checks = self.problem.problem.precheck(candidate)
        self.sequence += 1
        call_id = f"{self.sequence:06d}"
        context = dict(call_id=call_id, seed=seed, arm=arm, evaluation=evaluation, online=online,
                       phase=phase, proposal=proposal, candidate=point,
                       candidate_id=self.problem.space.key(candidate),
                       active_variables=active_names(self.problem.space, candidate), precheck=checks)
        reasons = violations(checks)
        if reasons:
            self.event("precheck_rejected", **context, reasons=reasons, charged=False)
            raise ValueError(f"Dispatch refused a precheck-rejected candidate: {reasons}")
        if self.max_calls is not None and self.count >= self.max_calls:
            raise RuntimeError("Worker call limit reached")
        if self.backend == "rcaide" and (
                self.expected_worker is None or evaluator_files(self.worker_root) != self.expected_worker):
            raise RuntimeError("Frozen physical worker identity changed")
        folder = self.output / "workers" / call_id
        folder.mkdir(parents=True, exist_ok=False)
        request, response = folder / "request.json", folder / "response.json"
        worker_module = "problems.x57.worker" if self.backend == "rcaide" else "ra_lsbo.x57_cold_mock"
        raw = {"status": "worker_no_result", "solver_success": False}
        payload = {"candidate": point, "config": self.problem.problem.config}
        if self.backend == "mock":
            payload["mock"] = {
                "arm": arm, "online": online, "evaluation": evaluation,
                "profile": self.mock_profile, "required": self.problem.required_constraints,
                "precheck": checks,
            }
        dump(request, payload)
        context["worker_dir"] = str(folder)
        self.event("dispatch_intent", **context, backend=self.backend)
        charged = False
        process = None
        finished_at = None
        try:
            with (folder / "solver.log").open("wb") as log:
                entry = ["-m", worker_module]
                if self.backend == "rcaide" and self.energy_stop:
                    entry = [str(Path(__file__).resolve().parents[1] / "run_x57_energy_stop_worker.py")]
                command = [sys.executable, "-B", "-u", "-X", "faulthandler", *entry,
                           str(request), str(response)]
                try:
                    process = subprocess.Popen(command, cwd=self.worker_root, stdout=log,
                                               stderr=subprocess.STDOUT)
                    charged = True
                    self.count += 1
                    self.event("worker_started", **context, backend=self.backend,
                               charged=True, pid=process.pid, started_at_utc=utc_now())
                    returncode = process.wait(timeout=self.timeout)
                    finished_at = utc_now()
                    if returncode:
                        raw = {**raw, "status": "worker_nonzero_exit", "returncode": process.returncode}
                    elif response.is_file():
                        result = json.loads(response.read_text(encoding="utf-8"))
                        if not isinstance(result, dict):
                            raise ValueError("Worker response must be a JSON object")
                        json.dumps(result, allow_nan=False)
                        raw = result
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    finished_at = utc_now()
                    raw = {**raw, "status": "timeout"}
                except (OSError, ValueError) as error:
                    raw = {**raw, "status": "invalid_worker_response" if charged else "worker_launch_error",
                           "message": str(error)}
        except BaseException:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            self.event("interrupted", **context, charged=charged,
                       finished_at_utc=(finished_at or utc_now()) if charged else None,
                       elapsed_seconds=time.perf_counter() - started)
            raise
        elapsed = time.perf_counter() - started
        self.elapsed += elapsed
        feedback = normalize(raw, self.problem.required_constraints, checks, self.backend)
        status = "interrupted" if raw.get("status") == "timeout" else "settled"
        settlement = dict(**context, charged=charged, elapsed_seconds=elapsed, feedback=feedback,
                          finished_at_utc=finished_at,
                          worker_result=raw)
        self.event(status, **settlement)
        return settlement
