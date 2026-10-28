"""Run each aircraft evaluation in a fresh interpreter, preserving crash logs."""

from pathlib import Path
import codecs
import json
import os
import subprocess
import sys
import threading
import time
import uuid

from lsbo.records import Evaluation


def _relay_solver_log(path, stop, started, timeout, heartbeat_seconds):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    next_message = started + heartbeat_seconds
    try:
        with path.open("rb") as reader:
            while True:
                chunk = reader.read(65536)
                if chunk:
                    print(decoder.decode(chunk), end="", flush=True)
                    continue
                if stop.is_set():
                    print(decoder.decode(b"", final=True), end="", flush=True)
                    break
                now = time.perf_counter()
                if now >= next_message:
                    print(f"[RCAIDE] Waiting for worker response: {now - started:.0f}s elapsed "
                          f"(timeout {timeout:g}s)", flush=True)
                    next_message = now + heartbeat_seconds
                stop.wait(0.1)
    except (OSError, UnicodeError) as error:
        print(f"[RCAIDE] Console relay unavailable ({type(error).__name__}); inspect {path}", flush=True)


def _run_logged(command, root, log_path, timeout, heartbeat_seconds=15):
    # Tail the on-disk log instead of a stdout pipe: a silent/crashed solver
    # cannot block the parent from enforcing its existing process timeout.
    stop = threading.Event()
    with log_path.open("w", encoding="utf-8") as log:
        relay = threading.Thread(target=_relay_solver_log,
                                 args=(log_path, stop, time.perf_counter(), timeout, heartbeat_seconds),
                                 daemon=True)
        relay.start()
        try:
            return subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                  check=False)
        finally:
            log.flush()
            stop.set()
            relay.join()


def evaluate_isolated(candidate, config, work_dir, *, diagnostic_mode=None):
    started = time.perf_counter()
    folder = Path(work_dir).resolve() / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    request = folder / "request.json"
    response = folder / "response.json"
    payload={"candidate":candidate.to_dict(),"config":config}
    if diagnostic_mode is not None:
        if diagnostic_mode!='powered_lift_only': raise ValueError('Unsupported diagnostic mode')
        payload['diagnostic_mode']=diagnostic_mode
    request.write_text(json.dumps(payload), encoding="utf-8")
    command = [sys.executable, "-u", "-X", "faulthandler", "-m", "problems.x57.worker", str(request), str(response)]
    root = Path(__file__).resolve().parents[2]
    print(f"[RCAIDE] Solver log: {folder / 'solver.log'}", flush=True)
    try:
        completed = _run_logged(command, root, folder / "solver.log", config["runtime"]["timeout_seconds"])
    except subprocess.TimeoutExpired:
        return Evaluation(None, False, status="timeout", metrics={"worker_dir": str(folder)},
                          elapsed_seconds=time.perf_counter() - started)
    if not response.exists():
        return Evaluation(None, False, status="worker_no_result",
                          metrics={"returncode": completed.returncode, "worker_dir": str(folder)},
                          elapsed_seconds=time.perf_counter() - started)
    try:
        result = Evaluation(**json.loads(response.read_text(encoding="utf-8")))
    except (ValueError, TypeError) as error:
        return Evaluation(None, False, status="invalid_worker_response", metrics={"message": str(error), "worker_dir": str(folder)})
    if completed.returncode != 0:
        result = Evaluation(None, False, status="worker_nonzero_exit", metrics={"returncode": completed.returncode})
    result.metrics["worker_dir"] = str(folder)
    return result
