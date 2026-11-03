"""Zero-truth spawn scheduler for infrastructure rehearsal only."""
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

WORKERS = (1, 2, 4, 8, 12, 16)


def _write_new(path, data):
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _worker(task, root):
    """Read immutable checkpoint bytes and perform synthetic hashing, never inference."""
    started = time.perf_counter()
    directory = Path(root) / task['id']
    directory.mkdir(exist_ok=False)
    try:
        _write_new(directory / 'input.json', task)
        if task.get('fail'):
            raise RuntimeError('injected rehearsal failure')
        with open(task['checkpoint'], 'rb') as stream:
            payload = stream.read()
        digest = hashlib.sha256(payload + str(task['seed']).encode()).digest()
        for _ in range(task['rounds']):
            digest = hashlib.sha256(digest + payload).digest()
        result = {'id': task['id'], 'digest': digest.hex(),
                  'elapsed_seconds': time.perf_counter() - started, 'status': 'complete'}
        _write_new(directory / 'result.json', result)
        return result
    except BaseException:
        with open(directory / 'error.txt', 'x', encoding='utf-8') as stream:
            traceback.print_exc(file=stream)
        raise


def schedule(tasks, output, workers, max_wall_seconds, drain_seconds=0):
    """Dispatch synthetic tasks once; stop dispatch on failure or deadline, drain in-flight."""
    if workers not in WORKERS or workers > (os.cpu_count() or 1):
        raise ValueError('unsupported worker count')
    if max_wall_seconds <= 0 or drain_seconds < 0 or drain_seconds >= max_wall_seconds:
        raise ValueError('invalid time limit')
    ids = [task['id'] for task in tasks]
    if len(ids) != len(set(ids)) or any(not x.isalnum() for x in ids):
        raise ValueError('task identities must be unique and alphanumeric')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_new(output / 'manifest.json', {'tasks': tasks, 'workers': workers,
        'max_wall_seconds': max_wall_seconds, 'drain_seconds': drain_seconds,
        'truth_calls': 0, 'kind': 'synthetic_checkpoint_hash_rehearsal'})
    begin = time.monotonic()
    pending, completed, errors, next_index = {}, [], [], 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        while pending or next_index < len(tasks):
            while (not errors and next_index < len(tasks) and len(pending) < workers
                   and time.monotonic() - begin < max_wall_seconds - drain_seconds):
                task = tasks[next_index]
                next_index += 1
                pending[pool.submit(_worker, task, str(output))] = task['id']
            if not pending:
                break
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                identity = pending.pop(future)
                try:
                    completed.append(future.result())
                except BaseException as exc:
                    errors.append({'id': identity, 'type': type(exc).__name__, 'message': str(exc)})
    elapsed = time.monotonic() - begin
    result = {'status': 'complete' if not errors and len(completed) == len(tasks)
              and elapsed <= max_wall_seconds else 'incomplete', 'elapsed_seconds': elapsed,
              'dispatched': next_index, 'completed': sorted(completed, key=lambda x: x['id']),
              'errors': errors, 'undispatched': len(tasks) - next_index, 'truth_calls': 0}
    _write_new(output / 'summary.json', result)
    return result
