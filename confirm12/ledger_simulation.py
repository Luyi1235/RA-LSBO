"""Spawn integration using the new call ledger and simulated adapter contract."""
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import hashlib
import time
import traceback
from . import call_ledger as calls
from . import adapters

CASES = ['aircraft_ordinary', 'aircraft_sparse']
ARMS = list(adapters.QUALIFICATION)


def _initial(job):
    db, case, seed = job
    records = []
    for index in range(12):
        candidate = {'case': case, 'seed': seed, 'initial': index}
        ident = calls.start(db, case, seed, 'initial', '', index, candidate)
        outcome = {'simulated_value': index}
        calls.settle(db, ident, 'settled', outcome)
        records.append({'case': case, 'seed': seed, 'role': 'initial', 'arm': '',
                        'candidate': candidate, 'outcome': outcome})
    return case, seed, records


def _trial(job):
    root, db, case, seed, arm, initial = job
    directory = Path(root) / case / str(seed) / arm
    directory.mkdir(parents=True, exist_ok=False)
    history = list(initial)
    adapter = adapters.create(arm, adapters.Trial(case, seed, arm), scope='simulation')
    for index in range(18):
        candidate, _ = adapter.propose(history, [{'value': index}, {'value': index + 1}])
        ident = calls.start(db, case, seed, 'online', arm, index, candidate)
        outcome = {'simulated_value': candidate['value']}
        calls.settle(db, ident, 'settled', outcome)
        history.append({'case': case, 'seed': seed, 'role': 'online', 'arm': arm,
                        'candidate': candidate, 'outcome': outcome})
    ident = calls.start(db, case, seed, 'verification', arm, 0, {'simulated_check': True})
    calls.settle(db, ident, 'settled', {'simulated_check': 'passed'})
    with (directory / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump({'scope': 'simulation', 'history': history}, stream, sort_keys=True)
    return case, seed, arm


def audit(output):
    output = Path(output)
    plan = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
    report = calls.audit(output / 'calls.sqlite', plan)
    with calls._read_connect(output / 'calls.sqlite') as db:
        rows = db.execute('SELECT case_name,seed,role,arm,idx,candidate,outcome FROM calls').fetchall()
    index = {(c, s, r, a, i): (json.loads(x), json.loads(y) if y else None)
             for c, s, r, a, i, x, y in rows}
    file_hashes = {}
    try:
        for case in plan['cases']:
            for seed in plan['seeds']:
                initial = [index[(case, seed, 'initial', '', i)] for i in range(plan['initial'])]
                for arm in plan['arms']:
                    path = output / case / str(seed) / arm / 'result.json'
                    raw = path.read_bytes()
                    file_hashes[path.relative_to(output).as_posix()] = hashlib.sha256(raw).hexdigest()
                    data = json.loads(raw)
                    history = data['history']
                    if data['scope'] != 'simulation' or len(history) != 30:
                        raise ValueError('invalid result scope or history length')
                    for i, record in enumerate(history):
                        role, owner, j = ('initial', '', i) if i < 12 else ('online', arm, i - 12)
                        if ((record['case'], record['seed'], record['role'], record['arm']) !=
                                (case, seed, role, owner) or
                                (record['candidate'], record['outcome']) != index[(case, seed, role, owner, j)]):
                            raise ValueError('result differs from ledger or contains foreign history')
                    if [(x['candidate'], x['outcome']) for x in history[:12]] != initial:
                        raise ValueError('shared initial mismatch')
                    index[(case, seed, 'verification', arm, 0)]
    except (KeyError, OSError, ValueError, TypeError) as exc:
        report['errors'].append('result reconciliation: ' + str(exc))
        report['integrity_passed'] = False
        report['campaign_complete'] = False
    report['result_file_sha256'] = file_hashes
    return report


def run(output, workers=12, max_wall_seconds=86400):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if workers != 12 or max_wall_seconds <= 0:
        raise ValueError('simulation requires twelve workers and a positive deadline')
    plan = {'cases': CASES, 'seeds': [1], 'arms': ARMS, 'initial': 12,
            'online': 18, 'verification': 1, 'max_calls': 290}
    (output / 'plan.json').write_text(json.dumps(plan, sort_keys=True), encoding='utf-8')
    db = str(output / 'calls.sqlite')
    calls.create(db, plan)
    start_time = time.monotonic()
    deadline = start_time + max_wall_seconds
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            initial_futures = [pool.submit(_initial, (db, case, 1)) for case in CASES]
            done, pending = concurrent.futures.wait(initial_futures, timeout=max(0, deadline - time.monotonic()))
            if pending:
                calls.stop(db)
                raise TimeoutError('simulation deadline reached during initialization')
            initial = [future.result() for future in initial_futures]
            print('SIMULATED shared initial completed: 24 calls', flush=True)
            jobs = iter((str(output), db, case, seed, arm, history)
                        for case, seed, history in initial for arm in ARMS)
            total, count, exhausted, pending = len(initial) * len(ARMS), 0, False, set()
            while pending or not exhausted:
                while len(pending) < workers and not exhausted and time.monotonic() < deadline:
                    try:
                        pending.add(pool.submit(_trial, next(jobs)))
                    except StopIteration:
                        exhausted = True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    calls.stop(db)
                    raise TimeoutError('simulation deadline reached; no more trials dispatched')
                if not pending:
                    break
                done, pending = concurrent.futures.wait(pending, timeout=remaining,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                if not done:
                    calls.stop(db)
                    raise TimeoutError('simulation deadline reached; no more trials dispatched')
                for future in done:
                    try:
                        case, seed, arm = future.result()
                    except BaseException:
                        calls.stop(db)
                        raise
                    count += 1
                    elapsed = time.monotonic() - start_time
                    with calls._connect(db) as connection:
                        charged = connection.execute('SELECT count(*) FROM calls').fetchone()[0]
                    eta = elapsed * (total - count) / count
                    print(f'SIMULATED {case} {seed} {arm}; trials {count}/{total}; '
                          f'calls {charged}/{plan["max_calls"]}; elapsed {elapsed:.1f}s; '
                          f'rough ETA {eta:.1f}s', flush=True)
        report = audit(output)
        if not report['integrity_passed'] or report['charged_calls'] != plan['max_calls']:
            raise RuntimeError('simulation audit incomplete')
        calls.finish_simulation(db, plan['max_calls'])
        report = audit(output)
        if not report['campaign_complete']:
            raise RuntimeError('simulation completion audit failed')
    except BaseException:
        calls.stop(db)
        with (output / 'error.txt').open('x', encoding='utf-8') as stream:
            traceback.print_exc(file=stream)
        raise
    with (output / 'audit.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    return report
