"""Two-stage Windows-spawn synthetic integration test, never an aircraft adapter."""
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import time
import traceback
from . import ledger

CASES = ('aircraft_ordinary', 'aircraft_sparse')
ARMS = ('ra_lsbo', 'full_latent', 'full_onehot', 'reverse_representation',
        'interval_as_missing', 'random', 'cbob_adapted')


def _evaluate(db, ident, role, candidate, fail=False):
    ledger.start(db, ident, role, candidate)
    if fail:
        ledger.stop(db)
        raise RuntimeError('injected failure after started; no retry')
    value = {'synthetic_value': int(ledger.digest(candidate)[:8], 16)}
    ledger.settle(db, ident, value)
    return {'candidate': candidate, 'outcome': value}


def _initial(job):
    root, db, case, seed = job
    folder = Path(root) / case / str(seed) / 'shared'
    folder.mkdir(parents=True, exist_ok=False)
    history = [_evaluate(db, f'{case}:{seed}:initial:{i}', 'initial',
                         {'case': case, 'seed': seed, 'index': i}) for i in range(12)]
    with (folder / 'history.json').open('x', encoding='utf-8') as stream:
        json.dump(history, stream, sort_keys=True)
    return (case, seed, history)


def _arm(job):
    root, db, case, seed, arm, initial = job
    folder = Path(root) / case / str(seed) / arm
    folder.mkdir(parents=True, exist_ok=False)
    history = list(initial)
    for i in range(18):
        history.append(_evaluate(db, f'{case}:{seed}:{arm}:online:{i}', 'online',
                     {'case': case, 'seed': seed, 'arm': arm, 'index': i,
                      'history_hash': ledger.digest(history)}))
    _evaluate(db, f'{case}:{seed}:{arm}:verification:0', 'verification',
              {'case': case, 'seed': seed, 'arm': arm, 'history_hash': ledger.digest(history)})
    with (folder / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump({'initial_hash': ledger.digest(initial), 'history': history}, stream, sort_keys=True)
    return (case, seed, arm)


def audit_simulation(output, plan, maximum):
    output = Path(output)
    report = ledger.audit(output / 'ledger.sqlite', plan, maximum)
    with ledger.connect(output / 'ledger.sqlite') as db:
        calls = {r[0]: (r[1], json.loads(r[2]), json.loads(r[3]) if r[3] else None)
                 for r in db.execute('SELECT id,role,candidate,outcome FROM calls')}
    expected_ids = set()
    for case in plan['cases']:
        for seed in plan['seeds']:
            shared_path = output / case / str(seed) / 'shared/history.json'
            try:
                initial = json.loads(shared_path.read_text(encoding='utf-8'))
                if len(initial) != 12 or any(x['candidate']['case'] != case or
                        x['candidate']['seed'] != seed for x in initial):
                    raise ValueError('invalid shared history')
                for index, record in enumerate(initial):
                    ident = f'{case}:{seed}:initial:{index}'
                    expected_ids.add(ident)
                    if calls.get(ident) != ('initial', record['candidate'], record['outcome']):
                        raise ValueError('shared history and ledger differ')
                for arm in plan['arms']:
                    result = json.loads((shared_path.parent.parent / arm / 'result.json').read_text(encoding='utf-8'))
                    history = result['history']
                    if result['initial_hash'] != ledger.digest(initial) or history[:12] != initial or len(history) != 30:
                        raise ValueError('shared history or length mismatch')
                    for index, record in enumerate(history[12:]):
                        c = record['candidate']
                        ident = f'{case}:{seed}:{arm}:online:{index}'
                        expected_ids.add(ident)
                        if (c['case'], c['seed'], c['arm'], c['index'], c['history_hash']) != (
                                case, seed, arm, index, ledger.digest(history[:12 + index])):
                            raise ValueError('algorithm history isolation mismatch')
                        if calls.get(ident) != ('online', c, record['outcome']):
                            raise ValueError('online history and ledger differ')
                    ident = f'{case}:{seed}:{arm}:verification:0'
                    expected_ids.add(ident)
                    verify = calls.get(ident)
                    if not verify or verify[0] != 'verification' or verify[1] != {
                            'case': case, 'seed': seed, 'arm': arm, 'history_hash': ledger.digest(history)}:
                        raise ValueError('verification and ledger differ')
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                report['errors'].append(f'{case}:{seed}: {exc}')
    if set(calls) != expected_ids:
        report['errors'].append('missing, duplicate or unexpected call identities')
    report['passed'] = not report['errors']
    return report


def run(output, seeds=(1,), workers=12, wall_seconds=300):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not 1 <= workers <= 12 or len(set(seeds)) != len(seeds) or wall_seconds <= 0:
        raise ValueError('invalid rehearsal settings')
    plan = {'kind': 'SYNTHETIC_ONLY', 'cases': CASES, 'seeds': list(seeds), 'arms': ARMS,
            'initial': 12, 'online': 18, 'verify': 1, 'workers': workers}
    (output / 'plan.json').write_text(ledger.canonical(plan), encoding='utf-8')
    db = str(output / 'ledger.sqlite')
    maximum = len(CASES) * len(seeds) * (12 + len(ARMS) * 19)
    ledger.create(db, maximum, plan)
    deadline = time.monotonic() + wall_seconds
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            shared = list(pool.map(_initial, [(str(output), db, case, seed)
                                             for case in CASES for seed in seeds]))
            print(f'Synthetic shared initial groups complete: {len(shared)}', flush=True)
            if time.monotonic() >= deadline:
                raise TimeoutError('deadline reached before online dispatch')
            jobs = iter((str(output), db, case, seed, arm, hist)
                        for case, seed, hist in shared for arm in ARMS)
            pending = set()
            exhausted = False
            while pending or not exhausted:
                while len(pending) < workers and not exhausted and time.monotonic() < deadline:
                    try:
                        pending.add(pool.submit(_arm, next(jobs)))
                    except StopIteration:
                        exhausted = True
                if not pending:
                    if not exhausted:
                        raise TimeoutError('deadline reached with undispatched trials')
                    break
                done, pending = concurrent.futures.wait(pending,
                            return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    try:
                        case, seed, arm = future.result()
                        print(f'Synthetic trial complete: {case} {seed} {arm}', flush=True)
                    except BaseException:
                        ledger.stop(db)
                        raise
                if time.monotonic() >= deadline and not exhausted:
                    raise TimeoutError('deadline reached; no more trials dispatched')
        report = audit_simulation(output, plan, maximum)
        if not report['passed']:
            raise RuntimeError('independent synthetic audit failed')
        ledger.finish(db)
        report['status'] = 'COMPLETE_SYNTHETIC'
    except BaseException:
        ledger.stop(db)
        with (output / 'error.txt').open('x', encoding='utf-8') as stream:
            traceback.print_exc(file=stream)
        raise
    with (output / 'audit.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    return report
