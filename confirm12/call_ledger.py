"""Write-once call-accounting prototype; no physical evaluator is provided.

Unknown calls remain charged. This module does not authorize an experiment.
"""
import json
import sqlite3
from pathlib import Path
from .ledger import canonical, digest

TERMINAL = ('settled', 'unknown', 'failed')


def _connect(path):
    uri = Path(path).resolve().as_uri() + '?mode=rw'
    db = sqlite3.connect(uri, uri=True, timeout=60, isolation_level=None)
    db.execute('PRAGMA busy_timeout=60000')
    db.execute('PRAGMA synchronous=FULL')
    return db


def _read_connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=60)


def _event(db, call_id, kind, payload_hash):
    last = db.execute('SELECT seq,chain_hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
    seq, prev = (last[0] + 1, last[1]) if last else (1, '0' * 64)
    chain = digest({'seq': seq, 'call_id': call_id, 'kind': kind,
                    'payload_hash': payload_hash, 'previous': prev})
    db.execute('INSERT INTO events VALUES (?,?,?,?,?,?)', (seq, call_id, kind, payload_hash, prev, chain))


def create(path, plan):
    if not isinstance(plan.get('max_calls'), int) or plan['max_calls'] <= 0:
        raise ValueError('positive frozen budget required')
    if not plan.get('cases') or not plan.get('arms') or not plan.get('seeds'):
        raise ValueError('frozen case, arm and seed lists required')
    if any(len(x) != len(set(x)) for x in (plan['cases'], plan['arms'], plan['seeds'])):
        raise ValueError('duplicate frozen identity')
    with Path(path).open('xb'):
        pass
    with _connect(path) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
          CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE calls(id TEXT PRIMARY KEY,case_name TEXT NOT NULL,seed INTEGER NOT NULL,
            role TEXT NOT NULL,arm TEXT NOT NULL,idx INTEGER NOT NULL,candidate TEXT NOT NULL,
            candidate_hash TEXT NOT NULL,state TEXT NOT NULL,outcome TEXT,outcome_hash TEXT,
            UNIQUE(case_name,seed,role,arm,idx));
          CREATE TABLE events(seq INTEGER PRIMARY KEY,call_id TEXT NOT NULL,kind TEXT NOT NULL,
            payload_hash TEXT NOT NULL,previous TEXT NOT NULL,chain_hash TEXT NOT NULL);
          CREATE TRIGGER event_immutable_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'immutable event'); END;
          CREATE TRIGGER event_immutable_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'immutable event'); END;
        ''')
        db.executemany('INSERT INTO meta VALUES (?,?)', [('plan', canonical(plan)),
             ('plan_hash', digest(plan)), ('status', 'RUNNING'), ('scope', 'SIMULATED_ONLY')])
        _event(db, 'campaign', 'created', digest(plan))


def _identity(plan, case, seed, role, arm, index):
    if case not in plan['cases'] or type(seed) is not int or seed not in plan['seeds']:
        raise ValueError('unregistered case or seed')
    if type(index) is not int:
        raise ValueError('invalid index')
    if role == 'initial':
        if arm != '' or not 0 <= index < plan['initial']:
            raise ValueError('invalid shared initial identity')
    elif role in ('online', 'verification'):
        if arm not in plan['arms'] or not 0 <= index < plan[role]:
            raise ValueError('invalid arm or index')
        allowed = plan.get('allowed_arms')
        if allowed is not None and arm not in allowed.get(case, {}).get(str(seed), []):
            raise ValueError('arm not registered for case and seed')
    else:
        raise ValueError('invalid role')
    return {'case': case, 'seed': seed, 'role': role, 'arm': arm, 'index': index}


def start(path, case, seed, role, arm, index, candidate):
    with _connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        meta = dict(db.execute('SELECT key,value FROM meta'))
        plan = json.loads(meta['plan'])
        if digest(plan) != meta['plan_hash'] or meta['status'] != 'RUNNING':
            raise RuntimeError('plan changed or campaign stopped')
        identity = _identity(plan, case, seed, role, arm, index)
        if db.execute('SELECT count(*) FROM calls').fetchone()[0] >= plan['max_calls']:
            raise RuntimeError('budget exhausted')
        if role != 'initial':
            initial = db.execute("SELECT count(*) FROM calls WHERE case_name=? AND seed=? "
                "AND role='initial' AND state='settled'", (case, seed)).fetchone()[0]
            if initial != plan['initial']:
                raise RuntimeError('shared initial history not fully settled')
            prior = db.execute('SELECT count(*) FROM calls WHERE case_name=? AND seed=? AND arm=? '
                "AND role='online' AND state='settled'", (case, seed, arm)).fetchone()[0]
            if (role == 'online' and prior != index) or (role == 'verification' and prior != plan['online']):
                raise RuntimeError('online history incomplete or out of order')
        call_id = digest({'plan_hash': meta['plan_hash'], 'identity': identity})
        value = canonical(candidate)
        db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                   (call_id, case, seed, role, arm, index, value, digest(candidate), 'started', None, None))
        _event(db, call_id, 'started', digest(candidate))
        db.execute('COMMIT')
        return call_id


def settle(path, call_id, state, outcome):
    if state not in TERMINAL:
        raise ValueError('unrecognized terminal state')
    with _connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        updated = db.execute("UPDATE calls SET state=?,outcome=?,outcome_hash=? WHERE id=? AND state='started'",
                             (state, canonical(outcome), digest(outcome), call_id))
        if updated.rowcount != 1:
            raise RuntimeError('unknown or already settled call')
        _event(db, call_id, state, digest(outcome))
        if state != 'settled':
            db.execute("UPDATE meta SET value='INCOMPLETE' WHERE key='status'")
        db.execute('COMMIT')


def stop(path):
    with _connect(path) as db:
        db.execute("UPDATE meta SET value='INCOMPLETE' WHERE key='status' AND value='RUNNING'")


def finish_simulation(path, expected_calls):
    with _connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        counts = dict(db.execute('SELECT state,count(*) FROM calls GROUP BY state'))
        if counts != {'settled': expected_calls}:
            raise RuntimeError('simulation contains missing or non-settled calls')
        updated = db.execute("UPDATE meta SET value='COMPLETE_SIMULATED' "
                             "WHERE key='status' AND value='RUNNING' AND "
                             "EXISTS (SELECT 1 FROM meta WHERE key='scope' AND value='SIMULATED_ONLY')")
        if updated.rowcount != 1:
            raise RuntimeError('campaign not running or not simulated')
        _event(db, 'campaign', 'complete_simulated', digest({'settled_calls': expected_calls}))
        db.execute('COMMIT')


def audit(path, expected_plan):
    """Read-only consistency check; never reclassifies unsettled or failed calls."""
    with _read_connect(path) as db:
        meta = dict(db.execute('SELECT key,value FROM meta'))
        rows = db.execute('SELECT id,case_name,seed,role,arm,idx,candidate,candidate_hash,state,outcome,outcome_hash FROM calls').fetchall()
        events = db.execute('SELECT seq,call_id,kind,payload_hash,previous,chain_hash FROM events ORDER BY seq').fetchall()
    errors, states, by_id = [], {}, {}
    if meta.get('plan_hash') != digest(expected_plan) or meta.get('plan') != canonical(expected_plan):
        errors.append('frozen plan mismatch')
    if meta.get('scope') != 'SIMULATED_ONLY':
        errors.append('unauthorized accounting scope')
    if len(rows) > expected_plan['max_calls']:
        errors.append('budget exceeded')
    previous = '0' * 64
    for index, (seq, ident, kind, payload, prev, chain) in enumerate(events, 1):
        computed = digest({'seq': seq, 'call_id': ident, 'kind': kind,
                           'payload_hash': payload, 'previous': prev})
        if seq != index or prev != previous or chain != computed:
            errors.append('event chain mismatch')
        previous = chain
        by_id.setdefault(ident, []).append((kind, payload))
    campaign_events = [('created', digest(expected_plan))]
    if meta.get('status') == 'COMPLETE_SIMULATED':
        campaign_events.append(('complete_simulated', digest({'settled_calls': len(rows)})))
    if by_id.pop('campaign', None) != campaign_events:
        errors.append('missing creation event')
    for ident, case, seed, role, arm, index, candidate, chash, state, outcome, ohash in rows:
        states[state] = states.get(state, 0) + 1
        try:
            expected_id = digest({'plan_hash': digest(expected_plan),
                                  'identity': _identity(expected_plan, case, seed, role, arm, index)})
            if expected_id != ident or digest(json.loads(candidate)) != chash:
                errors.append('call identity or candidate mismatch')
            expected_events = [('started', chash)]
            if state in TERMINAL:
                if outcome is None or digest(json.loads(outcome)) != ohash:
                    errors.append('outcome mismatch')
                expected_events.append((state, ohash))
            elif state != 'started':
                errors.append('unexpected state')
            if by_id.pop(ident, None) != expected_events:
                errors.append('call event mismatch')
        except (ValueError, TypeError, KeyError):
            errors.append('invalid call content')
    if by_id:
        errors.append('orphan events')
    if any(states.get(x, 0) for x in ('started', 'unknown', 'failed')):
        errors.append('unsettled, unknown or failed calls require review')
    return {'integrity_passed': not errors, 'campaign_complete': not errors and
            meta.get('status') == 'COMPLETE_SIMULATED',
            'errors': errors, 'states': states, 'charged_calls': len(rows),
            'maximum_calls': expected_plan['max_calls'], 'status': meta.get('status'),
            'event_chain_head': previous, 'scope': meta.get('scope')}
