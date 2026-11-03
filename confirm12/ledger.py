"""Transactional budget accounting for the simulation adapter only."""
import hashlib
import json
import sqlite3
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def connect(path):
    db = sqlite3.connect(str(path), timeout=60, isolation_level=None)
    db.execute('PRAGMA busy_timeout=60000')
    db.execute('PRAGMA synchronous=FULL')
    return db


def create(path, limit, plan):
    path = Path(path)
    with path.open('xb'):
        pass
    with connect(path) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
          CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE calls (id TEXT PRIMARY KEY, role TEXT NOT NULL, candidate TEXT NOT NULL,
            candidate_hash TEXT NOT NULL, state TEXT NOT NULL, outcome TEXT, outcome_hash TEXT);
          CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL,
            event TEXT NOT NULL, payload_hash TEXT NOT NULL);
        ''')
        db.executemany('INSERT INTO metadata VALUES (?,?)', [('limit', str(limit)),
             ('plan_hash', digest(plan)), ('status', 'RUNNING'), ('kind', 'SYNTHETIC_ONLY')])


def start(path, call_id, role, candidate):
    value = canonical(candidate)
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        status = db.execute("SELECT value FROM metadata WHERE key='status'").fetchone()[0]
        count = db.execute('SELECT count(*) FROM calls').fetchone()[0]
        limit = int(db.execute("SELECT value FROM metadata WHERE key='limit'").fetchone()[0])
        if status != 'RUNNING' or count >= limit:
            raise RuntimeError('campaign stopped or budget exhausted')
        db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?)',
                   (call_id, role, value, digest(candidate), 'started', None, None))
        db.execute('INSERT INTO events(id,event,payload_hash) VALUES (?,?,?)',
                   (call_id, 'started', digest(candidate)))
        db.execute('COMMIT')


def settle(path, call_id, outcome):
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        updated = db.execute("UPDATE calls SET state='settled', outcome=?, outcome_hash=? "
                             "WHERE id=? AND state='started'", (canonical(outcome), digest(outcome), call_id))
        if updated.rowcount != 1:
            raise RuntimeError('unknown or already settled call')
        db.execute('INSERT INTO events(id,event,payload_hash) VALUES (?,?,?)',
                   (call_id, 'settled', digest(outcome)))
        db.execute('COMMIT')


def stop(path):
    with connect(path) as db:
        db.execute("UPDATE metadata SET value='INCOMPLETE' WHERE key='status' AND value='RUNNING'")


def finish(path):
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        result = db.execute("UPDATE metadata SET value='COMPLETE_SYNTHETIC' "
                            "WHERE key='status' AND value='RUNNING'")
        if result.rowcount != 1:
            raise RuntimeError('cannot finish stopped campaign')
        db.execute('COMMIT')


def audit(path, plan, expected):
    with connect(path) as db:
        meta = dict(db.execute('SELECT key,value FROM metadata'))
        rows = db.execute('SELECT id,role,candidate,candidate_hash,state,outcome,outcome_hash FROM calls').fetchall()
        events = db.execute('SELECT id,event,payload_hash FROM events ORDER BY seq').fetchall()
    errors = []
    if meta.get('kind') != 'SYNTHETIC_ONLY' or meta.get('plan_hash') != digest(plan):
        errors.append('plan or adapter identity mismatch')
    if len(rows) != expected or len(rows) > int(meta['limit']):
        errors.append('unexpected call count or budget overflow')
    for ident, role, candidate, chash, state, outcome, ohash in rows:
        if digest(json.loads(candidate)) != chash or state != 'settled' or outcome is None:
            errors.append('invalid or unsettled call: ' + ident)
        elif digest(json.loads(outcome)) != ohash:
            errors.append('outcome hash mismatch: ' + ident)
    lookup = {}
    for ident, event, value in events:
        lookup.setdefault(ident, []).append((event, value))
    for ident, _, _, chash, _, _, ohash in rows:
        if lookup.get(ident) != [('started', chash), ('settled', ohash)]:
            errors.append('event mismatch: ' + ident)
    if len(lookup) != len(rows):
        errors.append('orphan events')
    return {'passed': not errors, 'errors': errors, 'started': len(rows),
            'settled': sum(r[4] == 'settled' for r in rows), 'limit': int(meta['limit']),
            'status': meta['status'], 'aircraft_calls': 0, 'rcaide_calls': 0}
