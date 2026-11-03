"""Independent read-only reconciliation for the PR #42 validation ledger.

Structural consistency never proves an evaluation was physical or approved.
This module does not reserve tickets, evaluate aircraft, or modify the ledger.
"""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

from ra_lsbo.partial_bo.records import DesignRecord
from . import structured_record_ledger as ledger


def audit(path, expected_plan):
    base = ledger.audit(path, expected_plan)
    uri = Path(path).resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as db:
        rows = db.execute('SELECT id,case_name,seed,role,arm,idx,candidate,state,record '
                          'FROM calls').fetchall()
        events = db.execute('SELECT seq,call_id,kind FROM events').fetchall()
    errors = list(base['errors'])
    started = {cid: seq for seq, cid, kind in events if kind == 'started'}
    settled = {cid: seq for seq, cid, kind in events if kind == 'settled'}
    initial, online, checks = defaultdict(list), defaultdict(list), []
    for cid, case, seed, role, arm, idx, candidate, state, raw_record in rows:
        row = (idx, cid, state, candidate, raw_record)
        if role == 'initial':
            initial[(case, seed)].append(row)
        elif role == 'online':
            online[(case, seed, arm)].append(row)
        elif role == 'verification':
            checks.append((case, seed, arm, row))
    for (case, seed, arm), items in online.items():
        shared = initial[(case, seed)]
        for idx, cid, _, _, _ in items:
            required = shared + [item for item in items if item[0] < idx]
            if (len(shared) != 12 or len(required) != 12 + idx or any(
                    item[1] not in settled or settled[item[1]] >= started.get(cid, -1)
                    for item in required)):
                errors.append('online started before complete settled prefix')
    for case, seed, arm, (idx, cid, _, candidate, _) in checks:
        shared = sorted(initial[(case, seed)])
        own = sorted(online[(case, seed, arm)])
        records = shared + own
        if (idx != 0 or len(shared) != 12 or len(own) != 18 or any(
                item[1] not in settled or settled[item[1]] >= started.get(cid, -1)
                for item in records)):
            errors.append('verification started before complete settled history')
            continue
        try:
            records = [DesignRecord.from_dict(json.loads(item[4])) for item in records]
            feasible = [record for record in records if record.usable_feasible]
            if not feasible or json.loads(candidate) != min(
                    feasible, key=lambda record: record.observation.objective.value).candidate:
                errors.append('verification differs from nominal best feasible design')
        except (ValueError, TypeError, KeyError, AttributeError):
            errors.append('verification history malformed')
    return {'structural_reconciliation_passed': not errors, 'errors': errors,
            'calls': base['calls'], 'states': base['states'],
            'scope': ledger.SCOPE, 'production_approved': False,
            'real_physics_provenance_verified': False,
            'campaign_complete': False}
