"""Portable, read-only import of ONE reviewed phase campaign.

Does not execute source snapshots, refit models, rewrite paths/hashes or call the
current strict load_plan on a foreign Windows checkout. This verifies a pinned
published evidence seal, not a substitute for its original runtime audit.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath

REVIEWED_FILES = {
    'plan.json': 'b0046d5cf0a1e76c6e3a12760115460c5ef5677b4062b178cb8984a693784950',
    'audit.json': 'dca96bc4ee699e27abacb7cbb3a76a8be75da97da913142fc5afbf2e15938391',
    'results.json': '95527bb0f87f99954b846afb5365d94e80d8e5b278c26289ad0dd5359d08c472',
    'registration.json': '729de3543185d9afb28de8323b74d8d4545bf989c17f009a38cc07d369d3f28f',
}
REVIEWED_COMMIT = '155f0b0237183d3feefa124e430c01bcd5db1450'


def prerequisite_receipt():
    return {'kind':'pinned_phase_evidence_import_v1', 'stage':'phase',
            'source_commit':REVIEWED_COMMIT, 'published_files_sha256':dict(REVIEWED_FILES),
            'settled_calls':8916, 'trial_count':280, 'checked_proposals':8400,
            'decision':'freeze_phase_control_test_representation_transfer',
            'algorithm_replay':False, 'new_model_calls':0}


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _read(root, relative):
    p = PurePosixPath(relative)
    if (p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative
            or not p.parts or str(p) != relative):
        raise ValueError('Unsafe evidence path')
    path = root.joinpath(*p.parts)
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Symlink evidence is not accepted')
    if not path.is_file():
        raise ValueError('Missing prerequisite evidence: '+relative)
    return path.read_bytes()


def verify_reviewed_phase(folder):
    root = Path(folder).absolute()
    def raw(name): return _read(root,name)
    def read(name): return json.loads(raw(name))
    def lines(name): return [json.loads(x) for x in raw(name).splitlines() if x.strip()]
    for name, expected in REVIEWED_FILES.items():
        if hashlib.sha256(raw(name)).hexdigest() != expected:
            raise ValueError('Not the reviewed phase artifact: '+name)
    plan, seal, results, registration = [read(n+'.json') for n in ('plan','audit','results','registration')]
    if (plan['source_commit'] != REVIEWED_COMMIT or plan['stage'] != 'phase'
            or plan['profile']['name'] != 'pilot' or not seal['passed']
            or registration['plan_sha256'] != REVIEWED_FILES['plan.json']):
        raise ValueError('Reviewed phase identity differs')
    names = {p.relative_to(root).as_posix() for p in root.rglob('*')
             if p.is_file() and p.suffix in {'.json','.jsonl'} and p.name != 'audit.json'}
    if names != set(seal['evidence_hashes']):
        raise ValueError('Prerequisite evidence set differs from its seal')
    for name, expected in seal['evidence_hashes'].items():
        if hashlib.sha256(raw(name)).hexdigest() != expected:
            raise ValueError('Prerequisite evidence hash changed: '+name)
    for expected in plan['sources'].values():
        if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('Invalid snapshot identity')
        if hashlib.sha256(raw('source_snapshot/'+expected+'.source')).hexdigest() != expected:
            raise ValueError('Prerequisite snapshot changed')
    events, ids, tags, roles = lines('calls.jsonl'), set(), set(), Counter()
    if len(events) != 2*8916:
        raise ValueError('Incomplete reviewed ledger')
    for number,(start,end) in enumerate(zip(events[::2],events[1::2]),1):
        payload = {k:start[k] for k in ('number','tag','role','candidate','contract','new_rcaide_mda')}
        record = end['record']
        if (start['event'] != 'started' or end['event'] != 'settled' or start['number'] != number
                or start['call_id'] != _digest(payload) or end['call_id'] != start['call_id']
                or start['call_id'] in ids or start['tag'] in tags or start['tag'] != end['tag']
                or record['candidate'] != start['candidate'] or end['record_digest'] != _digest(record)
                or record['observation']['design_id'] != _digest(start['candidate'])
                or record['observation']['contract_id'] != start['contract'] or start['new_rcaide_mda'] is not False):
            raise ValueError('Reviewed ledger linkage differs')
        ids.add(start['call_id']); tags.add(start['tag']); roles[start['role']] += 1
    if (dict(roles) != results['charged_by_role'] or results['charged_model_calls'] != 8916
            or seal['checked_proposals'] != 8400 or seal['trial_count'] != 280):
        raise ValueError('Reviewed ledger totals differ')
    # Recheck header bytes after scanning. Raw evidence is never rewritten.
    for name, expected in REVIEWED_FILES.items():
        if hashlib.sha256(raw(name)).hexdigest() != expected:
            raise ValueError('Prerequisite changed during verification')
    return prerequisite_receipt()
