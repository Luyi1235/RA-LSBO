"""Portable, read-only import of ONE reviewed historical campaign.

Does not execute source snapshots, refit models, rewrite paths/hashes or call the
current strict load_plan on a foreign Windows checkout. This verifies a pinned
published evidence seal, not a substitute for its original runtime audit.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath

REVIEWED_FILES = {
    'plan.json': '3b2013aadd6cc999483c04225e7e09760abadcac5f6178690ba0a97087c00af4',
    'audit.json': 'a66d885395f6584efad2a78127c67bba9ff3fd733e840b6efaaddf3be4aa1b44',
    'results.json': 'b54612217d5d48174ec4c5ba49ac7de38228112ee0de6968c46375eb98704398',
    'registration.json': 'c1c0519d0736671fd6f092ec852c463e6b9d16f4547e9fbfa4008febdb2ca6c6',
}
REVIEWED_COMMIT = 'cfdf7c30216585b3920f23723236708859435ce7'


def prerequisite_receipt():
    return {'kind':'pinned_discovery_evidence_import_v1', 'stage':'discovery',
            'source_commit':REVIEWED_COMMIT, 'published_files_sha256':dict(REVIEWED_FILES),
            'settled_calls':6516, 'trial_count':200, 'checked_proposals':6000,
            'decision':'withhold_promotion_test_phase_scoped_coverage',
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


def verify_reviewed_discovery(folder):
    root = Path(folder).absolute()
    def raw(name): return _read(root,name)
    def read(name): return json.loads(raw(name))
    def lines(name): return [json.loads(x) for x in raw(name).splitlines() if x.strip()]
    for name, expected in REVIEWED_FILES.items():
        if hashlib.sha256(raw(name)).hexdigest() != expected:
            raise ValueError('Not the reviewed discovery artifact: '+name)
    plan, seal, results, registration = [read(n+'.json') for n in ('plan','audit','results','registration')]
    if (plan['source_commit'] != REVIEWED_COMMIT or plan['stage'] != 'discovery'
            or plan['profile']['name'] != 'pilot' or not seal['passed']
            or registration['plan_sha256'] != REVIEWED_FILES['plan.json']):
        raise ValueError('Reviewed discovery identity differs')
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
    if len(events) != 2*6516:
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
    if (dict(roles) != results['charged_by_role'] or results['charged_model_calls'] != 6516
            or seal['checked_proposals'] != 6000 or seal['trial_count'] != 200):
        raise ValueError('Reviewed ledger totals differ')
    # Recheck header bytes after scanning. Raw evidence is never rewritten.
    for name, expected in REVIEWED_FILES.items():
        if hashlib.sha256(raw(name)).hexdigest() != expected:
            raise ValueError('Prerequisite changed during verification')
    return prerequisite_receipt()
