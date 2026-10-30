"""Read-only import of the reviewed 316-call aircraft qualification.

No source snapshot is executed. Paths in the historical Windows plan are not
rewritten. A matching seal is provenance, not physical or statistical validation.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

SOURCE_COMMIT = 'b8a94c324cd23a0ef9f15c668fe9a247e4ad9a00'
RESULT_COMMIT = '59643c3d544aaca5d97144e83b95b5ec782b8c60'
HEADERS = {
    'plan.json': '99fced2b8e9593a0e4d15c7c0a85476c38c9311c4ea7385574a8e130dab689bd',
    'audit.json': '9ec4fe8e3e2e9d76e80de1f48dc488761a83c4a79c2390152847b9569f9e7ece',
    'registration.json': 'cab9089f302bab150a8b7e9a44b2f9c79c83b934283bdedfc752d2f3cbe453a3',
    'results.json': '7e854b5e06e66ec0c22098b3a12879014c984cab38d2bc76de5809a597adb459',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def safe_read(root, relative):
    path = PurePosixPath(relative)
    if (not path.parts or path.is_absolute() or '..' in path.parts or str(path) != relative
            or '\\' in relative or ':' in relative):
        raise ValueError('Unsafe qualification evidence path')
    target = Path(root).absolute().joinpath(*path.parts)
    if any(p.is_symlink() for p in (target, *target.parents)) or not target.is_file():
        raise ValueError('Missing or linked qualification evidence: '+relative)
    return target.read_bytes()


def receipt():
    return {'kind': 'reviewed_aircraft_qualification_v1', 'source_commit': SOURCE_COMMIT,
            'result_commit': RESULT_COMMIT, 'headers_sha256': dict(HEADERS),
            'settled_calls': 316, 'evidence_files': 4, 'source_bindings': 156,
            'decision': 'bounded_nominal_exploration_only', 'sparse_below_10pct_supported': False,
            'algorithm_replay': False, 'physical_validation': False, 'new_evaluations': 0}


def satisfaction(value):
    if value['kind'] == 'exact':
        return value['value'] <= 0
    if value['kind'] == 'interval':
        if value['upper'] is not None and value['upper'] <= 0:
            return True
        if value['lower'] is not None and (value['lower'] > 0 or
                (value['lower'] == 0 and not value['lower_inclusive'])):
            return False
    return None


def physical_state(record):
    packet = record['observation']
    if record['diagnostics'].get('numerical_failure') or packet['status'] in {'unknown','infrastructure_error'}:
        return 'unresolved'
    labels = [satisfaction(v) for _,v in packet['constraints']]
    if False in labels or packet['status'] == 'hidden_failure':
        return 'known_infeasible'
    if packet['status'] == 'completed' and all(v is True for v in labels) and packet['objective']['kind'] == 'exact':
        return 'usable_feasible'
    return 'unresolved'


def verify(folder):
    root = Path(folder).absolute()
    raw = lambda name: safe_read(root, name)
    read = lambda name: json.loads(raw(name))
    for name, expected in HEADERS.items():
        if hashlib.sha256(raw(name)).hexdigest() != expected:
            raise ValueError('Not the fixed reviewed qualification: '+name)
    plan, audit, registration, results = [read(n+'.json') for n in ('plan','audit','registration','results')]
    if (plan['source_commit'] != SOURCE_COMMIT or plan['stage'] != 'qualification' or plan['size'] != 'full'
            or registration != {'plan_sha256': HEADERS['plan.json'], 'maximum_model_calls': 316}
            or audit['passed'] is not True or audit['settled_calls'] != 316):
        raise ValueError('Qualification identity changed')
    names = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
             and p.suffix in {'.json','.jsonl'} and p.name != 'audit.json'}
    if names != set(audit['evidence_hashes']):
        raise ValueError('Qualification sealed file set changed')
    for name, expected in audit['evidence_hashes'].items():
        if hashlib.sha256(raw(name)).hexdigest() != expected:
            raise ValueError('Qualification evidence changed: '+name)
    for expected in plan['sources'].values():
        if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('Invalid qualification snapshot identity')
        if hashlib.sha256(raw('source_snapshot/'+expected+'.source')).hexdigest() != expected:
            raise ValueError('Qualification snapshot changed')
    events = [json.loads(x) for x in raw('calls.jsonl').splitlines() if x.strip()]
    if len(events) != 632:
        raise ValueError('Incomplete qualification ledger')
    rows, roles, ids = {}, Counter(), set()
    for number,(start,end) in enumerate(zip(events[::2], events[1::2]),1):
        call = {k:start[k] for k in ('number','tag','role','candidate','contract','new_rcaide_mda')}
        r = end['record']
        if (start['event'] != 'started' or end['event'] != 'settled' or start['number'] != number
                or start['call_id'] != digest(call) or start['call_id'] in ids or start['tag'] in rows
                or end['call_id'] != start['call_id'] or end['tag'] != start['tag']
                or r['candidate'] != start['candidate'] or r['observation']['design_id'] != digest(r['candidate'])
                or r['observation']['contract_id'] != start['contract'] or end['record_digest'] != digest(r)
                or start['new_rcaide_mda'] is not False or not math.isfinite(end['seconds']) or end['seconds'] < 0):
            raise ValueError('Qualification ledger linkage changed')
        rows[start['tag']] = (r,start['role']); roles[start['role']] += 1; ids.add(start['call_id'])
    claimed = set()
    for call in plan['calls']:
        r,role = rows[call['tag']]
        if (call['tag'] in claimed or role != call['role'] or r['candidate'] != call['candidate']
                or r['observation']['contract_id'] != plan['tasks'][call['case']]['contracts'][str(call['step'])]):
            raise ValueError('Qualification call recipe differs')
        claimed.add(call['tag'])
    if claimed != set(rows) or dict(roles) != {'independent_panel':256, 'anchor_convergence':12, 'branch_convergence':48}:
        raise ValueError('Qualification calls are missing or unregistered')
    for case in plan['tasks']:
        panel = [rows[f'{case}/panel/{i}'][0] for i in range(128)]
        counts = Counter(physical_state(r) for r in panel)
        saved = results['cases'][case]['panel']
        if {k:counts[k] for k in ('usable_feasible','known_infeasible','unresolved')} != saved['counts']:
            raise ValueError('Qualification panel reconstruction differs')
    for name, expected in HEADERS.items():
        if hashlib.sha256(raw(name)).hexdigest() != expected:
            raise ValueError('Qualification changed during import')
    return receipt()
