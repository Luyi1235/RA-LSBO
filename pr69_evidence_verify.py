"""Raw-dictionary PR68 ledger verification without importing model packages.

The original DesignRecord package __init__ imports scientific code even for a
read-only verify. This module checks the registered mathematical observation
contract directly and reuses only frozen standard-library summary utilities.
"""
import hashlib
import json
import math
from pathlib import Path


def require(test, message):
    if not test:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()


def native_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid measured numeric value')
    return value


def feasible(record):
    obs, diag = record['observation'], record['diagnostics']
    require(obs['status'] == 'completed' and not diag.get('numerical_failure'), 'Untrusted record')
    require(obs['design_id'] == native_digest(record['candidate']), 'Native design identity mismatch')
    constraints = dict(obs['constraints'])
    require(len(obs['constraints']) == 2 and set(constraints) == {'region', 'resource'}, 'Wrong mathematical contract')
    region, resource, obj = constraints['region'], constraints['resource'], obs['objective']
    require(resource['kind'] == 'exact', 'Missing measured resource')
    resource_value = finite(resource['value'])
    if region['kind'] == 'interval':
        require(region['lower'] == 0 and region['lower_inclusive'] is False
                and region.get('upper') is None and obj['kind'] == 'missing', 'Invalid abort contract')
        require(region.get('value') is None and obj.get('value') is None, 'Hidden abort value')
        return False
    require(region['kind'] == 'exact' and finite(region['value']) <= 0 and obj['kind'] == 'exact',
            'Invalid completed mathematical observation')
    finite(obj['value'])
    return resource_value <= 0 and diag.get('certified_physical_infeasible') is not True


def best(records):
    return min((r['observation']['objective']['value'] for r in records if feasible(r)), default=None)


def verify(output):
    # These imports are stdlib-only. Do not import partial_bo/__init__.py.
    from ra_lsbo.experiments.information_value_stage1.io import verify as verify_manifest
    from ra_lsbo.experiments.information_value_stage3.protocol import ARMS, plan, summarize
    from ra_lsbo.experiments.information_value_stage3.runner import _decision
    root = Path(output)
    hashes = verify_manifest(root)
    def read(path):
        return json.loads((root/path).read_text(encoding='utf-8'))
    def lines(path):
        return [json.loads(s) for s in (root/path).read_text(encoding='utf-8').splitlines() if s.strip()]
    spec, report = read('plan.json'), read('report.json')
    require(spec == plan(spec['profile']), 'Unregistered plan')
    require(report['code_sha'] == read('source_receipt.json')['head_sha'], 'Source mismatch')
    events, count = lines('evaluations.jsonl'), spec['total_calls']
    require(len(events) == 2*count, 'Incomplete evaluation events')
    settled = {}
    for i in range(count):
        a, b = events[2*i:2*i+2]
        require(a['ticket'] == b['ticket'] == i and a['event'] == 'started' and b['event'] == 'settled', 'Broken ticket sequence')
        rec = b['record']
        feasible(rec)
        require(a['candidate'] == rec['candidate'] and a['candidate_id'] == b['candidate_id'] == fingerprint(rec['candidate'])
                and a['purpose'] == b['purpose'], 'Ledger candidate mismatch')
        settled[i] = b
    used, rows = set(), []
    for case in spec['cases']:
        for seed in spec['seeds']:
            init = read(Path('shared_initial')/case/(str(seed)+'.json'))
            require(len(init['records']) == len(init['tickets']) == spec['initial']
                    and fingerprint(init['records']) == init['hash'], 'Initial count/hash mismatch')
            for j, (ticket, rec) in enumerate(zip(init['tickets'], init['records'])):
                require(ticket not in used and settled[ticket]['record'] == rec
                        and settled[ticket]['purpose'] == {'phase': 'initial', 'case': case, 'seed': seed, 'index': j}, 'Initial ledger mismatch')
                used.add(ticket)
            for arm in ARMS:
                folder = Path('trials')/case/str(seed)/arm
                props, obs, result = lines(folder/'proposals.jsonl'), lines(folder/'observations.jsonl'), read(folder/'result.json')
                require(len(props) == len(obs) == spec['steps'], 'Wrong online count')
                history = list(init['records'])
                curve, path, decisions = [best(history)], [], []
                for i, (p, o) in enumerate(zip(props, obs)):
                    ticket, rec = o['ticket'], o['record']
                    require(ticket not in used and settled[ticket]['record'] == rec and settled[ticket]['purpose'] ==
                            {'phase': 'online', 'case': case, 'seed': seed, 'arm': arm, 'iteration': i}, 'Online ledger mismatch')
                    used.add(ticket)
                    require(p['iteration'] == o['iteration'] == i and p['candidate'] == rec['candidate']
                            and p['candidate_hash'] == o['candidate_hash'] == fingerprint(rec['candidate'])
                            and p['history_hash'] == fingerprint(history), 'Proposal/observation/prefix mismatch')
                    require(not any(r['candidate'] == rec['candidate'] for r in history), 'Repeated actual evaluation')
                    d = _decision(p['metadata'], rec['candidate'])
                    d.update(iteration=i, candidate_hash=fingerprint(rec['candidate']), nonfeasible=int(not feasible(rec)))
                    decisions.append(d)
                    history.append(rec)
                    curve.append(best(history))
                    path.append(fingerprint(rec['candidate']))
                require(result['curve'] == curve and result['path_hash'] == fingerprint(path)
                        and result['initial_hash'] == init['hash'] and result['decisions'] == decisions
                        and result['initial_best'] == curve[0] and result['final_best'] == curve[-1]
                        and result['nonfeasible'] == sum(d['nonfeasible'] for d in decisions)
                        and result['online_calls'] == spec['steps'], 'Trial summary mismatch')
                for name in ('fantasy_head_fits_started', 'fantasy_head_fits_completed'):
                    require(result[name] == sum(d[name] for d in decisions), 'Fantasy fit count mismatch')
                rows.append(result)
    require(used == set(range(count)) and report['trials'] == rows
            and report['summary'] == summarize(rows, spec), 'Campaign summary mismatch')
    require(report['actual_calls_started'] == report['actual_calls_settled'] == count, 'Bill mismatch')
    return {**hashes, 'trials': len(rows), 'actual_calls_started': count, 'actual_calls_settled': count,
            'initial_calls': spec['initial_calls'], 'online_calls': spec['online_calls'],
            'gp_fits_in_this_verification': 0, 'new_evaluations_in_this_verification': 0}
