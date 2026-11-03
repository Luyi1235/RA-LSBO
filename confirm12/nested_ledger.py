"""Accounting identity and audit for the nested confirmation design.

The underlying ledger remains SIMULATED_ONLY: no aircraft runner or approval
issuer is implemented here. Use only for zero-truth contract verification.
"""
import json
import math
from pathlib import Path
from . import call_ledger, design, nested_seed_plan
from .ledger import digest

SCHEMA = 'nested_accounting_dry_run_v1'


def _hex(value, length):
    return isinstance(value, str) and len(value) == length and all(c in '0123456789abcdef' for c in value)


def compile_plan(nested, *, source_sha, environment_sha, contracts):
    """Freeze proposed design identities; structural validity is not approval."""
    design.validate(nested)
    if nested != nested_seed_plan.candidate_plan():
        raise ValueError('nested roster is not the frozen, unapproved candidate plan')
    if not _hex(source_sha, 40) or not _hex(environment_sha, 64):
        raise ValueError('full source and environment identities required')
    if set(contracts) != set(design.CASES):
        raise ValueError('two task contracts required')
    for case in design.CASES:
        if set(contracts[case]) != {'20s', '5s'} or any(
                not _hex(value, 64) for value in contracts[case].values()):
            raise ValueError('both complete contract identities required')
    by_case = nested['seeds_by_case']
    allowed = {}
    for case in design.CASES:
        subset = set(nested['auxiliary_by_case'][case])
        allowed[case] = {str(seed): list(design.MAIN) + (list(design.AUXILIARY) if seed in subset else [])
                         for seed in by_case[case]}
    return {'schema': SCHEMA, 'cases': list(design.CASES),
            'seeds': [seed for case in design.CASES for seed in by_case[case]],
            'seeds_by_case': by_case, 'allowed_arms': allowed,
            'arms': list(design.MAIN + design.AUXILIARY), 'initial': 12, 'online': 18,
            'verification': 1, 'max_calls': 151040, 'nested_plan_hash': digest(nested),
            'source_sha': source_sha, 'environment_sha256': environment_sha,
            'contracts': contracts, 'phase': 'confirmation',
            'execution_authorized': False, 'scope': 'SIMULATED_ONLY'}


def create(path, plan):
    if plan.get('schema') != SCHEMA or plan.get('execution_authorized') is not False:
        raise ValueError('only an unapproved dry-run plan may create this ledger')
    call_ledger.create(path, plan)


def _check_incumbent(records, candidate):
    if len(records) != 30 or any(r[4] != 'settled' for r in records):
        raise ValueError('complete settled nominal history required')
    feasible = []
    for _, _, serialized, output, _ in records:
        observation = json.loads(output)
        if type(observation.get('feasible')) is not bool:
            raise ValueError('unknown feasibility cannot determine verification incumbent')
        if observation['feasible']:
            objective = observation.get('objective')
            if (not isinstance(objective, (int, float)) or isinstance(objective, bool)
                    or not math.isfinite(objective)):
                raise ValueError('feasible mock incumbent requires exact finite objective')
            feasible.append((objective, json.loads(serialized)['candidate']))
    if not feasible or candidate != min(feasible, key=lambda item: item[0])[1]:
        raise ValueError('verification must use the nominal incumbent, with no substitution')


def start(path, case, seed, role, arm, index, candidate):
    with call_ledger._read_connect(path) as db:
        plan = json.loads(dict(db.execute('SELECT key,value FROM meta'))['plan'])
        if role == 'verification':
            records = db.execute('SELECT role,idx,candidate,outcome,state FROM calls WHERE case_name=? '
                "AND seed=? AND ((role='initial' AND arm='') OR (role='online' AND arm=?)) "
                "ORDER BY CASE role WHEN 'initial' THEN 0 ELSE 1 END,idx", (case, seed, arm)).fetchall()
            _check_incumbent(records, candidate)
    if plan.get('schema') != SCHEMA:
        raise ValueError('wrong accounting schema')
    contract = plan['contracts'][case]['5s' if role == 'verification' else '20s']
    envelope = {'candidate': candidate, 'candidate_sha256': digest(candidate),
                'contract_id': contract, 'source_sha': plan['source_sha'],
                'environment_sha256': plan['environment_sha256'], 'phase': plan['phase'],
                'nested_plan_hash': plan['nested_plan_hash'], 'role': role}
    ident = call_ledger.start(path, case, seed, role, arm, index, envelope)
    return ident


def settle(path, call_id, state, result):
    """Record mock evidence only; no claim of real physical contract validation."""
    if not isinstance(result, dict) or result.get('scope') != 'mock':
        raise ValueError('only explicit mock observations accepted')
    call_ledger.settle(path, call_id, state, result)


def audit(path, plan):
    report = call_ledger.audit(path, plan)
    with call_ledger._read_connect(path) as db:
        rows = db.execute('SELECT case_name,seed,role,arm,candidate,outcome,state FROM calls').fetchall()
        histories = {}
        for case, seed, role, arm, serialized, output, state in rows:
            if role == 'verification':
                histories[(case, seed, arm)] = db.execute('SELECT role,idx,candidate,outcome,state FROM calls WHERE case_name=? '
                    "AND seed=? AND ((role='initial' AND arm='') OR (role='online' AND arm=?)) "
                    "ORDER BY CASE role WHEN 'initial' THEN 0 ELSE 1 END,idx", (case, seed, arm)).fetchall()
    for case, seed, role, arm, serialized, output, state in rows:
        try:
            env = json.loads(serialized)
            expected_contract = plan['contracts'][case]['5s' if role == 'verification' else '20s']
            if (env['candidate_sha256'] != digest(env['candidate']) or env['contract_id'] != expected_contract
                or env['source_sha'] != plan['source_sha'] or env['environment_sha256'] != plan['environment_sha256']
                or env['nested_plan_hash'] != plan['nested_plan_hash'] or env['phase'] != 'confirmation'
                or env['role'] != role):
                raise ValueError('bound identity mismatch')
            if state != 'started' and (not isinstance(json.loads(output), dict)
                                       or json.loads(output).get('scope') != 'mock'):
                raise ValueError('non-mock terminal evidence')
            if role == 'verification':
                _check_incumbent(histories[(case, seed, arm)], env['candidate'])
        except (KeyError, ValueError, TypeError):
            report['errors'].append('nested candidate/source/phase/contract mismatch')
    report['integrity_passed'] = not report['errors']
    report['campaign_complete'] = False
    report['production_qualified'] = False
    report['scope'] = 'SIMULATED_ONLY'
    return report
