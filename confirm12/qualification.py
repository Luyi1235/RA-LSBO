"""Read-only review of the previously registered zero-call qualification plan."""
import hashlib
import json
from pathlib import Path

from ra_lsbo.experiments import iid_qualification as iid

EXPECTED_SHA = '919d092ac675d9c854bffda08c7e397ee9f41415902471757194d14e916f66e2'


def inspect(path):
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA:
        raise ValueError('registered qualification plan hash mismatch')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if (plan['budget']['planned_simplified_aircraft_calls'] != 600 or
        plan['budget']['actual_simplified_aircraft_calls'] != 0 or plan['execution_authorized']):
        raise ValueError('qualification scope mismatch')
    tasks = {}
    for case, seed in iid.SEEDS.items():
        recorded = plan['tasks'][case]
        generated = iid.sample(seed, 300)
        if recorded['seed'] != seed or any(recorded[k] != v for k, v in generated.items()):
            raise ValueError('qualification candidates do not reproduce')
        tasks[case] = {'seed': seed, 'samples': len(generated['records']),
                       'ordered_candidate_digest': iid.digest([r['candidate_sha256'] for r in generated['records']]),
                       'attempts': generated['attempts']}
    if plan['historical_20_second_contracts'] != iid.HISTORICAL_CONTRACTS:
        raise ValueError('historical physical contract mismatch')
    source_checks = {}
    for relative, expected in plan['source_sha256'].items():
        source = iid.ROOT / relative
        raw = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        source_checks[relative] = {'expected': expected, 'actual': raw, 'match': expected == raw}
    return {'plan_sha256': actual, 'tasks': tasks, 'physical_contracts': iid.HISTORICAL_CONTRACTS,
            'source_checks': source_checks, 'seed_history_status': plan['seed_history_status'],
            'qualification_ready': False, 'new_aircraft_calls': 0, 'rcaide_calls': 0,
            'blockers': ['historical seed scan incomplete: unreadable records and invalid ZIPs',
                         'raw source identity mismatch where indicated', 'qualification not executed or approved']}
