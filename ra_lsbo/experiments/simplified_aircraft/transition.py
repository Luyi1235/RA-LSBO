"""Frozen phase-RA aircraft handoff: bounded smoke and model qualification only.

No optimization pilot, evidence-transfer ensemble, scenario risk claim, or RCAIDE
entry is exposed. Historical runners, evaluator equations and source records stay
unchanged. Qualifier observations are never optimizer training data.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from ...budget import digest
from ...partial_bo.problems import candidate_pool
from ...partial_bo.records import DesignRecord
from ..comparison.runner import evidence_hashes
from ..core.methods import settings_for
from ..core.metrics import binomial_interval, trajectory_metrics
from ..core.problems import AircraftCase
from ..core.protocol import BASE, ROOT, sha
from ..core.runner import Ledger, append, lines, plain, read, save
from ..discovery_stages.methods import Labelled, optimizer_for, specification
from ..discovery_stages.optimizer import DiscoveryBO
from ..discovery_stages.runner import validate_decision

SCHEMA = 'ra_lsbo_aircraft_transition_v1'
DOCUMENT = 'docs/AIRCRAFT_TRANSITION_PROTOCOL_20261030.md'
CASES = ('aircraft_ordinary', 'aircraft_sparse')
ARMS = ('random', 'ev_latent', 'ev_onehot', 'phase_exact',
        'ev_random_then_latent', 'original_lsbo')
SMOKE = {'name': 'smoke', 'seeds': [9901], 'initial': 6, 'horizon': 2,
         'pool': 32, 'encoder_epochs': 4, 'native_gp_steps': 2}
PANEL_SEED = 43101001


def problem_for(case, step=20.):
    if case not in CASES or step not in (20., 10., 5.):
        raise ValueError('Only fixed nominal reduced-aircraft tasks and 20/10/5 s steps')
    return AircraftCase(BASE, case.split('_')[1], integration_step=step)


def method_spec(method):
    if method not in ARMS:
        raise ValueError('Unregistered transition method; no adaptive weights')
    spec = specification('ev_latent' if method == 'phase_exact' else method, SMOKE)
    if method == 'phase_exact':
        spec = {**spec, 'feedback': 'exact'}
    return spec


def engine_for(problem, method, seed, folder):
    if method == 'phase_exact':
        return Labelled(DiscoveryBO(problem, BASE, seed, settings_for('ra_lsbo', SMOKE),
            cold_global=True, protect=True, feedback='exact', coverage_scope='cold'), method)
    return optimizer_for(problem, BASE, seed, method, SMOKE, folder)


def runtime():
    return {'python': sys.version, 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in
                         ('numpy', 'scipy', 'torch', 'gpytorch', 'botorch', 'scikit-learn')}}


def build_plan(stage, size='full'):
    if stage not in {'smoke', 'qualification'} or size not in {'full', 'ci'}:
        raise ValueError('Only smoke/qualification; no pilot, confirm, resume or MDA')
    files = set((ROOT/'ra_lsbo').rglob('*.py')) | set(BASE.rglob('*.py'))
    files |= {ROOT/DOCUMENT, ROOT/'requirements-models.txt', ROOT/'tests/test_aircraft_transition.py'}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True)
    sources = {str(p): sha(p) for p in sorted(files)}
    panel_count = 128 if size == 'full' else 2
    tasks, calls = {}, []
    for ci, case in enumerate(CASES):
        problem = problem_for(case)
        anchors = problem.anchors()
        configurations = problem.space.configurations()
        if len(configurations) != 12:
            raise ValueError('Aircraft branch contract changed; re-register the complete budget')
        probes = []
        for configuration in configurations:
            raw = problem.space.default().to_dict()
            raw['discrete'] = dict(configuration)
            probes.append(problem.canonical(raw))
        for c in anchors+probes:
            if any(not np.isfinite(v) or v > 0 for v in problem.precheck(c).values()):
                raise ValueError('Registered probe fails geometry; do not search for a favorable replacement')
        task = {'settings': deepcopy(problem.settings),
                'contracts': {str(step): problem_for(case, step).contract_id for step in (20., 10., 5.)},
                'configurations': configurations, 'anchors': anchors, 'branch_probes': probes}
        if stage == 'qualification':
            panel, rejected = candidate_pool(problem, PANEL_SEED, panel_count,
                                              {digest(c) for c in anchors+probes})
            task.update(panel=panel, exact_screen_rejections=rejected)
            # Same prospective designs in the two missions; labels are never used for sampling.
            for i, candidate in enumerate(panel):
                calls.append({'case': case, 'candidate': candidate, 'step': 20.,
                              'tag': f'{case}/panel/{i}', 'role': 'independent_panel'})
            for kind, candidates, steps in [('anchor', anchors, (20., 10., 5.)),
                                            ('branch', probes, (20., 5.))]:
                for i, candidate in enumerate(candidates):
                    for step in steps:
                        calls.append({'case': case, 'candidate': candidate, 'step': step,
                            'tag': f'{case}/{kind}/{i}/{int(step)}', 'role': kind+'_convergence'})
        else:
            task['initial'], task['exact_screen_rejections'] = candidate_pool(
                problem, 43102001+ci, SMOKE['initial'], {digest(c) for c in anchors+probes})
        tasks[case] = task
    roles = (dict(Counter(c['role'] for c in calls)) if stage == 'qualification'
             else {'shared_initial': 12, 'online': 24})
    return plain({'schema': SCHEMA, 'stage': stage, 'size': size,
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': commit.stdout.strip() if commit.returncode == 0 else 'unversioned-checkout',
        'runtime': runtime(), 'sources': sources, 'tasks': tasks, 'calls': calls,
        'profile': SMOKE if stage == 'smoke' else {'panel_per_case': panel_count, 'panel_seed': PANEL_SEED},
        'arms': list(ARMS) if stage == 'smoke' else [],
        'methods': {m: method_spec(m) for m in ARMS} if stage == 'smoke' else {},
        'budget': {**roles, 'maximum_model_calls': sum(roles.values()), 'maximum_rcaide_mda_calls': 0},
        'old_transfer_diagnostic_required': False, 'prerequisite_directory_required': False,
        'scope': 'Nominal model qualification and interface smoke only; no performance claims',
        'qualified_by_filename': False, 'automatic_pilot_authorization': False,
        'sparse_operational_threshold': .10, 'energy_step_relative_tolerance': .01,
        'sampling': 'uniform configuration and active coordinates BEFORE exact geometry rejection; '
                    'conditional accepted distribution, not necessarily equal branch proportions',
        'qualification_labels_available_to_optimizer': False,
        'observation_contract': 'original evaluator outputs retained, no midpoint or penalty imputation; '
            'numerical failures counted separately and not used as physical-infeasibility evidence'})


def prepare(folder, stage, size='full'):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError('Use a new directory, never an old transfer or aircraft run')
    plan = build_plan(stage, size)
    folder.mkdir(parents=True)
    (folder/'source_snapshot').mkdir()
    for path, identity in plan['sources'].items():
        target = folder/'source_snapshot'/(identity+'.source')
        if not target.exists():
            target.write_bytes(Path(path).read_bytes())
    save(folder/'plan.json', plan)
    save(folder/'registration.json', {'plan_sha256': sha(folder/'plan.json'),
                                     'maximum_model_calls': plan['budget']['maximum_model_calls']})
    return plan


def load_plan(folder):
    folder = Path(folder)
    plan = read(folder/'plan.json')
    if plan.get('schema') != SCHEMA:
        raise ValueError('Not a transition plan')
    if read(folder/'registration.json') != {'plan_sha256': sha(folder/'plan.json'),
                            'maximum_model_calls': plan['budget']['maximum_model_calls']}:
        raise ValueError('Registration changed')
    expected = build_plan(plan['stage'], plan['size'])
    expected.pop('registered_utc')
    saved = {k:v for k,v in plan.items() if k != 'registered_utc'}
    if digest(saved) != digest(expected):
        raise ValueError('Plan specification, source, runtime or commit changed')
    for identity in plan['sources'].values():
        if sha(folder/'source_snapshot'/(identity+'.source')) != identity:
            raise ValueError('Snapshot changed')
    return plan


def physical_state(record):
    # Do not convert numerical/infrastructure failure to physical infeasibility,
    # even though legacy process-level labels can call it hidden_failure.
    if (record.diagnostics.get('numerical_failure') or
            record.observation.status in {'unknown', 'infrastructure_error'}):
        return 'unresolved'
    if record.physical_feasible is False:
        return 'known_infeasible'
    if record.usable_feasible:
        return 'usable_feasible'
    return 'unresolved'


def summarize_panel(records):
    counts = Counter(physical_state(r) for r in records)
    n, feasible, unknown = len(records), counts['usable_feasible'], counts['unresolved']
    low, high = binomial_interval(feasible, n)[0], binomial_interval(feasible+unknown, n)[1]
    kinds = Counter()
    for r in records:
        for name, value in [('objective', r.observation.objective), *r.observation.constraints]:
            kinds[name+':'+value.kind] += 1
    branches = {}
    for r in records:
        key = digest(r.candidate['discrete'])
        entry = branches.setdefault(key, {'discrete': r.candidate['discrete'], 'n': 0,
                                          'usable_feasible': 0, 'unresolved': 0})
        entry['n'] += 1
        entry['usable_feasible'] += physical_state(r) == 'usable_feasible'
        entry['unresolved'] += physical_state(r) == 'unresolved'
    return {'n': n, 'counts': {k: counts[k] for k in ('usable_feasible','known_infeasible','unresolved')},
            'usable_fraction': feasible/n if n else None,
            'usable_binomial_95': binomial_interval(feasible, n),
            'physical_feasibility_partial_identification': [feasible/n,(feasible+unknown)/n] if n else [None,None],
            'unknown_conservative_binomial_envelope_95': [low, high],
            'sparse_below_10pct_supported': bool(n and high < .1),
            'numerical_failures': sum(bool(r.diagnostics.get('numerical_failure')) for r in records),
            'terminations': dict(Counter(r.diagnostics.get('termination_reason','unspecified') for r in records)),
            'output_kinds': dict(kinds), 'branches': list(branches.values()),
            'intervals_are_not_safety_guarantees': True}


def convergence(records):
    if any(r.diagnostics.get('numerical_failure') for r in records):
        return {'status': 'needs_review_numerical_failure', 'relative_energy_range': None}
    terminal = {(r.observation.status, r.diagnostics.get('termination_reason')) for r in records}
    if len(terminal) != 1:
        return {'status': 'needs_review_changed_termination', 'relative_energy_range': None}
    if any(r.observation.objective.kind != 'exact' for r in records):
        return {'status': 'inconclusive_partial_outputs', 'relative_energy_range': None}
    values = [r.observation.objective.value for r in records]
    relative = (max(values)-min(values))/max(abs(values[-1]), 1e-12)
    labels = [tuple((n,v.constraint_satisfaction()) for n,v in r.observation.constraints) for r in records]
    status = 'passed_screen' if relative <= .01 and all(x == labels[0] for x in labels) else 'needs_review'
    return {'status': status, 'relative_energy_range': relative,
            'energy_Wh': values, 'feasibility_labels_agree': all(x == labels[0] for x in labels)}


def qualify_result(plan, records):
    cases = {}
    for case, task in plan['tasks'].items():
        panel = [records[f'{case}/panel/{i}'] for i in range(len(task['panel']))]
        checks = {}
        for kind, candidates, steps in [('anchor',task['anchors'],(20,10,5)),
                                        ('branch',task['branch_probes'],(20,5))]:
            for i, candidate in enumerate(candidates):
                rows = [records[f'{case}/{kind}/{i}/{step}'] for step in steps]
                checks[f'{kind}/{i}'] = {'candidate': candidate, 'steps_s': steps,
                    **convergence(rows), 'physical_states': [physical_state(r) for r in rows],
                    'diagnostics': [r.diagnostics for r in rows]}
        cases[case] = {'panel': summarize_panel(panel), 'numerical_checks': checks,
                      'branch_count': len(task['configurations'])}
    return plain({'schema': SCHEMA, 'stage': 'qualification', 'cases': cases,
        'charged_model_calls': len(records), 'maximum_model_calls': plan['budget']['maximum_model_calls'],
        'new_rcaide_mda_calls': 0, 'automatic_pilot_authorization': False,
        'scientific_review_required': True, 'qualification_scope': 'parameter/termination screening; '
        'not validation against flight data, not proof of branch tradeoffs or global integration accuracy'})


def common_pool(problem, ci, iteration, history):
    return candidate_pool(problem, 43103001+10000*ci+iteration, SMOKE['pool'],
                          {r.observation.design_id for r in history})


def smoke_summary(initial, online):
    return plain(trajectory_metrics(initial, online, SMOKE['horizon']))


def run(folder, approved_call_limit):
    folder = Path(folder)
    plan = load_plan(folder)
    maximum = plan['budget']['maximum_model_calls']
    if type(approved_call_limit) is not int or approved_call_limit < maximum:
        raise ValueError(f'Explicit complete cap required: {maximum}')
    fd = os.open(folder/'execution.lock', os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.close(fd)  # Preserve even after completion; never resume or retry silently.
    ledger = Ledger(folder, maximum)
    if plan['stage'] == 'qualification':
        problems = {(c,s):problem_for(c,s) for c in CASES for s in (20.,10.,5.)}
        records = {call['tag']:ledger.evaluate(problems[(call['case'],call['step'])],
                   call['candidate'],call['tag'],call['role']) for call in plan['calls']}
        result = qualify_result(plan, records)
    else:
        results = []
        for ci, case in enumerate(CASES):
            problem = problem_for(case)
            initial = [ledger.evaluate(problem,c,f'{case}/initial/{i}','shared_initial')
                       for i,c in enumerate(plan['tasks'][case]['initial'])]
            for method in ARMS:
                output = folder/'optimization'/case/method
                output.mkdir(parents=True)
                save(output/'manifest.json', {'method': method, 'initial': [r.to_dict() for r in initial],
                                               'specification': plan['methods'][method]})
                engine, history, online = engine_for(problem, method, 9901, output), list(initial), []
                for i in range(SMOKE['horizon']):
                    pool,rejected = common_pool(problem,ci,i,history)
                    candidate, metadata = engine.propose(history,pool,i)
                    append(output/'proposals.jsonl', {'iteration': i,'candidate':candidate,'pool':pool,
                        'exact_screen_rejections':rejected,'prefix_digest':digest([r.to_dict() for r in history]),
                        'metadata':metadata})
                    row = ledger.evaluate(problem,candidate,f'{case}/{method}/online/{i}','online')
                    append(output/'observations.jsonl', row.to_dict())
                    engine.accept(row)
                    online.append(row); history.append(row)
                summary = {'case':case,'method':method,**smoke_summary(initial,online)}
                save(output/'summary.json',summary); results.append(summary)
        result = {'schema':SCHEMA,'stage':'smoke','summaries':results,
                  'charged_model_calls':ledger.count,'maximum_model_calls':maximum,
                  'new_rcaide_mda_calls':0,'performance_evidence':False}
    save(folder/'results.json', result)
    return result


def audit(folder, write=True):
    folder = Path(folder)
    plan, before = load_plan(folder), evidence_hashes(folder)
    events = lines(folder/'calls.jsonl')
    if len(events) != 2*plan['budget']['maximum_model_calls']:
        raise ValueError('Incomplete or unsettled campaign; retain all calls and locks')
    rows, roles = {}, Counter()
    for number,(start,end) in enumerate(zip(events[::2],events[1::2]),1):
        call = {k:start[k] for k in ('number','tag','role','candidate','contract','new_rcaide_mda')}
        row = DesignRecord.from_dict(end['record'])
        if (start['event'] != 'started' or end['event'] != 'settled' or start['number'] != number
            or start['call_id'] != digest(call) or end['call_id'] != start['call_id']
            or start['tag'] in rows or end['tag'] != start['tag'] or start['new_rcaide_mda'] is not False
            or row.candidate != start['candidate'] or row.observation.contract_id != start['contract']
            or end['record_digest'] != digest(row.to_dict()) or not np.isfinite(end['seconds']) or end['seconds'] < 0):
            raise ValueError('Ledger identity/ordering mismatch')
        rows[start['tag']] = (row,start['role']); roles[start['role']] += 1
    claimed = set()
    def claim(tag,role,problem,candidate):
        if tag in claimed or tag not in rows:
            raise ValueError('Missing or duplicate registered call')
        row, actual = rows[tag]
        if (actual != role or row.candidate != candidate or row.observation.contract_id != problem.contract_id
            or problem.canonical(candidate) != candidate
            or any(not np.isfinite(v) or v > 0 for v in problem.precheck(candidate).values())):
            raise ValueError('Registered call changed')
        claimed.add(tag); return row
    checked = 0
    if plan['stage'] == 'qualification':
        problems = {(c,s):problem_for(c,s) for c in CASES for s in (20.,10.,5.)}
        records = {c['tag']:claim(c['tag'],c['role'],problems[(c['case'],c['step'])],c['candidate'])
                   for c in plan['calls']}
        expected = qualify_result(plan,records)
    else:
        summaries = []
        for ci,case in enumerate(CASES):
            problem = problem_for(case)
            initial = [claim(f'{case}/initial/{i}','shared_initial',problem,c)
                       for i,c in enumerate(plan['tasks'][case]['initial'])]
            for method in ARMS:
                output = folder/'optimization'/case/method
                if read(output/'manifest.json') != {'method':method,'initial':[r.to_dict() for r in initial],
                                                    'specification':plan['methods'][method]}:
                    # Dataclass tuples become JSON lists; compare canonical JSON below.
                    if digest(read(output/'manifest.json')) != digest({'method':method,'initial':[r.to_dict() for r in initial],
                                                     'specification':plan['methods'][method]}):
                        raise ValueError('Shared initialization/manifest changed')
                proposals = lines(output/'proposals.jsonl')
                if len(proposals) != SMOKE['horizon']:
                    raise ValueError('Incomplete smoke trial')
                history, online = list(initial), []
                for i,p in enumerate(proposals):
                    pool,rejected = common_pool(problem,ci,i,history)
                    if (p['iteration'] != i or p['pool'] != pool or p['exact_screen_rejections'] != rejected
                        or p['prefix_digest'] != digest([r.to_dict() for r in history])):
                        raise ValueError('Proposal stream changed')
                    row = claim(f'{case}/{method}/online/{i}','online',problem,p['candidate'])
                    if row.observation.design_id in {r.observation.design_id for r in history}:
                        raise ValueError('Repeated evaluated design')
                    validate_decision(problem,plan,method,9901,history,p)
                    history.append(row); online.append(row); checked += 1
                summary = {'case':case,'method':method,**smoke_summary(initial,online)}
                if read(output/'summary.json') != summary or digest(lines(output/'observations.jsonl')) != digest([r.to_dict() for r in online]):
                    raise ValueError('Summary or observations changed')
                summaries.append(summary)
        expected = {'schema':SCHEMA,'stage':'smoke','summaries':summaries,
                    'charged_model_calls':len(rows),'maximum_model_calls':plan['budget']['maximum_model_calls'],
                    'new_rcaide_mda_calls':0,'performance_evidence':False}
    expected_roles = {k:v for k,v in plan['budget'].items() if not k.startswith('maximum_')}
    if (claimed != set(rows) or dict(roles) != expected_roles or read(folder/'results.json') != expected
            or before != evidence_hashes(folder)):
        raise ValueError('Unclaimed calls, budget, results or evidence changed')
    receipt = {'passed':True,'settled_calls':len(rows),'checked_proposals':checked,
               'source_bindings':len(plan['sources']),'evidence_hashes':before,
               'algorithm_replay':False,'physical_truth_validation':False,'automatic_pilot_authorization':False}
    if write:
        if (folder/'audit.json').exists():
            if read(folder/'audit.json') != receipt:
                raise ValueError('Audit seal changed; never overwrite')
        else:
            save(folder/'audit.json',receipt)
    return receipt


def report(folder):
    import json
    folder = Path(folder)
    if read(folder/'audit.json') != audit(folder,write=False):
        raise ValueError('Report requires unchanged freshly audited evidence')
    result = read(folder/'results.json')
    if result['stage'] == 'qualification':
        text = ['# Reduced-aircraft qualification (not an optimization comparison)',
                'Panel draws are independent of optimization. Anchors/probes are not part of random feasibility rates.',
                'Unknown-conservative envelopes are not safety certificates. Partial convergence is inconclusive.',
                'No automatic pilot authorization. Review labels, terminations, branches and numerical sensitivity.']
        for case, data in result['cases'].items():
            text += ['\n## '+case, json.dumps(data['panel'],ensure_ascii=False,indent=2),
                     'Integration screens: '+str(dict(Counter(c['status'] for c in data['numerical_checks'].values())))]
    else:
        text = ['# Reduced-aircraft interface smoke', 'Two online calls per method. No performance/ranking claim.',
                json.dumps(result['summaries'],ensure_ascii=False,indent=2)]
    text += [f"Calls: {result['charged_model_calls']}/{result['maximum_model_calls']}; new RCAIDE MDA: 0."]
    with (folder/'report.md').open('x',encoding='utf-8') as f:
        f.write('\n\n'.join(text)+'\n')
    return folder/'report.md'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','audit','report'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stage',choices=('smoke','qualification'))
    parser.add_argument('--size',choices=('full','ci'),default='full')
    parser.add_argument('--approved-call-limit',type=int)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        if args.stage is None:
            parser.error('prepare requires --stage')
        result = prepare(args.output,args.stage,args.size)
        print({'stage':result['stage'],'budget':result['budget']})
    elif args.command == 'run':
        result = run(args.output,args.approved_call_limit)
        print({k:result[k] for k in ('stage','charged_model_calls','maximum_model_calls','new_rcaide_mda_calls')})
    elif args.command == 'audit':
        result = audit(args.output); print({k:v for k,v in result.items() if k != 'evidence_hashes'})
    else:
        print(report(args.output))


if __name__ == '__main__':
    main()
