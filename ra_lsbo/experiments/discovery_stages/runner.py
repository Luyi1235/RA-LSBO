"""Single-writer staged campaigns and read-only evidence reconstruction (no replay)."""
from collections import Counter
import csv
import os
from pathlib import Path
import time

import numpy as np

from ...budget import digest
from ...partial_bo.problems import candidate_pool
from ...partial_bo.records import DesignRecord
from ..comparison.runner import evidence_hashes
from ..core.metrics import binomial_interval, paired_comparisons, qualification, trajectory_metrics
from ..core.problems import make_problem
from ..core.runner import Ledger, NATIVE_STOP, append, lines, plain, read, save
from .methods import STAGES, optimizer_for, specification
from .policies import COVERAGE_PERIOD, NEAR_TOLERANCE, coverage_pick, filter_near, nearest_distance
from .protocol import BASE, build_plan, sha, verify_plan


def prepare(folder, stage='discovery', profile='pilot', prerequisite=None):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError('Use a new result directory')
    plan = build_plan(stage, profile)
    if profile == 'pilot' and stage != 'discovery':
        if prerequisite is None:
            raise ValueError('Later pilot stages require --prerequisite: review the preceding stage first')
        previous = Path(prerequisite).resolve()
        old_plan = load_plan(previous)
        expected = 'discovery' if stage == 'representation' else 'representation'
        if old_plan['stage'] != expected or old_plan['profile']['name'] != 'pilot':
            raise ValueError('Prerequisite must be the preceding complete pilot, not a smoke')
        if read(previous/'audit.json') != audit(previous, write=False):
            raise ValueError('Prerequisite audit seal changed')
        plan['prerequisite'] = {'stage': expected, 'plan_sha256': sha(previous/'plan.json'),
            'audit_sha256': sha(previous/'audit.json'), 'results_sha256': sha(previous/'results.json'),
            'meaning': 'provenance gate only; manual scientific review is still required'}
    elif prerequisite is not None:
        raise ValueError('Prerequisite is only used by later pilot stages')
    verify_plan(plan)
    folder.mkdir(parents=True)
    snapshot = folder/'source_snapshot'
    snapshot.mkdir()
    for path, identity in plan['sources'].items():
        target = snapshot/(identity+'.source')
        if not target.exists():
            target.write_bytes(Path(path).read_bytes())
    save(folder/'plan.json', plan)
    save(folder/'registration.json', {'plan_sha256': sha(folder/'plan.json'),
        'maximum_model_calls': plan['budget']['maximum_model_calls'], 'new_model_calls': 0})
    return plan


def load_plan(folder):
    folder = Path(folder)
    plan, registration = read(folder/'plan.json'), read(folder/'registration.json')
    if registration != {'plan_sha256': sha(folder/'plan.json'),
            'maximum_model_calls': plan['budget']['maximum_model_calls'], 'new_model_calls': 0}:
        raise ValueError('Registration changed')
    verify_plan(plan)
    for identity in plan['sources'].values():
        if sha(folder/'source_snapshot'/(identity+'.source')) != identity:
            raise ValueError('Source snapshot changed')
    return plan


def pool_for(problem, plan, case, seed, iteration, history):
    return candidate_pool(problem, seed*10000+iteration+100000000*plan['profile']['cases'].index(case),
                          plan['profile']['pool'], {r.observation.design_id for r in history})


def discovery_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['case'], row['method']), []).append(row)
    result = []
    for (case, method), group in sorted(groups.items()):
        cold = [r for r in group if r['initial_kind'] == 'cold']
        warm = [r for r in group if r['initial_kind'] == 'warm']
        successes = sum(r['has_feasible'] for r in cold)
        result.append({'case': case, 'method': method, 'runs': len(group),
            'feasible_runs': sum(r['has_feasible'] for r in group),
            'cold_n': len(cold), 'cold_discovered': successes,
            'cold_discovery_fraction': successes/len(cold) if cold else None,
            'cold_binomial_95': binomial_interval(successes, len(cold)),
            'warm_n': len(warm), 'warm_improved': sum(r['best_feasible'] < r['initial_best'] for r in warm),
            'stopped_runs': sum(r['stopped'] for r in group),
            'unknown_online': sum(r['terminal_unknown'] for r in group)})
    return result


def result_packet(plan, summaries, qualifications, count, roles):
    return {'schema': plan['schema'], 'stage': plan['stage'], 'suite': 'mathematical',
        'profile': plan['profile']['name'], 'summaries': summaries, 'qualifications': qualifications,
        'discovery_summary': discovery_summary(summaries),
        'paired_comparisons': paired_comparisons(summaries, reference=plan['reference']),
        'charged_model_calls': count, 'charged_by_role': dict(roles),
        'maximum_model_calls': plan['budget']['maximum_model_calls'],
        'unused_calls_not_reassigned': plan['budget']['maximum_model_calls']-count,
        'new_rcaide_mda_calls': 0, 'publication_confirmation': False}


def run(folder, approved_call_limit=None):
    folder = Path(folder)
    plan = load_plan(folder)
    maximum = plan['budget']['maximum_model_calls']
    if type(approved_call_limit) is not int or approved_call_limit < maximum:
        raise ValueError(f'Explicit complete-call cap required: {maximum}')
    descriptor = os.open(str(folder/'execution.lock'), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    ledger = Ledger(folder, maximum)
    summaries, qualifications = [], {}
    for case in plan['profile']['cases']:
        problem, task = make_problem(BASE, case), plan['tasks'][case]
        anchors = [ledger.evaluate(problem, c, f'{case}/anchor/{i}', 'qualification_anchors')
                   for i,c in enumerate(task['anchors'])]
        panel = [ledger.evaluate(problem, c, f'{case}/panel/{i}', 'independent_panel')
                 for i,c in enumerate(task['panel'])]
        qualifications[case] = {**qualification(panel), 'anchors_usable': [bool(r.usable_feasible) for r in anchors]}
        save(folder/'qualification'/(case+'.json'), {'summary': qualifications[case],
            'anchors': [r.to_dict() for r in anchors], 'panel': [r.to_dict() for r in panel]})
        for seed in plan['profile']['seeds']:
            initial = [ledger.evaluate(problem, c, f'{case}/{seed}/random/initial/{i}', 'shared_initial')
                       for i,c in enumerate(task['initial'][str(seed)]['random']['random_candidates'])]
            for method in plan['arms']:
                output = folder/'optimization'/case/f'{seed}_{method}'
                output.mkdir(parents=True, exist_ok=False)
                save(output/'manifest.json', {'case': case, 'seed': seed, 'method': method,
                    'initial': [r.to_dict() for r in initial], 'initial_digest': digest([r.to_dict() for r in initial]),
                    'contract': problem.contract_id, 'specification': plan['methods'][method]})
                optimizer = optimizer_for(problem, BASE, seed, method, plan['profile'], output)
                history, online, start = list(initial), [], time.monotonic()
                for iteration in range(plan['profile']['horizon']):
                    pool, rejected = pool_for(problem, plan, case, seed, iteration, history)
                    candidate, metadata = optimizer.propose(history, pool, iteration)
                    append(output/'proposals.jsonl', {'iteration': iteration, 'candidate': candidate,
                        'pool': pool, 'exact_screen_rejections': rejected, 'metadata': metadata,
                        'prefix_digest': digest([r.to_dict() for r in history])})
                    record = ledger.evaluate(problem, candidate,
                        f'{case}/{seed}/random/{method}/online/{iteration}', 'online')
                    append(output/'observations.jsonl', record.to_dict())
                    history.append(record)
                    online.append(record)
                    optimizer.accept(record)
                    if optimizer.stop_reason:
                        break
                summary = {'case': case, 'seed': seed, 'method': method, 'initial_policy': 'random',
                    **trajectory_metrics(initial, online, plan['profile']['horizon'], bool(optimizer.stop_reason)),
                    'stop_reason': optimizer.stop_reason, 'optimization_seconds': time.monotonic()-start}
                save(output/'summary.json', summary)
                summaries.append(summary)
                print(f"{case} {seed} {method}: feasible={summary['has_feasible']} calls={len(online)}", flush=True)
    result = result_packet(plan, summaries, qualifications, ledger.count, ledger.roles)
    save(folder/'results.json', result)
    return result


def validate_decision(problem, plan, method, seed, history, proposal):
    """Verify guarded-policy invariants, not learned posterior fits or optimizer replay."""
    metadata = proposal['metadata']
    if metadata.get('method') != method:
        raise ValueError('Wrong proposal method')
    spec = plan['methods'][method]
    if spec['engine'] != 'discovery_bo_v1':
        return
    candidate, iteration = proposal['candidate'], proposal['iteration']
    cold = not any(r.usable_feasible for r in history)
    if metadata['cold_start'] != cold or metadata['raw_history_digest'] != digest([r.to_dict() for r in history]):
        raise ValueError('Staged phase or history changed')
    if metadata['feedback_view'] != spec['feedback'] or metadata['protection_enabled'] != spec['protect']:
        raise ValueError('Staged policy changed')
    if cold and spec['cold_global'] and (metadata['local_enabled'] or metadata.get('local_reports')):
        raise ValueError('Cold-local isolation violated')
    if metadata['pool'][metadata['selected_index']] != candidate:
        raise ValueError('Selected pool index changed')
    if spec['protect']:
        distance = nearest_distance(problem, candidate, [r.candidate for r in history])
        if distance is not None and distance <= NEAR_TOLERANCE:
            raise ValueError('Near-repeat protection violated')
        if metadata['near_tolerance'] != NEAR_TOLERANCE or metadata['selected_history_distance'] != distance:
            raise ValueError('Decision resolution or distance changed')
        novel, _, rejected = filter_near(problem, proposal['pool'], history)
        if metadata['global_near_rejections'] != rejected:
            raise ValueError('Near rejection evidence changed')
        scheduled = iteration % COVERAGE_PERIOD == 0
        reason = metadata['coverage_reason']
        if metadata['coverage_scheduled'] != scheduled:
            raise ValueError('Coverage schedule changed')
        if spec['random_discovery'] and cold:
            rng = np.random.default_rng(seed*10000+iteration+920000000)
            expected = novel[int(rng.integers(len(novel)))]
            if reason != 'random_discovery' or expected != candidate:
                raise ValueError('Random discovery decision changed')
        elif scheduled or reason == 'eligible_at_most_one':
            index, rule = coverage_pick(problem, novel, history, seed, iteration)
            if candidate != novel[index] or metadata['coverage_rule'] != rule:
                raise ValueError('Ungated coverage decision changed')
            if scheduled and reason != 'scheduled_coverage':
                raise ValueError('Scheduled coverage was gated')
            if reason == 'eligible_at_most_one' and sum(metadata['eligible']) > 1:
                raise ValueError('Invalid singleton fallback')
        elif reason is not None:
            raise ValueError('Unknown coverage reason')


def audit(folder, write=True):
    folder = Path(folder)
    plan = load_plan(folder)
    before = evidence_hashes(folder)
    events = lines(folder/'calls.jsonl')
    if len(events) % 2:
        raise ValueError('Unsettled model call; never delete or silently retry it')
    settled, roles, ids = {}, Counter(), set()
    for number, (start, end) in enumerate(zip(events[::2], events[1::2]), 1):
        call = {key: start[key] for key in ('number', 'tag', 'role', 'candidate', 'contract', 'new_rcaide_mda')}
        row = DesignRecord.from_dict(end['record'])
        if (start['event'] != 'started' or end['event'] != 'settled' or start['number'] != number
                or start['call_id'] != digest(call) or start['call_id'] in ids
                or end['call_id'] != start['call_id'] or end['tag'] != start['tag']
                or start['tag'] in settled or row.candidate != start['candidate']
                or row.observation.contract_id != start['contract'] or start['new_rcaide_mda'] is not False
                or end['record_digest'] != digest(row.to_dict())):
            raise ValueError('Call identity or ordering mismatch')
        settled[start['tag']] = (row, start['role'])
        roles[start['role']] += 1
        ids.add(start['call_id'])
    if len(ids) > plan['budget']['maximum_model_calls'] or any(n > plan['budget'].get(r,-1) for r,n in roles.items()):
        raise ValueError('Budget mismatch')
    claimed, summaries, qualifications, checked = set(), [], {}, 0
    def claim(tag, role, problem, candidate=None):
        if tag in claimed or tag not in settled:
            raise ValueError('Missing or repeated registered call')
        row, actual_role = settled[tag]
        if (actual_role != role or row.observation.contract_id != problem.contract_id
                or row.candidate != problem.canonical(row.candidate)
                or (candidate is not None and candidate != row.candidate)
                or any(not np.isfinite(v) or v > 0 for v in problem.precheck(row.candidate).values())):
            raise ValueError('Call role, task or candidate changed')
        claimed.add(tag)
        return row
    for case in plan['profile']['cases']:
        problem, task = make_problem(BASE, case), plan['tasks'][case]
        anchors = [claim(f'{case}/anchor/{i}', 'qualification_anchors', problem, c) for i,c in enumerate(task['anchors'])]
        panel = [claim(f'{case}/panel/{i}', 'independent_panel', problem, c) for i,c in enumerate(task['panel'])]
        qualifications[case] = {**qualification(panel), 'anchors_usable': [bool(r.usable_feasible) for r in anchors]}
        if read(folder/'qualification'/(case+'.json')) != plain({'summary': qualifications[case],
                'anchors': [r.to_dict() for r in anchors], 'panel': [r.to_dict() for r in panel]}):
            raise ValueError('Qualification evidence changed')
        for seed in plan['profile']['seeds']:
            initial = [claim(f'{case}/{seed}/random/initial/{i}', 'shared_initial', problem, c)
                       for i,c in enumerate(task['initial'][str(seed)]['random']['random_candidates'])]
            for method in plan['arms']:
                output = folder/'optimization'/case/f'{seed}_{method}'
                if read(output/'manifest.json') != plain({'case': case, 'seed': seed, 'method': method,
                        'initial': [r.to_dict() for r in initial], 'initial_digest': digest([r.to_dict() for r in initial]),
                        'contract': problem.contract_id, 'specification': plan['methods'][method]}):
                    raise ValueError('Trial manifest or common initial labels changed')
                proposals, saved = lines(output/'proposals.jsonl'), read(output/'summary.json')
                if not 0 < len(proposals) <= plan['profile']['horizon']:
                    raise ValueError('Invalid registered horizon')
                if len(proposals) < plan['profile']['horizon'] and not (method == 'original_lsbo' and saved['stop_reason'] == NATIVE_STOP):
                    raise ValueError('Incomplete trial horizon')
                history, online = list(initial), []
                for iteration, proposal in enumerate(proposals):
                    row = claim(f'{case}/{seed}/random/{method}/online/{iteration}', 'online', problem, proposal['candidate'])
                    pool, rejected = pool_for(problem, plan, case, seed, iteration, history)
                    if (proposal['iteration'] != iteration or proposal['prefix_digest'] != digest([r.to_dict() for r in history])
                            or proposal['pool'] != pool or proposal['exact_screen_rejections'] != rejected
                            or row.observation.design_id in {r.observation.design_id for r in history}):
                        raise ValueError('Proposal, prefix, pool or candidate identity changed')
                    validate_decision(problem, plan, method, seed, history, proposal)
                    online.append(row)
                    history.append(row)
                    checked += 1
                reason = saved['stop_reason']
                if reason is not None:
                    # Frozen native uses five consecutive missing objectives, counted after initialization.
                    streak = 0
                    stops = []
                    for i, row in enumerate(online):
                        streak = 0 if row.observation.objective.kind == 'exact' else streak+1
                        if streak >= 5:
                            stops.append(i)
                    if method != 'original_lsbo' or reason != NATIVE_STOP or stops != [len(online)-1]:
                        raise ValueError('Unsupported or unsubstantiated native stop')
                expected = {'case': case, 'seed': seed, 'method': method, 'initial_policy': 'random',
                    **trajectory_metrics(initial, online, plan['profile']['horizon'], reason is not None),
                    'stop_reason': reason}
                if (set(saved) != set(expected)|{'optimization_seconds'}
                        or any(saved[k] != v for k,v in expected.items())
                        or not np.isfinite(saved['optimization_seconds']) or saved['optimization_seconds'] < 0
                        or lines(output/'observations.jsonl') != plain([r.to_dict() for r in online])):
                    raise ValueError('Trial summary or observations changed')
                summaries.append(saved)
    if claimed != set(settled):
        raise ValueError('Unconsumed or unregistered call')
    if read(folder/'results.json') != plain(result_packet(plan, summaries, qualifications, len(ids), roles)):
        raise ValueError('Results differ from reconstructed evidence')
    if evidence_hashes(folder) != before:
        raise ValueError('Evidence changed during audit')
    result = {'passed': True, 'settled_calls': len(ids), 'checked_proposals': checked,
              'trial_count': len(summaries), 'evidence_hashes': before, 'algorithm_replay': False,
              'source_bindings': len(plan['sources']), 'policy_invariants_checked': True}
    if write:
        if (folder/'audit.json').exists():
            if read(folder/'audit.json') != result:
                raise ValueError('Existing audit seal changed; never overwrite')
        else:
            save(folder/'audit.json', result)
    return result


def report(folder):
    folder = Path(folder)
    if read(folder/'audit.json') != audit(folder, write=False):
        raise ValueError('Report requires unchanged audited evidence')
    plan, result = read(folder/'plan.json'), read(folder/'results.json')
    if (folder/'report.md').exists() or (folder/'summary.csv').exists():
        raise FileExistsError('Report already exists; preserve it')
    with (folder/'summary.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        fields = list(result['discovery_summary'][0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result['discovery_summary'])
    text = [f"# Staged study: {plan['stage']} ({plan['profile']['name']})", '',
            'Exploratory only; failure, unknown and native early stops are retained. No MDA.',
            f"Calls: {result['charged_model_calls']}/{result['maximum_model_calls']}; unused calls are not reassigned.", '',
            '| Case | Method | Feasible/all | Cold discovery/cold | Warm improved/warm |',
            '|---|---|---:|---:|---:|']
    for row in result['discovery_summary']:
        text.append(f"| {row['case']} | {row['method']} | {row['feasible_runs']}/{row['runs']} | "
                    f"{row['cold_discovered']}/{row['cold_n']} | {row['warm_improved']}/{row['warm_n']} |")
    text += ['', '0/0 means no runs in that stratum, not a zero success probability.',
             'Paired objective summaries in results.json condition on both methods feasible and retain denominators.',
             'Current math intervals mainly encode signs; sign and interval arms may be identical by construction.',
             'Software audit is not optimizer replay, calibration certification or evidence of superiority.']
    (folder/'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return folder/'report.md'
