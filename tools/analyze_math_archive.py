"""Read-only diagnosis of a frozen comparison archive. No model imports/evaluations.

This reconstructs counts and recorded decision diagnostics, not optimizer replay.
Run: python tools/analyze_math_archive.py ARCHIVE.zip OUTPUT_DIRECTORY
Outputs are new derived files; archive members are never rewritten or extracted.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import zipfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def satisfaction(value):
    if value['kind'] == 'exact':
        return value['value'] <= 0
    if value['kind'] == 'interval':
        lower, upper = value.get('lower'), value.get('upper')
        if upper is not None and upper <= 0:
            return True
        if lower is not None and (lower > 0 or (lower == 0 and not value['lower_inclusive'])):
            return False
    return None


def usable(record):
    run = record['observation']
    labels = [satisfaction(value) for _, value in run['constraints']]
    return (run['status'] == 'completed' and run['objective']['kind'] == 'exact'
            and record['diagnostics'].get('certified_physical_infeasible') is not True
            and bool(labels) and all(value is True for value in labels))


def median(values):
    values = [float(x) for x in values if x is not None and math.isfinite(x)]
    return statistics.median(values) if values else None


def stats(values):
    values = sorted(float(x) for x in values if x is not None and math.isfinite(x))
    return {'n': len(values), 'min': min(values, default=None), 'median': median(values),
            'max': max(values, default=None), 'mean': statistics.mean(values) if values else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    with zipfile.ZipFile(args.archive) as z:
        members = [info for info in z.infolist() if not info.is_dir()]
        check(len({i.filename for i in members}) == len(members), 'duplicate zip paths')
        for info in members:
            p = PurePosixPath(info.filename.replace('\\', '/'))
            check(not p.is_absolute() and '..' not in p.parts, 'unsafe archive path')
        if errors:
            raise ValueError(errors)
        roots = [i.filename[:-len('calls.jsonl')] for i in members if i.filename.endswith('/calls.jsonl') or i.filename == 'calls.jsonl']
        if len(roots) != 1:
            raise ValueError('Expected one campaign ledger')
        root = roots[0]
        names = {i.filename[len(root):] for i in members if i.filename.startswith(root)}
        def raw(path):
            return z.read(root+path)
        def read(path):
            return json.loads(raw(path).decode('utf-8-sig'))
        def lines(path):
            return [json.loads(line) for line in raw(path).decode('utf-8-sig').splitlines() if line.strip()]
        plan, audit, registration, results = [read(p+'.json') for p in ('plan', 'audit', 'registration', 'results')]
        hashes = audit.get('evidence_hashes', {})
        missing, mismatched = [], []
        for path, expected in hashes.items():
            if path not in names:
                missing.append(path)
            elif hashlib.sha256(raw(path)).hexdigest() != expected:
                mismatched.append(path)
        check(not missing and not mismatched, 'audit evidence hashes')
        check(hashlib.sha256(raw('plan.json')).hexdigest() == registration['plan_sha256'], 'registration plan hash')
        snapshots_bad = []
        source_map = {}
        for path, expected in plan['sources'].items():
            source = 'source_snapshot/'+expected+'.source'
            if source not in names or hashlib.sha256(raw(source)).hexdigest() != expected:
                snapshots_bad.append(source)
            relative = path.replace('\\', '/')
            if '/ra_lsbo/' in relative:
                relative = 'ra_lsbo/'+relative.split('/ra_lsbo/', 1)[1]
            elif '/tests/' in relative:
                relative = 'tests/'+relative.split('/tests/', 1)[1]
            elif '/docs/' in relative:
                relative = 'docs/'+relative.split('/docs/', 1)[1]
            source_map[relative] = expected
        check(not snapshots_bad, 'source snapshot hashes')
        events = lines('calls.jsonl')
        starts = [e for e in events if e['event'] == 'started']
        ends = [e for e in events if e['event'] == 'settled']
        roles, lookup = Counter(), {}
        check(len(events) == 2*len(starts) == 2*len(ends), 'unsettled ledger events')
        for index in range(0, len(events)-1, 2):
            a, b = events[index:index+2]
            check(a['event'] == 'started' and b['event'] == 'settled', 'event order')
            check(a['call_id'] == b['call_id'] and a['tag'] == b['tag'], 'start/settlement identity')
            payload = {key: a[key] for key in ('number', 'tag', 'role', 'candidate', 'contract', 'new_rcaide_mda')}
            check(digest(payload) == a['call_id'], 'call payload digest')
            check(a['number'] == index//2+1, 'call numbering')
            check(a['tag'] not in lookup, 'duplicate call tag')
            check(b['record']['candidate'] == a['candidate'], 'ledger candidate')
            check(digest(b['record']) == b['record_digest'], 'record digest')
            check(b['record']['observation']['design_id'] == digest(a['candidate']), 'design ID')
            check(b['record']['observation']['contract_id'] == a['contract'], 'contract ID')
            lookup[a['tag']] = b['record']
            roles[a['role']] += 1
        check(dict(roles) == results['charged_by_role'], 'aggregate role counts')
        check(len(starts) == results['charged_model_calls'], 'total calls')
        groups, trials, step_rows, examples = defaultdict(list), [], [], {}
        index_results = {(r['case'], r['seed'], r['method']): r for r in results['summaries']}
        initial_by_case_seed = {}
        for case in plan['profile']['cases']:
            for seed in plan['profile']['seeds']:
                for method in plan['arms']:
                    directory = f'optimization/{case}/{seed}_{method}'
                    manifest = read(directory+'/manifest.json')
                    props = lines(directory+'/proposals.jsonl')
                    obs = lines(directory+'/observations.jsonl')
                    summary = read(directory+'/summary.json')
                    initial = manifest['initial']
                    check(len(initial) == plan['profile']['initial'], 'initial horizon')
                    check(len(props) == len(obs) == plan['profile']['horizon'], 'online horizon')
                    check(digest(initial) == manifest['initial_digest'], 'initial digest')
                    init_digest = digest(initial)
                    previous = initial_by_case_seed.setdefault((case, seed), init_digest)
                    check(previous == init_digest, 'shared initial mismatch')
                    check(summary == index_results[(case, seed, method)], 'summary/results mismatch')
                    for j, r in enumerate(initial):
                        check(lookup[f'{case}/{seed}/random/initial/{j}'] == r, 'initial/ledger mismatch')
                    history, feasible_values = list(initial), [r['observation']['objective']['value'] for r in initial if usable(r)]
                    warm = bool(feasible_values)
                    first = next((j+1 for j, r in enumerate(initial) if usable(r)), None)
                    initial_best = min(feasible_values, default=None)
                    detail = {'case':case, 'seed':seed, 'method':method, 'warm':warm,
                              'initial_best':initial_best, 'initial_usable':len(feasible_values),
                              'local_selected':0, 'online_usable':0, 'local_solves':0,
                              'local_success':0, 'clip_warnings':0, 'local_errors':0,
                              'modes':Counter(), 'fallbacks':Counter(), 'branches':Counter(),
                              'local_status':Counter(), 'selected_joint':[], 'eligible_fraction':[],
                              'selected_support':[], 'selected_constraint_probabilities':defaultdict(list)}
                    seen = {digest(r['candidate']) for r in history}
                    for j, (p, r) in enumerate(zip(props, obs)):
                        check(p['iteration'] == j, 'proposal iteration')
                        check(p['candidate'] == r['candidate'], 'proposal/observation candidate')
                        check(r == lookup[f'{case}/{seed}/random/{method}/online/{j}'], 'online/ledger mismatch')
                        check(p['prefix_digest'] == digest(history), 'proposal prefix')
                        check(digest(r['candidate']) not in seen, 'repeated true evaluation')
                        before_feasible = bool(feasible_values)
                        feasible = usable(r)
                        if feasible:
                            feasible_values.append(r['observation']['objective']['value'])
                            detail['online_usable'] += 1
                            if first is None:
                                first = len(initial)+j+1
                        seen.add(digest(r['candidate']))
                        history.append(r)
                        m = p['metadata']
                        if method not in examples:
                            def slim(value):
                                if isinstance(value, dict):
                                    return {k: slim(v) for k,v in value.items()}
                                if isinstance(value, list):
                                    return {'length':len(value), 'first_two': [slim(x) for x in value[:2]]}
                                return value
                            examples[method] = {'metadata':slim(m), 'candidate':p['candidate'], 'observation':r,
                                                'pool_first':p['pool'][0]}
                        detail['modes'][str(m.get('mode'))] += 1
                        detail['fallbacks'][str(m.get('fallback'))] += 1
                        detail['branches'][json.dumps(p['candidate'].get('discrete'),sort_keys=True)] += 1
                        detail['local_selected'] += int(m.get('selected_from_local_augmentation', False))
                        pool = m.get('pool', p['pool'])
                        selected = next((i for i, c in enumerate(pool) if c == p['candidate']), None)
                        row = {'case':case,'seed':seed,'method':method,'iteration':j,'warm':warm,
                               'before_feasible':before_feasible,'feasible':feasible,
                               'candidate':p['candidate'],'mode':m.get('mode'),'fallback':m.get('fallback'),
                               'local_selected':m.get('selected_from_local_augmentation'),
                               'global_n':len(p['pool']),'n':len(pool),'selected_index':selected}
                        eligible = m.get('eligible')
                        if isinstance(eligible,list):
                            row['eligible_n'] = sum(eligible)
                            row['global_eligible_n'] = sum(eligible[:len(p['pool'])])
                            detail['eligible_fraction'].append(sum(eligible)/len(eligible))
                        if selected is not None:
                            for field in ('joint_probability_independence_proxy','completion_probability','scores','optimistic_violation'):
                                data = m.get(field)
                                if isinstance(data,list) and selected < len(data):
                                    row[field] = data[selected]
                            if row.get('joint_probability_independence_proxy') is not None:
                                detail['selected_joint'].append(row['joint_probability_independence_proxy'])
                            for key, value in m.get('per_constraint_probability',{}).items():
                                detail['selected_constraint_probabilities'][key].append(value[selected])
                            forecasts = m.get('forecasts',{})
                            row['constraints'] = {key:{k:v[selected] for k,v in f.items() if k in ('mean','variance')}
                                                  for key,f in forecasts.get('constraints',{}).items()}
                        for report in m.get('local_reports',[]):
                            detail['local_solves'] += 1
                            detail['local_success'] += int(report.get('success',False))
                            detail['clip_warnings'] += report.get('slsqp_clipping_warnings',0)
                            detail['local_errors'] += int(report.get('local_error') is not None)
                            detail['local_status'][str(report.get('status'))] += 1
                        step_rows.append(row)
                    best = min(feasible_values,default=None)
                    check(summary['best_feasible'] == best and summary['has_feasible'] == bool(feasible_values), 'recomputed best')
                    check(summary['first_feasible_total_call'] == first, 'recomputed first feasible')
                    detail.update(best=best, first=first, found=bool(feasible_values),
                                  optimization_seconds=summary.get('optimization_seconds'))
                    trials.append(detail)
                    groups[(case,method)].append(detail)
        check(len(trials) == len(results['summaries']), 'unexpected/missing summaries')
        integrity = {'archive_sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                     'archive_files':len(members),'uncompressed_bytes':sum(i.file_size for i in members),
                     'archive_bytes':args.archive.stat().st_size,'evidence_files':len(hashes),
                     'missing':missing,'mismatched':mismatched,'source_bindings':len(plan['sources']),
                     'unique_snapshot_hashes':len(set(plan['sources'].values())),
                     'snapshot_mismatches':snapshots_bad,'started':len(starts),'settled':len(ends),
                     'roles':dict(roles),'trials':len(trials),'proposals':len(step_rows),
                     'errors_n':len(errors),'errors':errors[:30], 'new_model_calls':0,'optimizer_replay':False}
        summaries = []
        for (case,method), rows in groups.items():
            cold = [r for r in rows if not r['warm']]
            warm = [r for r in rows if r['warm']]
            d = {'case':case,'method':method,'n':len(rows),'warm':len(warm),'cold':len(cold),
                 'cold_found':sum(r['found'] for r in cold),'feasible':sum(r['found'] for r in rows),
                 'conditional_median':median([r['best'] for r in rows]),
                 'warm_median':median([r['best'] for r in warm]),
                 'warm_improved':sum(r['best'] < r['initial_best'] for r in warm),
                 'online_usable':sum(r['online_usable'] for r in rows),
                 'failed_seeds':[r['seed'] for r in rows if not r['found']],
                 'cold_discovery_online_calls':[r['first']-plan['profile']['initial'] for r in cold if r['found']],
                 'optimization_seconds':stats([r['optimization_seconds'] for r in rows])}
            summaries.append(d)
        out = {'integrity':integrity,'group_summary':summaries,'trials':trials,'examples':examples,
               'qualifications':results['qualifications'],'source_map':source_map}
        for name, value in (('diagnosis.json',out),('steps.json',step_rows)):
            (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print('INTEGRITY '+json.dumps(integrity))
        for d in summaries:
            print('GROUP '+json.dumps(d))
        for method, sample in examples.items():
            print('EXAMPLE '+method+' '+json.dumps(sample,ensure_ascii=False))
        for case in plan['profile']['cases']:
            for phase in ('before_feasible','after_feasible'):
                rows = [r for r in step_rows if r['case']==case and r['method']=='ra_lsbo'
                        and r['before_feasible']==(phase=='after_feasible')]
                print('RA_PHASE '+json.dumps({'case':case,'phase':phase,'n':len(rows),
                    'modes':dict(Counter(str(r['mode']) for r in rows)),
                    'fallbacks':dict(Counter(str(r['fallback']) for r in rows)),
                    'local_selected':sum(bool(r['local_selected']) for r in rows),
                    'feasible':sum(r['feasible'] for r in rows),
                    'joint':stats([r.get('joint_probability_independence_proxy') for r in rows]),
                    'global_eligible':stats([r.get('global_eligible_n') for r in rows])}))
            for r in trials:
                if r['case']==case and r['method']=='ra_lsbo':
                    compact = dict(r)
                    for k in ('selected_joint','eligible_fraction','selected_support'):
                        compact[k]=stats(compact[k])
                    compact['selected_constraint_probabilities'] = {k:stats(v) for k,v in compact['selected_constraint_probabilities'].items()}
                    print('RA_TRIAL '+json.dumps(compact))
            seed = 9701 if case=='branin_sparse' else 9703
            for r in step_rows:
                if r['case']==case and r['seed']==seed and r['method']=='ra_lsbo':
                    print('COLD_TRACE '+json.dumps(r))
        print('QUALIFICATIONS '+json.dumps(results['qualifications']))
        if errors:
            raise ValueError(f'{len(errors)} integrity mismatches; do not interpret unverified results')


if __name__=='__main__':
    main()
