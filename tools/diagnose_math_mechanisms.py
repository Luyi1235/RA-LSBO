"""Post-hoc diagnostics from saved candidates/predictions only, not BO replay.

No evaluator, model fitting or latent retraining is invoked. Numerical proximity
thresholds are descriptive sensitivity checks, not new evaluation identities.
"""
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
import difflib


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


def stats(values):
    values = [float(x) for x in values if x is not None and math.isfinite(x)]
    return {'n': len(values), 'min': min(values, default=None),
            'median': statistics.median(values) if values else None,
            'max': max(values, default=None), 'mean': statistics.mean(values) if values else None}


def distance(a, b):
    if a['discrete'] != b['discrete']:
        return None
    keys = ('x', 'y', 'trim') if a['discrete']['architecture'] == 'augmented' else ('x', 'y')
    return max(abs(a['continuous'][k]-b['continuous'][k]) for k in keys)


def corner(candidate, tolerance=1e-6):
    return all(min(abs(candidate['continuous'][k]), abs(1-candidate['continuous'][k])) <= tolerance
               for k in ('x','y'))


def main():
    archive, output = Path(sys.argv[1]), Path(sys.argv[2])
    diagnosis = json.loads((output/'diagnosis.json').read_text())
    if diagnosis['integrity']['errors_n']:
        raise ValueError('Require verified archive before interpreting mechanisms')
    groups = defaultdict(list)
    records, diffs, per_seed = [], {}, []
    source_check = {'byte_equal':[], 'newline_only':[], 'different':[], 'not_in_checkout':[]}
    schema = Counter()
    with zipfile.ZipFile(archive) as z:
        root = next(n[:-len('calls.jsonl')] for n in z.namelist() if n.endswith('calls.jsonl'))
        def read(path):
            return json.loads(z.read(root+path).decode('utf-8-sig'))
        def lines(path):
            return [json.loads(x) for x in z.read(root+path).decode('utf-8-sig').splitlines() if x.strip()]
        plan=read('plan.json')
        for path,h in plan['sources'].items():
            relative=path.replace('\\','/').split('/RA-LSBO/',1)[-1]
            saved=z.read(root+'source_snapshot/'+h+'.source')
            current=Path(relative)
            if not current.is_file():
                source_check['not_in_checkout'].append(relative)
                continue
            actual=current.read_bytes()
            if actual==saved:
                source_check['byte_equal'].append(relative)
            elif actual.replace(b'\r\n',b'\n')==saved.replace(b'\r\n',b'\n'):
                source_check['newline_only'].append(relative)
            else:
                source_check['different'].append(relative)
                diffs[relative]=''.join(difflib.unified_diff(actual.decode('utf-8-sig').splitlines(True),
                       saved.decode('utf-8-sig').splitlines(True),fromfile='checkout/'+relative,tofile='run_snapshot/'+relative))
        for case in plan['profile']['cases']:
            for seed in plan['profile']['seeds']:
                for method in plan['arms']:
                    base=f'optimization/{case}/{seed}_{method}'
                    manifest=read(base+'/manifest.json')
                    history=list(manifest['initial'])
                    warm=any(usable(r) for r in history)
                    found=warm
                    pp=lines(base+'/proposals.jsonl'); oo=lines(base+'/observations.jsonl')
                    for j,(p,o) in enumerate(zip(pp,oo)):
                        m=p['metadata'];pool=m.get('pool',p['pool']);c=p['candidate'];index=next((i for i,x in enumerate(pool) if x==c),None)
                        dist=[d for r in history if (d:=distance(c,r['candidate'])) is not None]
                        row={'case':case,'seed':seed,'method':method,'iteration':j,
                             'cold_start':not warm,'before_first_feasible':not found,'feasible':usable(o),
                             'corner':corner(c), 'near_distance':min(dist,default=None),
                             'mode':m.get('mode',m.get('phase')),'fallback':m.get('fallback'),
                             'local_selected':bool(m.get('selected_from_local_augmentation',False)),
                             'selected_endpoint':any(r.get('endpoint')==c for r in m.get('local_reports',[])),
                             'global_corner_fraction':sum(corner(x) for x in p['pool'])/len(p['pool'])}
                        for name,v in o['observation']['constraints']:
                            schema[(method,name,v['kind'])]+=1
                        schema[(method,'objective',o['observation']['objective']['kind'])]+=1
                        if method=='ra_lsbo':
                            region=m['heads']['region'];row.update(
                                region_length=region['selected']['continuous_length'],
                                region_discrete_length=region['selected']['discrete_length'],
                                region_exact=region['counts']['n_exact'],
                                region_ep_converged=region['counts']['converged'],
                                region_all_candidates_ok=all(r['status']=='ok' for r in region['candidates']),
                                eligible_n=sum(m['eligible']),global_eligible_n=sum(m['eligible'][:len(p['pool'])]),
                                kappa=m['kappa'],boundary_slot=m['boundary_slot'],
                                clip_warnings=sum(r['slsqp_clipping_warnings'] for r in m['local_reports']),
                                local_errors=sum(r.get('local_error') is not None for r in m['local_reports']),
                                local_success=sum(r['success'] for r in m['local_reports']),
                                local_solves=len(m['local_reports']))
                            if index is not None:
                                row['selected_region_probability']=m['per_constraint_probability']['region'][index]
                                row['selected_joint']=m['joint_probability_independence_proxy'][index]
                                row['selected_completion']=m['completion_probability'][index]
                        if method in ('logcei','sego_utb_adapted') and index is not None:
                            ps=m.get('joint_probability')
                            if ps:row.update(selected_joint=ps[index],pool_joint_min=min(ps),pool_joint_max=max(ps))
                            es=m.get('log_ei')
                            if es:
                                finite=[v for v in es if v is not None]
                                scores=[None if v is None else v+math.log(ps[i]) for i,v in enumerate(es)]
                                finite_indices=[i for i,v in enumerate(scores) if v is not None and math.isfinite(v)]
                                row.update(logei_finite=len(finite),logei_min=min(finite,default=None),
                                    logei_max=max(finite,default=None),selected_logei=es[index],
                                    selected_matches_recorded_argmax=(index==max(finite_indices,key=lambda i:scores[i]) if finite_indices else None))
                            if 'eligible' in m:row['eligible_n']=sum(m['eligible'])
                        records.append(row);groups[(case,method)].append(row)
                        history.append(o);found=found or usable(o)
                    selected_rows=[r for r in groups[(case,method)] if r['seed']==seed]
                    per_seed.append({'case':case,'seed':seed,'method':method,'warm':warm,'found':found,
                        'corner_steps':sum(r['corner'] for r in selected_rows),
                        'near_1e6_steps':sum(r['near_distance'] is not None and r['near_distance']<=1e-6 for r in selected_rows),
                        'singleton_steps':sum(r.get('eligible_n')==1 for r in selected_rows),
                        'region_longest_steps':sum(r.get('region_length')==1 for r in selected_rows),
                        'selected_endpoints':sum(r['selected_endpoint'] for r in selected_rows),
                        'fallback_steps':sum(r['fallback'] is not None for r in selected_rows)})
    aggregates=[]
    for (case,method),rows in groups.items():
        failed={r['seed'] for r in per_seed if r['case']==case and r['method']==method and not r['found']}
        for label, selected in [('all',rows),('pre_first',[r for r in rows if r['before_first_feasible']]),
                                ('post_first',[r for r in rows if not r['before_first_feasible']]),
                                ('failed_runs',[r for r in rows if r['seed'] in failed])]:
            a={'case':case,'method':method,'phase':label,'n':len(selected),
               'feasible':sum(r['feasible'] for r in selected),'corners':sum(r['corner'] for r in selected),
               'near_1e8':sum(r['near_distance'] is not None and r['near_distance']<=1e-8 for r in selected),
               'near_1e6':sum(r['near_distance'] is not None and r['near_distance']<=1e-6 for r in selected),
               'local_selected':sum(r['local_selected'] for r in selected),
               'selected_endpoints':sum(r['selected_endpoint'] for r in selected),
               'modes':dict(Counter(str(r['mode']) for r in selected)),
               'fallbacks':dict(Counter(str(r['fallback']) for r in selected)),
               'singleton':sum(r.get('eligible_n')==1 for r in selected),
               'global_corner_fraction':stats([r['global_corner_fraction'] for r in selected]),
               'joint':stats([r.get('selected_joint') for r in selected])}
            if method=='ra_lsbo':
                a.update(region_lengths=dict(Counter(str(r['region_length']) for r in selected)),
                         region_ep_not_converged=sum(not r['region_ep_converged'] for r in selected),
                         region_bad_fit_candidates=sum(not r['region_all_candidates_ok'] for r in selected),
                         boundary_singleton=sum(r['boundary_slot'] and r['eligible_n']==1 for r in selected),
                         boundary_steps=sum(r['boundary_slot'] for r in selected),
                         clip_warnings=sum(r['clip_warnings'] for r in selected),
                         local_errors=sum(r['local_errors'] for r in selected),
                         local_success=sum(r['local_success'] for r in selected),
                         completion=stats([r['selected_completion'] for r in selected]))
            if method=='logcei':
                a.update(logei_argmax_mismatches=sum(r.get('selected_matches_recorded_argmax') is False for r in selected),
                         nonfinite_logei_candidate_steps=sum(r.get('logei_finite',128)!=128 for r in selected),
                         logei_selected=stats([r.get('selected_logei') for r in selected]),
                         within_pool_logei_span=stats([r['logei_max']-r['logei_min'] for r in selected if r.get('logei_max') is not None]))
            aggregates.append(a)
    pairs=[]; trials=diagnosis['trials']
    for case in plan['profile']['cases']:
        for method in plan['arms']:
            if method=='ra_lsbo':continue
            counts=Counter(); obj_diffs=[]
            for seed in plan['profile']['seeds']:
                a=next(r for r in trials if (r['case'],r['seed'],r['method'])==(case,seed,'ra_lsbo'))
                b=next(r for r in trials if (r['case'],r['seed'],r['method'])==(case,seed,method))
                counts['both_feasible' if a['found'] and b['found'] else 'only_ra' if a['found'] else 'only_other' if b['found'] else 'neither']+=1
                if a['found'] and b['found']:
                    delta=a['best']-b['best'];obj_diffs.append(delta)
                    key='tie' if math.isclose(a['best'],b['best'],rel_tol=1e-10,abs_tol=1e-12) else 'ra_win' if delta<0 else 'other_win'
                else:key='ra_win' if a['found'] else 'other_win' if b['found'] else 'tie'
                counts[key]+=1
            pairs.append({'case':case,'other':method,'n':len(plan['profile']['seeds']),'counts':dict(counts),
                          'conditional_delta_ra_minus_other':stats(obj_diffs),
                          'interpretation':'post-hoc feasibility-first descriptive pairing; not a registered confirmatory test'})
    out={'post_hoc':True,'new_model_calls':0,'optimizer_replay':False,'source_check':source_check,
         'aggregates':aggregates,'per_seed':per_seed,'pairings':pairs,
         'observation_kinds':{'/'.join(k):v for k,v in schema.items()}}
    (output/'mechanisms.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'source_differences.json').write_text(json.dumps(diffs,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SOURCE_COMPARISON',json.dumps({k:len(v) for k,v in source_check.items()}))
    print('SOURCE_DIFFERENCES',json.dumps(source_check['different']))
    for a in aggregates:
        if a['method']=='ra_lsbo':print('MECHANISM',json.dumps(a))
    for a in pairs:print('PAIR',json.dumps(a))


if __name__=='__main__':
    main()
