"""Reconstruct qualification diagnostics without fitting or evaluating any model."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ra_lsbo.experiments.simplified_aircraft import reviewed_qualification as q


def review(folder):
    folder=Path(folder);receipt=q.verify(folder)
    plan=json.loads((folder/'plan.json').read_bytes());saved=json.loads((folder/'results.json').read_bytes())
    events=[json.loads(x) for x in (folder/'calls.jsonl').read_bytes().splitlines() if x.strip()]
    rows={e['tag']:e['record'] for e in events if e['event']=='settled'}
    output={'receipt':receipt,'cases':{},'new_model_calls':0,'new_rcaide_mda_calls':0}
    for case,task in plan['tasks'].items():
        panel=[rows[f'{case}/panel/{i}'] for i in range(len(task['panel']))]
        counts=Counter(q.physical_state(r) for r in panel)
        violations=Counter(n for r in panel for n,v in r['observation']['constraints'] if q.satisfaction(v) is False)
        output_kinds=Counter(n+':'+v['kind'] for r in panel for n,v in
                            [('objective',r['observation']['objective']),*r['observation']['constraints']])
        terminations=Counter(r['diagnostics']['termination_reason'] for r in panel)
        for key,current in [('output_kinds',dict(output_kinds)),('terminations',dict(terminations))]:
            if saved['cases'][case]['panel'][key]!=current:raise ValueError('Saved panel does not reconstruct')
        strata=[]
        for parallel in (32,40,48):
            group=[r for r in panel if r['candidate']['discrete']['parallel_strings']==parallel]
            strata.append({'parallel_strings':parallel,'n':len(group),
                'usable_feasible':sum(q.physical_state(r)=='usable_feasible' for r in group),
                'mission_complete':sum(r['observation']['objective']['kind']=='exact' for r in group)})
        checks=[]
        for key,item in saved['cases'][case]['numerical_checks'].items():
            rs=[rows[f'{case}/{key}/{int(s)}'] for s in item['steps_s']]
            cons=[dict(r['observation']['constraints']) for r in rs]
            labels=[[q.satisfaction(c[n]) for n in sorted(c)] for c in cons]
            exact={n:max(c[n]['value'] for c in cons)-min(c[n]['value'] for c in cons)
                   for n in cons[0] if all(c[n]['kind']=='exact' for c in cons)}
            all_exact=all(r['observation']['objective']['kind']=='exact' for r in rs)
            energy=[r['observation']['objective']['value'] for r in rs] if all_exact else None
            relative=(max(energy)-min(energy))/max(abs(energy[-1]),1e-12) if all_exact else None
            reasons=[r['diagnostics']['termination_reason'] for r in rs]
            if any(r['diagnostics'].get('numerical_failure') for r in rs):status='needs_review_numerical_failure'
            elif len({(r['observation']['status'],r['diagnostics']['termination_reason']) for r in rs})!=1:status='needs_review_changed_termination'
            elif not all_exact:status='inconclusive_partial_outputs'
            else:status='passed_screen' if relative<=.01 and all(x==labels[0] for x in labels) else 'needs_review'
            if status!=item['status']:raise ValueError('Integration screen does not reconstruct')
            times=[r['diagnostics']['trace'][-1]['elapsed_s'] for r in rs]
            checks.append({'key':key,'status':status,'termination_reasons':reasons,
                'labels_agree':all(x==labels[0] for x in labels),'relative_energy_range':relative,
                'absolute_exact_constraint_ranges':exact,'terminal_time_range_s':max(times)-min(times),
                'physical_states':[q.physical_state(r) for r in rs]})
        rate=counts['usable_feasible']/len(panel)
        output['cases'][case]={'panel_counts':dict(counts),'n':len(panel),'explicit_violations':dict(violations),
            'termination_counts':dict(terminations),'output_kinds':dict(output_kinds),'battery_strata':strata,
            'usable_binomial_95':saved['cases'][case]['panel']['usable_binomial_95'],
            'cold_12_draws_plugin':(1-rate)**12,'cold_rate_is_planning_approximation':True,
            'numerical_screens':checks,'partial_screens_not_promoted_to_pass':True}
    pairs=Counter()
    for i in range(128):
        left,right=[rows[f'{c}/panel/{i}'] for c in ('aircraft_ordinary','aircraft_sparse')]
        if left['candidate']!=right['candidate']:raise ValueError('Mission panels were not matched')
        pairs[q.physical_state(left)+' -> '+q.physical_state(right)]+=1
    output['matched_mission_counts']=dict(pairs)
    if q.verify(folder)!=receipt:raise ValueError('Evidence changed during review')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('qualification',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();src=a.qualification.resolve();dst=a.output.resolve()
    if dst==src or src in dst.parents or dst in src.parents or dst.exists():
        raise ValueError('Use a new directory separate from the original evidence')
    facts=review(src);dst.mkdir(parents=True)
    (dst/'facts.json').write_text(json.dumps(facts,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print({'settled_calls':316,'cases':list(facts['cases']),'new_model_calls':0})


if __name__=='__main__':main()
