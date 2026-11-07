"""Pinned PR73 reconciliation and denominator/phase audit; ZERO model/evaluator work."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
from cold_coverage_core import require, first_feasible
from recourse_value_protocol import verify, ADAPTIVE, FIXED
from pr69_evidence_verify import fingerprint
from ra_lsbo.experiments.information_value_stage1.io import write_new, manifest

EVIDENCE='f02c936732e24aec656af7cb7ff0f8efb425ce94'
SOURCE='1949c3139d2256b35e1734af184248f551912ba0'


def run(evidence_root,output):
    ev=Path(evidence_root).resolve();out=Path(output).resolve()
    require(not out.exists() and ev not in out.parents and ev!=out,'New output outside evidence required')
    require(subprocess.check_output(['git','-C',str(ev),'rev-parse','HEAD'],text=True).strip()==EVIDENCE,'Wrong evidence SHA')
    require(not subprocess.check_output(['git','-C',str(ev),'status','--porcelain'],text=True).strip(),'Dirty evidence')
    root=ev/'results/recourse-value-pilot-20261107/full'
    before=verify(root)
    r=json.loads((root/'report.json').read_text(encoding='utf8'))
    require(r['code_sha']==SOURCE,'Wrong original code')
    rows=r['trials'];arms={};cold=[];discoveries=[];shared={}
    for arm in sorted({x['arm'] for x in rows}):
        group=[x for x in rows if x['arm']==arm]
        arms[arm]={'trials':len(group),'online_denominator':sum(x['online_calls'] for x in group),
            'initially_feasible':sum(x['initial_best'] is not None for x in group),
            'initially_cold':sum(x['initial_best'] is None for x in group),
            'final_feasible':sum(x['final_best'] is not None for x in group),
            'nonfeasible':sum(x['nonfeasible'] for x in group),
            'proposal_seconds':sum(x['proposal_seconds'] for x in group),
            'first_feasible_calls':{x['case']+'/'+str(x['seed']):first_feasible(x['curve']) for x in group}}
        for row in group:
            if row['initial_best'] is None:
                cold.append({'case':row['case'],'seed':row['seed'],'arm':arm,'final_best':row['final_best']})
            j=first_feasible(row['curve'])
            if row['initial_best'] is None and j is not None:
                discoveries.append({'case':row['case'],'seed':row['seed'],'arm':arm,'call':j,
                    'mode':row['decisions'][j-1]['mode'],'forecast_available':row['decisions'][j-1]['forecast_available']})
            if arm not in (ADAPTIVE,FIXED):continue
            path=root/'trials'/row['case']/str(row['seed'])/arm/'proposals.jsonl'
            props=[json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]
            for p in props:
                meta=p['metadata'];iv=meta.get('information_value_decision')
                if not iv:continue
                identity=(row['case'],row['seed'],p['iteration'],p['history_hash'],
                    fingerprint([p['protected_pool'],meta.get('forecasts'),meta.get('best_observed')]))
                shared.setdefault(identity,{})[arm]={'depth':iv['depth'],'selected':iv['selected_index'],
                    'decomposition':iv.get('recourse_decomposition')}
    common=[g for g in shared.values() if set(g)=={ADAPTIVE,FIXED}]
    depth2=[g for g in common if g[ADAPTIVE]['depth']==g[FIXED]['depth']==2]
    for g in depth2:
        require(g[ADAPTIVE]['decomposition']==g[FIXED]['decomposition'],'Same state has different score calculations')
    after=verify(root);require(before==after,'Input changed')
    result={'status':'COMPLETE_READ_ONLY_PR73_REVIEW','input_result_sha':EVIDENCE,'original_code_sha':SOURCE,
        'input_verify':before,'arms':arms,'initial_cold_trials':cold,'new_discoveries':discoveries,
        'exact_shared_model_states_including_final':len(common),'shared_depth2_states':len(depth2),
        'shared_depth2_different_root':sum(g[ADAPTIVE]['selected']!=g[FIXED]['selected'] for g in depth2),
        'summary_by_case':r['summary'],'new_evaluations':0,'new_gp_fits':0,'new_vae_training':0,
        'limits':['No pooled raw objectives or premiums across cases.',
            'Schedule observation is not a causal proof of coverage benefit.',
            'This review reuses archived records; no true counterfactual outcomes.']}
    out.mkdir(parents=True);write_new(out/'review.json',result);manifest(out)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.evidence_root,a.output)
    print(json.dumps({k:r[k] for k in ('status','exact_shared_model_states_including_final','shared_depth2_states','shared_depth2_different_root','new_discoveries')},indent=2))
