"""Fixed checkpoint metric sensitivity, not a new optimizer or encoder training.

Two representations (corpus_reset and onehot), two scalar metrics (legacy and
unit median legal-pair RMS distance), all 32 historical prefixes, two GP heads.
Exactly 256 grouped head fits (2,048 kernel candidates maximum), zero evaluator
calls, zero encoder fits. Outputs and targets remain development evidence.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

import numpy as np
from scipy.special import ndtr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.models import log_interval_probability
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.partial_refinement.posterior import GroupedMixedHead
from tools.review_structure_corpus_full import load_verified, PREFIX, sha

PILOT_SHA='35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd'
PILOT_PREFIX='aircraft_nominal_20261030_01/pilot/'
ARMS=('corpus_reset','onehot')
METRICS=('legacy','unit_pair_median')
HEADS=('objective','soc')
GP_FITS=256


def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def codebook_factor(codebook, metric):
    """Unweighted all-legal-config-pair geometry; no response input or tuning."""
    b=np.asarray(codebook,dtype=float)
    if b.ndim!=2 or min(b.shape)<1 or len(b)<2 or not np.isfinite(b).all():
        raise ValueError('Finite multi-row codebook required')
    if metric not in METRICS: raise ValueError('Unregistered metric')
    d=np.sqrt(np.mean((b[:,None,:]-b[None,:,:])**2,axis=-1))
    med=float(np.median(d[np.triu_indices(len(b),1)]))
    if not math.isfinite(med) or med<=1e-12: raise ValueError('Degenerate codebook; no automatic repair')
    return 1. if metric=='legacy' else med


class FrozenCodebook:
    """Exact finite-schema feature lookup; no decoder, neural fitting or repairs."""
    def __init__(self,variables,configurations,codebook,metric='legacy'):
        self.variables=deepcopy(variables);self.by_name={v['name']:v for v in variables}
        if len(self.by_name)!=len(variables): raise ValueError('Duplicate variable')
        self.cont=[v for v in variables if v['kind']=='continuous']
        self.disc=[v for v in variables if v['kind']!='continuous']
        self.configurations=deepcopy(configurations)
        self.codebook=np.asarray(codebook,float).copy()
        if len(configurations)!=len(self.codebook): raise ValueError('Codebook row mismatch')
        self.keys=[digest(c) for c in configurations]
        if len(set(self.keys))!=len(self.keys): raise ValueError('Duplicate canonical configuration')
        self.factor=codebook_factor(self.codebook,metric);self.metric=metric
        self.lookup={k:i for i,k in enumerate(self.keys)}

    def active(self,v,d,seen=()):
        if v['name'] in seen: raise ValueError('Cyclic activation')
        return all(self.active(self.by_name[parent],d,seen+(v['name'],)) and d[parent] in allowed
                   for parent,allowed in v.get('active_when',{}).items())

    def transform(self,candidates):
        out=[]
        for c in candidates:
            d,x=c['discrete'],c['continuous']
            if set(d)!={v['name'] for v in self.disc} or set(x)!={v['name'] for v in self.cont}:
                raise ValueError('Candidate schema mismatch')
            key=digest(d)
            if key not in self.lookup: raise ValueError('Unknown canonical configuration')
            units=[];masks=[]
            for v in self.cont:
                a=self.active(v,d);value=float(x[v['name']]);lo,hi=v['bounds']
                if not math.isfinite(value) or not lo<=value<=hi: raise ValueError('Invalid continuous coordinate')
                if not a and value!=float(v['default']): raise ValueError('Noncanonical inactive coordinate')
                units.append(((value-lo)/(hi-lo))*a);masks.append(float(a))
            out.append(np.r_[units,masks,self.codebook[self.lookup[key]]/self.factor])
        return np.asarray(out,dtype=float)


def score(mean,variance,noise,observation,head):
    if not all(math.isfinite(x) for x in (mean,variance,noise)) or variance < -1e-10 or noise<0:
        raise ValueError('Invalid prediction')
    var=max(0.,variance)+noise**2
    if var<=0: raise ValueError('Zero variance')
    sd=math.sqrt(var);o=observation
    result={'mean':mean,'variance':var,'kind':o.kind,'observation':asdict(o)}
    if o.kind=='missing': raise ValueError('Unregistered missing scoring target')
    if o.kind=='exact':
        err=mean-o.value
        result.update(nll=.5*(math.log(2*math.pi*var)+err*err/var),squared_error=err*err)
    else:
        lo=-np.inf if o.lower is None else (o.lower-mean)/sd
        hi=np.inf if o.upper is None else (o.upper-mean)/sd
        result['nll']=-log_interval_probability(lo,hi)
    if head=='soc' and o.constraint_satisfaction() is not None:
        prob=float(ndtr(-mean/sd));result.update(feasible_probability=prob,brier=(prob-float(o.constraint_satisfaction()))**2)
    return result


def run(corpus_archive,manifest,pilot_archive,output):
    output=Path(output)
    if output.exists(): raise FileExistsError('New output directory required')
    rows,verified=load_verified(corpus_archive,manifest)
    if sha(pilot_archive)!=PILOT_SHA: raise ValueError('Pilot archive identity mismatch')
    relevant=[r for r in rows if r['arm'] in ARMS]
    repo=Path(__file__).resolve().parents[1]
    imports=[*sorted((repo/'ra_lsbo').rglob('*.py')),Path(__file__),repo/'tools/review_structure_corpus_full.py',
             repo/'docs/STRUCTURE_CODEBOOK_METRIC_PROTOCOL_20261031.md']
    sources={p.relative_to(repo).as_posix():sha(p) for p in imports}
    result_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True)
    output.mkdir(parents=True)
    save(output/'plan.json',{'schema':'frozen_codebook_metric_probe_v1','corpus_sha':verified['archive_sha256'],
        'pilot_sha':PILOT_SHA,'arms':ARMS,'metrics':METRICS,'heads':HEADS,'expected_GP_heads':GP_FITS,
        'max_kernel_candidates':GP_FITS*8,'new_encoder_fits':0,'new_evaluations':0,'optimizer_runs':0,
        'source_commit':result_commit.stdout.strip() or 'unversioned','sources':sources,
        'python':sys.version,'platform':platform.platform(),
        'control_check':'absolute mean difference / output-scale <= 1e-6; variance / output-scale^2 <= 1e-6',
        'scope':'post-hoc same-history sensitivity, not confirmed gains; no retuning after results'})
    (output/'execution.lock').write_text('No resume, no retry, no automatic additional grid.\n')
    completed=[];attempts=0;checks=[]
    with zipfile.ZipFile(corpus_archive) as z,zipfile.ZipFile(pilot_archive) as pz:
        for r in relevant:
            case,seed,t,arm=(r[k] for k in ('case','seed','iteration','arm'))
            trial=PILOT_PREFIX+f'optimization/{case}/{seed}_ev_latent/'
            initial=json.loads(pz.read(trial+'manifest.json'))['initial']
            online=[json.loads(l) for l in pz.read(trial+'observations.jsonl').splitlines() if l.strip()]
            records=[DesignRecord.from_dict(o) for o in initial+online[:t]]
            targets=[DesignRecord.from_dict(o) for o in online[t:min(t+4,30)]]
            if digest([o.to_dict() for o in records])!=r['history_identity']: raise ValueError('Historical prefix mismatch')
            meta=json.loads(z.read(PREFIX+f'full/{case}_{seed}_p{t}/reset_checkpoint/metadata.json'))
            configs=meta['corpus']['configurations'];variables=meta['variables']
            if r['grammar'] is not None and configs!=r['grammar']['configurations']:
                raise ValueError('Codebook configuration order differs')
            for metric in METRICS:
                f=FrozenCodebook(variables,configs,r['normalized_codebook'],metric)
                x=f.transform([o.candidate for o in records]);xt=f.transform([o.candidate for o in targets]);hs=[]
                for hname in HEADS:
                    source=next(h for h in r['predictions'] if h['head']==hname)
                    if [o.candidate for o in targets]!=[s['candidate'] for s in source['scores']]: raise ValueError('Targets differ')
                    attempts+=1
                    if attempts>GP_FITS: raise RuntimeError('Fit budget exceeded')
                    with (output/'attempts.jsonl').open('a') as s:s.write(json.dumps({'attempt':attempts,'case':case,'seed':seed,'prefix':t,'arm':arm,'metric':metric,'head':hname})+'\n')
                    values=[o.observation.objective if hname=='objective' else o.observation.constraint('soc') for o in records]
                    tick=time.perf_counter()
                    head=GroupedMixedHead(x,values,continuous_count=len(f.cont),seed=seed,
                        continuous_grid=(.05,.15,.4,1.),discrete_grid=(.5,1.5),objective=hname=='objective',
                        reference_scale=40000. if hname=='objective' else 1.)
                    pred=head.predict(xt);scores=[]
                    for i,o in enumerate(targets):
                        obs=o.observation.objective if hname=='objective' else o.observation.constraint('soc')
                        if asdict(obs)!=source['scores'][i]['observation']: raise ValueError('Target observation differs')
                        scores.append({'candidate':o.candidate,'evidence_id':obs.evidence_id,
                            **score(float(pred['mean'][i]),float(pred['variance'][i]),.001*head.scale,obs,hname)})
                    hrow={'head':hname,'diagnostics':head.diagnostics(),'seconds':time.perf_counter()-tick,'scores':scores}
                    hs.append(hrow)
                    if metric=='legacy':
                        dm=max(abs(a['mean']-b['mean']) for a,b in zip(scores,source['scores']))/source['diagnostics']['scale']
                        dv=max(abs(a['variance']-b['variance']) for a,b in zip(scores,source['scores']))/source['diagnostics']['scale']**2
                        checks.append({'case':case,'seed':seed,'prefix':t,'arm':arm,'head':hname,'normalized_mean_error':dm,
                                       'normalized_variance_error':dv,'matched':bool(dm<=1e-6 and dv<=1e-6)})
                        if not checks[-1]['matched']:
                            save(output/'control_mismatch.json',checks[-1]);raise ValueError('Legacy forecast control mismatch; stop without relaxing tolerance')
                row={k:r[k] for k in ('case','seed','iteration','arm','feature_identity','history_identity')}
                row.update(metric=metric,scale_factor=f.factor,predictions=hs)
                completed.append(row)
                with (output/'rows.jsonl').open('a') as s:s.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    if attempts!=GP_FITS: raise ValueError('Incomplete head fits')
    if any(sha(repo/rel)!=h for rel,h in sources.items()): raise ValueError('Changed code during diagnostic')
    save(output/'control_checks.json',checks)
    save(output/'summary.json',{'GP_heads':attempts,'controls_matched':sum(c['matched'] for c in checks),
         'controls':len(checks),'new_encoder_fits':0,'new_evaluations':0,'new_RCAIDE_calls':0,'optimizer_runs':0,
         'failed_kernel_candidates':sum(c['status']!='ok' for r in completed for h in r['predictions'] for c in h['diagnostics']['candidates']),
         'scientific_pass':None,'automatic_pilot_authorization':False})
    evidence={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
    save(output/'audit.json',{'completed':True,'files':evidence,'no_retraining':True,'algorithm_replay':False})
    print({'GP_heads':attempts,'control_matches':len(checks),'new_evaluations':0})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('corpus_archive','manifest','pilot_archive','output'):p.add_argument(n,type=Path)
    p.add_argument('--approved-gp-heads',type=int,required=True);a=p.parse_args()
    if a.approved_gp_heads!=GP_FITS: raise ValueError('Exactly 256 offline GP head fits require approval')
    run(a.corpus_archive,a.manifest,a.pilot_archive,a.output)

if __name__=='__main__':main()
