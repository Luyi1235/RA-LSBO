"""Offline 2,000-configuration VAE comparison; no BO or evaluator calls."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.problems import AircraftCase, MathematicalCase
from ra_lsbo.experiments.core.protocol import BASE, ROOT
from ra_lsbo.features import load_base
from ra_lsbo.partial_bo.models import PrefixFeatures, log_interval_probability
from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft, HierarchicalBranin
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.partial_refinement.posterior import GroupedMixedHead
from ra_lsbo.structure_corpus import (CorpusRepresentation, feature_snapshot,
    native_observations, grammar_diagnostics)

ARCHIVE_SHA = '35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd'
PREFIX = 'aircraft_nominal_20261030_01/pilot/'
ARMS = ('old30', 'old480', 'corpus_reset', 'corpus_persistent', 'onehot')
HEADS = ('objective', 'soc')
CASES = ('aircraft_ordinary', 'aircraft_sparse')
CHECKPOINTS = (0, 29)
UPDATE_AT = (8, 16, 24)
DOCUMENT = 'docs/STRUCTURE_CORPUS_2000_PROTOCOL_20261031.md'


def blocked(*a, **k):
    raise AssertionError('Evaluation forbidden in the structural corpus study')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda: s.read(1048576), b''): h.update(b)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def limits(profile):
    if profile not in {'smoke', 'full'}: raise ValueError('Only fixed offline profiles')
    trials = 2 if profile == 'smoke' else 16
    return {'encoder_fit_calls': trials*10, 'onehot_transforms': trials*2,
            'GP_head_fit_calls': trials*2*len(ARMS)*len(HEADS),
            'GP_grid_candidates_max': trials*2*len(ARMS)*len(HEADS)*8,
            'new_evaluator_calls': 0, 'new_RCAIDE_calls': 0}


def forecast_score(mean, variance, noise_sd, observed, *, constraint=False):
    """Scores of the same observation type; never pool densities and masses."""
    if not all(math.isfinite(x) for x in (mean, variance, noise_sd)) or variance < -1e-10 or noise_sd < 0:
        raise ValueError('Invalid forecast moments')
    variance = max(variance, 0.) + noise_sd**2
    if variance <= 0: raise ValueError('Positive predictive observation variance required')
    sd = math.sqrt(variance)
    row = {'mean': mean, 'variance': variance, 'kind': observed.kind}
    if observed.kind == 'missing': return {**row, 'score_status': 'missing_not_imputed'}
    if observed.kind == 'exact':
        error = mean-observed.value
        row.update(score_status='scored', squared_error=error**2,
                   nll=.5*(math.log(2*math.pi*variance)+error**2/variance))
    else:
        lo = -np.inf if observed.lower is None else (observed.lower-mean)/sd
        hi = np.inf if observed.upper is None else (observed.upper-mean)/sd
        row.update(score_status='scored', nll=-log_interval_probability(lo, hi))
    if constraint and observed.constraint_satisfaction() is not None:
        prob = float(ndtr(-mean/sd))
        row.update(feasible_probability=prob, brier=(prob-float(observed.constraint_satisfaction()))**2)
    return row


def read_trial(archive, case, seed):
    root = PREFIX+f'optimization/{case}/{seed}_ev_latent/'
    m = json.loads(archive.read(root+'manifest.json'))
    online = [json.loads(l) for l in archive.read(root+'observations.jsonl').splitlines() if l.strip()]
    props = [json.loads(l) for l in archive.read(root+'proposals.jsonl').splitlines() if l.strip()]
    if len(m['initial']) != 12 or len(online) != 30 or len(props) != 30:
        raise ValueError('Historical trial incomplete')
    initial, online = [DesignRecord.from_dict(r) for r in m['initial']], [DesignRecord.from_dict(r) for r in online]
    for t, prop in enumerate(props):
        if digest([r.to_dict() for r in initial+online[:t]]) != prop['prefix_digest']:
            raise ValueError('Historical training prefix differs')
        if prop['candidate'] != online[t].candidate: raise ValueError('Proposal/observation differs')
    return initial, online, props


def feature_report(f):
    cs = []
    for d in f.space.configurations():
        c = f.space.default().to_dict(); c['discrete'] = d
        cs.append(f.space.canonical(f.modules['space'].Candidate(**c)).to_dict())
    block = f.transform(cs)[:, 2*len(f.space.continuous):]
    dist = np.sqrt(np.mean((block[:,None,:]-block[None,:,:])**2, axis=-1))
    return {'feature_identity': f.identity, 'state_identity': f.state_hash,
            'offset': f.offset.tolist(), 'scale': f.scale.tolist(),
            'normalized_codebook': block.tolist(), 'rms_distance_matrix': dist.tolist(),
            'grammar': None if f.encoder is None else grammar_diagnostics(f.encoder)}


def run(archive_path, output, profile, approved_encoder_fits, approved_gp_heads):
    cap = limits(profile)
    if approved_encoder_fits != cap['encoder_fit_calls'] or approved_gp_heads != cap['GP_head_fit_calls']:
        raise ValueError('Exact offline fit caps required; no hidden training grid')
    archive_path, output = Path(archive_path), Path(output)
    if output.exists(): raise FileExistsError('Use a new output directory')
    if sha(archive_path) != ARCHIVE_SHA: raise ValueError('Historical archive SHA256 differs')
    for cls in (AircraftCase, MathematicalCase, SimplifiedElectricAircraft, HierarchicalBranin): cls.evaluate = blocked
    torch.set_num_threads(1)
    seeds = [9911] if profile == 'smoke' else list(range(9911, 9919))
    sources = [*sorted((ROOT/'ra_lsbo').rglob('*.py')), *sorted(BASE.rglob('*.py')),
               Path(__file__), ROOT/DOCUMENT, ROOT/'tests/test_structure_corpus.py']
    sources = [p for p in sources if '__pycache__' not in p.parts]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    head = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    plan = {'schema':'structure_corpus_offline_v1','profile':profile,'input_archive_sha256':ARCHIVE_SHA,
        'cases':CASES,'seeds':seeds,'arms':ARMS,'heads':HEADS,'checkpoints':CHECKPOINTS,'update_at':UPDATE_AT,
        'structural_draws':2000,'duplicate_draws_retained':True,
        'source_commit':head.stdout.strip() if head.returncode==0 else 'unversioned','sources':hashes,
        'python':sys.version,'platform':platform.platform(),
        'packages':{m:importlib.metadata.version(m) for m in ('numpy','scipy','torch','gpytorch','botorch')},
        'limits':cap,'optimization_replay':False,'new_evaluator_calls':0}
    output.mkdir(parents=True);save(output/'plan.json',plan)
    (output/'execution.lock').write_text('No resume or automatic retries.\n',encoding='utf-8')
    counts=Counter(encoder_fit_calls=0,onehot_transforms=0,GP_head_fit_calls=0);results=[]
    def charge(kind):
        counts[kind]+=1
        if counts[kind]>cap[kind]: raise RuntimeError('Offline fit budget exceeded')
        with (output/'fit_attempts.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'kind':kind,'number':counts[kind]})+'\n')
    with zipfile.ZipFile(archive_path) as archive:
        historical=json.loads(archive.read(PREFIX+'plan.json'))
        for old,h in historical['sources'].items():
            raw=archive.read(PREFIX+'source_snapshot/'+h+'.source')
            if hashlib.sha256(raw).hexdigest()!=h: raise ValueError('Source snapshot changed')
            p=old.replace('\\','/')
            for marker in ('/ra_lsbo/','/runs/compact_mda_'):
                if marker in p:
                    rel=p[p.index(marker)+1:]
                    if rel.endswith('.py') and raw.replace(b'\r\n',b'\n')!=(ROOT/rel).read_bytes().replace(b'\r\n',b'\n'):
                        raise ValueError('Frozen source differs: '+rel)
                    break
        modules,_=load_base(BASE)
        for case in CASES:
            problem=AircraftCase(BASE,case.split('_')[1])
            for seed in seeds:
                initial,online,props=read_trial(archive,case,seed)
                meta=SimpleNamespace(space=problem.space,contract_id=initial[0].observation.contract_id)
                persistent=CorpusRepresentation(problem.space,seed)
                for iteration in CHECKPOINTS:
                    history=initial+online[:iteration]
                    if iteration:
                        for t in UPDATE_AT:
                            charge('encoder_fit_calls')
                            persistent.fit(native_observations(initial+online[:t],modules,problem.space),incremental=True)
                    else:
                        charge('encoder_fit_calls');persistent.fit(native_observations(history,modules,problem.space))
                    path=output/f'{case}_{seed}_p{iteration}';path.mkdir()
                    persistent.save_checkpoint(path/'persistent_checkpoint')
                    # Same existing candidates for every arm. Targets enter scores only after fits.
                    test=online[iteration:min(iteration+4,30)]
                    for arm in ARMS:
                        tick=time.monotonic();session=None
                        if arm in {'old30','old480','onehot'}:
                            charge('onehot_transforms' if arm=='onehot' else 'encoder_fit_calls')
                            f=PrefixFeatures(meta,BASE,history,seed,latent=arm!='onehot',epochs=480 if arm=='old480' else 30)
                        else:
                            session=persistent
                            if arm=='corpus_reset':
                                session=CorpusRepresentation(problem.space,seed);charge('encoder_fit_calls')
                                session.fit(native_observations(history,modules,problem.space))
                                session.save_checkpoint(path/'reset_checkpoint')
                            f=feature_snapshot(meta,BASE,history,session)
                        info={'case':case,'seed':seed,'iteration':iteration,'arm':arm,
                            'training_count':len(history),'history_identity':digest([r.to_dict() for r in history]),
                            'encoder_prefix_count':len(history) if session is None else len(session.history),
                            'feature_fit_seconds':time.monotonic()-tick,**feature_report(f),
                            'corpus':None if session is None else deepcopy(session.corpus),
                            'training_events':None if session is None else deepcopy(session.events),
                            'old_identity_match': (f.identity==props[iteration]['metadata'].get('feature_identity'))
                                if arm=='old30' and props[iteration]['metadata'].get('feature_identity') else None}
                        if iteration==0 and arm=='corpus_reset': first=f.transform([r.candidate for r in history])
                        if iteration==0 and arm=='corpus_persistent' and not np.array_equal(first,f.transform([r.candidate for r in history])):
                            raise ValueError('Matched initial reset/persistent features differ')
                        fits=[];x=f.transform([r.candidate for r in history]);xt=f.transform([r.candidate for r in test])
                        for name in HEADS:
                            charge('GP_head_fit_calls');tick=time.monotonic()
                            values=[r.observation.objective if name=='objective' else r.observation.constraint(name) for r in history]
                            if any(r.observation.status in {'unknown','infrastructure_error'} or r.diagnostics.get('numerical_failure') for r in history+test):
                                raise ValueError('Untrusted row requires a separately registered score policy')
                            h=GroupedMixedHead(x,values,continuous_count=len(problem.space.continuous),seed=seed,
                                continuous_grid=(.05,.15,.4,1.),discrete_grid=(.5,1.5),objective=name=='objective',
                                reference_scale=40000. if name=='objective' else 1.)
                            prediction=h.predict(xt);scores=[]
                            for i,target in enumerate(test):
                                obs=target.observation.objective if name=='objective' else target.observation.constraint(name)
                                score=forecast_score(float(prediction['mean'][i]),float(prediction['variance'][i]),
                                                     .001*h.scale,obs,constraint=name!='objective')
                                scores.append({'candidate':target.candidate,'evidence_id':obs.evidence_id,
                                               'observation':asdict(obs),**score})
                            fits.append({'head':name,'seconds':time.monotonic()-tick,'diagnostics':h.diagnostics(),'scores':scores})
                        info['predictions']=fits;save(path/(arm+'.json'),info);results.append(info)
                        with (output/'rows.jsonl').open('a',encoding='utf-8') as s:
                            s.write(json.dumps(info,sort_keys=True,allow_nan=False)+'\n')
    if sha(archive_path)!=ARCHIVE_SHA: raise ValueError('Archive changed during execution')
    if any(sha(ROOT/p)!=h for p,h in hashes.items()): raise ValueError('Source changed during execution')
    if any(counts[k]!=cap[k] for k in counts): raise ValueError('Incomplete offline fits')
    summary=[]
    for case in CASES:
        for arm in ARMS:
            for t in CHECKPOINTS:
                rs=[r for r in results if r['case']==case and r['arm']==arm and r['iteration']==t]
                summary.append({'case':case,'arm':arm,'iteration':t,'n_prefixes':len(rs),
                    'mean_active_reconstruction':None if arm=='onehot' else float(np.mean([r['grammar']['raw_mean_active_exact'] for r in rs])),
                    'mean_sampled_reconstruction':None if arm=='onehot' else float(np.mean([r['grammar']['raw_sampled_active_exact'] for r in rs])),
                    'all_mean_targets_met':None if arm=='onehot' else all(r['grammar']['mean_roundtrip_target_met'] for r in rs)})
    score_rows=[]
    for r in results:
        for head_result in r['predictions']:
            for kind in ('exact','interval','missing'):
                scores=[x for x in head_result['scores'] if x['kind']==kind]
                if not scores: continue
                entry={k:r[k] for k in ('case','seed','iteration','arm')}
                entry.update(head=head_result['head'],kind=kind,n=len(scores))
                for metric in ('nll','squared_error','brier'):
                    v=[x[metric] for x in scores if metric in x]
                    entry[metric+'_mean']=float(np.mean(v)) if v else None
                score_rows.append(entry)
    save(output/'scores.json',score_rows)
    save(output/'summary.json',{'counts':dict(counts),'structure_summary':summary,'new_evaluator_calls':0,
         'new_RCAIDE_calls':0,'closed_loop_gain_tested':False,'automatic_pilot_authorization':False})
    evidence={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
    save(output/'audit.json',{'completed':True,'files':evidence,'counts':dict(counts),'algorithm_replay':False})
    audit_output(output)
    print(json.dumps({'profile':profile,'counts':dict(counts),'new_evaluator_calls':0}))


def audit_output(output):
    """Independent read-only verification; no training or evaluation."""
    output=Path(output).resolve()
    a=json.loads((output/'audit.json').read_bytes())
    for relative,expected in a['files'].items():
        target=(output/relative).resolve()
        if output not in target.parents or target.is_symlink() or sha(target)!=expected:
            raise ValueError('Changed or unsafe offline evidence: '+relative)
    plan=json.loads((output/'plan.json').read_bytes())
    rows=[json.loads(s) for s in (output/'rows.jsonl').read_bytes().splitlines() if s.strip()]
    expected={(c,s,t,m) for c in plan['cases'] for s in plan['seeds'] for t in CHECKPOINTS for m in ARMS}
    keys=[(r['case'],r['seed'],r['iteration'],r['arm']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError('Missing or duplicate offline result')
    reference={}
    for r in rows:
        key=(r['case'],r['seed'],r['iteration'])
        target=[(h['head'],[(x['candidate'],x['evidence_id'],x['observation']) for x in h['scores']]) for h in r['predictions']]
        signature=digest({'history':r['history_identity'],'targets':target})
        if reference.setdefault(key,signature)!=signature: raise ValueError('Models did not use matched history/targets')
        if [h['head'] for h in r['predictions']]!=list(HEADS): raise ValueError('Missing forecast head')
        if r['corpus'] is not None and (r['corpus']['draws']!=2000 or len(r['corpus']['draw_indices'])!=2000):
            raise ValueError('Corpus budget changed')
    attempts=[json.loads(s) for s in (output/'fit_attempts.jsonl').read_bytes().splitlines() if s.strip()]
    counts=Counter(x['kind'] for x in attempts)
    if dict(counts)!=a['counts'] or any(counts[k]!=limits(plan['profile'])[k] for k in counts):
        raise ValueError('Offline accounting differs')
    for k,n in counts.items():
        if [x['number'] for x in attempts if x['kind']==k]!=list(range(1,n+1)):
            raise ValueError('Offline attempt sequence differs')
    return {'verified':True,'rows':len(rows),'counts':dict(counts),'new_evaluator_calls':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--profile',choices=['smoke','full'],required=True)
    p.add_argument('--approved-encoder-fits',type=int,required=True)
    p.add_argument('--approved-gp-heads',type=int,required=True)
    a=p.parse_args();run(a.archive,a.output,a.profile,a.approved_encoder_fits,a.approved_gp_heads)


if __name__=='__main__': main()
