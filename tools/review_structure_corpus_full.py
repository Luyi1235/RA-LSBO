"""Read-only review of the fixed Windows 2,000-corpus study. No model imports."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from statistics import mean, median
import zipfile

ARCHIVE_SHA = 'd7953cdcc91891eff8fa0ff7c1fc1f7103ccdb7d24a39cbd94651b4ef2a2628f'
MANIFEST_SHA = '0186cc780c5e184ee703e8378625c09589070ef594da872a152e566dcc597980'
PREFIX = 'RA-LSBO-structure-corpus-output-20261031_01/'
ARMS = ('old30','old480','corpus_reset','corpus_persistent','onehot')
CASES = ('aircraft_ordinary','aircraft_sparse')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda:s.read(1048576),b''): h.update(b)
    return h.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe archive member')
    return name


def load_verified(archive, manifest=None):
    if sha(archive) != ARCHIVE_SHA: raise ValueError('Unexpected input archive')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)): raise ValueError('Duplicate archive member')
        for name in names: safe_name(name)
        if manifest is not None:
            if sha(manifest) != MANIFEST_SHA: raise ValueError('Unexpected manifest')
            m = json.loads(Path(manifest).read_bytes())
            if set(names) != {PREFIX+f['path'] for f in m['files']}:
                raise ValueError('Manifest member set differs')
            for f in m['files']:
                b=z.read(PREFIX+f['path'])
                if len(b)!=f['bytes'] or hashlib.sha256(b).hexdigest()!=f['sha256']:
                    raise ValueError('Manifest mismatch: '+f['path'])
        a=json.loads(z.read(PREFIX+'full/audit.json'))
        for name,h in a['files'].items():
            safe_name(name)
            if hashlib.sha256(z.read(PREFIX+'full/'+name)).hexdigest()!=h:
                raise ValueError('Sealed evidence mismatch: '+name)
        rows=[json.loads(x) for x in z.read(PREFIX+'full/rows.jsonl').splitlines() if x.strip()]
        expected={(c,s,t,m) for c in CASES for s in range(9911,9919) for t in (0,29) for m in ARMS}
        keys=[(r['case'],r['seed'],r['iteration'],r['arm']) for r in rows]
        if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError('Row coverage differs')
        for r in rows:
            name=f"{r['case']}_{r['seed']}_p{r['iteration']}/{r['arm']}.json"
            if r!=json.loads(z.read(PREFIX+'full/'+name)): raise ValueError('Per-arm row differs')
        counts=Counter(json.loads(x)['kind'] for x in z.read(PREFIX+'full/fit_attempts.jsonl').splitlines())
        if dict(counts)!=a['counts']: raise ValueError('Fit ledger count differs')
        return rows, {'archive_files':len(names),'sealed_files':len(a['files']),'fit_counts':dict(counts),
                      'archive_sha256':ARCHIVE_SHA,'manifest_verified':manifest is not None}


def summarize(rows):
    groups=defaultdict(list); failures=[]; geometry=[]; ref={}; structure=[]
    for r in rows:
        signature=json.dumps([r['history_identity'],[(p['head'],[(x['candidate'],x['evidence_id'],x['observation'])
                    for x in p['scores']]) for p in r['predictions']]],sort_keys=True)
        key=(r['case'],r['seed'],r['iteration'])
        if ref.setdefault(key,signature)!=signature: raise ValueError('Unmatched histories or targets')
        if r['grammar'] is not None:
            structure.append({k:r[k] for k in ('case','seed','iteration','arm')}|{
                'reconstruction':r['grammar']['raw_mean_active_exact'],
                'sampled_reconstruction':r['grammar']['raw_sampled_active_exact'],
                'target_met':r['grammar']['mean_roundtrip_target_met']})
        for p in r['predictions']:
            if p['diagnostics']['selected']['status']!='ok': raise ValueError('Invalid selected head')
            for c in p['diagnostics']['candidates']:
                if c['status']!='ok': failures.append({k:r[k] for k in ('case','seed','iteration','arm')}|{'head':p['head'],**c})
            for x in p['scores']:
                if not math.isfinite(x['nll']): raise ValueError('Nonfinite observation score')
                for t in (r['iteration'],'pooled'):
                    groups[(r['case'],t,r['arm'],p['head'],x['kind'])].append(x)
    scores=[]
    for (c,t,m,h,k),xs in groups.items():
        entry={'case':c,'prefix':t,'arm':m,'head':h,'kind':k,'n':len(xs),
               'nll_mean':mean(x['nll'] for x in xs),'nll_median':median(x['nll'] for x in xs)}
        if k=='exact': entry['rmse']=math.sqrt(mean(x['squared_error'] for x in xs))
        if all('brier' in x for x in xs): entry['brier_mean']=mean(x['brier'] for x in xs)
        scores.append(entry)
    return {'structure':structure,'scores':scores,'failed_candidates':failures,
            'eligible_identity_count':sum(r['old_identity_match'] is not None for r in rows),
            'identity_matches':sum(r['old_identity_match'] is True for r in rows),
            'selected_heads':sum(len(r['predictions']) for r in rows),
            'kernel_candidates':sum(len(p['diagnostics']['candidates']) for r in rows for p in r['predictions']),
            'new_evaluator_calls':0,'new_encoder_fits':0,'new_GP_fits':0,
            'limitations':['post-hoc development analysis','pooled design counts are not independent seeds',
                          'density and probability-mass scores remain separate','no optimizer replay']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Use a new output file')
    rows,verification=load_verified(a.archive,a.manifest)
    result={'verification':verification,**summarize(rows)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(verification));print({'selected_heads':result['selected_heads'],'failed_candidates':len(result['failed_candidates'])})

if __name__=='__main__': main()
