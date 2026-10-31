"""Six fixed shared-history prediction controls. Zero evaluator/optimizer calls."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ra_lsbo import grammar_pretraining as gp
from ra_lsbo.budget import digest
from ra_lsbo.experiments.core.problems import AircraftCase, MathematicalCase
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft, HierarchicalBranin
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.partial_refinement.posterior import GroupedMixedHead
from ra_lsbo.risk_core.observations import OutputObservation

ARCHIVE_SHA = '35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd'
WINDOWS_SHA = '9b514cdcfa92dcab33345429e2a73f521a022a3f2fae041aec98ef83d578ea64'
PREFIX = 'aircraft_nominal_20261030_01/pilot/'
CASES = ('aircraft_ordinary', 'aircraft_sparse')
SEEDS = tuple(range(9911, 9919))
HEADS = ('objective', 'soc')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def blocked(*args, **kwargs):
    raise AssertionError('Physical evaluation forbidden in representation study')


def windows_facts(archive):
    if sha(archive) != WINDOWS_SHA: raise ValueError('Wrong Windows diagnostic archive')
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith('/diagnostic.json') or n == 'diagnostic.json']
        if len(names) != 1: raise ValueError('Ambiguous diagnostic.json')
        rows = json.loads(z.read(names[0]))['rows']
        eligible = [r for r in rows if r['representation']=='latent' and r['epochs']==30 and r['saved_feature_identity']]
        if len(eligible)!=29 or any(r['feature_identity']!=r['saved_feature_identity'] for r in eligible):
            raise ValueError('Original same-environment identities do not reconstruct')
        return {'archive_sha256': WINDOWS_SHA, 'eligible_identity_matches':len(eligible),
                'reconstruction': {str(e): {'n':len([r for r in rows if r['epochs']==e]),
                    'mean':statistics.mean(r['active_grammar_exact_fraction'] for r in rows if r['epochs']==e)}
                    for e in (30,120)}, 'cause_of_linux_identity_difference':'undetermined'}


def source_checks(z, plan):
    checked = 0
    for old, identity in plan['sources'].items():
        path = old.replace('\\','/')
        for marker in ('/ra_lsbo/', '/runs/compact_mda_'):
            if marker in path and path.endswith('.py'):
                relative = path[path.index(marker)+1:]
                original = z.read(PREFIX+'source_snapshot/'+identity+'.source')
                if hashlib.sha256(original).hexdigest()!=identity:
                    raise ValueError('Changed historical source')
                if original.replace(b'\r\n',b'\n')!=(ROOT/relative).read_bytes().replace(b'\r\n',b'\n'):
                    raise ValueError('Frozen implementation changed: '+relative)
                checked += 1
                break
    return checked


def trusted_output(record, name):
    packet = record.observation
    if packet.status in {'unknown','infrastructure_error'} or record.diagnostics.get('numerical_failure'):
        return OutputObservation('missing',packet.evidence_id+':'+name, reason='untrusted')
    return packet.objective if name=='objective' else packet.constraint(name)


def validation_rows(records, training):
    seen = {r.observation.design_id for r in training}
    return [r for r in records if r.observation.design_id not in seen]


def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['case'],row['iteration'],row['variant'])].append(row)
    result=[]
    for (case,iteration,variant), group in sorted(groups.items()):
        item={'case':case,'iteration':iteration,'variant':variant,'n_prefixes':len(group),'heads':{}}
        for head in HEADS:
            valid=[r['heads'][head]['scores'] for r in group if r['heads'][head]['status']=='ok']
            item['heads'][head]={'successful_fits':len(valid),'failed_fits':len(group)-len(valid),
                'median_over_prefixes':{metric:statistics.median(vals) if vals else None
                    for metric in ('exact_rmse','exact_nll','interval_nlpm','sign_brier')
                    for vals in [[s[metric] for s in valid if s[metric] is not None]]}}
        result.append(item)
    return result


def run(archive, windows_archive, output):
    if output.exists(): raise FileExistsError('Use a new diagnostic folder; no resume or overwrite')
    if sha(archive)!=ARCHIVE_SHA: raise ValueError('Wrong pilot archive')
    original_windows=windows_facts(windows_archive)
    for cls in (AircraftCase,MathematicalCase,SimplifiedElectricAircraft,HierarchicalBranin): cls.evaluate=blocked
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    sources={str(p.relative_to(ROOT)):sha(p) for p in
        list((ROOT/'ra_lsbo').rglob('*.py'))+list(BASE.rglob('*.py'))+
        [Path(__file__),ROOT/'docs/GRAMMAR_PRETRAINING_PROTOCOL_20261031.md']}
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
    plan={'source_commit':commit.stdout.strip(),'source_hashes':sources,'input_sha256':ARCHIVE_SHA,
          'windows_diagnostic':original_windows,'cases':CASES,'seeds':SEEDS,'prefixes':[0,29],
          'variants':gp.VARIANTS,'heads':HEADS,'pretrain_updates':gp.PRETRAIN_UPDATES,
          'prefix_updates':gp.PREFIX_UPDATES,'maximum_grouped_head_fits':384,
          'maximum_kernel_candidate_fits':3072,'maximum_encoder_training_calls':104,
          'new_model_calls':0,'new_rcaide_mda_calls':0,'optimizer_calls':0,
          'runtime':{'python':sys.version,'platform':platform.platform(),
                     'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','torch','gpytorch','botorch')}},
          'claim':'post-hoc development prediction check; not BO replay or superiority confirmation'}
    save(output/'plan.json',plan)
    rows=[];cache={};counts={'pretrain_fits':0,'prefix_encoder_fits':0,'onehot_conversions':0,'grouped_head_fits':0,'failed_heads':0}
    with zipfile.ZipFile(archive) as z:
        load=lambda name:json.loads(z.read(PREFIX+name))
        lines=lambda name:[json.loads(x) for x in z.read(PREFIX+name).splitlines() if x.strip()]
        old_plan=load('plan.json')
        if tuple(old_plan['cases'])!=CASES or tuple(old_plan['profile']['seeds'])!=SEEDS:
            raise ValueError('Unexpected study cohort')
        checked=source_checks(z,old_plan)
        for case in CASES:
            problem=AircraftCase(BASE,case.split('_')[1])
            for seed in SEEDS:
                trial=f'optimization/{case}/{seed}_ev_latent/'
                initial=load(trial+'manifest.json')['initial']; online=lines(trial+'observations.jsonl')
                props=lines(trial+'proposals.jsonl')
                validation=[DesignRecord.from_dict(r) for r in lines(f'optimization/{case}/{seed}_random/observations.jsonl')]
                if len(validation)!=30: raise ValueError('Incomplete random-arm validation records')
                for iteration in (0,29):
                    records=[DesignRecord.from_dict(r) for r in initial+online[:iteration]]
                    if digest([r.to_dict() for r in records])!=props[iteration]['prefix_digest']:
                        raise ValueError('Historical prefix mismatch')
                    fit_problem=SimpleNamespace(space=problem.space,contract_id=records[0].observation.contract_id)
                    legacy=PrefixFeatures(fit_problem,BASE,records,seed,latent=True,epochs=30)
                    joint=PrefixFeatures(fit_problem,BASE,records,seed,latent=True,epochs=630)
                    if seed not in cache:
                        checkpoint=gp.pretrain_grammar(legacy,seed);cache[seed]=checkpoint;counts['pretrain_fits']+=1
                        save(output/f'grammar_{seed}.json',{'state_digest':checkpoint._grammar_digest,
                            'schema':checkpoint._grammar_schema,'settings':asdict(checkpoint.settings),
                            'metrics':gp.reconstruction(checkpoint),'trace':checkpoint.training_trace})
                        weights={n:t.detach().cpu().numpy() for n,t in checkpoint.model.state_dict().items()}
                        weights['generator_state']=checkpoint.generator.get_state().numpy()
                        np.savez_compressed(output/f'grammar_{seed}.npz',**weights)
                    pre=gp.from_checkpoint(legacy,records,cache[seed],seed)
                    one=PrefixFeatures(fit_problem,BASE,records,seed,latent=False,epochs=30)
                    counts['prefix_encoder_fits']+=3;counts['onehot_conversions']+=1
                    features={'legacy':legacy,'legacy_schema_scale':gp.scaled_copy(legacy,records),
                              'joint_630':joint,'pretrained':pre,
                              'pretrained_schema_scale':gp.scaled_copy(pre,records),'onehot':one}
                    test=validation_rows(validation,records)
                    if not test: raise ValueError('Empty evaluation-only cohort')
                    for variant in gp.VARIANTS:
                        f=features[variant];train_x=f.transform([r.candidate for r in records]);test_x=f.transform([r.candidate for r in test])
                        row={'case':case,'seed':seed,'iteration':iteration,'variant':variant,
                             'prefix_digest':props[iteration]['prefix_digest'],'training_count':len(records),
                             'validation_design_ids':[r.observation.design_id for r in test],
                             'validation_count':len(test),'excluded_training_duplicates':30-len(test),
                             'feature_geometry':gp.geometry(f),'heads':{}}
                        if variant=='legacy':
                            saved=props[iteration]['metadata'].get('feature_identity')
                            row.update(saved_feature_identity=saved,original_identity_matches=f.identity==saved if saved else None)
                        for name in HEADS:
                            obs=[trusted_output(r,name) for r in records]
                            target=[trusted_output(r,name) for r in test]
                            started=time.monotonic();counts['grouped_head_fits']+=1
                            try:
                                head=GroupedMixedHead(train_x,obs,continuous_count=len(problem.space.continuous),
                                    seed=seed+(100 if name=='objective' else 5), objective=name=='objective',
                                    reference_scale=problem.objective_reference_scale if name=='objective' else 1.)
                                prediction=head.predict(test_x)
                                scores=gp.score_observations(target,prediction['mean'],prediction['variance'],
                                    constraint=name=='soc',exact_noise=head.scale*head.head.model.settings.exact_noise)
                                row['heads'][name]={'status':'ok','seconds':time.monotonic()-started,
                                    'scores':scores,'diagnostics':head.diagnostics(),
                                    'mean':prediction['mean'].tolist(),'variance':prediction['variance'].tolist(),
                                    'observations':[asdict(v) for v in target]}
                            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                                counts['failed_heads']+=1
                                row['heads'][name]={'status':'failed','error':type(error).__name__+': '+str(error)}
                        with (output/'rows.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                        rows.append(row)
                    print({'case':case,'seed':seed,'iteration':iteration,'head_fits':counts['grouped_head_fits']},flush=True)
    if sha(archive)!=ARCHIVE_SHA or sha(windows_archive)!=WINDOWS_SHA:
        raise ValueError('Input changed during analysis')
    if any(sha(ROOT/p)!=h for p,h in sources.items()): raise ValueError('Source changed during analysis')
    summary={'counts':counts,'source_bindings_checked':checked,'windows_review':original_windows,
             'new_model_calls':0,'new_rcaide_mda_calls':0,'optimizer_replay':False,'rows':len(rows),
             'groups':aggregate(rows),'automatic_pilot_authorization':False}
    save(output/'summary.json',summary)
    sealed={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    save(output/'audit.json',{'evidence_hashes':sealed,'complete':len(rows)==192 and counts['grouped_head_fits']==384,
         'all_heads_fitted':counts['failed_heads']==0,'algorithm_replay':False,'physical_evaluation_calls':0})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pilot_archive',type=Path);p.add_argument('windows_archive',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.pilot_archive,a.windows_archive,a.output)


if __name__=='__main__':main()
