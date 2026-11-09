"""Resealed-copy adversarial checks; never change original evidence or fit."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from c1_protocol import digest
from risk_learning.io import read,write,seal
from c1_verify import verify,lines


def dump_lines(path,rows):
    path.write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows),encoding='utf-8')


def check(root):
    root=Path(root);baseline=verify(root);rejected=[]
    modes=('prefix','frozen_training','selection','risk_probability','objective_feedback','ledger','event_order','bill')
    for mode in modes:
        with tempfile.TemporaryDirectory() as td:
            work=Path(td)/'copy';shutil.copytree(root,work)
            journal=next(iter(sorted((work/'units').rglob('records.jsonl'))));rs=lines(journal)
            if mode in ('prefix','frozen_training','selection'):
                entry=next(r for r in rs if '/trials/ei_frozen/proposal_' in r['ref'])
                r=entry['value']
                if mode=='prefix':r['history_sha256']='0'*64
                elif mode=='frozen_training':r['model_training']='own_prefix'
                else:r['decision']['selected_index']=(r['decision']['selected_index']+1)%len(r['decision']['scores'])
            elif mode=='risk_probability':
                entry=next(r for r in rs if '/predictions/' in r['ref'] and r['value']['prediction'] is not None)
                entry['value']['prediction']['probabilities']['joint'][0]=.123456789
            elif mode=='objective_feedback':
                entry=next(r for r in rs if '/observation_' in r['ref'])
                entry['value']['observed']['objective']=dict(kind='exact',value=987654.)
            if mode in modes[:5]:
                dump_lines(journal,rs)
                events=lines(work/'events.jsonl')
                for e in events:
                    if e.get('ref')==entry['ref'] and 'record_sha256' in e:
                        e['record_sha256']=digest(entry['value'])
                dump_lines(work/'events.jsonl',events)
            elif mode=='ledger':
                es=lines(work/'evaluations.jsonl');es.pop();dump_lines(work/'evaluations.jsonl',es)
            elif mode=='event_order':
                es=lines(work/'events.jsonl');es[0],es[-1]=es[-1],es[0];dump_lines(work/'events.jsonl',es)
            else:
                p=work/'report.json';r=read(p);r['bill']['gp_head_started']+=1;p.unlink();write(p,r)
            (work/'manifest.json').unlink();seal(work)
            try:verify(work)
            except ValueError:rejected.append(mode)
            else:raise AssertionError('Resealed tampering accepted: '+mode)
    return dict(status='REJECTED_ALL_C1_TAMPERING',checks=rejected,original_verified=baseline,new_math_calls=0,new_gp_fits=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    print(json.dumps(check(p.parse_args().input),indent=2))
