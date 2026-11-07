"""Reseal tampered COPY and require rejection. Never alters input or fits."""
import argparse
from pathlib import Path
import tempfile
import shutil
from risk_learning.io import read,write,seal
from b1_verify import verify


def check(root):
    root=Path(root); baseline=verify(root)
    first=next(p for p in sorted((root/'states').rglob('*.json')) if read(p)['status']=='OK')
    rejected=[]
    for mode in ('incumbent','ranking','risk_probability','ledger','seal_order','objective_labels'):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/'copy';shutil.copytree(root,work)
            state=work/first.relative_to(root); row=read(state)
            if mode=='incumbent': row['decision']['incumbent']=123456.
            elif mode=='ranking': row['decision']['rankings']['risk_ei'].reverse()
            elif mode=='risk_probability': row['predictions']['risk']['probabilities']['joint'][0]=.123456789
            elif mode=='objective_labels': row['receipt']['objective']['exact_objective_rows']+=1
            if mode in ('ledger','seal_order'):
                p=work/('evaluations.jsonl' if mode=='ledger' else 'timeline.jsonl')
                lines=p.read_text(encoding='utf-8').splitlines()
                lines[0],lines[-1]=lines[-1],lines[0]
                p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
            else:
                state.unlink();write(state,row)
            (work/'manifest.json').unlink();seal(work)
            try: verify(work)
            except ValueError: rejected.append(mode)
            else: raise AssertionError('Tampering accepted: '+mode)
    return dict(status='REJECTED_ALL_RESEALED_TAMPERING',checks=rejected,
                original_verified=baseline,new_math_calls=0,new_gp_fits=0)


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path)
    print(json.dumps(check(p.parse_args().input),indent=2))
