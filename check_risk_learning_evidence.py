"""Read-only positive verification and tamper tests on a copied CI artifact."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

from risk_learning.io import read, seal
from risk_learning.study import verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    args=p.parse_args()
    with patch('risk_learning.study.truth',side_effect=AssertionError('Verifier evaluated truth')):
        result=verify(args.input)
    cases=('metric','training_binding','ledger','prediction','extra_file')
    for case in cases:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'copy';shutil.copytree(args.input,root)
            path=next((root/'fits').rglob('native_censored.json'))
            row=read(path)
            if case=='metric':
                row['metrics']['all']['joint']['brier']+=.1
            elif case=='training_binding':
                row['receipt']['training_input_sha256']='0'*64
            elif case=='prediction':
                # Resealing AND recalculating metrics must not hide changed model probabilities.
                row['predictions']['probabilities']['region'][0]=.123456
                from risk_learning.core import score
                test=read(root/'datasets'/row['task']/('r'+str(row['repeat']))/'test.json')
                row['metrics']=score(test,row['predictions'])
            elif case=='ledger':
                events=(root/'evaluations.jsonl').read_text(encoding='utf-8').splitlines()
                (root/'evaluations.jsonl').write_text('\n'.join(events[:-1])+'\n',encoding='utf-8')
            else:
                (root/'extra.txt').write_text('unexpected',encoding='utf-8')
            path.write_text(json.dumps(row),encoding='utf-8')
            if case!='extra_file':
                (root/'manifest.json').unlink();seal(root)
            try:
                verify(root)
            except ValueError:
                pass
            else:
                raise AssertionError('Tamper not detected: '+case)
    print(json.dumps(dict(positive_verification=result,resealed_tamper_cases_passed=list(cases),
                         new_evaluations=0,new_fits=0),indent=2))


if __name__=='__main__': main()
