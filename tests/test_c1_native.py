import copy
import fnmatch
import inspect
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from c1_native import fit
from c1_protocol import bill,digest,information,ei
from c1_verify import check_fit,check_prediction
from risk_learning.core import observed
from ra_lsbo.experiments.simplified_aircraft import dualview_optimizer as native


def synthetic_rows(n=12):
    out=[]
    for i in range(n):
        x=.05+.9*i/max(1,n-1)
        c=dict(branch=0,x=x,y=.3,trim=.5)
        target=dict(region=x-.5,resource=-.2,objective=(x-.3)**2+.5,feasible=x<=.5)
        out.append(observed(c,target))
    return out


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=synthetic_rows();cls.queries=[dict(branch=0,x=x,y=.3,trim=.5) for x in (.2,.7)]
        cls.bill=bill();cls.model,cls.receipt=fit(copy.deepcopy(cls.rows),42,cls.bill)
        before=dict(cls.bill);cls.pred,cls.error=cls.model.predict(cls.queries,cls.bill)
        cls.pred_delta={k:cls.bill[k]-before[k] for k in before}
        cls.fit_delta=dict(before)
    def test_three_heads(self):
        self.assertEqual(self.fit_delta['gp_head_completed'],3);self.assertEqual(self.fit_delta['gp_kernel_started'],24)
    def test_fit_receipt(self):check_fit(dict(receipt=self.receipt,bill_delta=self.fit_delta),self.rows,42,False)
    def test_prediction_receipt(self):
        check_prediction(dict(model_ref='m',query_sha256=digest(self.queries),prediction=self.pred,error=None,bill_delta=self.pred_delta),'m',self.receipt,self.queries)
    def test_matches_original_shared_model(self):
        from b1_native import fit_shared
        b=bill();p,r=fit_shared(self.rows,self.queries,42,b)
        np.testing.assert_allclose(p['risk']['probabilities']['joint'],self.pred['probabilities']['joint'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(p['objective']['mean'],self.pred['objective']['mean'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(p['objective']['variance'],self.pred['objective']['variance'],rtol=1e-10,atol=1e-12)
    def test_ei_matches_original(self):
        from b1_core import expected_improvement
        for m,v,b in [(0.,1.,0.),(1.,0.,2.),(10.,.2,1.),(-.2,.5,.1)]:
            self.assertEqual(ei(m,v,b),expected_improvement(m,v,b))
    def test_frozen_predict_does_not_refit(self):
        b=bill()
        with patch.object(native,'StableGroupedHead',side_effect=AssertionError('No refit')):
            p,e=self.model.predict(self.queries,b)
        self.assertIsNone(e);self.assertEqual(b['gp_head_started'],0)
        self.assertEqual(p,self.pred)
    def test_risk_only_two_heads(self):
        b=bill();m,r=fit(self.rows,42,b,risk_only=True)
        self.assertEqual(set(r['heads']),{'region','resource'});check_fit(dict(receipt=r,bill_delta=b),self.rows,42,True)
    def test_all_missing_objectives(self):
        rows=[r for r in self.rows if r['objective']['kind']=='missing'];b=bill();m,r=fit(rows,42,b)
        self.assertEqual(set(r['heads']),{'region','resource'});check_fit(dict(receipt=r,bill_delta=b),rows,42,False)
    def test_single_objective(self):
        rows=[self.rows[0]]+self.rows[6:];b=bill();m,r=fit(rows,42,b)
        check_fit(dict(receipt=r,bill_delta=b),rows,42,False)
    def test_all_kernel_failure_kept(self):
        b=bill()
        with patch.object(native,'EvidenceGP',side_effect=np.linalg.LinAlgError('injected')):
            m,r=fit(self.rows,42,b)
        self.assertIsNone(m);self.assertEqual(b['gp_kernel_errors'],8)
        check_fit(dict(receipt=r,bill_delta=b),self.rows,42,False)
    def test_prediction_failure_is_separate(self):
        b=bill()
        error=native.ModelNumericalFailure('region',[dict(phase='prediction',status='test')])
        with patch.object(self.model.heads['region'],'predict',side_effect=error):p,e=self.model.predict(self.queries,b)
        self.assertIsNone(p);self.assertEqual((b['prediction_errors'],b['gp_head_errors']),(1,0))
    def test_no_truth_or_vae_fitting(self):
        b=bill();rows=copy.deepcopy(self.rows)
        with patch('risk_learning.core.truth',side_effect=AssertionError('hidden truth')):
            m,r=fit(rows,42,b)
        self.assertEqual(rows,self.rows);self.assertEqual(b['vae_fits'],0)
        self.assertEqual(list(inspect.signature(fit).parameters),['rows','seed','bill','risk_only'])
    def test_prefix_identity_changes_with_new_data(self):
        b=bill();rows=self.rows+[observed(dict(branch=0,x=.25,y=.31,trim=.5),dict(region=-.25,resource=-.2,objective=.9,feasible=True))]
        m,r=fit(rows,42,b)
        self.assertNotEqual(r['training_sha256'],self.receipt['training_sha256'])
        self.assertEqual(r['training_information']['rows'],13)
    def test_unexpected_failure_preserves_attempt(self):
        from c1_study import run
        def evaluator(task,c):return dict(region=-.1,resource=-.2,objective=1.,feasible=True)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'e'
            with self.assertRaisesRegex(RuntimeError,'unexpected'):
                run(out,'ci',dict(code_sha='a'*40),fitter=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('unexpected')),evaluator=evaluator)
            f=json.loads((out/'failure.json').read_text())
            self.assertEqual(f['bill']['math_settled'],48);self.assertTrue((out/'manifest.json').exists())
    def test_old_workflow_filters_do_not_match(self):
        root=Path(__file__).resolve().parents[1]
        new=['c1_protocol.py','c1_native.py','c1_study.py','c1_verify.py','check_c1_evidence.py','run_c1_online.py',
             'tests/test_c1_core.py','tests/test_c1_native.py','docs/PHASE_C1_ONLINE_LEARNING.txt','.github/workflows/c1-online-learning.yml']
        def pr_patterns(content):
            # The historical comparison workflow is branch-restricted PUSH only.
            # Read the actual PR event, not an unrelated event's paths block.
            event=re.search(r'(?ms)^  pull_request:[ \t]*\n(.*?)(?=^  \S|^\S|\Z)',content)
            if not event:
                return []
            paths=re.search(r'(?m)^    paths:[ \t]*\n((?:      .*\n)+)',event.group(1))
            return ([x.strip().strip("'\"") for x in re.findall(r'^      - (.+)$',paths.group(1),re.M)]
                    if paths else [])
        push="on:\n  push:\n    branches: [old-fix]\n    paths:\n      - 'tests/**'\n  workflow_dispatch:\n"
        pr="on:\n  pull_request:\n    paths:\n      - 'b1_*.py'\n      - 'tests/test_b1_*.py'\n  push:\n    paths:\n      - 'tests/**'\npermissions:\n  contents: read\n"
        self.assertEqual(pr_patterns(push),[])
        self.assertEqual(pr_patterns(pr),['b1_*.py','tests/test_b1_*.py'])
        self.assertTrue(fnmatch.fnmatchcase('tests/test_b1_audit.py',pr_patterns(pr)[1]))
        for f in (root/'.github/workflows').glob('*.yml'):
            if f.name=='c1-online-learning.yml':continue
            content=f.read_text(encoding='utf-8')
            for name in new:
                self.assertFalse(any(fnmatch.fnmatchcase(name,p) for p in pr_patterns(content)),
                                 'Old pull-request CI matches '+f.name+': '+name)

if __name__=='__main__':unittest.main()
