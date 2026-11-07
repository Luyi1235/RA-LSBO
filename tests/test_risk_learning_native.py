"""Native integration, not mocked surrogate tests. Requires existing model dependencies."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from risk_learning.core import TASKS, candidates, features, native_candidate, observed, truth
from risk_learning.native import predict
from risk_learning.study import new_bill, run, verify
from risk_learning.io import read, seal


class NativeTests(unittest.TestCase):
    def test_legacy_evaluator_and_feature_parity(self):
        from ra_lsbo.experiments.core.problems import MathematicalCase
        from ra_lsbo.experiments.core.protocol import BASE
        from ra_lsbo.partial_bo.models import PrefixFeatures
        for task,family in zip(TASKS[:2],('branin','rosenbrock')):
            problem=MathematicalCase(BASE,family,'sparse')
            cs=candidates(751991,32)
            native=[problem.evaluate(native_candidate(c)) for c in cs]
            feature=PrefixFeatures(problem,BASE,native[:12],751991,latent=False)
            np.testing.assert_allclose(feature.transform([r.candidate for r in native]),[features(c) for c in cs],rtol=0,atol=0)
            for c,r in zip(cs,native):
                target=truth(task,c); packet=observed(c,target)
                self.assertEqual(r.usable_feasible,target['feasible'])
                for head in ('region','resource','objective'):
                    value=r.observation.objective if head=='objective' else r.observation.constraint(head)
                    self.assertEqual(value.kind,packet[head]['kind'])
                    if value.kind=='exact': self.assertAlmostEqual(value.value,packet[head]['value'],places=9)
                    elif value.kind=='interval':
                        self.assertEqual(value.lower,0.);self.assertFalse(value.lower_inclusive)
    def test_censored_fit_does_not_call_truth(self):
        cs=candidates(751993,12);rows=[observed(c,truth(TASKS[2],c)) for c in cs]
        bill=new_bill();before=deepcopy(rows)
        with patch('risk_learning.core.truth',side_effect=AssertionError('truth leaked')):
            guess,receipt=predict(rows,candidates(751994,16),'native_censored',0,bill)
        self.assertIsNotNone(guess);self.assertEqual(rows,before)
        self.assertEqual(bill['gp_kernel_started'],16)
        self.assertEqual(bill['gp_head_completed'],2)
        self.assertFalse(receipt['objective_labels_used'])
    def test_binary_and_privileged_real_fits(self):
        cs=candidates(751995,12);targets=[truth(TASKS[2],c) for c in cs]
        rows=[observed(c,t) for c,t in zip(cs,targets)]
        for method in ('native_sign','privileged_exact'):
            kwargs={'privileged':targets} if method=='privileged_exact' else {}
            p,r=predict(rows,candidates(751996,8),method,0,new_bill(),**kwargs)
            self.assertIsNotNone(p)
            self.assertTrue(all(0<=v<=1 for a in p['probabilities'].values() for v in a))
            self.assertEqual(r['privileged_training'],method=='privileged_exact')
    def test_all_negative_cold_feedback_is_retained(self):
        cs=[dict(branch=0,x=.02+.01*i,y=.02,trim=.5) for i in range(12)]
        rows=[observed(c,truth(TASKS[0],c)) for c in cs]
        self.assertTrue(all(r['region']['kind']=='interval' for r in rows))
        guess,receipt=predict(rows,candidates(751997,8),'native_censored',0,new_bill())
        self.assertIsNotNone(guess)
        self.assertEqual(receipt['training_rows'],12)

    def test_maximum_training_size_real_censored(self):
        cs=candidates(751998,192)
        rows=[observed(c,truth(TASKS[3],c)) for c in cs]
        bill=new_bill()
        p,r=predict(rows,candidates(751999,8),'native_censored',0,bill)
        self.assertIsNotNone(p)
        self.assertEqual(r['training_rows'],192)
        self.assertEqual(bill['gp_head_completed'],2)
        self.assertEqual(bill['gp_kernel_started'],16)


if __name__=='__main__': unittest.main()
