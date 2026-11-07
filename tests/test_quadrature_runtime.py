"""Require the actual frozen RA-LSBO scientific stack; never skip missing imports."""
import copy
import unittest
import numpy as np
from test_information_value_stage3_runtime import make_factory
from ra_lsbo.experiments.information_value_stage3.planner import one_step, shortlist
from pr70_future_pool import choose_next
from quadrature_audit_numeric import compute_rule, points, verify_rule
from quadrature_audit_protocol import close


class SamePoolFactory:
    """Real EP conditioning; only the query view removes the evaluated root."""
    def __init__(self): self.native,self.forecast=make_factory()
    def support(self,i):
        return {'model_candidates':[c for j,c in enumerate(self.native.candidates) if j!=i]}
    def condition(self,i,outcome,**kwargs):
        pred=self.native.condition(i,outcome,**kwargs)
        def exclude(h):
            if h is None:return None
            return {**h,'mean':np.delete(h['mean'],i),'variance':np.delete(h['variance'],i)}
        return {'constraints':{n:exclude(h) for n,h in pred['constraints'].items()},
                'objective':exclude(pred.get('objective'))}
    def diagnostics(self):return self.native.diagnostics()


class RuntimeTests(unittest.TestCase):
    def test_four_node_equivalence_with_frozen_pr70(self):
        a=SamePoolFactory();b=SamePoolFactory()
        old_index,old=choose_next(a.forecast,5.,2,a,seed=123,iteration=0)
        firsts=shortlist(b.forecast,one_step(b.forecast,5.))
        new=compute_rule(b.forecast,5.,b,seed=123,iteration=0,n=4,replicate=0,first_indices=firsts)
        self.assertEqual(new['status'],'OK');self.assertIsNone(old['fallback'])
        self.assertEqual(old_index,new['selected_index'])
        close(new['scores'],{str(i):old['scores'][i] for i in firsts})
        close(new['base_scores'],old['myopic_scores'])
    def test_real_16_node_fits_and_reconciliation(self):
        f=SamePoolFactory();firsts=shortlist(f.forecast,one_step(f.forecast,5.))
        r=compute_rule(f.forecast,5.,f,seed=123,iteration=0,n=16,replicate=0,first_indices=firsts)
        self.assertEqual(r['status'],'OK');self.assertGreater(f.native.completed,0)
        self.assertEqual(f.native.completed,f.native.started)
        verify_rule(r,5.,firsts,{str(i):3 for i in firsts})
    def test_native_sobol_first_four(self):
        from ra_lsbo.experiments.information_value_stage3.planner import nodes
        np.testing.assert_array_equal(nodes(998901,3),points(998901,3,64,0)[:4])
    def test_real_predictions_are_not_mutated(self):
        f=SamePoolFactory();before=copy.deepcopy(f.forecast)
        firsts=shortlist(f.forecast,one_step(f.forecast,5.))
        compute_rule(f.forecast,5.,f,seed=123,iteration=0,n=4,replicate=0,first_indices=firsts)
        for name in before['constraints']:
            for field in ('mean','variance'):
                np.testing.assert_array_equal(before['constraints'][name][field],f.forecast['constraints'][name][field])
    def test_true_evaluator_is_blocked(self):
        from run_pr70_quadrature_diagnostic import model_only_guard
        from ra_lsbo.experiments.core.problems import MathematicalCase
        bill={'phase':'test','gp_started':0,'gp_completed':0,'gp_errors':0,'by_phase':{},
              'blocked_evaluator_attempts':0,'blocked_vae_training_attempts':0}
        with model_only_guard(bill,1000):
            with self.assertRaises(RuntimeError): MathematicalCase.evaluate(None,{})
        self.assertEqual(bill['blocked_evaluator_attempts'],1)
    def test_vae_training_is_blocked(self):
        from run_pr70_quadrature_diagnostic import model_only_guard
        from ra_lsbo.structure_corpus import CorpusRepresentation
        bill={'phase':'test','gp_started':0,'gp_completed':0,'gp_errors':0,'by_phase':{},
              'blocked_evaluator_attempts':0,'blocked_vae_training_attempts':0}
        with model_only_guard(bill,1000):
            with self.assertRaises(RuntimeError): CorpusRepresentation.fit(None,[])
        self.assertEqual(bill['blocked_vae_training_attempts'],1)

if __name__=='__main__':unittest.main()
