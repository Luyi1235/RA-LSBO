"""Real EP tests use synthetic feedback; no research evaluator is called."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from b1_core import digest,decisions,incumbent_from_observed,objective_payload,PARENT
from b1_native import fit_shared
from b1_verify import check_prediction
from risk_learning.core import candidates,observed
from risk_learning.study import new_bill


def synthetic_rows(n=12,exact=4):
    out=[]
    for i,c in enumerate(candidates(781900,n)):
        # A synthetic observation fixture, NOT a research-task evaluation.
        target=dict(region=-.1*(i+1) if i<exact else 1.,resource=-.2,
                    objective=2.+i*.2,feasible=i<exact)
        out.append(observed(c,target))
    return out


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=synthetic_rows();cls.q=candidates(781901,16);cls.bill=new_bill()
        cls.before=deepcopy(cls.rows)
        with patch('risk_learning.core.truth',side_effect=AssertionError('Truth forbidden in fit')), \
             patch('risk_learning.study.truth',side_effect=AssertionError('Truth forbidden in fit')):
            cls.pred,cls.rec=fit_shared(cls.rows,cls.q,781900,cls.bill)
        if cls.pred is None: raise AssertionError('Synthetic native EP fit failed')
    def test_three_heads_and_kernel_bill(self):
        self.assertEqual((self.bill['gp_head_started'],self.bill['gp_head_completed'],self.bill['gp_kernel_started']),(3,3,24))
        self.assertEqual(self.bill['math_started'],0)
    def test_original_rows_not_mutated(self): self.assertEqual(self.rows,self.before)
    def test_only_observed_objective_used(self):
        self.assertEqual(self.rec['objective']['exact_objective_rows'],4)
        self.assertEqual(self.rec['objective']['training_input_sha256'],digest(objective_payload(self.rows)))
    def test_model_free_prediction_contract(self):
        row=dict(receipt=self.rec,predictions=self.pred,status='OK',bill_delta=self.bill,
                 objective_scale=self.rec['objective']['head']['scale'])
        check_prediction(row,self.rows,self.q)
    def test_shared_objective_not_refitted_by_policy(self):
        before=dict(self.bill)
        d=decisions(self.pred['risk']['candidate_ids'],self.pred['risk']['probabilities']['joint'],
                    self.pred['objective'],incumbent_from_observed(self.rows),'fixture')
        self.assertEqual(before,self.bill)
        self.assertEqual(d['scores']['risk_ei'],[e*p for e,p in zip(d['scores']['objective_ei'],d['scores']['risk_only'])])
    def test_no_truth_or_objective_imputation_in_receipts(self):
        self.assertFalse(self.rec['risk']['objective_labels_used']);self.assertFalse(self.rec['risk']['privileged_training'])
        self.assertTrue(self.rec['no_objective_truth_imputation'])
    def test_single_sided_cold_two_heads(self):
        rows=synthetic_rows(exact=0);bill=new_bill();pred,rec=fit_shared(rows,self.q,781900,bill)
        self.assertIsNotNone(pred);self.assertIsNone(pred['objective']);self.assertEqual(bill['gp_head_completed'],2)
        check_prediction(dict(receipt=rec,predictions=pred,status='OK',bill_delta=bill,objective_scale=1.),rows,self.q)
    def test_single_exact_objective_scale(self):
        rows=synthetic_rows(exact=1);bill=new_bill();pred,rec=fit_shared(rows,self.q,781900,bill)
        self.assertIsNotNone(pred);self.assertEqual(rec['objective']['head']['scale'],2.)
    def test_legacy_b0_entry_unchanged(self):
        from decision_value_core import fixed_plan,rank_candidates
        self.assertEqual(fixed_plan()['source_model_rows'],360)
        self.assertEqual(sorted(rank_candidates([digest('x'),digest('y')],[.1,.2])),[0,1])
    def test_new_model_refuses_privileged_truth_argument(self):
        with self.assertRaises(TypeError): fit_shared(self.rows,self.q,0,new_bill(),privileged=[])
    def test_writer_seals_every_n_before_pool_truth(self):
        import b1_study as study
        from risk_learning.io import read
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run'
            def collect(task,seed,count,role,unit,where,bill):
                if role=='pool':
                    self.assertTrue((where/'states'/unit/'n12.json').exists())
                    self.assertTrue((where/'states'/unit/'n48.json').exists())
                out=[]
                for i,c in enumerate(candidates(seed,count)):
                    target=dict(region=-.2,resource=-.1,objective=1.+i*.01,feasible=True)
                    out.append(dict(candidate=c,candidate_id=digest(c),truth=target,observed=observed(c,target),call_id=unit+'/'+role+'/'+str(i)))
                bill['math_started']+=count;bill['math_settled']+=count
                return out
            def fitter(rows,queries,seed,bill):
                self.assertFalse(any('truth' in r for r in rows))
                self.assertFalse(any('objective' in q for q in queries))
                return None,{'status':'MODEL_NUMERICAL_FAILURE'}
            with patch.object(study,'collect',collect):
                r=study.run(root,'ci',{'code_sha':'a'*40},fitter=fitter)
            self.assertEqual(r['failed_states'],4)
            self.assertEqual(r['status'],'COMPLETE_WITH_MODEL_FAILURES')
            self.assertEqual(read(root/'report.json')['model_states'],4)
    def test_mutating_fitter_stops_and_preserves_failure(self):
        import b1_study as study
        from risk_learning.io import read
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'run'
            def collect(task,seed,count,role,unit,where,bill):
                self.assertEqual(role,'train')
                return [dict(candidate=r['candidate'],candidate_id=r['candidate_id'],observed=r,truth={}) for r in synthetic_rows(count)]
            def bad(rows,queries,seed,bill):
                rows[0]['candidate']['x']=.123456789
                return None,{}
            with patch.object(study,'collect',collect):
                with self.assertRaisesRegex(ValueError,'mutated'):
                    study.run(root,'ci',{'code_sha':'a'*40},fitter=bad)
            self.assertEqual(read(root/'failure.json')['status'],'STOPPED_ERROR')
            self.assertTrue((root/'manifest.json').exists())


if __name__=='__main__': unittest.main()
