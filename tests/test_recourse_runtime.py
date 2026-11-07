"""Actual frozen EP adapter parity; imports fail rather than skip without models."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from test_quadrature_runtime import SamePoolFactory
from pr70_future_pool import choose_next
from recourse_value_engine import choose_recourse
from recourse_value_core import ADAPTIVE,FIXED,decompose
from ra_lsbo.experiments.information_value_stage3.planner import FantasyFitError


class RuntimeTests(unittest.TestCase):
    def run_rule(self,arm,best=5.,remaining=2):
        f=SamePoolFactory()
        i,r=choose_recourse(f.forecast,best,remaining,f,arm=arm,seed=123,iteration=0)
        return f,i,r
    def parity(self,best):
        old=SamePoolFactory();i,r=choose_next(old.forecast,best,2,old,seed=123,iteration=0)
        new,j,s=self.run_rule(ADAPTIVE,best)
        self.assertEqual(i,j);self.assertEqual(r['scores'],s['scores'])
        self.assertEqual(old.native.started,new.native.started)
        self.assertIsNone(s['fallback'])
    def test_warm_exact_PR70_parity(self):self.parity(5.)
    def test_cold_exact_PR70_parity(self):self.parity(None)
    def test_both_rules_use_identical_updates_and_vectors(self):
        a,_,r=self.run_rule(ADAPTIVE);b,_,s=self.run_rule(FIXED)
        self.assertEqual(a.native.started,b.native.started);self.assertGreater(a.native.started,0)
        self.assertEqual(r['fantasies'],s['fantasies']);self.assertEqual(r['recourse_decomposition'],s['recourse_decomposition'])
    def test_fixed_root_maxes_fixed_score(self):
        _,i,r=self.run_rule(FIXED)
        self.assertEqual(i,r['recourse_decomposition']['fixed_index'])
    def test_real_raw_decomposition_reconstructs(self):
        _,_,r=self.run_rule(ADAPTIVE)
        self.assertEqual(r['recourse_decomposition'],decompose(r['myopic_scores'],r['first_action_indices'],r['fantasies'],
                         {i:len(r['future_supports'][str(i)]['model_candidates']) for i in r['first_action_indices']}))
    def test_last_adaptive_call_has_no_fits(self):
        f,i,r=self.run_rule(ADAPTIVE,remaining=1)
        self.assertEqual(f.native.started,0);self.assertEqual(i,r['myopic_index']);self.assertEqual(r['depth'],1)
    def test_last_fixed_call_has_no_fits(self):
        f,i,r=self.run_rule(FIXED,remaining=1)
        self.assertEqual(f.native.started,0);self.assertEqual(i,r['myopic_index']);self.assertEqual(r['depth'],1)
    def test_hidden_abort_values_never_present(self):
        _,_,r=self.run_rule(ADAPTIVE)
        for row in r['fantasies']:
            if row['observation']['aborted']:
                self.assertEqual(row['observation']['objective'],{'kind':'missing'})
                self.assertNotIn('value',row['observation']['region'])
    def test_cold_success_terminal_vectors_are_zero(self):
        _,_,r=self.run_rule(ADAPTIVE,None)
        for row in r['fantasies']:
            if not row['conditioned']:
                self.assertEqual(set(row['future_values']),{0.});self.assertIsNone(row['second_index'])
    def test_input_predictions_not_mutated(self):
        f=SamePoolFactory();before=copy.deepcopy(f.forecast)
        choose_recourse(f.forecast,5.,2,f,arm=FIXED,seed=123,iteration=0)
        for n in before['constraints']:
            for field in ('mean','variance'):np.testing.assert_array_equal(before['constraints'][n][field],f.forecast['constraints'][n][field])
    def test_numeric_failure_discards_entire_plan(self):
        f=SamePoolFactory()
        with patch.object(f,'condition',side_effect=FantasyFitError('fixture')):
            i,r=choose_recourse(f.forecast,5.,2,f,arm=FIXED,seed=123,iteration=0)
        self.assertEqual(i,r['myopic_index']);self.assertEqual(r['scores'],r['myopic_scores']);self.assertIsNone(r['recourse_decomposition'])
    def test_programming_error_is_not_numeric_fallback(self):
        f=SamePoolFactory()
        with patch.object(f,'condition',side_effect=KeyError('fixture')):
            with self.assertRaises(KeyError):choose_recourse(f.forecast,5.,2,f,arm=ADAPTIVE,seed=123,iteration=0)
    def test_true_evaluator_not_accessed_by_planner(self):
        from ra_lsbo.experiments.core.problems import MathematicalCase
        with patch.object(MathematicalCase,'evaluate',side_effect=AssertionError('evaluator forbidden')):
            self.run_rule(ADAPTIVE)

    def bound_receipt(self):
        from ra_lsbo.budget import digest
        from recourse_value_protocol import check_receipt
        f=SamePoolFactory()
        original_support=f.support
        def support(i):
            d=original_support(i);points=d['model_candidates']
            return {**d,'iteration':1,'raw_pool_sha256':digest(points),
                    'model_pool_sha256':digest(points),'generation_uses_observations':False,
                    'first_candidate_sha256':digest(f.native.candidates[i])}
        f.support=support
        i,r=choose_recourse(f.forecast,5.,2,f,arm=ADAPTIVE,seed=123,iteration=0)
        cs=f.native.candidates
        r.update(candidate_pool_hash=digest(cs),selected_candidate_hash=digest(cs[i]),
                 selected_predicted_failure=.2,fantasy_fits=f.diagnostics())
        meta={'pool':cs,'scores':r['scores'],'mode':'fixture','information_value_decision':r}
        return meta,cs[i],check_receipt
    def test_actual_receipt_reconciles(self):
        m,c,check=self.bound_receipt();d=check(m,c,0,2,ADAPTIVE)
        self.assertTrue(d['forecast_available']);self.assertIsNotNone(d['selected_recourse_premium'])
    def test_premium_tamper_rejected(self):
        m,c,check=self.bound_receipt();r=m['information_value_decision']
        r['recourse_decomposition']['roots'][str(r['selected_index'])]['recourse_premium']+=1
        with self.assertRaises(ValueError):check(m,c,0,2,ADAPTIVE)
    def test_full_vector_tamper_rejected(self):
        m,c,check=self.bound_receipt();m['information_value_decision']['fantasies'][0]['future_values'][0]+=1
        with self.assertRaises(ValueError):check(m,c,0,2,ADAPTIVE)
    def test_abort_leak_tamper_rejected(self):
        m,c,check=self.bound_receipt()
        rows=[x for x in m['information_value_decision']['fantasies'] if x['observation']['aborted']]
        self.assertTrue(rows);rows[0]['observation']['objective']={'kind':'exact','value':0.}
        with self.assertRaises(ValueError):check(m,c,0,2,ADAPTIVE)
    def test_wrong_selected_support_rejected(self):
        m,c,check=self.bound_receipt();m['information_value_decision']['selected_future_support']=None
        with self.assertRaises(ValueError):check(m,c,0,2,ADAPTIVE)

if __name__=='__main__':unittest.main()
