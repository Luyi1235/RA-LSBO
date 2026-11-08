import copy
import inspect
import math
import unittest

from pr78_bottleneck_core import (ACTUAL, BUDGETS, PRIVILEGED, NEW_BILL, analyze_state,
    counts, digest, distribution, outcome, privileged_orders, summarize, winner)


def fixture(cold=False):
    ids=[digest(i) for i in range(12)]
    pool=[dict(candidate_id=cid, truth=dict(objective=float(i+1), feasible=(i%3==0)))
          for i,cid in enumerate(ids)]
    base=list(range(12))
    state=dict(task='tradeoff_s0',repeat=0,n=48,status='OK',objective_scale=2.,
        labels=dict(objective=dict(exact=3,missing=45,interval=0),joint_feasible=2),
        decision=dict(candidate_ids=ids,incumbent=None if cold else 5.,
                      rankings={p:base[:] for p in ACTUAL},scores=dict(objective_ei=[1.]*12)),
        predictions=dict(risk=dict(probabilities=dict(joint=[.5]*12)),
                         objective=dict(mean=[float(i) for i in range(12)],variance=[1.]*12)))
    return state,pool


class BottleneckTests(unittest.TestCase):
    def test_digest_finite(self):
        with self.assertRaises(ValueError): digest(float('nan'))

    def test_budget_roster(self):
        self.assertEqual(BUDGETS,(1,2,4,6,12))

    def test_zero_new_cost(self):
        self.assertTrue(NEW_BILL and all(v==0 for v in NEW_BILL.values()))

    def test_finite_pool_upper_bound(self):
        s,p=fixture(); r=analyze_state(s,p)
        for k in BUDGETS:
            self.assertEqual(r['by_budget'][str(k)]['outcomes']['known_both']['best_after'],1.)

    def test_all_permutations(self):
        s,p=fixture();r=analyze_state(s,p)
        self.assertEqual(set(r['rankings']),set(ACTUAL+PRIVILEGED))
        for order in r['rankings'].values(): self.assertEqual(sorted(order),list(range(12)))

    def test_known_feasible_first_at_zero_ei(self):
        s,p=fixture();s['decision']['scores']['objective_ei']=[0.]*12
        r=analyze_state(s,p)
        for name in ('known_feasibility_ei','known_both','known_feasibility_mean'):
            self.assertTrue(all(p[i]['truth']['feasible'] for i in r['rankings'][name][:4]))

    def test_budget_padding(self):
        s,p=fixture();r=analyze_state(s,p)
        self.assertEqual(r['by_budget']['12']['outcomes']['known_both']['selected_feasible'],4)

    def test_no_pool_opportunity(self):
        s,p=fixture();s['decision']['incumbent']=0.
        r=analyze_state(s,p)
        self.assertEqual(r['by_budget']['6']['tie_category'],'tie_no_pool_opportunity')

    def test_zero_feasible_pool(self):
        s,p=fixture()
        for x in p:x['truth']['feasible']=False
        r=analyze_state(s,p)
        self.assertIsNone(r['feasible_objective_rmse'])
        self.assertEqual(r['pool_opportunity'],0.)

    def test_negative_information_swap_retained(self):
        s,p=fixture()
        s['decision']['scores']['objective_ei'][3]=100.
        r=analyze_state(s,p)
        self.assertLess(r['by_budget']['1']['signed_information_swaps']['known_feasibility_minus_learned'],0.)

    def test_swap_arithmetic(self):
        s,p=fixture();r=analyze_state(s,p)
        for k in BUDGETS:
            q=r['by_budget'][str(k)]['signed_information_swaps']
            self.assertAlmostEqual(q['known_feasibility_minus_learned']+q['known_objective_minus_learned']+
                                   q['joint_interaction'],q['total_remaining_gap'])
            self.assertTrue(q['not_causal_error_shares'])

    def test_cold_has_no_oracle_claim(self):
        s,p=fixture(True);r=analyze_state(s,p)
        self.assertEqual(r['audit_status'],'COLD_ACTUAL_ONLY')
        self.assertEqual(set(r['rankings']),set(ACTUAL))
        self.assertNotIn('signed_information_swaps',r['by_budget']['6'])

    def test_cold_shared_fallback_required(self):
        s,p=fixture(True);s['decision']['rankings']['risk_ei'].reverse()
        with self.assertRaises(ValueError): analyze_state(s,p)

    def test_failed_state_not_dropped(self):
        s,p=fixture();s['status']='MODEL_NUMERICAL_FAILURE'
        r=analyze_state(s,p)
        self.assertEqual(counts([r],6)['failed_states'],1)

    def test_input_not_mutated(self):
        s,p=fixture();before=copy.deepcopy((s,p));analyze_state(s,p)
        self.assertEqual((s,p),before)

    def test_pool_order_guard(self):
        s,p=fixture();p.reverse()
        with self.assertRaises(ValueError): analyze_state(s,p)

    def test_negative_probability_guard(self):
        s,p=fixture();s['predictions']['risk']['probabilities']['joint'][0]=-.1
        with self.assertRaises(ValueError): analyze_state(s,p)

    def test_nonfinite_guard(self):
        s,p=fixture();s['predictions']['risk']['probabilities']['joint'][0]=math.nan
        with self.assertRaises(ValueError): analyze_state(s,p)

    def test_boolean_score_guard(self):
        s,p=fixture();s['decision']['scores']['objective_ei'][0]=True
        with self.assertRaises(ValueError): analyze_state(s,p)

    def test_warm_objective_required(self):
        s,p=fixture();s['predictions']['objective']=None
        with self.assertRaises(ValueError): analyze_state(s,p)

    def test_outcome_rejects_duplicate_rank(self):
        s,p=fixture()
        with self.assertRaises(ValueError):outcome([0]*12,p,6,5.,[.5]*12)

    def test_null_not_zero(self):
        s,p=fixture(True)
        for x in p:x['truth']['feasible']=False
        r=analyze_state(s,p)
        self.assertIsNone(r['by_budget']['6']['outcomes']['risk_ei']['best_after'])

    def test_comparison_tolerance(self):
        self.assertEqual(winner(1.,1.+1e-12),0)
        self.assertEqual(winner(1.,2.),1)
        self.assertEqual(winner(None,1.),-1)

    def test_no_raw_cross_task_mean(self):
        s,p=fixture();r=summarize([analyze_state(s,p)])
        self.assertTrue(r['tradeoff_k6_descriptive']['48']['no_raw_cross_task_objective_mean'])
        self.assertEqual(r['primary_key'],'tradeoff_k6_descriptive/48')

    def test_objective_label_count_is_not_training_n(self):
        s,p=fixture();r=analyze_state(s,p)
        self.assertEqual(r['observable_objective_labels'],3)
        self.assertEqual(r['n'],48)

    def test_distribution_empty(self):
        self.assertEqual(distribution([]),dict(n=0,minimum=None,median=None,maximum=None))

    def test_rank_mean_differs_from_ei(self):
        s,p=fixture();s['decision']['scores']['objective_ei'][3]=100.
        r=analyze_state(s,p)
        self.assertEqual(r['rankings']['known_feasibility_mean'][0],0)
        self.assertEqual(r['rankings']['known_feasibility_ei'][0],3)

    def test_known_objective_is_explicit_privileged_interface(self):
        self.assertIn('truth',inspect.signature(privileged_orders).parameters)
        self.assertNotIn('evaluator',inspect.signature(privileged_orders).parameters)


if __name__=='__main__':
    unittest.main()
