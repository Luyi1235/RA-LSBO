"""Standard-library-only finite scenario and protocol regression tests."""
import copy
import math
import unittest
from recourse_value_core import decompose, winner
from recourse_value_protocol import plan


class CoreTests(unittest.TestCase):
    def example(self):
        return [0.4,0.3],[0,1],[
            {'first_index':0,'weight':.5,'future_values':[1.,0.]},
            {'first_index':0,'weight':.5,'future_values':[0.,1.]},
            {'first_index':1,'weight':.5,'future_values':[.8,.8]},
            {'first_index':1,'weight':.5,'future_values':[.8,.8]}],{0:2,1:2}
    def test_adaptive_information_changes_root(self):
        r=decompose(*self.example());self.assertEqual(r['adaptive_index'],0);self.assertEqual(r['fixed_index'],1)
    def test_premium_is_expectation_max_difference(self):
        r=decompose(*self.example());self.assertEqual(r['roots']['0']['recourse_premium'],.5)
    def test_constant_optimal_second_has_zero_premium(self):
        self.assertEqual(decompose(*self.example())['roots']['1']['recourse_premium'],0.)
    def test_root_sums(self):
        r=decompose(*self.example())['roots']['0'];self.assertEqual(r['adaptive_score'],1.4);self.assertEqual(r['fixed_score'],.9)
    def test_fixed_second_deterministic_tie(self):
        self.assertEqual(decompose(*self.example())['roots']['0']['fixed_second_index'],0)
    def test_single_second_no_value(self):
        r=decompose([.1],[0],[{'first_index':0,'weight':1.,'future_values':[.7]}],{0:1})
        self.assertEqual(r['roots']['0']['recourse_premium'],0.)
    def test_terminal_zero_vectors(self):
        r=decompose([.7],[0],[{'first_index':0,'weight':1.,'future_values':[0.,0.]}],{0:2})
        self.assertEqual(r['roots']['0']['adaptive_score'],.7)
    def test_nonuniform_weights(self):
        a=self.example();a[2][0]['weight']=.2;a[2][1]['weight']=.8
        r=decompose(*a)['roots']['0'];self.assertAlmostEqual(r['fixed_continuation'],.8);self.assertAlmostEqual(r['recourse_premium'],.2)
    def test_order_and_input_unchanged(self):
        a=self.example();before=copy.deepcopy(a);decompose(*a);self.assertEqual(a,before)
    def test_missing_root_rejected(self):
        a=self.example();a[2].pop();
        with self.assertRaises(ValueError):decompose(*a)
    def test_negative_weight_rejected(self):
        a=self.example();a[2][0]['weight']=-.5
        with self.assertRaises(ValueError):decompose(*a)
    def test_nan_rejected(self):
        a=self.example();a[2][0]['future_values'][0]=math.nan
        with self.assertRaises(ValueError):decompose(*a)
    def test_negative_future_rejected(self):
        a=self.example();a[2][0]['future_values'][0]=-.1
        with self.assertRaises(ValueError):decompose(*a)
    def test_changed_support_rejected(self):
        a=self.example();a[3][0]=3
        with self.assertRaises(ValueError):decompose(*a)
    def test_duplicate_roots_rejected(self):
        a=self.example();a[1].append(0)
        with self.assertRaises(ValueError):decompose(*a)
    def test_myopic_must_be_screened(self):
        with self.assertRaises(ValueError):decompose([1.,0.],[1],[{'first_index':1,'weight':1.,'future_values':[.1]}],{1:1})
    def test_nonfinite_base_rejected(self):
        a=self.example();a[0][0]=math.inf
        with self.assertRaises(ValueError):decompose(*a)
    def test_bool_is_not_value(self):
        with self.assertRaises(ValueError):winner({0:True})
    def test_tie_uses_original_candidate_index(self):
        self.assertEqual(winner({5:1.,2:1.}),2)
    def test_finite_jensen_many_scenarios(self):
        import random
        rng=random.Random(7)
        for _ in range(20):
            rows=[{'first_index':0,'weight':.125,'future_values':[rng.random() for _ in range(7)]} for _ in range(8)]
            self.assertGreaterEqual(decompose([.2],[0],rows,{0:7})['roots']['0']['recourse_premium'],0)
    def test_ci_budget(self):self.assertEqual(plan('ci')['total_calls'],40)
    def test_smoke_budget(self):self.assertEqual(plan('smoke')['total_calls'],80)
    def test_full_budget(self):
        p=plan('full');self.assertEqual((p['initial_calls'],p['online_calls'],p['total_calls']),(384,768,1152))
    def test_seed_blocks_disjoint(self):
        c,s,f=[set(plan(p)['seeds']) for p in ('ci','smoke','full')];self.assertFalse(c&s or c&f or s&f)
    def test_unregistered_profile_rejected(self):
        with self.assertRaises(ValueError):plan('custom')

if __name__=='__main__':unittest.main()
