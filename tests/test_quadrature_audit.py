import copy
import math
import unittest
from types import SimpleNamespace
import numpy as np
from quadrature_audit_protocol import plan, close, summarize, key
from quadrature_audit_numeric import points, compute_rule, verify_rule

class NumericFailure(RuntimeError):
    pass

class Outcome:
    feasible = False
    objective = None
    def packet(self):
        return {'aborted': True, 'region': {'kind': 'interval', 'lower': 0., 'lower_inclusive': False},
                'resource': {'kind': 'exact', 'value': -.2}, 'objective': {'kind': 'missing'}}

class Factory:
    def __init__(self, failure=None): self.calls=0; self.failure=failure
    def support(self, i): return {'model_candidates': [0,1]}
    def condition(self, i, o, **kw):
        self.calls += 1
        if self.failure: raise self.failure('deliberate')
        return {'values': [.4+i*.1, .1]}
    def diagnostics(self): return {'head_fits_started': self.calls}

K = SimpleNamespace(one_step=lambda f,b: np.array(f['values']), shortlist=lambda f,b: [0,1],
                    nodes=lambda s,i: points(s,i,4,0), FantasyFitError=NumericFailure,
                    outcomes=lambda f,i,u,w: [(1/len(u),Outcome()) for _ in u])

def rule(n=4, replicate=0, failure=None):
    return compute_rule({'values': [.4,.2]}, None, Factory(failure), seed=99, iteration=0,
                        n=n, replicate=replicate, first_indices=[0,1], kernel=K)

class ProtocolTests(unittest.TestCase):
    def test_smoke_count(self): self.assertEqual(len(plan('smoke')['states']),4)
    def test_full_count(self): self.assertEqual(len(plan('full')['states']),32)
    def test_smoke_subset(self): self.assertTrue(all(s in plan('full')['states'] for s in plan('smoke')['states']))
    def test_budget_is_zero(self): self.assertEqual(plan('full')['new_evaluation_budget'],0)
    def test_no_vae_training(self): self.assertEqual(plan('full')['new_vae_training_budget'],0)
    def test_profile_invalid(self):
        with self.assertRaises(ValueError): plan('pilot')
    def test_nested_nodes(self):
        np.testing.assert_array_equal(points(9,3,4,0),points(9,3,64,0)[:4])
    def test_replicate_differs(self): self.assertFalse(np.array_equal(points(9,3,64,0),points(9,3,64,1)))
    def test_nodes_reproducible(self): np.testing.assert_array_equal(points(9,3,16,0),points(9,3,16,0))
    def test_non_power_of_two_rejected(self):
        with self.assertRaises(ValueError): points(9,3,12,0)
    def test_bad_replicate_rejected(self):
        with self.assertRaises(ValueError): points(9,3,16,2)
    def test_finite_interior_nodes(self):
        p=points(9,3,64,0);self.assertTrue(np.isfinite(p).all() and (p>0).all() and (p<1).all())
    def test_constant_contribution_invariance(self):
        close(rule(4)['scores'],rule(64)['scores'])
    def test_formula_and_winner(self):
        r=rule();self.assertEqual(r['selected_index'],0);close(r['scores'],{'0':.8,'1':.7})
    def test_raw_reconciliation(self): verify_rule(rule(),None,[0,1],{'0':2,'1':2})
    def test_changed_score_rejected(self):
        r=rule();r['scores']['0']+=.2
        with self.assertRaises(ValueError): verify_rule(r,None,[0,1],{'0':2,'1':2})
    def test_wrong_second_rejected(self):
        r=rule();r['outcomes'][0]['second_index']=2
        with self.assertRaises(ValueError): verify_rule(r,None,[0,1],{'0':2,'1':2})
    def test_abort_value_leak_rejected(self):
        r=rule();r['outcomes'][0]['observation']['objective']={'kind':'exact','value':1.}
        with self.assertRaises(ValueError): verify_rule(r,None,[0,1],{'0':2,'1':2})
    def test_numerical_failure_not_fake_winner(self):
        r=rule(failure=NumericFailure);self.assertEqual(r['status'],'NUMERICAL_FAILURE');self.assertIsNone(r['selected_index'])
    def test_programming_error_propagates(self):
        with self.assertRaises(KeyError): rule(failure=KeyError)
    def test_replay_numeric_tolerance(self): close([1.,None], [1.+1e-9,None])
    def test_replay_missingness_rejected(self):
        with self.assertRaises(ValueError): close(None,0.)
    def test_nan_rejected(self):
        with self.assertRaises(ValueError): close(float('nan'),0.)
    def test_empty_summary_null_comparisons(self): self.assertEqual(summarize([])['branin_ordinary']['states'],0)
    def test_summary_winner_change(self):
        rules={key(n,r):rule(n,r) for n,r in ((4,0),(16,0),(64,0),(64,1))}
        states=[{'case':'branin_ordinary','best':None,'rules':rules}]
        self.assertEqual(summarize(states)['branin_ordinary']['comparisons']['n4_r0_vs_n64_r0']['winner_changes'],0)
    def test_inputs_not_mutated(self):
        f={'values':[.4,.2]};old=copy.deepcopy(f)
        compute_rule(f,None,Factory(),seed=99,iteration=0,n=4,replicate=0,first_indices=[0,1],kernel=K)
        self.assertEqual(f,old)

if __name__ == '__main__': unittest.main()
