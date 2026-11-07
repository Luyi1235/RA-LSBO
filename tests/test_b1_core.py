import inspect
import math
import unittest
from b1_core import (PROFILES,TASKS,POLICIES,BUDGETS,plan,digest,expected_improvement,
    incumbent_from_observed,objective_payload,decisions,assess,compare,summarize)


def ids(n=3): return [digest(['candidate',i]) for i in range(n)]
def row(f=5.,g=-1.,r=-1.,missing=False):
    return dict(candidate={'x':.3},candidate_id=digest('c'),
        objective={'kind':'missing'} if missing else {'kind':'exact','value':f},
        region={'kind':'exact','value':g},resource={'kind':'exact','value':r},truth='FORBIDDEN')
def pool():
    return [dict(candidate_id=c,truth=dict(objective=f,feasible=y))
            for c,f,y in zip(ids(),(3.,1.,7.),(True,False,True))]


class CoreTests(unittest.TestCase):
    def test_ci_budget(self):
        p=plan('ci');self.assertEqual((p['train_calls'],p['audit_pool_calls'],p['total_math_calls']),(96,64,160))
    def test_smoke_budget(self):
        p=plan('smoke');self.assertEqual((p['train_calls'],p['audit_pool_calls'],p['model_states']),(288,384,12))
    def test_full_budget(self):
        p=plan('full');self.assertEqual((p['train_calls'],p['audit_pool_calls'],p['total_math_calls']),(9216,12288,21504))
        self.assertEqual((p['model_states'],p['scored_rows'],p['maximum_gp_heads']),(144,2880,432))
    def test_disjoint_profiles_and_datasets(self):
        all_seeds=[]
        for name in PROFILES:
            for u in plan(name)['units']: all_seeds.extend((u['train_seed'],u['pool_seed']))
        self.assertEqual(len(all_seeds),len(set(all_seeds)))
    def test_primary_fixed(self):
        p=plan('full');self.assertEqual((p['primary_n'],p['primary_k']),(48,6))
        self.assertEqual(p['primary_tasks'],list(TASKS[2:]));self.assertFalse(p['online_BO'])
    def test_reserved_not_released(self):
        self.assertFalse(set('tradeoff_s'+str(i) for i in range(4,8)) & set(TASKS))
    def test_unknown_profile(self):
        with self.assertRaises(ValueError): plan('adaptive')
    def test_ei_standard_normal(self):
        self.assertAlmostEqual(expected_improvement(0.,1.,0.),1/math.sqrt(2*math.pi))
    def test_ei_deterministic(self):
        self.assertEqual(expected_improvement(2.,0.,3.),1.)
        self.assertEqual(expected_improvement(4.,0.,3.),0.)
    def test_ei_monotone_incumbent(self):
        self.assertGreater(expected_improvement(2.,1.,3.),expected_improvement(2.,1.,1.))
    def test_ei_affine_units(self):
        self.assertAlmostEqual(expected_improvement(12.,9.,15.),3*expected_improvement(2.,1.,3.))
    def test_ei_tail_nonnegative(self):
        for z in (-40,-12,-8,0,8,40): self.assertGreaterEqual(expected_improvement(-z,1.,0.),0.)
    def test_bad_variance(self):
        with self.assertRaises(ValueError): expected_improvement(0.,-1.,0.)
    def test_nonfinite_and_bool(self):
        for x in (float('nan'),float('inf'),True):
            with self.assertRaises(ValueError): expected_improvement(x,1.,0.)
    def test_incumbent_observable_only(self):
        self.assertEqual(incumbent_from_observed([row(),row(1.,r=1.),row(0.,missing=True)]),5.)
    def test_no_incumbent(self): self.assertIsNone(incumbent_from_observed([row(r=1.),row(missing=True)]))
    def test_payload_excludes_hidden_constraints(self):
        self.assertEqual(set(objective_payload([row()])[0]),{'candidate','candidate_id','objective'})
    def test_selector_no_truth_parameters(self):
        self.assertEqual(list(inspect.signature(decisions).parameters),['ids','probabilities','objective','incumbent','salt'])
    def test_cold_matched_fallback(self):
        d=decisions(ids(),[.1,.9,.2],None,None,'unit')
        self.assertEqual(d['rankings']['objective_ei'],d['rankings']['uniform_hash'])
        self.assertEqual(d['rankings']['risk_ei'],d['rankings']['uniform_hash'])
        self.assertEqual(d['rankings']['risk_only'][0],1)
    def test_flat_positive_risk_preserves_ei_ranking(self):
        d=decisions(ids(),[.3]*3,dict(mean=[1.,2.,3.],variance=[0.,0.,0.]),5.,'x')
        self.assertEqual(d['rankings']['objective_ei'],d['rankings']['risk_ei'])
    def test_risk_changes_action(self):
        d=decisions(ids(),[.01,.9,.1],dict(mean=[1.,2.,3.],variance=[0.,0.,0.]),5.,'x')
        self.assertEqual(d['rankings']['objective_ei'][0],0);self.assertEqual(d['rankings']['risk_ei'][0],1)
    def test_id_and_probability_validation(self):
        for ci,p in [(['x'],[.5]),(ids(),[.1]),(ids(),[.1,1.1,.3]),([ids()[0]]*3,[.1]*3)]:
            with self.assertRaises(ValueError): decisions(ci,p,None,None,'x')
    def test_warm_needs_objective(self):
        with self.assertRaises(ValueError): decisions(ids(),[.1]*3,None,2.,'x')
    def test_tie_order_independent_of_input_order(self):
        a=decisions(ids(),[.2]*3,None,None,'x'); b=decisions(ids()[::-1],[.2]*3,None,None,'x')
        aa=[a['candidate_ids'][i] for i in a['rankings']['uniform_hash']]
        bb=[b['candidate_ids'][i] for i in b['rankings']['uniform_hash']]
        self.assertEqual(aa,bb)
    def test_objective_missing_not_imputed(self):
        r=assess(pool(),[1,0,2],1,None,1.)
        self.assertIsNone(r['best_after']);self.assertIsNone(r['improvement']);self.assertEqual(r['selected_nonfeasible'],1)
    def test_warm_cannot_lose_incumbent(self):
        r=assess(pool(),[2,0,1],1,5.,2.)
        self.assertEqual(r['best_after'],5.);self.assertEqual(r['improvement'],0.)
    def test_opportunity_fraction(self):
        r=assess(pool(),[0,1,2],1,5.,2.)
        self.assertEqual((r['improvement'],r['standardized_improvement'],r['opportunity_fraction']),(2.,1.,1.))
    def test_no_pool_opportunity(self): self.assertIsNone(assess(pool(),[0,1,2],1,2.,1.)['opportunity_fraction'])
    def test_cold_discovery(self):
        r=assess(pool(),[0,1,2],1,None,1.);self.assertTrue(r['new_discovery']);self.assertFalse(r['initial_warm'])
    def test_reject_bad_selection(self):
        for order in ([0,0,2],[0,1],[False,1,2],[0,1,2,2]):
            with self.assertRaises(ValueError): assess(pool(),order,1,None,1.)
    def test_budget_and_scale(self):
        for k,s in ((0,1.),(4,1.),(1,0.)):
            with self.assertRaises(ValueError): assess(pool(),[0,1,2],k,None,s)
    def test_feasible_first(self):
        self.assertEqual(compare({'best_after':None},{'best_after':None}),0)
        self.assertEqual(compare({'best_after':1.},{'best_after':None}),1)
        self.assertEqual(compare({'best_after':2.},{'best_after':1.}),-1)
    def test_summary_failure_retained(self):
        s=summarize([dict(task=TASKS[0],n=48,status='MODEL_NUMERICAL_FAILURE')])
        self.assertEqual(len(s),len(BUDGETS))
        for r in s.values():
            self.assertEqual(r['failed_model_states'],1);self.assertEqual(r['all']['valid_pairs'],0)
    def test_summary_pairing_and_no_cross_task_mean(self):
        p=pool()*4
        p=[dict(r,candidate_id=digest(['p',i])) for i,r in enumerate(p)]
        d=decisions([r['candidate_id'] for r in p],[.5]*12,dict(mean=[0.]*12,variance=[1.]*12),5.,'x')
        outcomes={a:{str(k):assess(p,d['rankings'][a],k,5.,1.) for k in BUDGETS} for a in POLICIES}
        s=summarize([dict(task=TASKS[0],repeat=0,n=48,status='OK',decision=d,outcomes=outcomes)])
        r=s[TASKS[0]+'/n48/k6'];self.assertEqual(r['initial_warm']['risk_vs_objective_wlt'],[0,0,1])
        self.assertEqual(r['initial_cold']['valid_pairs'],0);self.assertEqual(r['all']['paired_objectives'][0]['risk_minus_no_risk'],0.)


if __name__=='__main__': unittest.main()
