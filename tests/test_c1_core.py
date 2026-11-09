import copy
import inspect
import math
import unittest
from c1_protocol import *


def record(i,region=-.2,resource=-.2,objective=1.):
    c=dict(branch=i%4,x=.1+i*.01,y=.3,trim=.5)
    return dict(candidate=c,candidate_id=digest(c),
        region=dict(kind='exact',value=region) if region<=0 else dict(kind='interval',lower=0.,lower_inclusive=False),
        resource=dict(kind='exact',value=resource),
        objective=dict(kind='exact',value=objective) if region<=0 else dict(kind='missing',reason='controlled_region_abort'))


def prediction(ids,p=(.01,.9)):
    return dict(candidate_ids=ids,probabilities=dict(joint=list(p)),
                objective=dict(mean=[0.,.5],variance=[0.,0.]))


class CoreTests(unittest.TestCase):
    def setUp(self):self.ids=[digest('a'),digest('b')]
    def test_ci_budget(self):self.assertEqual(plan('ci')['total_math_calls'],120)
    def test_smoke_budget(self):self.assertEqual(plan('smoke')['total_math_calls'],264)
    def test_full_budget(self):
        p=plan('full');self.assertEqual((p['initial_calls'],p['online_calls'],p['total_math_calls']),(1536,4608,6144))
    def test_full_gp_cap(self):self.assertEqual((plan('full')['max_gp_heads'],plan('full')['max_gp_kernels']),(5984,47872))
    def test_profiles_disjoint(self):
        seen=set()
        for name in PROFILES:
            p=plan(name)
            for u in p['units']:
                seeds=set(range(u['seed'],u['seed']+p['steps']+1))
                self.assertFalse(seen & seeds);seen |= seeds
    def test_tasks_and_arms(self):
        self.assertEqual(len(ARMS),6);self.assertEqual(TASKS,tuple('tradeoff_s%d'%i for i in range(4)))
    def test_no_reserved(self):self.assertFalse(plan('full')['retained_task_s4_s7'])
    def test_checkpoints(self):self.assertEqual(plan('full')['checkpoints'],[6,12,24])
    def test_unknown_profile(self):
        with self.assertRaises(ValueError):plan('pilot')
    def test_ei_standard_normal(self):self.assertAlmostEqual(ei(0.,1.,0.),1/math.sqrt(2*math.pi))
    def test_ei_deterministic(self):self.assertEqual(ei(.5,0.,1.),.5)
    def test_ei_bad_variance(self):
        with self.assertRaises(ValueError):ei(0.,-1.,0.)
    def test_ei_nonfinite(self):
        for x in (float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):ei(x,1.,0.)
    def test_risk_changes_action(self):
        p=prediction(self.ids)
        self.assertEqual(select('ei_updated',self.ids,[0,1],p,1.,'s')['selected_index'],0)
        self.assertEqual(select('risk_ei_updated',self.ids,[0,1],p,1.,'s')['selected_index'],1)
    def test_cold_shared_fallback(self):
        p=prediction(self.ids);a=[select(x,self.ids,[0,1],p,None,'s') for x in ARMS[:4]]
        self.assertTrue(all(x==a[0] for x in a))
    def test_missing_objective(self):
        p=prediction(self.ids);p['objective']=None
        self.assertEqual(select('ei_frozen',self.ids,[0,1],p,1.,'s')['mode'],'no_objective_model_hash')
    def test_failed_fit_hash(self):
        a=select('ei_frozen',self.ids,[0,1],None,1.,'s');b=select('uniform_hash',self.ids,[0,1],None,1.,'s')
        self.assertEqual(a['selected_index'],b['selected_index']);self.assertEqual(a['mode'],'model_failure_hash')
    def test_risk_only_no_objective(self):
        p=prediction(self.ids);p['objective']=None
        self.assertEqual(select('risk_only_updated',self.ids,[0,1],p,None,'s')['selected_index'],1)
    def test_duplicate_excluded(self):
        self.assertEqual(select('ei_updated',self.ids,[1],prediction(self.ids),1.,'s')['selected_index'],1)
    def test_bad_pool(self):
        for available in ([],[True],[2],[0,0]):
            with self.assertRaises(ValueError):select('ei_updated',self.ids,available,None,1.,'s')
    def test_misaligned_ids(self):
        p=prediction(list(reversed(self.ids)))
        with self.assertRaises(ValueError):select('ei_updated',self.ids,[0,1],p,1.,'s')
    def test_invalid_probability(self):
        for p in (-.1,1.1,float('nan'),True):
            with self.assertRaises(ValueError):select('ei_updated',self.ids,[0,1],prediction(self.ids,(p,.5)),1.,'s')
    def test_selector_has_no_truth_interface(self):
        self.assertEqual(list(inspect.signature(select).parameters),['arm','ids','available','prediction','incumbent','salt'])
    def test_live_incumbent(self):
        a=select('ei_frozen',self.ids,[0,1],prediction(self.ids),1.,'s')
        b=select('ei_frozen',self.ids,[0,1],prediction(self.ids),.1,'s')
        self.assertNotEqual(a['scores'],b['scores'])
    def test_information_vs_feasibility(self):
        r=[record(0),record(1,resource=.5),record(2,region=.5)]
        self.assertEqual(information(r)['exact_objectives'],2);self.assertEqual(information(r)['feasible'],1)
    def test_best_only_feasible(self):self.assertEqual(best([record(0,objective=3.),record(1,resource=.5,objective=-100.)]),3.)
    def test_cold_best_missing(self):self.assertIsNone(best([record(0,region=.5)]))
    def test_initial_warm_first_zero(self):
        r=trial_result('x',0,'a',[record(0)],[],[]);self.assertEqual(r['first_feasible_call'],0)
    def test_cold_first_discovery(self):
        p=[dict(decision=dict(mode='uniform_hash'))]*2
        r=trial_result('x',0,'a',[record(0,region=.5)],[record(1,region=.5),record(2)],p)
        self.assertEqual(r['first_feasible_call'],2)
    def test_no_discovery_null(self):self.assertIsNone(trial_result('x',0,'a',[record(0,region=.5)],[],[])['first_feasible_call'])
    def test_feasible_first(self):
        self.assertEqual(outcome(None,1.),-1);self.assertEqual(outcome(1.,None),1);self.assertEqual(outcome(None,None),0)
    def test_ties(self):self.assertEqual(outcome(1.,1.+1e-12),0)
    def test_frozen_vs_updated_flags(self):
        self.assertEqual([a for a in ARMS if is_updated(a)],['ei_updated','risk_ei_updated','risk_only_updated'])
    def test_summary_keeps_fallback_and_cold(self):
        spec=plan('ci');rows=[]
        for u in spec['units']:
            for a in ARMS:
                initial=[record(0,region=.5)]
                online=[record(1,region=.5),record(2,region=.5)]
                props=[dict(decision=dict(mode='model_failure_hash'))]*2
                rows.append(trial_result(u['task'],u['repeat'],a,initial,online,props))
        s=summarize(rows,spec)
        self.assertEqual(s['tradeoff_s0/k2/initial_cold']['arms']['ei_updated']['n'],1)
        self.assertEqual(s['tradeoff_s0/k2/initial_cold']['arms']['ei_updated']['model_failure_calls'],2)
        with self.assertRaises(ValueError):summarize(rows[:-1],spec)

if __name__=='__main__':unittest.main()
