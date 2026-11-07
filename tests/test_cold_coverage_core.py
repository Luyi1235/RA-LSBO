import copy
import math
import unittest
from cold_coverage_core import *


class CoreTests(unittest.TestCase):
    def test_ci_budget(self): self.assertEqual(plan('ci')['total_calls'],36)
    def test_smoke_budget(self): self.assertEqual(plan('smoke')['total_calls'],72)
    def test_full_budget(self):
        p=plan('full');self.assertEqual((p['initial_calls'],p['online_calls'],p['total_calls']),(384,576,960))
    def test_full_is_sparse_only(self): self.assertEqual(plan('full')['cases'],['branin_sparse','rosenbrock_sparse'])
    def test_disjoint_seeds(self):
        self.assertFalse(set(plan('full')['seeds']) & set(plan('smoke')['seeds']))
    def test_wrong_profile(self):
        with self.assertRaises(ValueError):plan('pilot')
    def test_schedule_first_and_sixth(self): self.assertEqual([i for i in range(6) if scheduled(True,i)],[0,5])
    def test_no_warm_schedule(self): self.assertFalse(any(scheduled(False,i) for i in range(12)))
    def test_schedule_rejects_int_phase(self):
        with self.assertRaises(ValueError):scheduled(1,0)
    def test_schedule_rejects_negative(self):
        with self.assertRaises(ValueError):scheduled(True,-1)
    def test_initial_solution_is_zero(self): self.assertEqual(first_feasible([3.,3.,2.]),0)
    def test_late_discovery(self): self.assertEqual(first_feasible([None,None,3.]),2)
    def test_never_discovered_is_null(self): self.assertIsNone(first_feasible([None,None]))
    def test_solution_cannot_disappear(self):
        with self.assertRaises(ValueError):first_feasible([1.,None])
    def test_curve_cannot_worsen(self):
        with self.assertRaises(ValueError):first_feasible([1.,2.])
    def test_nonfinite_curve(self):
        with self.assertRaises(ValueError):first_feasible([None,math.nan])
    def test_feasible_first(self): self.assertEqual(compare(3.,None),1)
    def test_both_missing_tie(self): self.assertEqual(compare(None,None),0)
    def test_lower_is_better(self): self.assertEqual(compare(3.,4.),1)
    def test_nonfinite_not_discovery(self):
        with self.assertRaises(ValueError):compare(math.inf,None)

    def fixture(self,cold=False):
        spec=plan('ci');rows=[]
        for c in spec['cases']:
            for s in spec['seeds']:
                for a in ARMS:
                    curve=[None,None,2.] if cold else [3.,3.,2.]
                    r={'case':c,'seed':s,'arm':a,'initial_hash':'same','initial_best':curve[0],
                        'final_best':curve[-1],'curve':curve,'first_feasible_call':first_feasible(curve),
                        'path_hash':'same','online_calls':2,'nonfeasible':0,'proposal_seconds':1.,
                        'scheduled_coverage_calls':0,'degenerate_coverage_calls':0,'model_failure_calls':0,
                        'cold_decisions':int(cold)*2,'model_free_calls':0}
                    rows.append(r)
        return rows,spec
    def test_warm_stratum_is_not_discovery(self):
        rows,spec=self.fixture();r=summarize(rows,spec)
        self.assertEqual(r[spec['cases'][0]]['initial_cold']['groups'],0)
        self.assertEqual(r[spec['cases'][0]]['initial_warm']['arms'][CONTROL]['discoveries'],0)
    def test_cold_stratum_denominator(self):
        rows,spec=self.fixture(True);r=summarize(rows,spec)
        self.assertEqual(r[spec['cases'][0]]['initial_cold']['groups'],1)
    def test_warm_path_difference_rejected(self):
        rows,spec=self.fixture();rows[1]['path_hash']='different'
        with self.assertRaises(ValueError):summarize(rows,spec)
    def test_initial_mismatch_rejected(self):
        rows,spec=self.fixture();rows[1]['initial_hash']='different'
        with self.assertRaises(ValueError):summarize(rows,spec)
    def test_duplicate_trial_rejected(self):
        rows,spec=self.fixture()
        with self.assertRaises(ValueError):summarize(rows+[rows[0]],spec)
    def test_missing_trial_rejected(self):
        rows,spec=self.fixture()
        with self.assertRaises(ValueError):summarize(rows[:-1],spec)
    def test_zero_stratum_has_null_delta(self):
        rows,spec=self.fixture();r=summarize(rows,spec)
        pair=r[spec['cases'][0]]['initial_cold']['pairs'][TREATMENT+'_vs_'+CONTROL]
        self.assertIsNone(pair['both_feasible_objective_delta_mean'])
    def test_inputs_not_mutated(self):
        rows,spec=self.fixture(True);before=copy.deepcopy(rows);summarize(rows,spec);self.assertEqual(rows,before)


if __name__=='__main__':unittest.main()
