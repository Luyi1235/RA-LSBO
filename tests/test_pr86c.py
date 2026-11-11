"""PR86-C protocol, semantics and acquisition regression tests (no paid math runs)."""
import math
import unittest

from risk_learning.core import TASKS, candidates, observed, truth, known_label
from pr86c_protocol import HELDOUT, METHODS, plan
from pr86c_evaluator import truth_for_evaluation
from pr86c_models import outcome_state, exact_feasible_objectives
from pr86c_acquisition import entropy, expected_improvement, score_candidates, choose
from pr86c_analysis import compare_objective


class ProtocolTests(unittest.TestCase):
    def test_frozen_bills(self):
        ci=plan("ci")
        full=plan("full")
        self.assertEqual(ci["total_math_calls"],272)
        self.assertEqual(full["total_math_calls"],21120)
        self.assertEqual(full["shared_anchor_calls"],360)
        self.assertEqual(full["arm_online_calls"],5400)
        self.assertEqual(full["heldout_test_calls"],15360)
        self.assertEqual(full["expected_risk_refits"],5550)
        self.assertEqual(len(full["units"]),30)
        self.assertFalse(any(t in HELDOUT for t in ci["tasks"]) if "tasks" in ci
                         else any(u["task"] in HELDOUT for u in ci["units"]))
        self.assertEqual(len(METHODS),5)

    def test_new_seeds_and_holdouts(self):
        self.assertEqual(set(HELDOUT),{"tradeoff_s4","tradeoff_s5","tradeoff_s6","tradeoff_s7"})
        self.assertFalse(any(u["task"] in HELDOUT for u in plan("smoke")["units"]))
        self.assertTrue(all(u["task"] in TASKS+HELDOUT for u in plan("full")["units"]))
        self.assertEqual(len({(u["task"],u["repeat"]) for u in plan("full")["units"]}),30)

    def test_synthetic_evaluator_semantics(self):
        c=candidates(121,5)
        for task in TASKS:
            for point in c:
                self.assertEqual(truth(task,point),truth_for_evaluation(task,point))
        for task in HELDOUT:
            for point in c:
                row=truth_for_evaluation(task,point)
                self.assertTrue(all(math.isfinite(v) for k,v in row.items() if k!="feasible"))
                self.assertEqual(row["feasible"],row["region"]<=0 and row["resource"]<=0)
        with self.assertRaises(ValueError):
            truth_for_evaluation("tradeoff_s8",c[0])


class ObjectiveContractTests(unittest.TestCase):
    def test_censored_objective_not_imputed(self):
        c=candidates(123,1)[0]
        truth_record=dict(region=0.5,resource=-0.2,objective=-9999.0,feasible=False)
        obs=observed(c,truth_record)
        self.assertEqual(obs["region"]["kind"],"interval")
        self.assertEqual(obs["objective"]["kind"],"missing")
        self.assertFalse(known_label(obs,"region"))
        self.assertEqual(outcome_state([obs])["exact_objective_count"],0)
        self.assertIsNone(outcome_state([obs])["best_exact_feasible"])

    def test_feasible_vs_infeasible_exact_objective(self):
        c=candidates(124,3)
        first=observed(c[0],dict(region=-0.3,resource=0.3,objective=-100.0,feasible=False))
        second=observed(c[1],dict(region=-0.1,resource=-0.4,objective=5.0,feasible=True))
        third=observed(c[2],dict(region=0.4,resource=-0.4,objective=-200.0,feasible=False))
        state=outcome_state([first,second,third])
        self.assertEqual(state["best_exact_feasible"],5.0)
        self.assertEqual(state["exact_objective_count"],2)
        self.assertEqual(state["missing_objective_count"],1)
        self.assertEqual(exact_feasible_objectives([first,second,third]),[5.0])


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.cs=candidates(5555,4)
        self.f=dict(probabilities={
            "region":[0.9,0.4,0.6,0.4],
            "resource":[0.9,0.8,0.5,0.8],
            "joint":[0.81,0.32,0.30,0.32],
        })
        self.objective=dict(mean=[10.0,0.0,8.0,0.0],
                            variance=[0.01,0.01,0.01,0.01])

    def test_entropy_ei_and_ties(self):
        self.assertEqual(entropy(0.0),0.0)
        self.assertGreater(entropy(0.5),entropy(0.1))
        self.assertAlmostEqual(expected_improvement(4.0,0.0,5.0),1.0)
        self.assertEqual(expected_improvement(6.0,0.0,5.0),0.0)
        self.assertGreater(expected_improvement(4.0,0.04,5.0),0)

    def test_constraint_ei_target_alignment(self):
        scores,rule=score_candidates("constrained_ei",self.f,self.objective,5.0)
        self.assertEqual(rule,"constrained_expected_improvement")
        self.assertGreater(scores[1],scores[0])
        self.assertEqual(scores[1],scores[3])
        new,_=score_candidates("objective_weighted_boundary",self.f,self.objective,5.0)
        self.assertGreater(new[1],new[0])
        self.assertEqual(new[1],new[3])

    def test_cold_fallback_does_not_hallucinate_objective(self):
        scores,rule=score_candidates("constrained_ei",self.f,None,None)
        self.assertEqual(scores,self.f["probabilities"]["joint"])
        self.assertEqual(rule,"cold_feasibility")
        scores,rule=score_candidates("objective_weighted_boundary",self.f,None,None)
        self.assertEqual(len(scores),len(self.cs))
        self.assertIn("cold",rule)

    def test_selection_replay(self):
        for method in METHODS:
            a,receipt=choose(method,self.cs,self.f,self.objective,5.0,99,5)
            b,replay=choose(method,self.cs,self.f,self.objective,5.0,99,5)
            self.assertEqual(a,b)
            self.assertEqual(receipt,replay)
            self.assertEqual(receipt["selected_candidate_sha256"],
                             __import__("risk_learning.core",fromlist=["digest"]).digest(self.cs[a]))
        with self.assertRaises(ValueError):
            score_candidates("boundary",dict(probabilities={
                "region":[0.8],"resource":[0.8],"joint":[0.9]}),None,None)

    def test_no_imputed_objective_pairing(self):
        lhs=dict(best_exact_feasible=None)
        rhs=dict(best_exact_feasible=5.0)
        self.assertEqual(compare_objective(lhs,rhs),(-1,None))
        self.assertEqual(compare_objective(lhs,lhs),(0,None))
        self.assertEqual(compare_objective(rhs,dict(best_exact_feasible=6.0)),(1,-1.0))


if __name__=="__main__":
    unittest.main()
