"""PR86-D frozen protocol, nested sample, objective observability and selector tests."""
import math
import unittest

from risk_learning.core import TASKS, candidates, observed
from pr86d_protocol import plan, METHODS, ANCHOR_SIZES
from pr86c_evaluator import truth_for_evaluation
from pr86c_models import outcome_state, exact_feasible_objectives
from pr86c_acquisition import choose
from pr86c_analysis import compare_objective


class ProtocolTests(unittest.TestCase):
    def test_frozen_budget(self):
        ci=plan("ci")
        full=plan("full")
        self.assertEqual(ci["total_math_calls"],388)
        self.assertEqual(ci["shared_anchor_calls"],24)
        self.assertEqual(ci["arm_online_calls"],300)
        self.assertEqual(ci["heldout_test_calls"],64)
        self.assertEqual(ci["expected_risk_refits"],310)
        self.assertEqual(full["total_math_calls"],25080)
        self.assertEqual(full["shared_anchor_calls"],720)
        self.assertEqual(full["arm_online_calls"],9000)
        self.assertEqual(full["heldout_test_calls"],15360)
        self.assertEqual(full["expected_risk_refits"],9300)
        self.assertEqual(full["expected_risk_heads"],18600)
        self.assertEqual(full["expected_risk_kernel_constructions"],148800)
        self.assertEqual(len(full["units"]),30)
        self.assertEqual(len(METHODS),5)

    def test_equal_total_budget_and_nested_anchor_prefix(self):
        p=plan("full")
        self.assertEqual(ANCHOR_SIZES,(12,24))
        self.assertEqual(p["online_per_method_by_anchor_size"],{"12":36,"24":24})
        self.assertEqual({k+v for k,v in [(12,36),(24,24)]},{48})
        for unit in p["units"]:
            a=candidates(unit["anchor_seed"],24)
            self.assertEqual(len(a[:12]),12)
            self.assertEqual(a[:12],candidates(unit["anchor_seed"],12))
            self.assertEqual(len(a[12:]),12)

    def test_seed_blocks_are_disjoint(self):
        ci=plan("ci")
        full=plan("full")
        def seeds(p):
            return {v for u in p["units"] for v in
                    (u["anchor_seed"],u["pool_seed"],u["test_seed"])}
        self.assertFalse(seeds(ci)&seeds(full))
        self.assertEqual(len(seeds(full)),len(full["units"])*3)

    def test_known_task_family_and_new_seed_profile(self):
        full=plan("full")
        self.assertEqual(set(full["tasks"]),set(TASKS+("tradeoff_s4","tradeoff_s5","tradeoff_s6","tradeoff_s7")))
        self.assertTrue(all(u["anchor_seed"]>=1190000 for u in full["units"]))


class ObjectiveObservabilityTests(unittest.TestCase):
    def test_region_violation_hides_objective_but_resource_violation_does_not(self):
        cs=candidates(12001,2)
        region_censored=observed(cs[0],dict(region=0.2,resource=-0.2,
                                           objective=-999.0,feasible=False))
        resource_infeasible=observed(cs[1],dict(region=-0.2,resource=0.2,
                                                objective=5.0,feasible=False))
        self.assertEqual(region_censored["objective"]["kind"],"missing")
        self.assertEqual(resource_infeasible["objective"]["kind"],"exact")
        state=outcome_state([region_censored,resource_infeasible])
        self.assertEqual(state["exact_objective_count"],1)
        self.assertFalse(state["feasible_found"])
        self.assertEqual(exact_feasible_objectives([region_censored,resource_infeasible]),[])

    def test_exact_feasible_incumbent_requires_both_constraints(self):
        cs=candidates(12002,3)
        rows=[
            observed(cs[0],dict(region=-0.1,resource=0.3,objective=1.0,feasible=False)),
            observed(cs[1],dict(region=0.3,resource=-0.1,objective=0.0,feasible=False)),
            observed(cs[2],dict(region=-0.2,resource=-0.2,objective=4.0,feasible=True)),
        ]
        state=outcome_state(rows)
        self.assertEqual(state["best_exact_feasible"],4.0)
        self.assertEqual(state["exact_objective_count"],2)
        self.assertEqual(state["exact_feasible_count"],1)

    def test_heldout_tasks_are_reproducible_but_distinct_from_released(self):
        c=candidates(12003,1)[0]
        a=truth_for_evaluation("tradeoff_s4",c)
        b=truth_for_evaluation("tradeoff_s4",c)
        self.assertEqual(a,b)
        self.assertTrue(all(math.isfinite(a[k]) for k in ("region","resource","objective")))


class AcquisitionAndPairingTests(unittest.TestCase):
    def test_all_methods_replay_deterministically(self):
        cs=candidates(12004,16)
        forecast={"probabilities":{"region":[.2,.4,.6,.8]*4,
            "resource":[.8,.6,.4,.2]*4,"joint":[.16,.24,.24,.16]*4}}
        objective={"mean":[float(i) for i in range(16)],
                   "variance":[.2]*16}
        for method in METHODS:
            a,ra=choose(method,cs,forecast,objective,5.0,123,7)
            b,rb=choose(method,cs,forecast,objective,5.0,123,7)
            self.assertEqual(a,b)
            self.assertEqual(ra,rb)
            self.assertEqual(ra["selected_candidate_sha256"],
                __import__("risk_learning.core",fromlist=["digest"]).digest(cs[a]))

    def test_null_pairing_never_invents_objective_penalty(self):
        no=dict(best_exact_feasible=None)
        feasible=dict(best_exact_feasible=4.0)
        self.assertEqual(compare_objective(no,no),(0,None))
        self.assertEqual(compare_objective(feasible,no),(1,None))
        self.assertEqual(compare_objective(no,feasible),(-1,None))
        self.assertEqual(compare_objective(dict(best_exact_feasible=3.0),
                                            dict(best_exact_feasible=4.0)),
                         (1,-1.0))


class VerifierContractTests(unittest.TestCase):
    def test_cold_verifier_accepts_final_and_checkpoint_receipts(self):
        """Prevent NameError when a stored fit has no acquisition proposal."""
        from risk_learning.core import digest
        from risk_learning.native import training_payload
        from pr86d_verify import check_fit

        cs=candidates(12005,3)
        current=[observed(c,dict(region=1.0,resource=-.2,
             objective=-100.,feasible=False)) for c in cs[:2]]
        available=[cs[2]]
        test=[]
        diag={"candidates":[{"status":"ok"} for _ in range(8)]}
        risk_receipt=dict(status="OK",objective_labels_used=False,
            privileged_training=False,training_rows=2,
            training_input_sha256=digest(training_payload(current,"native_censored")),
            query_ids=[digest(cs[2])],heads={"region":diag,"resource":diag})
        objective_receipt=dict(status="INSUFFICIENT_EXACT_OBJECTIVES",
            training_rows=2,exact_training_rows=0,
            privileged_training=False,missing_labels_imputed=False,
            training_input_sha256=digest([]),query_ids=[digest(cs[2])])
        # Independent checkpoints and the final fit have no decision object.
        check_fit(current,available,test,risk_receipt,objective_receipt)
        probabilities={"region":[.5],"resource":[.8],"joint":[.4]}
        forecast={"probabilities":probabilities}
        proposal=dict(candidate_forecast=forecast,
                      forecast_sha256=digest(forecast),objective_forecast=None)
        check_fit(current,available,test,risk_receipt,objective_receipt,proposal)
        bad=dict(proposal,candidate_forecast={
            "probabilities":{"region":[.5],"resource":[.8],"joint":[.2]}})
        with self.assertRaisesRegex(ValueError,"Joint risk"):
            check_fit(current,available,test,risk_receipt,objective_receipt,bad)


if __name__=="__main__":
    unittest.main()
