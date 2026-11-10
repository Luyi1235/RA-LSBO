import unittest, math
from pr86b_protocol import plan, METHODS
from pr86b_acquisition import (bald_binary_information, anchor_prevalence,
    anchor_intercept_calibrate, calibrate_prediction, choose, metrics, brier_skill_score)
from risk_learning.core import candidates, observed, truth

class ProtocolTests(unittest.TestCase):
    def test_frozen_budgets(self):
        self.assertEqual(plan("ci")["total_math_calls"],272)
        self.assertEqual(plan("smoke")["total_math_calls"],1920)
        self.assertEqual(plan("full")["total_math_calls"],12672)
        self.assertEqual(plan("full")["expected_model_refits"],3330)
        self.assertEqual(plan("full")["expected_gp_heads"],6660)
        self.assertEqual(plan("full")["expected_gp_kernel_constructions"],53280)
    def test_profile_seeds_are_disjoint(self):
        plans=[plan(x) for x in ("ci","smoke","full")]
        seeds=[(u["anchor_seed"],u["pool_seed"],u["test_seed"]) for p in plans for u in p["units"]]
        flat=[x for row in seeds for x in row]
        self.assertEqual(len(flat),len(set(flat)))
    def test_bald_information_is_nonnegative_and_bounded(self):
        for m in (-2.,0.,1.):
            for v in (0.,.1,1.,4.):
                value=bald_binary_information(m,v,1.)
                self.assertGreaterEqual(value,0.)
                self.assertLessEqual(value,math.log(2.)+1e-12)
        self.assertEqual(bald_binary_information(0.,0.,1.),0.)
    def test_anchor_prevalence_uses_only_observed_packets(self):
        cs=candidates(12345,12)
        rows=[observed(c,truth("tradeoff_s0",c)) for c in cs]
        rates=anchor_prevalence(rows)
        self.assertEqual(rates["n"],12)
        self.assertTrue(0<rates["joint"]<1)
        self.assertNotIn("truth",rows[0])
    def test_intercept_calibration_matches_anchor_rate(self):
        raw=[.05,.1,.2,.4,.6,.8,.95]
        adjusted,shift=anchor_intercept_calibrate(raw,.17)
        self.assertTrue(math.isfinite(shift))
        self.assertAlmostEqual(sum(adjusted)/len(adjusted),.17,places=10)
    def test_calibration_preserves_independent_joint_contract(self):
        p={"probabilities":{"region":[.2,.7],"resource":[.3,.8],"joint":[.06,.56]}}
        q={"region":.4,"resource":.6,"joint":.3}
        calibrated,_=calibrate_prediction(p,q)
        self.assertEqual(calibrated["joint"],[a*b for a,b in zip(calibrated["region"],calibrated["resource"])])
    def test_all_acquisition_methods_return_bound_candidate(self):
        cs=candidates(991,24)
        forecast={"probabilities":{"region":[.4]*24,"resource":[.5]*24,"joint":[.2]*24},
            "gaussian":{"region":{"mean":[0.]*24,"variance":[1.]*24,"observation_noise_variance":1.},
                        "resource":{"mean":[0.]*24,"variance":[1.]*24,"observation_noise_variance":0.}}}
        for method in METHODS:
            i,r=choose(method,cs,cs[:3],forecast,100,0)
            self.assertIn(cs[i],cs)
            self.assertEqual(r["selected_candidate_sha256"],__import__("risk_learning.core",fromlist=["digest"]).digest(cs[i]))
            self.assertEqual(len(r["scores"]),len(cs))
    def test_reserved_calibration_anchor_uses_independent_pool(self):
        main=candidates(993,20)
        anchor_pool=candidates(994,4)
        f={"probabilities":{"region":[.5]*20,"resource":[.5]*20,"joint":[.25]*20},
           "gaussian":{"region":{"mean":[0.]*20,"variance":[1.]*20,"observation_noise_variance":1.},
                       "resource":{"mean":[0.]*20,"variance":[1.]*20,"observation_noise_variance":0.}}}
        i,receipt=choose("calibration_aware_voi",main,main[:2],f,88,3,
                         calibration_candidates=anchor_pool)
        self.assertEqual(receipt["selected_from"],"calibration_anchor_pool")
        self.assertEqual(receipt["selected_candidate_sha256"],
                         __import__("risk_learning.core",fromlist=["digest"]).digest(anchor_pool[i]))
        self.assertNotIn(anchor_pool[i],main)
    def test_random_and_reserved_anchor_are_deterministic(self):
        cs=candidates(992,20)
        f={"probabilities":{"region":[.5]*20,"resource":[.5]*20,"joint":[.25]*20},
           "gaussian":{"region":{"mean":[0.]*20,"variance":[1.]*20,"observation_noise_variance":1.},
                       "resource":{"mean":[0.]*20,"variance":[1.]*20,"observation_noise_variance":0.}}}
        a=choose("iid_random",cs,cs[:2],f,88,2)
        b=choose("iid_random",cs,cs[:2],f,88,2)
        self.assertEqual(a,b)
        x=choose("calibration_aware_voi",cs,cs[:2],f,88,3)
        self.assertEqual(x[1]["rule"],"reserved_random_calibration_anchor")
    def test_metrics_and_bss(self):
        m=metrics([True,False,True,False],[.8,.2,.6,.4])
        self.assertEqual(m["auroc"],1.)
        self.assertGreater(m["brier"],0.)
        self.assertGreater(brier_skill_score(.05,.08),0.)
        self.assertLess(brier_skill_score(.1,.08),0.)
    def test_single_class_auroc_is_null(self):
        self.assertIsNone(metrics([True,True],[.8,.4])["auroc"])

if __name__=="__main__":
    unittest.main()
