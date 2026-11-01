"""RA-LSBO risk-decoupled candidate: unit and protocol gates.

No aircraft or RCAIDE evaluator is invoked in these tests.
"""
from unittest.mock import patch
import unittest

import numpy as np
from numpy.testing import assert_allclose

from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.experiments.simplified_aircraft import dualview_protocol as pr
from ra_lsbo.experiments.simplified_aircraft import dualview_optimizer as dv
from ra_lsbo.experiments.simplified_aircraft.dualview_metrics import soc_prequential
from types import SimpleNamespace
import math


class DualViewTests(unittest.TestCase):
    def test_prospective_roster_and_budget(self):
        self.assertEqual(pr.ARMS,("random","onehot_stable","corpus_stable",
                                  "hybrid_onehot","hybrid_guard"))
        self.assertEqual(pr.CORPUS_ARMS,("corpus_stable","hybrid_onehot","hybrid_guard"))
        self.assertEqual(pr.budget_for(pr.profile_for("smoke"))["maximum_model_calls"],42)
        self.assertEqual(pr.budget_for(pr.profile_for("pilot"))["maximum_model_calls"],856)
        self.assertEqual(pr.profile_for("pilot")["seeds"],[9961,9962,9963,9964])
        self.assertEqual(pr.profile_for("pilot")["horizon"],18)
        for method in pr.ARMS[1:]:
            self.assertEqual(pr.model_settings(method,pr.profile_for("smoke")).continuous_grid,
                             (.05,.15,.4,1.))
            self.assertEqual(pr.model_settings(method,pr.profile_for("smoke")).discrete_grid,
                             (.5,1.5))
        for name in pr.ARMS[1:]:
            spec=pr.method_spec(name,pr.profile_for("pilot"))
            self.assertTrue(spec["gp_stable_tail_moments"])
            self.assertEqual(spec["gp_grid_continuous"],[.05,.15,.4,1.])
            self.assertEqual(spec["gp_grid_discrete"],[.5,1.5])
            self.assertEqual(spec["failure_policy"],
                "on_no_valid_response_head_deterministic_coverage_no_prior_invention")
        self.assertEqual(pr.method_spec("hybrid_onehot",pr.profile_for("pilot"))[
            "response_policy"]["objective"],"corpus_latent")

    def test_risk_intersection_uses_less_optimistic_probability(self):
        onehot={"mean":np.array([-.4,.6,0.]),
                "variance":np.array([.01,.01,.04])}
        latent={"mean":np.array([.2,-.3,0.]),
                "variance":np.array([.01,.01,.04])}
        output=dv.conservative_soc_moments(onehot,latent)
        assert_allclose(output["mean"],[.2,.6,0.])
        assert_allclose(output["variance"],[.01,.01,.04])
        assert_allclose(output["selected_latent"],[1,0,0])
        self.assertTrue(np.all(output["view_disagreement_feasibility"]>=0))
        self.assertTrue(np.all(output["view_disagreement_feasibility"]<=1))

    def test_degenerate_gaussian_risk_is_finite(self):
        one={"mean":np.array([-.2,.2]),"variance":np.zeros(2)}
        latent={"mean":np.array([.1,-.1]),"variance":np.ones(2)}
        pred=dv.conservative_soc_moments(one,latent)
        self.assertTrue(np.isfinite(pred["mean"]).all())
        self.assertTrue(np.isfinite(pred["variance"]).all())
        self.assertEqual(pred["selected_latent"].tolist(),[1,0])

    def test_mixed_likelihood_normalization_preserves_censor_bounds(self):
        obs=[OutputObservation("exact","a",value=-.4),
             OutputObservation("interval","b",lower=.2),
             OutputObservation("missing","c",reason="not_observed")]
        center,scale,norm=dv._normalize(obs,objective=False)
        self.assertEqual(center,0.)
        self.assertEqual(scale,.4)
        self.assertEqual(norm[1].kind,"interval")
        self.assertAlmostEqual(norm[1].lower,.5)
        self.assertIsNone(norm[1].upper)
        self.assertEqual(norm[2].kind,"missing")

    def test_all_gp_kernels_fail_explicitly_with_original_error_records(self):
        x=np.array([[0.,0.,0.],[.2,.1,.3],[.5,.7,.2]])
        obs=[OutputObservation("exact",str(i),value=float(i)) for i in range(3)]
        with patch.object(dv,"EvidenceGP",side_effect=RuntimeError("EP fixture fail")):
            with self.assertRaises(dv.ModelNumericalFailure) as cm:
                dv.StableGroupedHead(x,obs,continuous_count=1,seed=7,name="soc")
        self.assertEqual(cm.exception.head,"soc")
        self.assertEqual(len(cm.exception.failures),8)
        self.assertTrue(all(e["error_message"]=="EP fixture fail"
                            for e in cm.exception.failures))

    def test_one_head_success_exact_only_keeps_original_kernel_roster(self):
        x=np.array([[0.,0.,0.],[.2,.1,.3],[.5,.7,.2],[.8,.6,.4]])
        obs=[OutputObservation("exact",str(i),value=float(i)/10) for i in range(4)]
        h=dv.StableGroupedHead(x,obs,continuous_count=1,seed=7,name="exact")
        self.assertEqual(len(h.candidates),8)
        self.assertTrue(any(p["status"]=="ok" for p in h.candidates))
        result=h.predict(x)
        self.assertTrue(np.isfinite(result["mean"]).all())
        self.assertTrue(np.isfinite(result["variance"]).all())
        self.assertTrue(np.all(result["variance"]>=0))

    def test_fallback_requires_explicit_model_error_and_uses_fixed_coverage(self):
        class Stub:
            pending=None
            next_iteration=0
            seed=19
            problem=object()
            def _finish(self,candidate,records,meta,prefix):
                self.finished=True
                return candidate,meta
        stub=Stub()
        pool=[{"x":1},{"x":2}]
        failure=dv.ModelNumericalFailure("constraint:soc",[
            {"continuous_length":.05,"discrete_length":.5,
             "status":"RuntimeError","error_message":"test"}])
        with patch.object(dv,"filter_near",return_value=(pool,[0,1],[])), \
             patch.object(dv,"coverage_pick",return_value=(1,{"rule":"fixture"})):
            selected,meta=dv._registered_failure_proposal(stub,[],pool,0,failure)
        self.assertEqual(selected,pool[1])
        self.assertTrue(stub.finished)
        self.assertEqual(meta["coverage_reason"],"model_failure")
        self.assertEqual(meta["model_failure"]["head"],"constraint:soc")
        self.assertEqual(meta["model_failure"]["candidate_failures"],failure.failures)
        self.assertFalse(meta["posterior_fit_skipped"])

    def test_prequential_scoring_separates_density_from_interval_mass(self):
        candidate={"x":1}
        exact=SimpleNamespace(candidate=candidate,
            observation=SimpleNamespace(constraint=lambda _:OutputObservation(
                "exact","e",value=.1)))
        interval=SimpleNamespace(candidate=candidate,
            observation=SimpleNamespace(constraint=lambda _:OutputObservation(
                "interval","i",lower=0.)))
        def prop():
            return {"candidate":candidate,"metadata":{"selected_index":0,
                "coverage_reason":None,
                "forecasts":{"constraints":{"soc":{"mean":[0.],"variance":[1.],"scale":1.}}}}}
        report=soc_prequential([prop(),prop()],[exact,interval])
        self.assertEqual(report["exact"],1)
        self.assertEqual(report["interval"],1)
        self.assertAlmostEqual(report["exact_density_nll_mean"],
            .5*math.log(2*math.pi)+.005,places=9)
        self.assertAlmostEqual(report["interval_mass_nll_mean"],math.log(2),places=9)
        self.assertIsNotNone(report["exact_95pct_coverage_numerator"])
        self.assertEqual(report["model_failure_fallback"],0)

    def test_prequential_skips_genuine_unfitted_coverage(self):
        row=SimpleNamespace(candidate={"x":1},
            observation=SimpleNamespace(constraint=lambda _:OutputObservation("exact","e",value=1.)))
        report=soc_prequential([{"candidate":{"x":1},
            "metadata":{"coverage_reason":"model_failure"}}],[row])
        self.assertEqual(report["skipped_no_forecast"],1)
        self.assertEqual(report["model_failure_fallback"],1)
        self.assertIsNone(report["exact_density_nll_mean"])

    def test_structure_sampler_not_changed_and_hybrid_has_no_hidden_weights(self):
        spec=pr.method_spec("hybrid_guard",pr.profile_for("pilot"))
        self.assertEqual(spec["response_policy"]["soc_min_feasibility"],
                         "both_latent_and_onehot")
        self.assertEqual(spec["corpus_training"]["draws"],2000)
        self.assertEqual(spec["corpus_training"]["fit_before_online_indices"],[0,8,16])
        self.assertNotIn("adaptive_weight",spec)


if __name__=="__main__":
    unittest.main()
