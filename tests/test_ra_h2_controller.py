"""Unit and adversarial tests for RA-LSBO-H2 delayed prequential risk control.

No aircraft, GP fitting, VAE, or historical pilot data is touched.
"""
from statistics import NormalDist
from unittest.mock import patch
import json
import math
import unittest

from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.experiments.simplified_aircraft.h2_controller import (
    CausalRiskController,Gaussian,H2Settings,proper_evidence_score)
from ra_lsbo.experiments.simplified_aircraft import h2_synthetic as bench


def normal_from_probability(p):
    return Gaussian(-NormalDist().inv_cdf(p),1.)


def forecast(onehot=.6,latent=.4):
    return {"onehot":normal_from_probability(onehot),
            "latent":normal_from_probability(latent)}


def exact(y,i):
    return OutputObservation("exact",f"e{i}",value=-.1 if y else +.1)


class H2ControllerTests(unittest.TestCase):
    def test_precommit_is_earlier_than_outcome_update(self):
        c=CausalRiskController()
        s=c.state()
        d=c.begin(0,"candidate_0","history_sha_before_t",forecast())
        self.assertEqual(d["selected_view"],"onehot")
        self.assertEqual(d["decision_state_before"],s)
        self.assertFalse(d["decision_state_before"]["awaiting_observation"])
        self.assertEqual(c.state()["resolved_pairs"],0)
        with self.assertRaisesRegex(RuntimeError,"unsettled"):
            c.begin(1,"candidate_1","future_prefix",forecast())
        event=c.settle(0,"candidate_0",exact(True,0))
        self.assertTrue(event["feedback_pair_scored"])
        self.assertEqual(c.state()["resolved_pairs"],1)
        self.assertEqual(len(c.journal()["events"]),1)
        self.assertTrue(c.state()["onehot_weight"]>0)

    def test_future_response_kind_is_unavailable_at_decision(self):
        one=CausalRiskController()
        two=CausalRiskController()
        f=forecast(.3,.7)
        pre=one.begin(0,"same_candidate","same_history",f)
        alternate=two.begin(0,"same_candidate","same_history",f)
        self.assertEqual(pre,alternate)
        one.settle(0,"same_candidate",OutputObservation("exact","actual",value=.2))
        two.settle(0,"same_candidate",OutputObservation(
            "interval","censored",lower=0.,lower_inclusive=False))
        self.assertEqual(one.state()["resolved_pairs"],two.state()["resolved_pairs"])
        self.assertAlmostEqual(one.state()["onehot_weight"],two.state()["onehot_weight"])

    def test_later_latent_superiority_switches_after_resolved_history(self):
        c=CausalRiskController(H2Settings(minimum_resolved_pairs=3))
        heads=[]
        for t in range(12):
            # Onehot persistently overestimates feasibility of a failing
            # design; the latent forecast is accurate.
            heads.append(c.begin(t,str(t),f"h{t}",forecast(.95,.05))["selected_view"])
            c.settle(t,str(t),exact(False,t))
        self.assertEqual(heads[:3],["onehot"]*3)
        self.assertIn("latent",heads[3:])
        self.assertEqual(c.state()["resolved_pairs"],12)
        self.assertEqual(c.state()["currently_preferred"],"latent")
        self.assertTrue(0<c.state()["onehot_weight"]<.5)

    def test_censored_resolved_observation_updates_brier(self):
        c=CausalRiskController()
        c.begin(0,"A","p",forecast(.8,.3))
        event=c.settle(0,"A",OutputObservation("interval","upper",upper=0.))
        self.assertTrue(event["resolved_feasible"])
        self.assertTrue(event["feedback_pair_scored"])
        self.assertAlmostEqual(event["brier_by_view"]["onehot"],.04,places=8)
        self.assertEqual(event["proper_evidence_score_diagnostic"]["onehot"]["score_type"],
                         "interval_mass_nll")

    def test_ambiguous_interval_and_missing_do_not_update(self):
        c=CausalRiskController()
        c.begin(0,"A","p0",forecast())
        e=c.settle(0,"A",OutputObservation("interval","ambiguous",lower=-1.,upper=1.))
        self.assertIsNone(e["resolved_feasible"])
        self.assertFalse(e["feedback_pair_scored"])
        w=c.state()["onehot_weight"]
        c.begin(1,"B","p1",forecast())
        e=c.settle(1,"B",OutputObservation("missing","missing",reason="unknown"))
        self.assertIsNone(e["resolved_feasible"])
        self.assertEqual(c.state()["resolved_pairs"],0)
        self.assertEqual(w,c.state()["onehot_weight"])

    def test_single_head_outage_no_pair_update(self):
        c=CausalRiskController()
        d=c.begin(0,"A","p",{"onehot":None,"latent":normal_from_probability(.7)})
        self.assertEqual(d["selected_view"],"latent")
        self.assertEqual(d["reason"],"one_valid_head_no_brier_pair")
        e=c.settle(0,"A",exact(True,0))
        self.assertFalse(e["feedback_pair_scored"])
        self.assertEqual(c.state()["resolved_pairs"],0)

    def test_both_heads_outage_does_not_invent_gaussian_prior(self):
        c=CausalRiskController()
        d=c.begin(0,"A","p",{"onehot":None,"latent":None})
        self.assertIsNone(d["selected_view"])
        self.assertEqual(d["reason"],"no_valid_soc_head_registered_coverage_required")
        e=c.settle(0,"A",exact(False,0))
        self.assertFalse(e["feedback_pair_scored"])
        self.assertEqual(c.state()["resolved_pairs"],0)

    def test_mismatch_double_settlement_and_order_are_rejected(self):
        c=CausalRiskController()
        with self.assertRaises(ValueError):
            c.begin(1,"A","p",forecast())
        c.begin(0,"A","p",forecast())
        with self.assertRaises(ValueError):
            c.settle(0,"different_candidate",exact(True,0))
        self.assertTrue(c.state()["awaiting_observation"])
        c.settle(0,"A",exact(True,0))
        with self.assertRaises(RuntimeError):
            c.settle(0,"A",exact(True,0))

    def test_brier_bound_and_no_arbitrary_nll_mixing(self):
        c=CausalRiskController()
        for t in range(4):
            c.begin(t,str(t),"prefix",forecast(.99,.01))
            e=c.settle(t,str(t),OutputObservation("interval",f"c{t}",lower=0.,lower_inclusive=False))
            self.assertTrue(0<=e["brier_by_view"]["onehot"]<=1)
            self.assertTrue(0<=e["brier_by_view"]["latent"]<=1)
            self.assertEqual(e["proper_evidence_score_diagnostic"]["latent"]["score_type"],
                             "interval_mass_nll")
        self.assertTrue(0<c.state()["onehot_weight"]<1)
        # Hard log-odds cap prevents any weight from becoming exact 0 or 1.
        self.assertLess(c.state()["onehot_weight"],.99)

    def test_exact_density_nll_can_be_negative(self):
        g=Gaussian(-.1,.0001)
        x=proper_evidence_score(g,OutputObservation("exact","e",value=-.1))
        self.assertLess(x["nll"],0)
        self.assertEqual(x["score_type"],"exact_density_nll")

    def test_settings_reject_unbounded_or_noninteger_history(self):
        for kw in ({"learning_rate":float("nan")},{"prior_onehot":1.},
                   {"max_steps":True},{"minimum_resolved_pairs":0}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                H2Settings(**kw)

    def test_journal_copy_does_not_mutate_next_decision(self):
        c=CausalRiskController()
        c.begin(0,"A","p",forecast(.2,.8))
        journal=c.journal()
        journal["pending"]["expert_moments"]["onehot"]["mean"]=900.
        self.assertNotEqual(c.journal()["pending"]["expert_moments"]["onehot"]["mean"],900.)
        c.settle(0,"A",exact(False,0))

    def test_synthetic_hierarchy_conditional_activation_and_repeatability(self):
        a=bench._candidate_pool(7301,0)
        b=bench._candidate_pool(7301,0)
        self.assertEqual(a,b)
        self.assertEqual(len(a),bench.POOL)
        for x in a:
            self.assertEqual(x["conditional_z"] is None,x["mode"]=="eco")
            self.assertIn(x["family"],bench.FAMILIES)
            self.assertIn(x["level"],bench.LEVELS)
        with patch.object(bench,"_true_margin",side_effect=RuntimeError("future truth blocked")):
            self.assertTrue(math.isfinite(bench._expert_predictions(a[0])["latent"].mean))

    def test_complete_synthetic_protocol_has_no_gp_or_aircraft(self):
        payload=bench.run_benchmark()
        self.assertEqual(len(payload["trials"]),len(bench.CASES)*len(bench.SEEDS)*len(bench.ARMS))
        self.assertEqual(payload["synthetic_oracle_calls"],
            len(bench.CASES)*len(bench.SEEDS)*len(bench.ARMS)*bench.HORIZON)
        self.assertEqual(payload["GP_fits"],0)
        self.assertEqual(payload["VAE_fits"],0)
        self.assertEqual(payload["aircraft_calls"],0)
        for row in payload["trials"]:
            self.assertEqual(row["synthetic_calls"],bench.HORIZON)
            self.assertEqual(row["head_outages"],2)
            if row["method"]=="h2_causal":
                self.assertLess(row["h2_online_brier_updates"],bench.HORIZON)
                self.assertGreater(row["h2_online_brier_updates"],0)
                self.assertTrue(0<row["terminal_H2_state"]["onehot_weight"]<1)


if __name__=="__main__":
    unittest.main()
