"""No-model tests of the RA-LSBO-H frozen cross-policy panel scoring."""

import math
import unittest

from ra_lsbo.risk_core.observations import OutputObservation
from tools.ra_h_soc_panel_run import _score,_paired_report,_OriginalPredictionProxy


class FrozenSocPanelTests(unittest.TestCase):
    def test_exact_gaussian_nll_is_density_not_mass(self):
        o=OutputObservation("exact","e",value=0.1)
        a=_score({"mean":0.,"variance":1.},o)
        self.assertEqual(a["score_type"],"exact_SOC_density_NLL")
        self.assertAlmostEqual(a["score_nll"],.5*math.log(2*math.pi)+.005)
        self.assertEqual(a["central_95_covered"],True)
        self.assertAlmostEqual(a["brier"],(.5-0.)**2)

    def test_censored_interval_uses_normal_mass(self):
        o=OutputObservation("interval","i",lower=0.)
        a=_score({"mean":0.,"variance":1.},o)
        self.assertEqual(a["score_type"],"interval_SOC_mass_NLL")
        self.assertAlmostEqual(a["score_nll"],math.log(2.))
        self.assertIsNone(a["central_95_covered"])
        self.assertIsNone(a["standardized_error"])

    def test_no_prediction_is_not_fabricated(self):
        o=OutputObservation("exact","e",value=-.3)
        self.assertEqual(_score(None,o)["status"],"head_unavailable")
        self.assertEqual(_score({"mean":0.,"variance":0.},o)["status"],
                         "zero_or_invalid_variance")

    def test_paired_scores_keep_exact_and_interval_disjoint(self):
        rows=[{"case":"aircraft_sparse","seed":9961,"iteration":1,"targets":[
            {"policy":"hybrid_onehot","excluded_as_training_design":False,
             "soc_evidence_kind":"exact","predictions":{
                 "hierarchical_onehot":{"status":"ok","score_nll":1.},
                 "frozen_corpus_latent":{"status":"ok","score_nll":3.}}},
            {"policy":"hybrid_onehot","excluded_as_training_design":False,
             "soc_evidence_kind":"interval","predictions":{
                 "hierarchical_onehot":{"status":"ok","score_nll":.5},
                 "frozen_corpus_latent":{"status":"ok","score_nll":2.5}}},
            {"policy":"random","excluded_as_training_design":True,
             "soc_evidence_kind":"exact","predictions":{}},
        ]}]
        data=_paired_report(rows)
        grp=data["groups"]
        self.assertEqual(grp["aircraft_sparse/exact/hybrid_onehot"]["pooled_matched_records"],1)
        self.assertEqual(grp["aircraft_sparse/exact/hybrid_onehot"][
            "seed_mean_of_mean_differences"],2.)
        self.assertEqual(grp["aircraft_sparse/interval/hybrid_onehot"][
            "seed_mean_of_mean_differences"],2.)
        self.assertEqual(len(grp),2)

    def test_saved_baseline_proxy_does_not_refit(self):
        proxy=_OriginalPredictionProxy({"mean":-.1,"variance":.01})
        self.assertEqual(proxy.predict(None),{"mean":[-.1],"variance":[.01]})

if __name__=="__main__":
    unittest.main()
