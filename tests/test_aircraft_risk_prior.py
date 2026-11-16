import unittest

import numpy as np

from aircraft_risk_prior import SharedRiskEvidenceGP, layout_features
from aircraft_risk_protocol import plan
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings
from ra_lsbo.risk_core.observations import OutputObservation


class AircraftRiskPriorTests(unittest.TestCase):
    def test_layout_kernel_has_unit_variance_and_is_positive_semidefinite(self):
        inputs = layout_features(
            np.array([[0.], [10.], [10.]]),
            [{"discrete": {"prop_layout": layout}} for layout in ("wingtip", "wingtip", "distributed")],
        )
        missing = [OutputObservation("missing", str(i), reason="unobserved") for i in range(3)]
        settings = MixedGPSettings(length_scale=1.)
        layout = SharedRiskEvidenceGP(inputs, missing, settings=settings, sharing="layout")
        global_only = SharedRiskEvidenceGP(inputs, missing, settings=settings, sharing="global")
        covariance = layout._kernel(inputs, inputs)
        np.testing.assert_array_equal(np.diag(covariance), np.ones(3))
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), 0.)
        self.assertAlmostEqual(covariance[0, 1] - covariance[0, 2], .25)
        self.assertAlmostEqual(global_only._kernel(inputs, inputs)[0, 2], covariance[0, 1])
        np.testing.assert_array_equal(layout.predict(inputs)["variance"], np.ones(3))

    def test_layout_label_transfers_to_unseen_designs_and_prior_round_trips(self):
        inputs = np.array([[0., 0.]])
        observed_abort = [OutputObservation("interval", "abort", lower=0., lower_inclusive=False)]
        settings = MixedGPSettings(length_scale=1., interval_noise=1., stable_tail_moments=True)
        model = SharedRiskEvidenceGP(inputs, observed_abort, settings=settings, sharing="layout")
        query = np.array([[10., 0.], [10., 1.]])
        prediction = model.predict(query)
        self.assertGreater(prediction["mean"][0], prediction["mean"][1])
        self.assertGreater(prediction["mean"][1], 0.)
        cloned = SharedRiskEvidenceGP.from_dict(model.to_dict())
        self.assertEqual(cloned.to_dict(), model.to_dict())
        np.testing.assert_array_equal(cloned.predict(query)["mean"], prediction["mean"])

    def test_coverage_variants_share_the_same_evaluated_prefix(self):
        for scenario in ("nominal", "headwind"):
            spec = plan("hierarchical-prior-" + scenario)
            self.assertEqual(spec["risk_prior_coverage_reuse"], ["risk_global_coverage", "risk_layout_coverage"])
            self.assertEqual(spec["learning_checkpoints"], [8, 12, 16])
            self.assertEqual(spec["budget"], {"initial": 96, "online": 864, "heldout": 1536})
            self.assertTrue(set(spec["seeds"]).isdisjoint(plan("stage-information-"+scenario)["seeds"]))
        self.assertEqual(sum(plan("hierarchical-prior-smoke")["budget"].values()), 29)
