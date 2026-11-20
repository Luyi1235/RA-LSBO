"""Check the endpoints of fixed cross-state kernel sharing."""

import unittest

import numpy as np

from aircraft_current_state_transfer import CompletionStateEvidenceGP
from ra_lsbo.partial_bo.models import EvidenceGP
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings
from ra_lsbo.risk_core.observations import OutputObservation


class CurrentStateTransferTests(unittest.TestCase):
    def setUp(self):
        self.history = np.array([[0.], [.25]])
        self.query = np.array([[.1], [.35]])
        self.history_state = np.column_stack((self.history, [0., 1.]))
        self.query_completed = np.column_stack((self.query, [0., 0.]))
        self.observations = [OutputObservation("exact", "completed", value=-.2),
                             OutputObservation("interval", "stopped", lower=.3)]
        self.settings = MixedGPSettings(stable_tail_moments=True)

    def predict(self, rho):
        model = CompletionStateEvidenceGP(self.history_state, self.observations,
                                          settings=self.settings, cross_state=rho)
        return model.predict(self.query_completed)

    def test_no_sharing_recovers_the_exact_only_posterior(self):
        baseline = EvidenceGP(self.history[:1], self.observations[:1], settings=self.settings)
        result = self.predict(0.)
        for name, values in baseline.predict(self.query).items():
            np.testing.assert_allclose(result[name], values, atol=1e-12)

    def test_full_sharing_recovers_the_pooled_posterior(self):
        baseline = EvidenceGP(self.history, self.observations, settings=self.settings)
        result = self.predict(1.)
        for name, values in baseline.predict(self.query).items():
            np.testing.assert_allclose(result[name], values, atol=1e-12)

    def test_stop_only_evidence_moves_the_completed_head_when_shared(self):
        rows = self.history_state[1:]
        model = CompletionStateEvidenceGP(rows, self.observations[1:], settings=self.settings,
                                          cross_state=.5)
        result = model.predict(np.array([[.25, 0.]]))
        self.assertGreater(result["mean"][0], 0.)
        self.assertLess(result["variance"][0], self.settings.signal_variance)


if __name__ == "__main__":
    unittest.main()
