import unittest
from unittest.mock import Mock, patch

import numpy as np

from ra_lsbo.discovery_selection import log_feasibility, select_feasibility_first


class DiscoverySelectionTests(unittest.TestCase):
    def test_log_score_avoids_product_underflow(self):
        scores = log_feasibility([1e-200, 1e-201], [[1e-200, 1e-200]])
        self.assertEqual(int(np.argmax(scores)), 0)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.isneginf(log_feasibility([0.], [[1.]])[0]))

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            log_feasibility([1.1], [[.5]])
        with self.assertRaises(ValueError):
            log_feasibility([.1, .2], [[.5]])

    def test_discovery_includes_completion_and_every_expensive_head(self):
        responses, completion = Mock(), Mock()
        responses.features.snapshot_id = completion.snapshot_id = "same"
        responses.diagnostics = {"constraint:a": {}, "constraint:cheap": {}, "objective": {}}
        responses.models = {"constraint:a": object()}
        completion.models = {"completion": object()}
        completion.model_id = "risk"
        completion.predict.return_value = {"heads": {"completion": {"probability": [.9, .8]}}}
        responses.predict.return_value = {"constraint:a": {"p_satisfied_raw": [.1, .9]}}
        pool = [{"candidate": {"x": 1}}, {"candidate": {"x": 2}}]
        proposal, metadata = select_feasibility_first(pool, [], responses, completion, ["cheap"], 1)
        self.assertEqual(proposal["candidate"], {"x": 2})
        self.assertEqual(metadata["constraint_names"], ["constraint:a"])
        self.assertFalse(metadata["probability_guarantee"])

    def test_after_first_feasible_returns_to_unchanged_posterior_policy(self):
        with patch("ra_lsbo.discovery_selection.select_posterior", return_value=("original", {})) as select:
            result = select_feasibility_first([], [{"risk": {"feasible_label": True, "objective": 1.}}],
                                              None, None, [], 7)
            self.assertEqual(result, ("original", {}))
            select.assert_called_once()


if __name__ == "__main__":
    unittest.main()
