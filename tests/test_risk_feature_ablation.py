from copy import deepcopy
import importlib.util
import unittest

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    import numpy as np
    import torch
    from ra_lsbo.features import LatentFeatures
    from tests.test_models import BASE, VARIABLES, history
    from tools.risk_feature_ablation import MODES, feature_matrix, score_records


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class RiskFeatureAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.features = LatentFeatures(BASE, VARIABLES, seed=0).fit(history(4), epochs=1)

    def test_all_encodings_canonicalize_inactive_coordinates(self):
        first = deepcopy(history(1)[0]["candidate"])
        second = deepcopy(first)
        second["continuous"]["spacing"] = .99
        second["discrete"]["count"] = 4
        for mode in MODES:
            matrix = feature_matrix(self.features, [first, second], mode)
            np.testing.assert_allclose(matrix[0], matrix[1], rtol=1e-12, atol=1e-14)

    def test_hierarchical_onehot_masks_inactive_discrete_child(self):
        candidate = history(1)[0]["candidate"]
        matrix = feature_matrix(self.features, [candidate], "hierarchical_onehot")
        self.assertEqual(matrix.shape, (1, 10))
        np.testing.assert_array_equal(matrix[0, -3:], [0., 0., 0.])
        self.assertEqual(feature_matrix(self.features, [], "plain_onehot").shape, (0, 6))
        self.assertEqual(feature_matrix(self.features, [candidate], "continuous_mask").shape, (1, 4))

    def test_scoring_keeps_unknown_accepted_and_unavailable_calibration(self):
        rows = [
            {"raw": .9, "calibrated": None, "label": False},
            {"raw": .9, "calibrated": None, "label": None}]
        scores = score_records(rows)
        self.assertEqual(scores["raw"]["samples"], 1)
        self.assertEqual(scores["raw"]["infeasible_among_accepted"], 1)
        self.assertEqual(scores["raw"]["unknown_accepted"], 1)
        self.assertEqual(scores["calibrated"]["predictions"], 0)
        self.assertIsNone(scores["calibrated"]["brier"])


if __name__ == "__main__":
    unittest.main()
