import unittest

import numpy as np

from ra_lsbo.normalized_margin_joint import normalized_members


class NormalizedMarginTests(unittest.TestCase):
    def test_float32_near_unit_completion_becomes_a_valid_distribution(self):
        p = np.array([[[1.00000008, 0., 1e-9]], [[.99999999, 1e-8, 0.]]])
        repaired = normalized_members(p)
        self.assertTrue((repaired >= 0.).all())
        self.assertTrue((repaired <= 1.).all())
        np.testing.assert_allclose(repaired.sum(-1), 1., atol=1e-15)
        self.assertLess(np.max(np.abs(repaired-p)), 1e-6)

    def test_normalization_does_not_hide_large_probability_errors(self):
        for p in (np.array([[[1.01, 0.]]]), np.array([[[-1e-9, 1.]]]), np.array([[[np.nan, 1.]]])):
            with self.assertRaises(ValueError):
                normalized_members(p)


if __name__ == "__main__":
    unittest.main()
