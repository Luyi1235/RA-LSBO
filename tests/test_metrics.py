import math
import unittest

from ra_lsbo.metrics import binary_metrics


class MetricTests(unittest.TestCase):
    def test_perfect_predictions_and_endpoint_bins(self):
        result = binary_metrics([0., 1.], [0, 1], bins=5)
        self.assertEqual(result["brier"], 0.)
        self.assertEqual(result["ece"], 0.)
        self.assertEqual(result["reliability"][0]["samples"], 1)
        self.assertEqual(result["reliability"][-1]["samples"], 1)
        self.assertTrue(math.isfinite(result["log_loss"]))

    def test_uniform_predictions(self):
        result = binary_metrics([.5, .5], [0, 1])
        self.assertEqual(result["brier"], .25)
        self.assertAlmostEqual(result["log_loss"], math.log(2))
        self.assertEqual(result["ece"], 0.)

    def test_no_observations_is_not_perfect_calibration(self):
        self.assertIsNone(binary_metrics([], [])["brier"])

    def test_bad_inputs_rejected(self):
        for p, y in (([float("nan")], [0]), ([-.1], [1]), ([.1], []), ([.1], [True])):
            with self.assertRaises(ValueError):
                binary_metrics(p, y)


if __name__ == "__main__":
    unittest.main()
