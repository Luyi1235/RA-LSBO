"""Regression tests for the current bound reconstructed from charge traces."""

import unittest

from run_aircraft_current_prefix_audit import peak_current_lower


class PrefixCurrentTests(unittest.TestCase):
    def test_uses_largest_segment_average(self):
        trace = [{"elapsed_s": 360., "soc": .8}, {"elapsed_s": 1080., "soc": .6},
                 {"elapsed_s": 1440., "soc": .45}]
        self.assertAlmostEqual(peak_current_lower(trace, .9, 3.), 4.5)

    def test_zero_duration_marker_preserves_the_previous_bound(self):
        trace = [{"elapsed_s": 360., "soc": .8}, {"elapsed_s": 360., "soc": .8}]
        self.assertAlmostEqual(peak_current_lower(trace, .9, 3.), 3.)

    def test_rejects_discharge_without_positive_elapsed_time(self):
        with self.assertRaisesRegex(ValueError, "without elapsed time"):
            peak_current_lower([{"elapsed_s": 0., "soc": .7}], .9, 3.)


if __name__ == "__main__":
    unittest.main()
