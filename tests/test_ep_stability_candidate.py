"""Focused numerical tests: no aircraft evaluator, BO or VAE training."""
from dataclasses import replace
import math
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

from ra_lsbo.risk_core.ep_numerics import extreme_tail_moments
from ra_lsbo.risk_core.mixed_gp import (
    MixedGPSettings, MixedObservationGP, _truncated_moments)
from ra_lsbo.risk_core.observations import OutputObservation


def independently_integrated_tail(a, upper):
    """Shift to u=a*(z-a), integrate exp(-u-u**2/(2*a**2))."""
    cap = min(a * (upper - a), 55.) if math.isfinite(upper) else 55.
    def moment(k):
        return quad(lambda u: u**k * math.exp(-u-u*u/(2*a*a)),
                    0, cap, epsabs=1e-13, epsrel=2e-13)[0]
    z, one, two = moment(0), moment(1), moment(2)
    mean_excess = one / (z*a)
    return a + mean_excess, (two/z-(one/z)**2) / (a*a)


class StableTailMomentsTests(unittest.TestCase):
    def test_far_right_and_left_tails_match_independent_shifted_integration(self):
        for a, b in ((8., math.inf), (40., math.inf), (500., math.inf),
                     (40., 40.5), (100., 100.001), (8., 12.)):
            with self.subTest(a=a, b=b):
                expected = independently_integrated_tail(a, b)
                actual = extreme_tail_moments(a, b)
                assert_allclose(actual, expected, rtol=2e-7, atol=2e-11)
                reflected = extreme_tail_moments(-b, -a)
                assert_allclose(reflected, [-actual[0], actual[1]],
                                rtol=0, atol=1e-12)
                self.assertGreater(actual[1], 0)

    def test_unchanged_default_and_opt_in_range(self):
        self.assertIsNone(extreme_tail_moments(-1., 2.))
        self.assertIsNone(extreme_tail_moments(-5., math.inf))
        for a, b in ((-1., 2.), (0., math.inf), (-math.inf, 0.)):
            assert_allclose(_truncated_moments(a, b, True),
                            _truncated_moments(a, b, False), rtol=0, atol=0)
        for a, b in ((40., math.inf), (-math.inf, -40.)):
            m, v = _truncated_moments(a, b, stable_tail=True)
            self.assertTrue(math.isfinite(m) and math.isfinite(v) and v > 0)

    def test_invalid_interval_explicit_failure(self):
        for a, b in ((10., 10.), (math.nan, math.inf),
                     (math.inf, math.inf)):
            with self.subTest(a=a, b=b), self.assertRaises(FloatingPointError):
                extreme_tail_moments(a, b)

    def test_settings_validate_opt_in_and_retain_legacy_default(self):
        default = MixedGPSettings()
        self.assertFalse(default.stable_tail_moments)
        self.assertFalse(default.record_ep_trace)
        for value in (1, "true", None):
            with self.assertRaises(ValueError):
                MixedGPSettings(stable_tail_moments=value)
            with self.assertRaises(ValueError):
                MixedGPSettings(record_ep_trace=value)

    def test_single_site_matches_prior_behavior_where_no_extreme_tail(self):
        samples = [OutputObservation("interval", "lower_bound", lower=0.)]
        x = [[0.]]
        old = MixedObservationGP(x, samples, settings=MixedGPSettings())
        new = MixedObservationGP(x, samples, settings=MixedGPSettings(
            stable_tail_moments=True, record_ep_trace=True))
        for key in ("mean", "variance"):
            assert_allclose(old.predict(x)[key], new.predict(x)[key], atol=1e-13, rtol=0)
        self.assertTrue(new.diagnostics["converged"])
        self.assertEqual(new.diagnostics["ep_trace"][-1]["sweep"],
                         new.diagnostics["sweeps"])

    def test_trace_on_failure_does_not_silently_accept_nonconverged_EP(self):
        evidence = [OutputObservation("interval", "sign", lower=0.)]
        with self.assertRaisesRegex(RuntimeError, "did not converge") as captured:
            MixedObservationGP([[0.]], evidence,
                               settings=MixedGPSettings(max_sweeps=1,
                                   record_ep_trace=True, stable_tail_moments=True))
        error = captured.exception
        self.assertEqual(error.ep_sweeps, 1)
        self.assertEqual(len(error.ep_trace), 1)
        self.assertGreater(error.ep_last_site_change, 0)
        self.assertEqual(error.ep_trace[0]["sweep"], 1)


if __name__ == "__main__":
    unittest.main()
