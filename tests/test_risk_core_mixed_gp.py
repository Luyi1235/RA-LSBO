from dataclasses import replace
import json
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm

from ra_lsbo.risk_core.mixed_gp import (
    MixedGPSettings, MixedObservationGP, eicb_factor)
from ra_lsbo.risk_core.observations import OutputObservation


def exact(value):
    return OutputObservation("exact", "numeric", value=value)


def interval(lower=None, upper=None):
    return OutputObservation("interval", "proven-bound", lower=lower, upper=upper)


def matern(left, right, scale, variance):
    radius = np.sqrt(5) * np.linalg.norm(
        (left[:, None] - right[None, :]) / scale, axis=2)
    return variance * (1 + radius + radius**2 / 3) * np.exp(-radius)


class MixedGPTests(unittest.TestCase):
    def test_exact_only_matches_independent_closed_form_physical_units(self):
        x = np.array([[0., 0.], [.4, .2], [1., .8]])
        test = np.array([[.15, .2], [.7, .3], [2., 1.]])
        y = np.array([120., 102., 135.])
        settings = MixedGPSettings(
            length_scale=(.4, .7), signal_variance=400., prior_mean=110.,
            exact_noise=.3, jitter=1e-9)
        model = MixedObservationGP(x, [exact(v) for v in y], settings=settings)
        kernel = matern(x, x, np.array(settings.length_scale), 400.)
        covariance = kernel + np.eye(3) * (.3**2 + settings.jitter)
        cross = matern(test, x, np.array(settings.length_scale), 400.)
        expected_mean = 110 + cross @ np.linalg.solve(covariance, y - 110)
        expected_variance = 400 - np.sum(
            cross * np.linalg.solve(covariance, cross.T).T, axis=1)
        prediction = model.predict(test)
        assert_allclose(prediction["mean"], expected_mean, rtol=1e-12)
        assert_allclose(prediction["variance"], expected_variance, rtol=1e-12)

    def test_single_sign_matches_known_truncated_normal(self):
        for sign in (-1, 1):
            evidence = interval(lower=0.) if sign == 1 else interval(upper=0.)
            model = MixedObservationGP([[0.]], [evidence])
            prediction = model.predict([[0.]])
            assert_allclose(prediction["mean"], [sign * np.sqrt(2 / np.pi)], atol=2e-8)
            assert_allclose(prediction["variance"], [1 - 2 / np.pi], atol=2e-8)

    def test_finite_interval_and_censor_match_independent_quadrature(self):
        settings = MixedGPSettings(
            prior_mean=1.2, signal_variance=2.3, jitter=1e-12)
        for lower, upper in ((-.7, .9), (2., np.inf), (-np.inf, -.5)):
            evidence = interval(
                lower=None if np.isneginf(lower) else lower,
                upper=None if np.isposinf(upper) else upper)
            density = lambda z: norm.pdf(z, loc=1.2, scale=np.sqrt(2.3))
            mass = quad(density, lower, upper, epsabs=1e-12)[0]
            mean = quad(lambda z: z * density(z), lower, upper, epsabs=1e-12)[0] / mass
            variance = quad(
                lambda z: (z - mean)**2 * density(z),
                lower, upper, epsabs=1e-12)[0] / mass
            prediction = MixedObservationGP(
                [[0.]], [evidence], settings=settings).predict([[0.]])
            assert_allclose(prediction["mean"], [mean], atol=2e-8)
            assert_allclose(prediction["variance"], [variance], atol=2e-8)

    def test_soft_interval_probit_matches_quadrature(self):
        settings = MixedGPSettings(interval_noise=.4)
        for lower, upper in ((.3, np.inf), (-.4, .8)):
            def density(z):
                return norm.pdf(z) * (
                    ndtr((upper - z) / .4) - ndtr((lower - z) / .4))
            mass = quad(density, -10, 10)[0]
            mean = quad(lambda z: z * density(z), -10, 10)[0] / mass
            variance = quad(lambda z: (z - mean)**2 * density(z), -10, 10)[0] / mass
            evidence = interval(lower, None if np.isposinf(upper) else upper)
            prediction = MixedObservationGP(
                [[0.]], [evidence], settings=settings).predict([[0.]])
            assert_allclose(prediction["mean"], [mean], atol=2e-8)
            assert_allclose(prediction["variance"], [variance], atol=2e-8)

    def test_numeric_plus_one_interval_matches_conditional_quadrature(self):
        settings = MixedGPSettings(exact_noise=.1, jitter=1e-12)
        x = np.array([[0.], [.4]])
        covariance = matern(x, x, .5, 1.)
        mu = covariance[1, 0] * -1. / 1.01
        var = 1 - covariance[1, 0]**2 / 1.01
        density = lambda z: norm.pdf(z, mu, np.sqrt(var))
        mass = quad(density, .2, np.inf)[0]
        expected_mean = quad(lambda z: z * density(z), .2, np.inf)[0] / mass
        expected_variance = quad(
            lambda z: (z - expected_mean)**2 * density(z), .2, np.inf)[0] / mass
        prediction = MixedObservationGP(
            x, [exact(-1.), interval(lower=.2)], settings=settings).predict([[.4]])
        assert_allclose(prediction["mean"], [expected_mean], atol=2e-8)
        assert_allclose(prediction["variance"], [expected_variance], atol=2e-8)

    def test_missing_rows_do_not_change_predictions(self):
        missing = OutputObservation("missing", "unobserved", reason="solver_stopped")
        evidence = [exact(-.5), interval(lower=0.)]
        test = [[.2], [.8], [100.]]
        base = MixedObservationGP([[0.], [1.]], evidence).predict(test)
        augmented = MixedObservationGP(
            [[0.], [1.], [100.]], evidence + [missing]).predict(test)
        for key in base:
            assert_allclose(base[key], augmented[key], atol=0, rtol=0)

    def test_no_evidence_returns_configured_prior_with_explicit_diagnostics(self):
        missing = OutputObservation("missing", "unobserved", reason="solver_stopped")
        settings = MixedGPSettings(
            prior_mean=12., signal_variance=9., length_scale=(.2, .7))
        for x, evidence in ((np.empty((0, 2)), []), ([[3., 4.]], [missing])):
            model = MixedObservationGP(x, evidence, settings=settings)
            prediction = model.predict([[0., 1.], [2., 3.]])
            assert_allclose(prediction["mean"], [12., 12.], atol=0, rtol=0)
            assert_allclose(prediction["variance"], [9., 9.], atol=0, rtol=0)
            self.assertTrue(model.supports_empty_observations)
            self.assertTrue(model.diagnostics["prior_only"])
            self.assertEqual(model.diagnostics["inference"], "prior")
            self.assertEqual(model.diagnostics["length_scale"], [.2, .7])
            self.assertEqual(model.diagnostics["normalization"]["output"], "identity")
            self.assertTrue(model.diagnostics["converged"])
            self.assertEqual(model.diagnostics["sweeps"], 0)
            restored = MixedObservationGP.from_dict(json.loads(json.dumps(model.to_dict())))
            assert_allclose(restored.predict([[0., 1.]])["mean"], [12.], atol=0, rtol=0)

    def test_single_class_is_finite_and_reverts_to_fixed_prior_far_away(self):
        model = MixedObservationGP(
            [[0.], [.4], [.8]], [interval(lower=0.)] * 3)
        prediction = model.predict([[0.], [.4], [.8], [100.]])
        self.assertTrue(np.all(prediction["mean"][:3] > 0))
        self.assertTrue(np.all(prediction["variance"][:3] < 1))
        assert_allclose(prediction["mean"][-1], 0, atol=1e-14)
        assert_allclose(prediction["variance"][-1], 1, atol=1e-14)
        self.assertTrue(model.diagnostics["converged"])

    def test_multiple_opposite_sign_sites_use_spatial_evidence(self):
        model = MixedObservationGP(
            [[0.], [.5], [1.]],
            [interval(upper=0.), interval(-.2, .2), interval(lower=0.)])
        prediction = model.predict([[0.], [.5], [1.]])
        self.assertLess(prediction["mean"][0], 0)
        self.assertGreater(prediction["mean"][2], 0)
        self.assertLess(abs(prediction["mean"][1]), .05)
        self.assertTrue(np.all(prediction["variance"] >= 0))
        self.assertLess(prediction["variance"][1], .02)

    def test_extreme_tail_and_narrow_interval_remain_finite(self):
        for lower, upper in ((40., None), (-40.001, -40.), (10., 10.00001), (0., 1e-8)):
            model = MixedObservationGP(
                [[0.]], [interval(lower, upper)],
                settings=MixedGPSettings(jitter=1e-14))
            prediction = model.predict([[0.]])
            self.assertTrue(np.all(np.isfinite(prediction["mean"])))
            self.assertTrue(np.all(np.isfinite(prediction["variance"])))
            self.assertGreaterEqual(prediction["mean"][0], lower - 1e-8)
            if upper is not None:
                self.assertLessEqual(prediction["mean"][0], upper + 1e-8)
            self.assertTrue(np.all(prediction["variance"] >= 0))

    def test_serialization_refits_deterministically_and_preserves_evidence(self):
        evidence = [exact(-.4), interval(.1, .8)]
        model = MixedObservationGP(
            [[0.], [.7]], evidence, seed=42,
            settings=MixedGPSettings(length_scale=(.6,)))
        state = json.loads(json.dumps(model.to_dict(), allow_nan=False))
        self.assertIsNone(state["observations"][1]["value"])
        restored = MixedObservationGP.from_dict(state)
        self.assertEqual(restored.observations, tuple(evidence))
        for key, value in model.predict([[.2], [1.]]).items():
            assert_allclose(restored.predict([[.2], [1.]])[key], value, atol=0, rtol=0)

    def test_nonfinite_malformed_and_nonconverged_fail_explicitly(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                MixedObservationGP([[bad]], [exact(0.)])
            with self.assertRaises(ValueError):
                MixedGPSettings(prior_mean=bad)
            with self.assertRaises(ValueError):
                MixedGPSettings(length_scale=bad)
        model = MixedObservationGP([[0.]], [exact(1.)])
        for bad in ([[np.nan]], [[np.inf]], [[0., 1.]], [0.]):
            with self.assertRaises(ValueError):
                model.predict(bad)
        with self.assertRaises(ValueError):
            MixedObservationGP([[0.]], [])
        with self.assertRaises(ValueError):
            MixedObservationGP([[0.]], [1.])
        with self.assertRaises(ValueError):
            MixedObservationGP([[0., 1.]], [exact(0.)],
                               settings=MixedGPSettings(length_scale=(.1,)))
        with self.assertRaises(RuntimeError):
            MixedObservationGP([[0.]], [interval(lower=0.)],
                               settings=MixedGPSettings(max_sweeps=1))
        with self.assertRaisesRegex(ValueError, "Contradictory"):
            MixedObservationGP([[0.], [0.]], [interval(lower=1.), interval(upper=0.)])

    def test_endpoint_inclusion_does_not_change_continuous_likelihood(self):
        closed = interval(lower=0.)
        strict = replace(closed, lower_inclusive=False)
        left = MixedObservationGP([[0.]], [closed]).predict([[0.]])
        right = MixedObservationGP([[0.]], [strict]).predict([[0.]])
        for key in left:
            assert_allclose(left[key], right[key], atol=0, rtol=0)

    def test_duplicate_hard_intervals_are_idempotent_and_intersect(self):
        for bounds, merged in (
                ([interval(lower=0.), interval(lower=0.)], interval(lower=0.)),
                ([interval(lower=0.), interval(.5, 1.5)], interval(.5, 1.5))):
            model = MixedObservationGP([[0.], [0.]], bounds)
            prediction = model.predict([[0.], [.6]])
            expected = MixedObservationGP([[0.]], [merged]).predict([[0.], [.6]])
            for key in expected:
                assert_allclose(prediction[key], expected[key], atol=0, rtol=0)
            self.assertEqual(model.diagnostics["n_interval"], 2)
            self.assertEqual(model.diagnostics["n_interval_sites"], 1)
            self.assertEqual(len(model.to_dict()["observations"]), 2)

    def test_single_interval_propagates_known_moments_to_unobserved_location(self):
        model = MixedObservationGP([[0.]], [interval(lower=0.)])
        query = np.array([[.4], [1.]])
        correlation = matern(query, np.array([[0.]]), .5, 1.)[:, 0]
        prediction = model.predict(query)
        assert_allclose(prediction["mean"], correlation * np.sqrt(2 / np.pi), atol=2e-8)
        assert_allclose(prediction["variance"], 1 - correlation**2 * 2 / np.pi, atol=2e-8)


class EICBTests(unittest.TestCase):
    def test_formula_and_probability_bounds(self):
        mean = np.array([-3., -.3, 0., .3, 3.])
        variance = np.array([.1, .8, 1., .8, .1])
        ratio = mean / np.sqrt(variance)
        expected = np.clip(
            (1 + norm.cdf(1.96 - ratio) - norm.cdf(-1.96 - ratio))
            * norm.cdf(-ratio), 0, 1)
        actual = eicb_factor(mean, variance)
        assert_allclose(actual, expected)
        self.assertTrue(np.all(actual >= norm.cdf(-ratio)))
        self.assertTrue(np.all(actual <= 1))
        assert_allclose(eicb_factor(mean, variance, beta=0), norm.cdf(-ratio))

    def test_zero_variance_broadcast_and_invalid_inputs(self):
        assert_allclose(eicb_factor([-1., 0., 1.], 0.), [1., 1., 0.])
        assert_allclose(eicb_factor(1., [0., 1.]), [0., eicb_factor(1., 1.)])
        for mean, variance in ((np.nan, 1.), (0., np.inf), (0., -1.)):
            with self.assertRaises(ValueError):
                eicb_factor(mean, variance)
        for beta in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                eicb_factor(0., 1., beta=beta)


if __name__ == "__main__":
    unittest.main()
