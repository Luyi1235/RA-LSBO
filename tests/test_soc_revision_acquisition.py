"""Independent formula checks for exploration versus exploitation risk."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import ndtr

from ra_lsbo.partial_bo.selector import PartialBOSettings, choose_acquisition as frozen_choose
from ra_lsbo.risk_core.mixed_gp import eicb_factor
from ra_lsbo.risk_core.optimizer import expected_improvement
from ra_lsbo.soc_revision.acquisition import (
    average, choose_acquisition, pointwise_acquisition, quantities)


def fixture():
    return {"completion": {"mean": np.array([-10., -10.]), "variance": np.zeros(2)},
        "constraints": {"soc": {"mean": np.array([0., -1.6448536269514722]),
            "variance": np.ones(2), "scale": 1.}},
        "objective": {"mean": np.array([8.5, 9.]), "variance": np.zeros(2)}}


def subset(prediction, indices):
    def select(forecast):
        result = {key: np.asarray(value)[indices] for key, value in forecast.items()
                  if key not in {"components", "scale"}}
        if "scale" in forecast:
            result["scale"] = forecast["scale"]
        if "components" in forecast:
            c = forecast["components"]
            result["components"] = {"weights": c["weights"], "mean": c["mean"][:, indices],
                                    "variance": c["variance"][:, indices]}
        return result
    return {"completion": select(prediction["completion"]),
            "constraints": {name: select(value) for name, value in prediction["constraints"].items()},
            "objective": None if prediction["objective"] is None else select(prediction["objective"])}


class RiskAcquisitionTests(unittest.TestCase):
    def test_balanced_factor_is_not_a_probability(self):
        self.assertAlmostEqual(float(eicb_factor(0., 1.)), .9750021048517795)
        self.assertAlmostEqual(float(ndtr(0.)), .5)

    def test_exploitation_no_longer_rewards_prior_uncertainty_as_safety(self):
        prediction, settings = fixture(), PartialBOSettings(horizon=12)
        old, old_meta = choose_acquisition(prediction, 10., 0, settings, False)
        new, new_meta = choose_acquisition(prediction, 10., 0, settings, True)
        self.assertEqual(old, 0)
        self.assertEqual(new, 1)
        self.assertEqual(old_meta["mode"], "balanced_feasible_improvement")
        self.assertEqual(new_meta["mode"], "probability_weighted_feasible_improvement")
        self.assertAlmostEqual(new_meta["per_constraint_probability"]["soc"][0], .5)
        self.assertFalse(new_meta["probability_guarantee"])

    def test_discovery_and_boundary_allocation_are_retained(self):
        prediction, settings = fixture(), PartialBOSettings(horizon=12)
        self.assertEqual(choose_acquisition(prediction, None, 0, settings, True)[1]["mode"], "feasible_discovery")
        slots = [choose_acquisition(prediction, 10., iteration, settings, True)[1]["boundary_slot"]
                 for iteration in range(12)]
        self.assertEqual([index for index, slot in enumerate(slots) if slot], [4, 9])
        first = choose_acquisition(prediction, 10., 4, settings, True)
        second = choose_acquisition(prediction, 10., 4, settings, False)
        self.assertEqual(first[0], second[0])
        assert_allclose(first[1]["scores"], second[1]["scores"])

    def test_componentwise_cdf_and_ei_not_moment_matched(self):
        weights, means, variance = np.array([.2, .8]), np.array([[-3.], [2.]]), np.array([[.04], [.09]])
        mean = weights @ means
        total_variance = weights @ (variance+means**2)-mean**2
        head = {"mean": mean, "variance": total_variance,
                "components": {"weights": weights, "mean": means, "variance": variance}}
        probability = average(head, lambda mean, var: ndtr(-mean/np.sqrt(var)))
        expected = weights @ ndtr(-means/np.sqrt(variance))
        assert_allclose(probability, expected)
        self.assertGreater(abs(float(probability[0]-ndtr(-mean/np.sqrt(total_variance))[0])), .05)
        improvement = average(head, lambda mean, var: expected_improvement(mean, var, 0.))
        assert_allclose(improvement, weights @ expected_improvement(means, variance, 0.))

    def test_pointwise_scores_are_batch_independent_with_mixture(self):
        prediction = fixture()
        base = prediction["constraints"]["soc"]
        base["components"] = {"weights": np.array([.3, .7]),
            "mean": np.array([base["mean"]-.1, base["mean"]+.2]),
            "variance": np.array([base["variance"], base["variance"]*.8])}
        for mode in ("feasible_discovery", "balanced_feasible_improvement",
                     "probability_weighted_feasible_improvement", "budgeted_boundary_information",
                     "posterior_support_restoration"):
            score, margins = pointwise_acquisition(prediction, 10., mode, 2., .5)
            for index in range(2):
                one, support = pointwise_acquisition(subset(prediction, [index]), 10., mode, 2., .5)
                assert_allclose(one, score[[index]])
                assert_allclose(support, margins[[index]])

    def test_single_component_control_matches_frozen_selector(self):
        random = np.random.default_rng(82626)
        prediction = {"completion": {"mean": random.normal(size=17), "variance": random.uniform(.01, 1., 17)},
            "constraints": {name: {"mean": random.normal(size=17), "variance": random.uniform(.01, 1., 17),
                                    "scale": scale} for name, scale in (("soc", .05), ("stall", .2))},
            "objective": {"mean": random.normal(size=17), "variance": random.uniform(.01, 1., 17)}}
        settings = PartialBOSettings(horizon=12)
        for impossible in (False, True):
            if impossible:
                for forecast in prediction["constraints"].values():
                    forecast["mean"] += 100.
            for iteration in range(12):
                for best in (None, -1.):
                    frozen, old = frozen_choose(prediction, best, iteration, settings)
                    revised, new = choose_acquisition(prediction, best, iteration, settings, False)
                    self.assertEqual(frozen, revised)
                    self.assertEqual(old["mode"], new["mode"])
                    self.assertEqual(old["fallback"], new["fallback"])
                    self.assertEqual(old["eligible"], new["eligible"])
                    assert_allclose([x for x in old["scores"] if x is not None],
                                    [x for x in new["scores"] if x is not None], rtol=1e-12, atol=1e-12)

    def test_degenerate_constraint_probability_is_exact(self):
        prediction = fixture()
        prediction["constraints"]["soc"].update(mean=np.array([0., .01]), variance=np.zeros(2))
        result = quantities(prediction, 10., 2., .5)
        assert_allclose(result["probabilities"]["soc"], [1., 0.])


if __name__ == "__main__":
    unittest.main()
