import unittest

import numpy as np

from aircraft_risk_transfer_value import comparison_value, feedback_probabilities


class TestTransferValue(unittest.TestCase):
    def test_feedback_must_distinguish_models_that_disagree_on_action(self):
        weights = [.5, .5]
        actions = [[.8, .4], [.4, .8]]
        feedback = [[[.5, .5], [1., 0.]], [[.5, .5], [0., 1.]]]
        np.testing.assert_allclose(comparison_value(weights, actions, feedback), [0., .2])
        np.testing.assert_allclose(comparison_value(weights, [[.8, .4], [.8, .5]], feedback), 0.)

    def test_distinguishing_models_has_no_gain_if_choice_will_not_change(self):
        feedback = [[[.6, .4]], [[.4, .6]]]
        np.testing.assert_allclose(comparison_value([.9, .1], [[.8, .4], [.2, .9]], feedback), 0.)

    def test_timing_cells_cover_the_normal_and_coarsen_to_completion(self):
        mean = np.array([[0., .4]])
        sd = np.array([[.2, .1]])
        full = feedback_probabilities(mean, sd)
        coarse = feedback_probabilities(mean, sd, (0.,))
        np.testing.assert_allclose(full.sum(axis=2), 1.)
        np.testing.assert_allclose(full[..., 0], coarse[..., 0])
        np.testing.assert_allclose(full[..., 1:].sum(axis=2), coarse[..., 1])


if __name__ == "__main__":
    unittest.main()
