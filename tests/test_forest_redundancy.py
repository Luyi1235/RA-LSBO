"""A matching finite forecast sample must not be mistaken for model equivalence."""

from copy import deepcopy
import unittest

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from tools.diagnose_system_components import forest_complement_check


def forests():
    x = np.asarray([[-2.], [-1.], [1.], [2.]])
    y = np.asarray([0, 0, 1, 1])
    first = RandomForestClassifier(n_estimators=2, bootstrap=False, random_state=0).fit(x, y)
    second = RandomForestClassifier(n_estimators=2, bootstrap=False, random_state=0).fit(x, 1-y)
    return first, second


class ForestRedundancyTests(unittest.TestCase):
    def test_partition_change_is_detected_even_when_test_scores_match(self):
        first, second = forests()
        self.assertTrue(forest_complement_check(first, second)["complement_for_all_inputs_within_tolerance"])
        second.estimators_[0].tree_.threshold[0] += .25
        x = np.asarray([[-2.], [2.]])
        np.testing.assert_allclose(first.predict_proba(x)[:, 1], 1-second.predict_proba(x)[:, 1])
        self.assertFalse(forest_complement_check(first, second)["complement_for_all_inputs_within_tolerance"])

    def test_noncomplementary_leaf_or_feature_dimensions_are_rejected(self):
        first, second = forests()
        second.estimators_[0].tree_.value[1, 0] = [.25, .75]
        self.assertFalse(forest_complement_check(first, second)["complement_for_all_inputs_within_tolerance"])
        first, second = forests()
        second.n_features_in_ = 2
        self.assertFalse(forest_complement_check(first, second)["complement_for_all_inputs_within_tolerance"])


if __name__ == "__main__":
    unittest.main()
