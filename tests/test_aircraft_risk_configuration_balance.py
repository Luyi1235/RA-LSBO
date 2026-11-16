import unittest

from aircraft_risk_active import balance_configuration_scores, configuration_diagnostics
from aircraft_risk_protocol import plan


def design(config, x):
    return {"discrete": {"configuration": config}, "continuous": {"x": x}}


class AircraftRiskConfigurationBalanceTests(unittest.TestCase):
    def test_gate_uses_acquired_counts_and_keeps_within_group_scores(self):
        old = design("A", 0)
        remaining = [design("A", 1), design("B", 1), design("B", 2), design("C", 1)]
        scores = [.25, .015625, .03125, .0625]
        self.assertEqual(balance_configuration_scores(scores, remaining, [old]),
                         [-1., .015625, .03125, .0625])
        evaluated = [old, design("B", 0), design("C", 0), design("D", 0)]
        self.assertEqual(balance_configuration_scores(scores, remaining, evaluated), scores)
        self.assertEqual(balance_configuration_scores(scores, remaining, evaluated+[design("B", 3)]),
                         [.25, -1., -1., .0625])

    def test_diagnostics_weight_configurations_and_skip_unknown_truth(self):
        rows = [(design("A", 0), True, .5), (design("A", 1), True, .5),
                (design("B", 0), False, .25), (design("C", 0), None, .9)]
        actual = configuration_diagnostics(rows, [design("A", 2), design("A", 3)])
        self.assertEqual(actual, {"training_configurations": 1, "max_training_configuration_count": 2,
                                  "configuration_probability_bias": -.125, "unseen_configuration_brier": .0625,
                                  "unseen_scored_configurations": 1})
        self.assertIsNone(configuration_diagnostics(rows[:2], [design("A", 2)])["unseen_configuration_brier"])

    def test_formal_profiles_are_paired_with_fresh_seeds(self):
        nominal, headwind = (plan("config-balance-"+case) for case in ("nominal", "headwind"))
        self.assertEqual(nominal["seeds"], headwind["seeds"])
        for spec in (nominal, headwind):
            self.assertEqual(spec["configuration_balanced_arms"], ["risk_balanced_coverage", "risk_balanced_active"])
            self.assertEqual(spec["budget"], {"initial": 96, "online": 1440, "heldout": 1536})
            self.assertEqual(spec["learning_checkpoints"], [8, 12, 16])
            self.assertTrue(set(spec["seeds"]).isdisjoint(plan("hierarchical-prior-nominal")["seeds"]))
