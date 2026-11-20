"""Check that decision gating uses both predictive gains and no held-out labels."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_aircraft_decision_value_audit as audit


class DecisionValueTests(unittest.TestCase):
    def test_larger_two_call_score_alone_does_not_trigger_query(self):
        probes = [{"x": 0}, {"x": 1}]
        values = [{"adaptive": .3, "decision_information": .02},
                  {"adaptive": .4, "decision_information": .01}]
        chosen, detail = audit.gate(probes, values, 0)
        self.assertEqual(chosen, 0)
        self.assertEqual(detail["reason"], "no_additional_decision_information")

    def test_more_information_does_not_trigger_below_greedy_total(self):
        probes = [{"x": 0}, {"x": 1}]
        values = [{"adaptive": .3, "decision_information": .02},
                  {"adaptive": .29, "decision_information": .04}]
        chosen, detail = audit.gate(probes, values, 0)
        self.assertEqual(chosen, 0)
        self.assertEqual(detail["reason"], "gain_cannot_offset_cost")

    def test_more_information_with_net_gain_triggers_query(self):
        probes = [{"x": 0}, {"x": 1}]
        values = [{"adaptive": .3, "decision_information": .02},
                  {"adaptive": .32, "decision_information": .04}]
        chosen, detail = audit.gate(probes, values, 0)
        self.assertEqual(chosen, 1)
        self.assertEqual(detail["reason"], "activated")

    def test_panel_partitions_only_on_screened_design_order(self):
        class Record:
            def __init__(self, value):
                self.candidate = {"x": value}

            @property
            def usable_feasible(self):
                raise AssertionError("Selection must not see usable labels")

        records = [Record(i) for i in range(5)]
        with patch.object(audit, "passes_screen", side_effect=lambda _, x: x["x"] != 1):
            probes, followups = audit.split_panel(None, records)
        self.assertEqual(probes, [records[0], records[3]])
        self.assertEqual(followups, [records[2], records[4]])

    def test_actual_followup_gain_can_be_negative(self):
        first = SimpleNamespace(usable_feasible=False)
        pair = [SimpleNamespace(usable_feasible=True), SimpleNamespace(usable_feasible=False)]
        before = {"usable": [.6, .4]}
        receipt = {"after": {"usable": [.3, .7]}}
        result = audit.settle(first, pair, [{"x": 0}, {"x": 1}], receipt, "usable", before)
        self.assertFalse(result["hit"])
        self.assertEqual(result["rank_correction"], -1)


if __name__ == "__main__":
    unittest.main()
