import unittest

from tools.validate_mechanism_transfer import split_groups


class FailureAreaSplitTests(unittest.TestCase):
    def test_anchor_and_all_neighbors_removed_together(self):
        plan = {"probes": [
            {"index": 1}, {"index": 2}, {"index": 3, "anchor_index": 1},
            {"index": 4, "anchor_index": 2}, {"index": 5, "anchor_index": 1}]}
        results = [{"index": i, "risk": {"state": "hidden_failure"}} for i in range(1, 6)]
        training, test = split_groups(plan, results, 1)
        self.assertEqual([r["index"] for r in training], [2, 4])
        self.assertEqual([r["index"] for r in test], [1, 3, 5])

    def test_missing_actual_anchor_cannot_be_claimed_as_held_out(self):
        with self.assertRaises(ValueError):
            split_groups({"probes": [{"index": 1}]}, [{"index": 1, "risk": None}], 1)
