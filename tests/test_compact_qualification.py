from pathlib import Path
import tempfile
import unittest

from ra_lsbo.budget import BudgetPolicy, RiskLedger
from tools.qualify_compact_mda import study_charged, wilson


class CompactQualificationTests(unittest.TestCase):
    def test_all_zero_is_not_an_empty_region_certificate(self):
        lower, upper = wilson(0, 30)
        self.assertAlmostEqual(lower, 0.)
        self.assertGreater(upper, .10)

    def test_sparse_confirmation_gate_has_nontrivial_uncertainty(self):
        self.assertLess(wilson(4, 60)[1], .20)
        self.assertGreater(wilson(8, 60)[1], .20)
        with self.assertRaises(ValueError):
            wilson(1, 0)

    def test_global_count_includes_unsettled_started_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with RiskLedger(root/"pilot"/"budget.jsonl", BudgetPolicy(2, 2., 2, 0, 0)) as ledger:
                token = ledger.reserve({"continuous": {}, "discrete": {}}, "bootstrap",
                                       {"basis": "unavailable"}, "unresolved_test")
                ledger.start(token)
            self.assertEqual(study_charged(root), 1)


if __name__ == "__main__":
    unittest.main()
