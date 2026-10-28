from pathlib import Path
import tempfile
import unittest

from tools.run_partial_bo_development import (
    ARMS, BASE, HORIZON, INITIAL, MAXIMUM, SEEDS, TASKS, CallLedger,
    append, metrics, wilson)
from ra_lsbo.partial_bo.problems import HierarchicalBranin
from ra_lsbo.partial_bo.records import without_intervals
from ra_lsbo.risk_core.observations import OutputObservation


class StudyAccountingTests(unittest.TestCase):
    def test_complete_budget_includes_qualification_and_integration(self):
        self.assertEqual(MAXIMUM, len(TASKS)*(32+len(SEEDS)*(INITIAL+len(ARMS)*HORIZON))+4)
        self.assertEqual(MAXIMUM, 476)
        self.assertLessEqual(MAXIMUM, 500)

    def test_ledger_reserves_before_evaluation_and_never_auto_retries(self):
        problem = HierarchicalBranin(BASE)
        candidate = problem.anchors()[0]
        record = problem.record(candidate, 3., {"region": -.3, "resource": -.2},
                                {"termination_reason": "mock_complete"})

        class MockEvaluator:
            contract_id = problem.contract_id

            def __init__(self):
                self.calls = 0

            def evaluate(self, raw):
                self.calls += 1
                return record

        mock = MockEvaluator()
        with tempfile.TemporaryDirectory() as temporary:
            ledger = CallLedger(Path(temporary), 1)
            self.assertEqual(ledger.evaluate(mock, candidate, "a"), record)
            with self.assertRaisesRegex(RuntimeError, "retry"):
                ledger.evaluate(mock, candidate, "a")
            with self.assertRaisesRegex(RuntimeError, "budget"):
                ledger.evaluate(mock, candidate, "b")
            self.assertEqual(mock.calls, 1)
        with tempfile.TemporaryDirectory() as temporary:
            ledger = CallLedger(Path(temporary), 2)
            append(ledger.path, {"event": "started", "call_id": "unsettled", "tag": "pending"})
            with self.assertRaisesRegex(RuntimeError, "Unsettled"):
                ledger.evaluate(mock, candidate, "a")
            self.assertEqual(mock.calls, 1)

    def test_bounds_never_become_an_incumbent_and_stop_is_explicit(self):
        problem = HierarchicalBranin(BASE)
        candidate = problem.anchors()[0]
        partial = problem.record(candidate, OutputObservation("interval", "prefix", lower=1.),
            {"region": OutputObservation("interval", "sign", lower=.1), "resource": -.2},
            {"termination_reason": "physical_abort"})
        exact = problem.record(candidate, 5., {"region": -.1, "resource": -.2},
                               {"termination_reason": "mock_complete"})
        result = metrics([], [partial], True)
        self.assertIsNone(result["endpoint_best"])
        self.assertEqual(result["physical_infeasible_online"], 1)
        self.assertTrue(result["policy_termination_absorbing"])
        self.assertEqual(metrics([], [partial, exact])["first_online_feasible"], 2)
        self.assertEqual(metrics([], [without_intervals(partial)])["endpoint_best"], None)

    def test_zero_observed_feasibility_is_not_a_zero_population_rate(self):
        lower, upper = wilson(0, 30)
        self.assertAlmostEqual(lower, 0.)
        self.assertGreater(upper, .1)


if __name__ == "__main__":
    unittest.main()
