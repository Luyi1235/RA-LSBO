import unittest

from ra_lsbo.risk_core.observations import OutputObservation
from tools.run_partial_refinement_study import (
    ARMS, HORIZON, INITIAL, MAXIMUM, SEEDS, TASKS, convergence_result, problem_for)


class RefinementStudyGateTests(unittest.TestCase):
    def test_budget_counts_every_new_label_and_does_not_charge_reused_truth(self):
        self.assertEqual(MAXIMUM, len(TASKS)*len(SEEDS)*len(ARMS)*HORIZON+len(SEEDS)*INITIAL+30+4)
        self.assertEqual(MAXIMUM, 430)
        self.assertLessEqual(MAXIMUM, 500)

    def fixture(self):
        problem = problem_for("aircraft_140km")
        candidate = problem.anchors()[0]
        constraints = {name: -.1 for name in problem.required_constraints}
        return problem, candidate, constraints

    def test_integration_gate_rejects_soc_drift_even_when_energy_is_unchanged(self):
        problem, candidate, constraints = self.fixture()
        coarse = problem.record(candidate, 30000., constraints, {"termination_reason": "mock_complete"})
        finer = problem.record(candidate, 30000., {**constraints, "soc": -.11},
                               {"termination_reason": "mock_complete"})
        self.assertFalse(convergence_result(0, coarse, [coarse, finer])["passed"])
        self.assertTrue(convergence_result(0, coarse, [coarse, coarse])["passed"])

    def test_partial_objective_cannot_pass_integration_convergence(self):
        problem, candidate, constraints = self.fixture()
        coarse = problem.record(candidate, OutputObservation("interval", "prefix", lower=100.), constraints,
                                {"termination_reason": "physical_abort"})
        result = convergence_result(0, coarse, [coarse, coarse])
        self.assertFalse(result["passed"])
        self.assertIsNone(result["energy_relative_error"])


if __name__ == "__main__":
    unittest.main()
