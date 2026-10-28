from pathlib import Path
import unittest

from ra_lsbo.controlled_sparse import SparseInteriorFiveModeProblem
from ra_lsbo.records import RiskEvaluation


class SparseInteriorCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parents[2]/"LSBO/X57-LSBO-refactored"
        if not base.exists():
            raise unittest.SkipTest("Local original LSBO required")
        cls.problem = SparseInteriorFiveModeProblem(base)

    def candidate(self, x, y):
        value = self.problem.space.default().to_dict()
        value["continuous"].update(x=x, y=y)
        return self.problem.canonical(value)

    def test_interior_hole_is_true_hidden_failure_and_censors_energy(self):
        candidate = self.candidate(.78, .69)
        margin = ((.78-.69)/.12)**2+((.69-.63)/.15)**2-1.
        self.assertLess(margin, 0)
        risk = self.problem.evaluate(candidate)
        RiskEvaluation(**risk)
        self.assertEqual(risk["hidden_labels"]["mission_solver_nonconvergence"], 1)
        self.assertIsNone(risk["hidden_labels"]["energy_audit_failure"])
        self.assertIsNone(risk["objective"])
        self.assertIsNone(risk["constraints"]["envelope"])

    def test_outside_hole_can_return_fully_observed_feasible_result(self):
        risk = self.problem.evaluate(self.candidate(.69, .63))
        RiskEvaluation(**risk)
        self.assertTrue(risk["feasible_label"])
        self.assertTrue(all(risk["constraint_observed"].values()))
