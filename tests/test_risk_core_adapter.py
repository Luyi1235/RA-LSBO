from copy import deepcopy
from pathlib import Path
import unittest

from ra_lsbo.controlled_sparse import SparseInteriorFiveModeProblem
from ra_lsbo.features import validate_history
from ra_lsbo.risk_core.adapter import CoreFeatures, global_pool, observable_row, packet


BASE = (Path(__file__).resolve().parents[1]/"runs"/
        "compact_mda_20261021T081006Z_20eb608d"/"qualification_recovered_20261025"/"frozen_base")


class RiskCoreAdapterTests(unittest.TestCase):
    def setUp(self):
        self.problem = SparseInteriorFiveModeProblem(BASE)

    def row(self, x=.4, y=.5):
        c = self.problem.space.sample(__import__("numpy").random.default_rng(7)).to_dict()
        c["continuous"].update(x=x, y=y)
        c = self.problem.canonical(c)
        return {"evaluation": 1, "candidate": c, "risk": self.problem.evaluate(c),
                "evaluator_contract_sha256": self.problem.contract_id}

    def test_sign_sensor_preserves_truth_and_creates_no_numerical_target(self):
        truth = self.row()
        original = deepcopy(truth)
        view = observable_row(truth, True)
        self.assertEqual(truth, original)
        self.assertFalse(truth["risk"]["feasible_label"])
        validate_history([view])
        value = packet(view).constraint("envelope")
        self.assertEqual(value.kind, "interval")
        self.assertIsNone(value.value)
        self.assertFalse(value.constraint_satisfaction())
        self.assertIsNotNone(packet(view).objective.value)

    def test_solver_failure_never_creates_physical_constraint_bounds(self):
        view = observable_row(self.row(.1, .2), True)
        self.assertEqual(view["risk"]["evidence"]["verified_constraint_bounds"], {})
        p = packet(view)
        self.assertEqual(p.status, "hidden_failure")
        self.assertTrue(all(o.kind == "missing" for n, o in p.constraints
                            if n not in self.problem.analytical_constraints))

    def test_repeated_adapter_preserves_existing_interval_evidence(self):
        view = observable_row(self.row(), True)
        self.assertEqual(view, observable_row(view, True))
        self.assertEqual(view, observable_row(view, False))

    def test_global_pool_is_deterministic_and_avoids_seen(self):
        first = global_pool(self.problem, 21, 5, set())
        self.assertEqual(first, global_pool(self.problem, 21, 5, set()))
        from ra_lsbo.budget import digest
        excluded = global_pool(self.problem, 21, 5, {digest(first[0])})
        self.assertNotIn(first[0], excluded)

    def test_onehot_transform_uses_no_truth_or_candidate_pool_extrema(self):
        row = observable_row(self.row(), True)
        features = CoreFeatures(self.problem, BASE, [row], 2)
        first = features.transform([row["candidate"]])
        pool = global_pool(self.problem, 33, 8, set())
        import numpy as np
        np.testing.assert_array_equal(
            first, features.transform([row["candidate"]]+pool)[:1])


if __name__ == "__main__":
    unittest.main()
