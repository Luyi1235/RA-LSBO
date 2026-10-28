from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ra_lsbo.controlled_sparse import SparseInteriorFiveModeProblem
from ra_lsbo.risk_core.adapter import global_pool, observable_row
from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.risk_core.optimizer import (
    CoreSelector, FAMILIES, expected_improvement, probability, response_gp)


BASE = (Path(__file__).resolve().parents[1]/"runs"/
        "compact_mda_20261021T081006Z_20eb608d"/"qualification_recovered_20261025"/"frozen_base")


class RiskCoreOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem = SparseInteriorFiveModeProblem(BASE)
        cls.rows = []
        for i, (x, y) in enumerate(((.4, .5), (.69, .63), (.2, .2), (.8, .75)), 1):
            candidate = {"continuous": {"x": x, "y": y, "spacing": .18},
                         "discrete": {"layout": "distributed", "count": 2}}
            candidate = cls.problem.canonical(candidate)
            cls.rows.append({
                "evaluation": i, "candidate": candidate, "risk": cls.problem.evaluate(candidate),
                "evaluator_contract_sha256": cls.problem.contract_id})

    def test_physical_objective_scale_and_offset_are_retained(self):
        observations = [OutputObservation("exact", "a", value=1000.),
                        OutputObservation("exact", "b", value=1001.)]
        x = np.array([[0.], [1.]])
        model = response_gp(x, observations, 0, objective=True)
        np.testing.assert_allclose(model.predict(x)["mean"], [1000., 1001.], atol=1e-6)
        self.assertGreater(model.predict([[10.]])["mean"][0], 900.)

    def test_minimization_and_constraint_sign_conventions(self):
        np.testing.assert_array_less(probability([1., 2.], [1., 1.]), .5)
        self.assertGreater(expected_improvement(np.array([-1.]), np.array([1.]), 0.)[0],
                           expected_improvement(np.array([1.]), np.array([1.]), 0.)[0])

    def test_all_families_propose_from_same_partial_history_and_settle(self):
        rows = [observable_row(r, True) for r in self.rows]
        pool = global_pool(self.problem, 122, 12, set())
        for family in FAMILIES:
            with self.subTest(family=family):
                selector = CoreSelector(family, self.problem, BASE, 81)
                candidate, metadata = selector.propose(rows, pool, 0)
                self.assertIn(candidate, pool)
                self.assertGreater(metadata["observation_counts"]["envelope"]["interval"], 0)
                self.assertFalse(metadata["probability_guarantee"])
                with self.assertRaises(RuntimeError):
                    selector.propose(rows, pool, 1)
                wrong = deepcopy(rows[0])
                with self.assertRaises(ValueError):
                    selector.accept(wrong)
                settled = deepcopy(rows[0])
                settled["candidate"] = candidate
                selector.accept(settled)
                self.assertIsNone(selector.pending)

    def test_chronological_calibration_can_replay_without_truth_calls(self):
        rows = [observable_row(r, True) for r in self.rows]
        pool = global_pool(self.problem, 123, 12, set())
        selectors = [CoreSelector("calibrated_eicb", self.problem, BASE, 82) for _ in range(2)]
        outputs = [s.propose(rows, pool, 0) for s in selectors]
        self.assertEqual(outputs[0][0], outputs[1][0])
        self.assertEqual(outputs[0][1], outputs[1][1])

    def test_boundary_exploitation_responds_to_completion_risk(self):
        rows = [observable_row(r, True) for r in self.rows]
        pool = global_pool(self.problem, 124, 2, set())
        with patch("ra_lsbo.risk_core.optimizer.BoundaryEnsemble") as classifier:
            classifier.return_value.predict.return_value = {
                "probability": np.ones(2), "upper": np.ones(2)}
            with patch("ra_lsbo.risk_core.optimizer.expected_improvement",
                       return_value=np.ones(2)):
                indices = []
                for completion in (np.array([.9, .1]), np.array([.1, .9])):
                    selector = CoreSelector("ensemble_boundary", self.problem, BASE, 84)
                    with patch.object(selector, "_completion", return_value=(completion, {})):
                        _, metadata = selector.propose(rows, pool, 0)
                    indices.append(metadata["index"])
        self.assertEqual(indices, [0, 1])


if __name__ == "__main__":
    unittest.main()
