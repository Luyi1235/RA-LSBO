from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ra_lsbo.adaptive_risk import (JointRiskEnsemble, JointRiskSettings, ReferenceTemperature,
                                   entropy_decomposition, partial_event_loss, terminal_sets)
from ra_lsbo.adaptive_system import AdaptiveLearning, LocalRegion, local_pool, select_adaptive
from ra_lsbo.mechanism_system import FiveModeProblem, SystemSettings
from ra_lsbo.structured_encoder import StructuredFeatures
from ra_lsbo.system_experiment import sample_rows
from tools.run_adaptive_research import _trial, problem_for
from ra_lsbo.system_experiment import save_new


BASE = Path(__file__).resolve().parents[2]/"LSBO"/"X57-LSBO-refactored"


class AdaptiveResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem = FiveModeProblem(BASE)
        cls.rows = sample_rows(cls.problem, 140, 927)
        cls.variables = [asdict(v) for v in cls.problem.space.variables]
        cls.settings = SystemSettings(calls=8, failure_cap=5, calibration_window=40,
                                      encoder_epochs=3, incremental_epochs=2, gp_steps=2, pool_size=16)
        cls.risk_settings = JointRiskSettings(members=2, epochs=3, width=16, embedding=4)
        cls.features = StructuredFeatures(BASE, cls.variables, settings=cls.settings.encoder()).fit(cls.rows[:100])
        cls.model = JointRiskEnsemble(cls.rows[:100], cls.features, settings=cls.risk_settings)

    def test_serial_failures_are_single_events_with_censored_stages(self):
        allowed = terminal_sets(self.rows)
        self.assertTrue((allowed.sum(1) == 1).all())
        self.assertTrue((allowed.sum(0) > 0).all())
        aggregate = terminal_sets(self.rows, True)
        np.testing.assert_array_equal(aggregate[:, 2], allowed[:, 2:].any(1))

    def test_partial_outcome_uses_probability_sum_and_unknown_has_no_loss(self):
        row = deepcopy(next(r for r in self.rows if r["risk"]["objective_observed"]))
        row["risk"]["feasible_label"] = None
        row["risk"]["state"] = "incomplete"
        allowed = terminal_sets([row])
        self.assertEqual(allowed.tolist(), [[True, True, False, False, False, False, False]])
        logits = torch.zeros((2, 7), requires_grad=True)
        labels = torch.tensor(np.vstack((allowed, np.zeros((1, 7), dtype=bool))))
        loss = partial_event_loss(logits, labels)
        self.assertAlmostEqual(float(loss.detach()), np.log(7/2), places=6)
        loss.backward()
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.)

    def test_ensemble_information_does_not_confuse_shared_entropy_with_disagreement(self):
        shared = np.array([[[.5, .5]], [[.5, .5]]])
        opposite = np.array([[[1., 0.]], [[0., 1.]]])
        total, _, information = entropy_decomposition(shared)
        self.assertGreater(total[0], .6)
        self.assertAlmostEqual(information[0], 0.)
        self.assertAlmostEqual(entropy_decomposition(opposite)[2][0], np.log(2))
        with self.assertRaises(ValueError):
            entropy_decomposition(np.array([[[.5, .8]]]))

    def test_probabilities_normalize_and_checkpoint_replays(self):
        candidates = [r["candidate"] for r in self.rows[-5:]]
        predicted = self.model.predict(candidates)
        np.testing.assert_allclose(np.sum(predicted["probabilities"], axis=1), 1., atol=1e-6)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"risk.pt"
            self.model.save_checkpoint(path)
            restored = JointRiskEnsemble.load_checkpoint(path, self.features)
            self.assertEqual(predicted, restored.predict(candidates))

    def test_reference_is_disjoint_and_normalized_transform_preserves_other_ratios(self):
        with self.assertRaises(ValueError):
            ReferenceTemperature(self.model, self.rows[:40])
        calibration = ReferenceTemperature(self.model, self.rows[100:])
        self.assertEqual(calibration.status, "fitted_reference_temperature")
        original = np.array([[.8, .1, .02, .02, .02, .02, .02]])
        transformed = calibration.transform(original, self.model.model_id)
        self.assertAlmostEqual(transformed.sum(), 1.)
        self.assertAlmostEqual(transformed[0, 1]/transformed[0, 2], 5.)
        with self.assertRaises(ValueError):
            calibration.transform(original, "other_model")
        calibration.temperature += .1
        with self.assertRaises(RuntimeError):
            calibration.transform(original, self.model.model_id)

    def test_margin_targets_only_count_observed_nonanalytical_constraints(self):
        for name in self.model.margin_names:
            expected = sum(r["risk"]["constraint_observed"].get(name, False) for r in self.rows[:100])
            self.assertEqual(self.model.diagnostics["margin_observations"][name], expected)
        self.assertNotIn("packing", self.model.margin_names)

    def test_unseen_modes_are_explicitly_unsupported(self):
        completed = [dict(r, evaluation=i+1) for i, r in enumerate(
            [r for r in self.rows if r["risk"]["objective_observed"]])]
        features = StructuredFeatures(BASE, self.variables, settings=self.settings.encoder()).fit(completed)
        model = JointRiskEnsemble(completed, features, settings=self.risk_settings)
        self.assertEqual(len(model.diagnostics["unsupported_events"]), 5)

    def test_new_labels_enter_next_fit_without_moving_reference(self):
        learner = AdaptiveLearning("adaptive_full", self.settings, range(101, 141), 0,
                                   self.risk_settings)
        _, before = learner.fit(self.rows, self.variables, BASE)
        new_row = deepcopy(sample_rows(self.problem, 1, 939)[0])
        new_row["evaluation"] = 141
        models, after = learner.fit(self.rows+[new_row], self.variables, BASE)
        self.assertEqual(after["training_indices"], list(range(1, 101))+[141])
        self.assertEqual(before["reference_history_id"], after["reference_history_id"])
        self.assertEqual(models[0].training_indices[-1], 141)
        for diagnostic in after["response_diagnostics"].values():
            self.assertFalse(set(diagnostic["training_indices"]) & set(range(101, 141)))

    def test_no_calibration_ablation_preserves_raw_model_and_training_ids(self):
        a = AdaptiveLearning("adaptive_full", self.settings, range(101, 141), 4, self.risk_settings)
        b = AdaptiveLearning("adaptive_no_calibration", self.settings, range(101, 141), 4,
                             self.risk_settings)
        am, ad = a.fit(self.rows, self.variables, BASE)
        bm, bd = b.fit(self.rows, self.variables, BASE)
        self.assertEqual(am[0].model_id, bm[0].model_id)
        self.assertEqual(ad["training_indices"], bd["training_indices"])
        self.assertEqual(am[2].calibration_id, bm[2].calibration_id)
        pool, _ = local_pool(self.problem, np.random.default_rng(2), 16, set(), self.rows, LocalRegion())
        _, selection = select_adaptive(pool, self.rows, bm, 1, "adaptive_no_calibration",
                                       self.settings, 0, 0., 0)
        self.assertFalse(selection["calibration_applied"])
        self.assertEqual(selection["raw_final"], selection["used_final"])

    def test_local_region_contracts_expands_and_keeps_canonical_candidates(self):
        region = LocalRegion()
        for _ in range(4):
            region.observe(100., {"objective": None, "feasible_label": False})
        self.assertEqual(region.width, .2)
        for _ in range(2):
            region.observe(100., {"objective": 99., "feasible_label": True})
        self.assertAlmostEqual(region.width, .3)
        pool, _ = local_pool(self.problem, np.random.default_rng(2), 20, set(), self.rows, region)
        self.assertEqual(len(pool), 20)
        for item in pool:
            self.assertEqual(item["candidate"], self.problem.canonical(item["candidate"]))
            self.assertTrue(all(v <= 0 for v in self.problem.precheck(item["candidate"]).values()))

    def test_controlled_runner_charges_hard_calls_and_has_no_physical_case(self):
        with self.assertRaises(ValueError):
            problem_for("physical", BASE)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/"five_mode").mkdir()
            output = folder/"trial"
            output.mkdir()
            save_new(folder/"five_mode/initial_2.json", self.rows)
            save_new(folder/"five_mode/test_2.json", self.rows[-5:])
            settings = asdict(self.settings)
            settings.update(calls=2, failure_cap=2)
            result = _trial({"settings": settings, "base_project": str(BASE)}, folder, output,
                            self.problem, "five_mode", "adaptive_full", 2)
            self.assertIn(result["status"], ("complete", "stopped_by_policy"))
            self.assertEqual(result["evaluations"], 2)
            self.assertEqual(result["ledger"]["risk_charge"], 2.)


if __name__ == "__main__":
    unittest.main()
