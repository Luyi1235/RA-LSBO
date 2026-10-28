from copy import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ra_lsbo.mechanism_system import FiveModeProblem
from ra_lsbo.posterior_risk import CompletionPosterior, PosteriorSettings
from ra_lsbo.posterior_selection import choose_sample, discovery_anchor, posterior_pool, select_posterior
from ra_lsbo.structured_encoder import EncoderSettings, StructuredFeatures
from ra_lsbo.structured_models import StructuredResponseHeads
from ra_lsbo.system_experiment import sample_rows


BASE = Path(__file__).resolve().parents[2]/"LSBO/X57-LSBO-refactored"


class PosteriorSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem = FiveModeProblem(BASE)
        cls.rows = sample_rows(cls.problem, 90, 7189)
        cls.features = StructuredFeatures(BASE, [asdict(v) for v in cls.problem.space.variables],
                                         settings=EncoderSettings(initial_epochs=3)).fit(cls.rows)
        cls.model = CompletionPosterior(cls.rows, cls.features,
                                        settings=PosteriorSettings(steps=3, inducing=12))
        cls.responses = StructuredResponseHeads(cls.rows, cls.features, steps=2)

    def test_best_sampled_feasible_objective(self):
        index, mode, _ = choose_sample([100., 2., 1.], [[-1.], [-1.], [-1.]],
                                      [True, True, False], [1.])
        self.assertEqual(index, 1)
        self.assertEqual(mode, "sampled_feasible_improvement")

    def test_no_feasible_uses_violation_before_objective(self):
        index, mode, violation = choose_sample([-100., 2., 3.], [[100.], [.1], [.2]],
                                               [True, True, False], [1.])
        self.assertEqual(index, 1)
        self.assertEqual(mode, "sampled_feasibility_restoration")
        self.assertAlmostEqual(violation[2], 1.2)

    def test_constraint_units_do_not_change_restoration_ranking(self):
        args = ([4., 3.], [[3., 1.], [1., 3.]], [True, True])
        a = choose_sample(*args, [1., 10.])
        b = choose_sample(args[0], [[300., 1.], [100., 3.]], args[2], [100., 10.])
        self.assertEqual(a[0], b[0])
        np.testing.assert_allclose(a[2], b[2])

    def test_unsupported_mechanisms_are_not_synthetic_successes(self):
        completed = [dict(r, evaluation=i+1) for i, r in enumerate(
            [r for r in self.rows if r["risk"]["objective_observed"]])]
        features = StructuredFeatures(BASE, self.features.variables,
                                     settings=EncoderSettings(initial_epochs=2)).fit(completed)
        model = CompletionPosterior(completed, features,
                                    settings=PosteriorSettings(steps=2, inducing=8))
        prediction = model.predict([completed[0]["candidate"]])
        self.assertEqual(len(prediction["unsupported"]), 5)
        self.assertTrue(all(prediction["heads"][n] is None for n in prediction["unsupported"]))

    def test_unknown_labels_are_excluded_and_checkpoint_replays(self):
        for name, diagnostic in self.model.diagnostics.items():
            key = "hidden_success" if name == "completion" else "hidden_labels"
            expected = [r["evaluation"] for r in self.rows if (
                r["risk"][key] if name == "completion" else r["risk"][key][name]) is not None]
            self.assertEqual(diagnostic["training_indices"], expected)
        candidates = [r["candidate"] for r in self.rows[-4:]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"posterior.pt"
            self.model.save_checkpoint(path)
            restored = CompletionPosterior.load_checkpoint(path, self.features, self.rows)
            np.testing.assert_allclose(
                self.model.predict(candidates)["heads"]["completion"]["probability"],
                restored.predict(candidates)["heads"]["completion"]["probability"], atol=1e-10)
            np.testing.assert_array_equal(self.model.sample_completion(candidates, 55)[0],
                                          restored.sample_completion(candidates, 55)[0])

    def test_no_feasible_anchor_uses_real_observed_margins(self):
        rows = [r for r in self.rows if r["risk"]["feasible_label"] is False]
        anchor, reason = discovery_anchor(rows, self.problem.analytical_constraints)
        self.assertIsNotNone(anchor)
        self.assertEqual(reason, "least_observed_violation")
        pool, _ = posterior_pool(self.problem, np.random.default_rng(3), 20, set(), rows)
        self.assertTrue(any(p["source"] == "local_least_observed_violation" for p in pool))

    def test_posterior_selection_has_no_learned_analytical_constraint(self):
        pool, _ = posterior_pool(self.problem, np.random.default_rng(5), 20, set(), self.rows)
        proposal, metadata = select_posterior(pool, self.rows, self.responses, self.model,
                                             self.problem.analytical_constraints, 75)
        self.assertEqual(metadata["candidate_key"], proposal["evidence"]["candidate_key"])
        self.assertNotIn("constraint:packing", metadata["constraint_names"])
        self.assertEqual(metadata["calibration_status"], "not_applied")
        self.assertFalse(metadata["probability_guarantee"])

    def test_missing_expensive_model_bootstraps_instead_of_assuming_satisfaction(self):
        pool, _ = posterior_pool(self.problem, np.random.default_rng(6), 8, set(), self.rows)
        responses = copy(self.responses)
        responses.models = dict(self.responses.models)
        del responses.models["constraint:envelope"]
        _, metadata = select_posterior(pool, self.rows, responses, self.model,
                                       self.problem.analytical_constraints, 75)
        self.assertEqual(metadata["reason"], "unsupported_expensive_heads_bootstrap")


if __name__ == "__main__":
    unittest.main()
