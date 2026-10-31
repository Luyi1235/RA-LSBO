from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    import numpy as np
    import torch
    from ra_lsbo.calibration import CalibratedRiskHeads
    from ra_lsbo.structured_encoder import EncoderSettings, StructuredFeatures, VIEWS
    from ra_lsbo.structured_models import (
        StructuredLearningStrategy, StructuredResponseHeads, StructuredRiskHeads, fit_structured_models)
    from tests.test_models import VARIABLES, history
    # Portable, tracked fixture: never depend on a developer sibling checkout.
    BASE = (Path(__file__).resolve().parents[1]
            / "runs/compact_mda_20261021T081006Z_20eb608d"
            / "qualification_recovered_20261025/frozen_base")
    from tools.risk_feature_ablation import feature_matrix


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class StructuredEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make(self, rows=None, settings=None):
        config = settings or EncoderSettings(initial_epochs=4, update_epochs=2, width=12)
        return StructuredFeatures(BASE, VARIABLES, seed=3, settings=config).fit(
            history(6) if rows is None else rows)

    def test_layouts_match_dimensions_and_onehot_reference(self):
        features = self.make()
        candidates = [r["candidate"] for r in history(4)]
        expected = {"latent": 8, "paper_gp": 6, "plain_onehot": 6,
                    "hierarchical_onehot": 10, "continuous_mask": 4}
        for view, dimensions in expected.items():
            self.assertEqual(features.layout(view)["dimensions"], dimensions)
            self.assertEqual(features.transform_view([], view).shape, (0, dimensions))
            self.assertEqual(features.transform_view(candidates, view).shape, (4, dimensions))
            self.assertEqual(len(set(features.layout(view)["columns"])), dimensions)
            if view in ("plain_onehot", "hierarchical_onehot", "continuous_mask"):
                np.testing.assert_array_equal(features.transform_view(candidates, view).numpy(),
                                              feature_matrix(features, candidates, view))

    def test_every_view_ignores_inactive_values_and_rejects_active_bounds(self):
        features = self.make()
        first = deepcopy(history(1)[0]["candidate"])
        second = deepcopy(first)
        second["continuous"]["spacing"], second["discrete"]["count"] = 999., 4
        for view in VIEWS:
            matrix = features.transform_view([first, second], view)
            torch.testing.assert_close(matrix[0], matrix[1], rtol=1e-13, atol=1e-14)
        first["continuous"]["x"] = 1.01
        with self.assertRaisesRegex(ValueError, "bounds"):
            features.transform([first])

    def test_transitive_masks_and_decoder_do_not_reactivate_grandchildren(self):
        variables = [
            {"name": "root", "kind": "nominal", "default": "off", "values": ["off", "on"]},
            {"name": "child", "kind": "nominal", "default": "yes", "values": ["yes", "no"],
             "active_when": {"root": ["on"]}},
            {"name": "leaf", "kind": "ordinal", "default": 2, "values": [2, 4],
             "active_when": {"child": ["yes"]}},
            {"name": "x", "kind": "continuous", "default": .5, "bounds": [0., 1.],
             "active_when": {"leaf": [2]}}]
        features = StructuredFeatures(BASE, variables, settings=EncoderSettings(width=8))
        model = features.representation.model
        masks = model.discrete_mask(torch.tensor([[0, 0, 0], [1, 0, 0]]))
        torch.testing.assert_close(masks, torch.tensor([[1., 0., 0.], [1., 1., 1.]], dtype=torch.double))
        self.assertFalse(features.space.active(features.space.continuous[0],
                                              {"root": "off", "child": "yes", "leaf": 2}))
        with torch.no_grad():
            for layer in model.decoder:
                layer.weight.zero_()
                layer.bias.copy_(torch.tensor([10., -10.]))
            model.decoder[-1].bias.copy_(torch.tensor([-10., 10.]))
        decoded = features.representation.decode(torch.zeros(4), np.random.default_rng(0))
        self.assertEqual(decoded, {"root": "off", "child": "yes", "leaf": 2})

    def test_inactive_decoder_heads_have_zero_reconstruction_gradient(self):
        features = self.make()
        generator = torch.Generator().manual_seed(4)
        model = features.representation.model
        model.zero_grad()
        terms = model.loss_terms(torch.tensor([[0, 0], [0, 1]]), generator)
        terms["reconstruction"].backward()
        torch.testing.assert_close(model.decoder[1].weight.grad,
                                   torch.zeros_like(model.decoder[1].weight.grad))
        self.assertGreater(float(model.decoder[0].weight.grad.abs().sum()), 0.)

    def test_loss_trace_matches_formula_and_fixed_annealing(self):
        features = self.make()
        trace = features.representation.training_trace
        self.assertEqual(trace[0]["weights"]["performance"], 0.)
        self.assertEqual(trace[-1]["weights"]["performance"], 1.)
        self.assertEqual(trace[-1]["weights"]["entropy"], 0.)
        for row in trace:
            losses, weights = row["losses"], row["weights"]
            self.assertAlmostEqual(
                losses["total"], losses["reconstruction"]+weights["mmd"]*losses["mmd"]
                +weights["performance"]*losses["performance"]-weights["entropy"]*losses["entropy"])
            self.assertTrue(np.isfinite(list(losses.values())).all())
        self.assertEqual(features.objective_indices, [1, 2, 4, 5, 6])

    def test_performance_prior_only_uses_z_in_both_joint_layouts(self):
        for view in ("paper_gp", "latent"):
            config = EncoderSettings(initial_epochs=4, width=12, joint_view=view)
            features = self.make(settings=config)
            x = features.transform([history(1)[0]["candidate"]]).requires_grad_()
            shifted = x.detach().clone()
            shifted[:, :-4] += 100.
            head = features.representation.model.performance
            torch.testing.assert_close(head(x), head(shifted), rtol=0, atol=0)
            head(x).sum().backward()
            torch.testing.assert_close(x.grad[:, :-4], torch.zeros_like(x.grad[:, :-4]))
            self.assertGreater(float(x.grad[:, -4:].abs().sum()), 0.)

    def test_seed_is_reproducible_and_does_not_change_global_torch_rng(self):
        before = torch.random.get_rng_state().clone()
        first, second = self.make(), self.make()
        torch.testing.assert_close(before, torch.random.get_rng_state())
        self.assertEqual(first.snapshot_id, second.snapshot_id)
        torch.testing.assert_close(first.transform([history(1)[0]["candidate"]]),
                                   second.transform([history(1)[0]["candidate"]]), rtol=0, atol=0)

    def test_no_performance_ablation_and_failed_objective_handling(self):
        settings = EncoderSettings(initial_epochs=3, width=12, performance_weight=0.)
        features = self.make(settings=settings)
        self.assertFalse(features.representation.performance_fitted)
        self.assertTrue(all(row["losses"]["performance"] == 0.
                            for row in features.representation.training_trace))
        failed = self.make(rows=history(3)[2:])
        self.assertEqual(failed.objective_indices, [])
        self.assertFalse(failed.representation.performance_fitted)

    def test_incremental_update_rejects_changed_prefix_and_invalidates_models(self):
        rows = history(6)
        features = self.make(rows=rows[:4])
        risks = StructuredRiskHeads(rows[:4], features)
        old_snapshot = features.snapshot_id
        changed = deepcopy(rows)
        changed[0]["risk"]["objective"] += 1.
        with self.assertRaisesRegex(ValueError, "prefix"):
            features.update(changed)
        features.update(rows)
        self.assertNotEqual(old_snapshot, features.snapshot_id)
        self.assertEqual(features.training_indices, list(range(1, 7)))
        self.assertTrue(all(r["incremental"] for r in features.representation.training_trace))
        self.assertTrue(all(r["weights"]["performance"] == 1.
                            for r in features.representation.training_trace))
        with self.assertRaisesRegex(RuntimeError, "changed"):
            risks.predict([rows[0]["candidate"]])

    def test_direct_model_and_grammar_mutations_are_detected(self):
        features = self.make()
        with torch.no_grad():
            features.representation.model.mu.bias.add_(.01)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            features.transform([])
        features = self.make()
        features.representation.grammar[0, 0] = 1
        with self.assertRaisesRegex(RuntimeError, "changed"):
            features.transform([])

    def test_checkpoint_round_trip_and_training_identity(self):
        features = self.make()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"encoder.pt"
            features.save_checkpoint(path)
            loaded = StructuredFeatures.load_checkpoint(path, BASE, expected_history=history(6))
            self.assertEqual(features.snapshot_id, loaded.snapshot_id)
            for view in VIEWS:
                torch.testing.assert_close(
                    features.transform_view([history(1)[0]["candidate"]], view),
                    loaded.transform_view([history(1)[0]["candidate"]], view), rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                features.save_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "history"):
                StructuredFeatures.load_checkpoint(path, BASE, expected_history=history(5))
            loaded.update(history(7))
            self.assertEqual(loaded.training_indices, list(range(1, 8)))

    def test_checkpoint_rejects_tampered_weights_and_training_trace(self):
        features = self.make()
        with tempfile.TemporaryDirectory() as directory:
            path, bad = Path(directory)/"encoder.pt", Path(directory)/"bad.pt"
            features.save_checkpoint(path)
            payload = torch.load(path, weights_only=True)
            payload["model"]["mu.bias"].add_(1.)
            torch.save(payload, bad)
            with self.assertRaisesRegex(RuntimeError, "changed"):
                StructuredFeatures.load_checkpoint(bad, BASE)
            payload = torch.load(path, weights_only=True)
            metadata = json.loads(payload["metadata_json"])
            metadata["training_trace"][0]["losses"]["total"] += 1.
            payload["metadata_json"] = json.dumps(metadata)
            torch.save(payload, bad)
            with self.assertRaisesRegex(RuntimeError, "changed"):
                StructuredFeatures.load_checkpoint(bad, BASE)

    def test_gp_risk_and_calibration_interfaces_share_encoder_and_prefix(self):
        rows = history(10)
        for view, latent_dim in ((v, d) for v in ("paper_gp", "latent") for d in (4, 8)):
            settings = EncoderSettings(initial_epochs=4, width=12, joint_view=view, latent_dim=latent_dim)
            features, risks, responses, calibrated = fit_structured_models(
                BASE, VARIABLES, rows[:6], rows[6:], settings,
                risk_view="hierarchical_onehot", gp_steps=2)
            candidates = [rows[-1]["candidate"]]
            self.assertEqual(responses.offset.shape[0], features.layout()["dimensions"])
            self.assertTrue(responses.diagnostics["objective"]["prior_used"])
            self.assertEqual(responses.diagnostics["objective"]["prior_input"], "latent_only")
            self.assertTrue(np.isfinite(responses.predict(candidates)["objective"]["mean"]).all())
            self.assertEqual(risks.diagnostics["final_feasibility"]["input_dimensions"], 10)
            self.assertIsNone(calibrated.predict(candidates)["risk_probabilities_calibrated"]["final_feasibility"])
            different = StructuredRiskHeads(rows[:6], features, view="latent")
            self.assertNotEqual(risks.model_id, different.model_id)
            with self.assertRaisesRegex(ValueError, "after training"):
                CalibratedRiskHeads(risks, rows[5:])

    def test_refit_resets_weights_and_future_labels_cannot_affect_prefix(self):
        first = self.make(rows=history(4))
        prefix = first.transform([history(1)[0]["candidate"]]).clone()
        first.fit(history(6), epochs=2)
        first.fit(history(4), epochs=4)
        torch.testing.assert_close(prefix, first.transform([history(1)[0]["candidate"]]), rtol=0, atol=0)

    def test_invalid_settings_and_histories_fail_explicitly(self):
        with self.assertRaises(ValueError):
            EncoderSettings(latent_dim=0)
        with self.assertRaises(ValueError):
            EncoderSettings(warmup_fraction=.8, ramp_fraction=.5)
        with self.assertRaises(ValueError):
            EncoderSettings(performance_weight=float("nan"))
        with self.assertRaises(ValueError):
            EncoderSettings(joint_view="hierarchical_onehot")
        with self.assertRaisesRegex(ValueError, "Nonempty"):
            self.make(rows=[])
        rows = history(4)
        rows[-1]["evaluator_contract_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Mixed"):
            self.make(rows=rows)

    def test_structured_strategy_runs_inside_existing_loop_without_future_labels(self):
        from ra_lsbo.optimizer import OptimizerSettings, run_optimizer
        from ra_lsbo.synthetic import SyntheticProblem
        settings = OptimizerSettings(budget=16, bootstrap=10, calibration_window=4,
                                     pool_size=8, epochs=2, gp_steps=2, update_every=2)
        strategy = StructuredLearningStrategy(
            EncoderSettings(initial_epochs=4, update_epochs=2, width=12),
            risk_view="hierarchical_onehot", incremental=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"loop"
            rows, summary = run_optimizer(SyntheticProblem(BASE), BASE, settings,
                                          output, model_strategy=strategy)
            self.assertEqual(summary["evaluations"], 16)
            fits = [json.loads(line) for line in (output/"model_fits.jsonl").read_text().splitlines()]
            self.assertGreaterEqual(len(fits), 2)
            self.assertEqual(fits[0]["encoder_update"], "reset")
            self.assertEqual(fits[1]["encoder_update"], "append_only_incremental")
            for fit in fits:
                self.assertLess(max(fit["training_indices"]), min(fit["calibration_indices"]))
                self.assertLess(max(fit["calibration_indices"]), fit["before_evaluation"])
            self.assertTrue(all(r["proposal"]["evidence"]["basis"] != "empirical_calibrated"
                                for r in rows[settings.bootstrap:]))


if __name__ == "__main__":
    unittest.main()
