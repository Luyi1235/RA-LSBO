"""Posterior-only fixtures: never evaluate a design or launch an MDA."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.models import LearnedMixedHead
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.partial_refinement.posterior import (
    GroupedMixedHead, RefinementPosterior, group_divisors)
from ra_lsbo.risk_core.observations import OutputObservation, RunObservation
from ra_lsbo.soc_revision.posterior import AveragedGroupedHead, RiskPosterior


MODULE = "ra_lsbo.soc_revision.posterior"


def fake_head(evidence, mean=(0., 0.), variance=(1., 1.)):
    return SimpleNamespace(
        selected={"log_evidence": evidence}, center=0., scale=1.,
        identity="head:"+str(evidence),
        model=SimpleNamespace(diagnostics={"exact": 0, "interval": 0, "missing": 2}),
        predict=Mock(return_value={"mean": np.asarray(mean), "variance": np.asarray(variance)}))


class AveragedGroupedHeadTests(unittest.TestCase):
    def setUp(self):
        self.inputs = np.asarray([[0., 1., 0.], [1., 1., 1.]])
        self.observations = [OutputObservation("missing", "missing:"+str(index), reason="unmeasured")
                             for index in range(2)]

    def fit_fake(self, heads, **kwargs):
        with patch(MODULE+".LearnedMixedHead", side_effect=heads):
            return AveragedGroupedHead(self.inputs, self.observations,
                continuous_count=1, continuous_grid=(.2,),
                discrete_grid=tuple(range(1, len(heads)+1)), **kwargs)

    def test_weights_and_total_moments_match_hand_calculation(self):
        head = self.fit_fake([fake_head(0., (0., 2.), (1., 4.)),
                              fake_head(np.log(3.), (4., 6.), (9., 16.))])
        prediction = head.predict(self.inputs)
        np.testing.assert_allclose(head.weights, [.25, .75])
        np.testing.assert_allclose(prediction["mean"], [3., 5.])
        np.testing.assert_allclose(prediction["variance"], [10., 16.])
        components = prediction["components"]
        np.testing.assert_array_equal(components["mean"], [[0., 2.], [4., 6.]])
        np.testing.assert_array_equal(components["variance"], [[1., 4.], [9., 16.]])
        np.testing.assert_allclose(components["weights"], [.25, .75])
        direct = components["weights"] @ (components["variance"]+components["mean"]**2)
        np.testing.assert_allclose(prediction["variance"], direct-prediction["mean"]**2)

    def test_between_model_uncertainty_and_large_offset_stability(self):
        for offset in (0., 1e12):
            with self.subTest(offset=offset):
                head = self.fit_fake([fake_head(0., (offset-2.,)*2, (0.,)*2),
                                      fake_head(0., (offset+2.,)*2, (0.,)*2)])
                prediction = head.predict(self.inputs)
                np.testing.assert_array_equal(prediction["mean"], [offset, offset])
                np.testing.assert_allclose(prediction["variance"], [4., 4.])

    def test_default_eight_metrics_and_frozen_inner_length(self):
        heads = [fake_head(0.) for _ in range(8)]
        inputs = np.ones((2, 7))
        with patch(MODULE+".LearnedMixedHead", side_effect=heads) as learned:
            head = AveragedGroupedHead(inputs, self.observations, continuous_count=2,
                                       seed=19, completion=True)
        self.assertEqual(learned.call_count, 8)
        for call, continuous, discrete in zip(learned.call_args_list,
                np.repeat((.05, .15, .4, 1.), 2), np.tile((.5, 1.5), 4)):
            divisor = group_divisors(7, 2, continuous, discrete)
            np.testing.assert_array_equal(call.args[0], inputs/divisor)
            self.assertEqual(tuple(call.args[1]), tuple(self.observations))
            self.assertEqual(call.kwargs, {"seed": 19, "length_grid": (1/np.sqrt(7),), "completion": True})
        np.testing.assert_allclose(head.weights, np.full(8, .125))
        self.assertEqual(head.predict(inputs)["components"]["mean"].shape, (8, 2))

    def test_softmax_stays_finite_and_retains_valid_zero_weight_models(self):
        head = self.fit_fake([fake_head(1e300), fake_head(-1e300), fake_head(1e300)])
        np.testing.assert_array_equal(head.weights, [.5, 0., .5])
        self.assertTrue(np.isfinite(head.weights).all())
        prediction = head.predict(self.inputs)
        self.assertEqual(prediction["components"]["mean"].shape, (3, 2))
        self.assertEqual(head.heads[1].predict.call_count, 1)

    def test_failures_have_zero_diagnostic_weight_without_invented_components(self):
        head = self.fit_fake([fake_head(0.), RuntimeError("EP failed"),
                              fake_head(float("nan")), fake_head(np.log(3.))])
        diagnostics = head.diagnostics()
        np.testing.assert_allclose(diagnostics["weights"], [.25, 0., 0., .75])
        self.assertEqual([entry["status"] for entry in diagnostics["candidates"]],
                         ["ok", "RuntimeError", "FloatingPointError", "ok"])
        self.assertEqual(diagnostics["candidates"][1]["weight"], 0.)
        self.assertIsNone(diagnostics["candidates"][2]["log_evidence"])
        prediction = head.predict(self.inputs)
        self.assertEqual(prediction["components"]["mean"].shape, (2, 2))
        np.testing.assert_allclose(prediction["components"]["weights"], [.25, .75])
        self.assertEqual(diagnostics["selected"]["discrete_length"], 4)
        self.assertTrue({"identity", "scale", "center", "counts", "selected", "candidates", "weights"}
                        <= set(diagnostics))
        self.assertIn("approximate EP evidence averaging", diagnostics["evidence_interpretation"])
        self.assertIn("not a safety certificate", diagnostics["evidence_interpretation"])
        json.dumps(diagnostics, allow_nan=False)

    def test_all_failed_fits_raise(self):
        for failure in (RuntimeError("failed"), np.linalg.LinAlgError("singular"),
                        FloatingPointError("nonfinite"), fake_head(float("inf"))):
            with self.subTest(failure=str(failure)), self.assertRaisesRegex(RuntimeError, "No grouped kernel"):
                self.fit_fake([failure])

    def test_invalid_predictions_raise_instead_of_changing_weights(self):
        cases = [((np.nan, 0.), (1., 1.), FloatingPointError),
                 ((np.inf, 0.), (1., 1.), FloatingPointError),
                 ((0., 0.), (np.inf, 1.), FloatingPointError),
                 ((0., 0.), (-.01, 1.), FloatingPointError),
                 ((0.,), (1., 1.), ValueError),
                 ((1e308, -1e308), (1., 1.), FloatingPointError)]
        for mean, variance, error in cases:
            with self.subTest(mean=mean, variance=variance):
                head = self.fit_fake([fake_head(0.), fake_head(0., mean, variance)])
                with self.assertRaises(error):
                    head.predict(self.inputs)
                np.testing.assert_array_equal(head.weights, [.5, .5])

    def test_only_roundoff_negative_variance_is_tolerated(self):
        head = self.fit_fake([fake_head(0., variance=(-1e-16, 0.))])
        np.testing.assert_array_equal(head.predict(self.inputs)["variance"], [0., 0.])

    def test_invalid_grids_inputs_and_prediction_dimensions_raise(self):
        for kwargs in ({"continuous_grid": ()}, {"discrete_grid": ()},
                       {"continuous_grid": (0.,)}, {"discrete_grid": (np.inf,)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AveragedGroupedHead(self.inputs, self.observations, continuous_count=1, **kwargs)
        for inputs in (np.zeros(3), np.full((2, 3), np.nan)):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                AveragedGroupedHead(inputs, self.observations, continuous_count=1)
        head = self.fit_fake([fake_head(0.)])
        for inputs in (np.zeros((2, 4)), np.full((2, 3), np.inf)):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                head.predict(inputs)


class RealEPTests(unittest.TestCase):
    def fixture(self):
        inputs = np.asarray([[0., 1., 0.], [.3, 1., 0.], [.7, 1., 1.], [1., 1., 1.]])
        observations = [OutputObservation("exact", "zero", value=0.),
                        OutputObservation("interval", "upper", upper=-.05),
                        OutputObservation("interval", "lower", lower=.1, lower_inclusive=False),
                        OutputObservation("missing", "absent", reason="not_measured")]
        return inputs, observations

    def test_real_ep_mixture_matches_separate_frozen_components(self):
        inputs, observations = self.fixture()
        head = AveragedGroupedHead(inputs, observations, continuous_count=1, seed=7)
        means, variances, evidence = [], [], []
        for continuous in (.05, .15, .4, 1.):
            for discrete in (.5, 1.5):
                divisor = group_divisors(3, 1, continuous, discrete)
                component = LearnedMixedHead(inputs/divisor, observations, seed=7,
                                             length_grid=(1/np.sqrt(3),))
                prediction = component.predict(inputs/divisor)
                means.append(prediction["mean"])
                variances.append(prediction["variance"])
                evidence.append(component.selected["log_evidence"])
                self.assertTrue(component.model.diagnostics["converged"])
        weights = np.exp(np.asarray(evidence)-np.max(evidence))
        weights /= weights.sum()
        prediction = head.predict(inputs)
        mean = weights @ np.asarray(means)
        variance = weights @ (np.asarray(variances)+np.asarray(means)**2)-mean**2
        np.testing.assert_allclose(head.weights, weights)
        np.testing.assert_allclose(prediction["mean"], mean)
        np.testing.assert_allclose(prediction["variance"], variance, atol=1e-12)
        repeated = AveragedGroupedHead(inputs, observations, continuous_count=1, seed=7)
        self.assertEqual(head.identity, repeated.identity)

    def test_single_group_matches_frozen_head_in_physical_objective_units(self):
        inputs, _ = self.fixture()
        observations = [OutputObservation("exact", "a", value=10.),
                        OutputObservation("exact", "b", value=14.),
                        OutputObservation("interval", "c", lower=16., upper=18.),
                        OutputObservation("missing", "d", reason="absent")]
        kwargs = {"continuous_count": 1, "continuous_grid": (.4,), "discrete_grid": (1.5,),
                  "objective": True, "reference_scale": 50., "seed": 13}
        averaged = AveragedGroupedHead(inputs, observations, **kwargs)
        frozen = GroupedMixedHead(inputs, observations, **kwargs)
        for key in ("mean", "variance"):
            np.testing.assert_array_equal(averaged.predict(inputs)[key], frozen.predict(inputs)[key])
        self.assertEqual(averaged.scale, frozen.scale)
        self.assertEqual(averaged.center, frozen.head.center)

    def test_exact_zero_and_missing_are_distinct_without_numeric_imputation(self):
        inputs, _ = self.fixture()
        missing = [OutputObservation("missing", "absent:"+str(index), reason="unobserved")
                   for index in range(len(inputs))]
        prior = AveragedGroupedHead(inputs, missing, continuous_count=1)
        np.testing.assert_array_equal(prior.predict(inputs)["mean"], np.zeros(4))
        np.testing.assert_allclose(prior.predict(inputs)["variance"], np.ones(4))
        np.testing.assert_allclose(prior.weights, np.full(8, .125))
        observed = [OutputObservation("exact", "measured_zero", value=0.)]+missing[1:]
        head = AveragedGroupedHead(inputs, observed, continuous_count=1)
        self.assertLess(head.predict(inputs)["variance"][0], 1e-6)
        for component in head.heads:
            self.assertEqual([value.kind for value in component.model.observations],
                             ["exact", "missing", "missing", "missing"])
            for value in component.model.observations[1:]:
                self.assertIsNone(value.value)
                self.assertIsNone(value.lower)
                self.assertIsNone(value.upper)

    def test_interval_normalization_preserves_bounds_and_inclusion(self):
        inputs, _ = self.fixture()
        observations = [OutputObservation("exact", "a", value=10.),
                        OutputObservation("exact", "b", value=14.),
                        OutputObservation("interval", "c", lower=16., upper=18., lower_inclusive=False),
                        OutputObservation("interval", "d", upper=8., upper_inclusive=False)]
        head = AveragedGroupedHead(inputs, observations, continuous_count=1,
                                   continuous_grid=(.4,), discrete_grid=(1.5,), objective=True)
        self.assertEqual((head.center, head.scale), (12., 2.))
        normalized = head.head.model.observations
        self.assertEqual([value.value for value in normalized[:2]], [-1., 1.])
        self.assertEqual((normalized[2].lower, normalized[2].upper), (2., 3.))
        self.assertFalse(normalized[2].lower_inclusive)
        self.assertIsNone(normalized[2].value)
        self.assertIsNone(normalized[3].lower)
        self.assertEqual(normalized[3].upper, -2.)
        self.assertFalse(normalized[3].upper_inclusive)
        self.assertIsNone(normalized[3].value)

    def test_completion_probit_uses_frozen_unit_noise(self):
        inputs, _ = self.fixture()
        labels = [OutputObservation("interval", "success", upper=0.),
                  OutputObservation("interval", "failure", lower=0., lower_inclusive=False),
                  OutputObservation("missing", "unknown", reason="untrusted_terminal_status"),
                  OutputObservation("missing", "infrastructure", reason="untrusted_terminal_status")]
        head = AveragedGroupedHead(inputs, labels, continuous_count=1, completion=True)
        for component in head.heads:
            self.assertEqual((component.center, component.scale), (0., 1.))
            self.assertEqual(component.model.settings.interval_noise, 1.)
            self.assertEqual(component.model.settings.length_scale, 1.)


class FixtureFeatures:
    def __init__(self, problem, base, records, seed, latent, epochs):
        self.arguments = (problem, base, records, seed, latent, epochs)
        self.identity = digest([record.to_dict() for record in records])

    def transform(self, candidates):
        return np.asarray([[candidate["continuous"]["x"], 1., candidate["discrete"]["branch"]]
                           for candidate in candidates])


class RiskPosteriorTests(unittest.TestCase):
    def setUp(self):
        self.problem = SimpleNamespace(
            space=SimpleNamespace(continuous=("x",)), required_constraints=("g", "analytic"),
            analytical_constraints=("analytic",), objective_reference_scale=10.,
            evaluate=Mock(side_effect=AssertionError("Evaluation is forbidden")))
        self.base = object()
        self.settings = SimpleNamespace(average_kernel=True, kernel="grouped", latent=False,
                                       encoder_epochs=3, continuous_grid=(.15,), discrete_grid=(.5,))

    def record(self, index, objective, constraint, status="completed", analytic=-1.):
        candidate = {"continuous": {"x": index/10}, "discrete": {"branch": index % 2}}
        objective = (objective if isinstance(objective, OutputObservation) else
                     OutputObservation("exact", "objective:"+str(index), value=objective))
        constraint = (constraint if isinstance(constraint, OutputObservation) else
                      OutputObservation("exact", "g:"+str(index), value=constraint))
        packet = RunObservation(digest(candidate), "fixture", "run:"+str(index), status, objective,
            (("g", constraint), ("analytic", OutputObservation("exact", "analytic:"+str(index), value=analytic))),
            failure_cause="declared_failure" if status == "hidden_failure" else None)
        return DesignRecord(candidate, packet, {})

    def fit(self, records):
        with patch(MODULE+".PrefixFeatures", FixtureFeatures):
            return RiskPosterior(self.problem, self.base, records, 17, self.settings)

    def test_evidence_filtering_labels_settings_and_exact_feasible_best(self):
        records = [self.record(0, 0., 0.),
                   self.record(1, OutputObservation("interval", "objective_interval", lower=-10., upper=-5.),
                               OutputObservation("interval", "g_interval", upper=-.1)),
                   self.record(2, -10., 1., "hidden_failure"),
                   self.record(3, -100., -100., "unknown"),
                   self.record(4, -200., -200., "infrastructure_error"),
                   self.record(5, OutputObservation("missing", "objective_missing", reason="not_returned"),
                               OutputObservation("missing", "g_missing", reason="not_returned")),
                   self.record(6, -1., -.1, analytic=1.)]
        with patch(MODULE+".AveragedGroupedHead", return_value=fake_head(0.)) as head:
            posterior = self.fit(records)
        self.assertEqual(head.call_count, 3)
        completion, constraint, objective = head.call_args_list
        labels = completion.args[1]
        self.assertEqual([value.kind for value in labels],
                         ["interval", "interval", "interval", "missing", "missing", "interval", "interval"])
        self.assertEqual(labels[0].upper, 0.)
        self.assertEqual(labels[2].lower, 0.)
        self.assertFalse(labels[2].lower_inclusive)
        for call in (constraint, objective):
            values = call.args[1]
            self.assertEqual(values[0].value, 0.)
            self.assertEqual(values[1].kind, "interval")
            self.assertIsNone(values[1].value)
            for index in (3, 4, 5):
                self.assertEqual(values[index].kind, "missing")
                self.assertIsNone(values[index].value)
                self.assertIsNone(values[index].lower)
                self.assertIsNone(values[index].upper)
        self.assertEqual(set(posterior.constraints), {"g"})
        self.assertEqual(posterior.best, 0.)
        self.assertEqual(posterior.history_digest, digest([record.to_dict() for record in records]))
        self.assertEqual(posterior.features.arguments,
                         (self.problem, self.base, records, 17, False, 3))
        for call, seed in zip(head.call_args_list, (17, 18, 117)):
            self.assertEqual(call.kwargs["seed"], seed)
            self.assertEqual(call.kwargs["continuous_count"], 1)
            self.assertEqual(call.kwargs["continuous_grid"], self.settings.continuous_grid)
            self.assertEqual(call.kwargs["discrete_grid"], self.settings.discrete_grid)
        self.assertTrue(completion.kwargs["completion"])
        self.assertTrue(objective.kwargs["objective"])
        self.assertEqual(objective.kwargs["reference_scale"], 10.)
        self.problem.evaluate.assert_not_called()

    def test_no_usable_objective_does_not_create_an_objective_head(self):
        records = [self.record(0, OutputObservation("missing", "absent", reason="not_returned"), -.2),
                   self.record(1, -100., -.2, "unknown"),
                   self.record(2, -200., -.2, "infrastructure_error")]
        posterior = self.fit(records)
        self.assertIsNone(posterior.objective)
        self.assertIsNone(posterior.best)
        self.assertIsNone(posterior.predict([records[0].candidate])["objective"])
        self.assertNotIn("objective", posterior.diagnostics()["heads"])

    def test_no_exact_feasible_objective_has_no_best(self):
        records = [self.record(0, OutputObservation("interval", "bound", lower=1.), -.2),
                   self.record(1, -2., 1.), self.record(2, -3., -.2, "hidden_failure")]
        posterior = self.fit(records)
        self.assertIsNotNone(posterior.objective)
        self.assertIsNone(posterior.best)

    def test_disabled_average_matches_frozen_posterior(self):
        self.settings.average_kernel = False
        records = [self.record(0, 1., -.2), self.record(1, 3., -.1),
                   self.record(2, OutputObservation("missing", "skip", reason="declared_abort"),
                               OutputObservation("interval", "g_positive", lower=0., lower_inclusive=False))]
        with patch(MODULE+".AveragedGroupedHead", side_effect=AssertionError("Averaging disabled")):
            posterior = self.fit(records)
        with patch("ra_lsbo.partial_refinement.posterior.PrefixFeatures", FixtureFeatures):
            frozen = RefinementPosterior(self.problem, self.base, records, 17, self.settings)
        self.assertIsInstance(posterior.completion, GroupedMixedHead)
        self.assertEqual(posterior.diagnostics(), frozen.diagnostics())
        candidates = [record.candidate for record in records]
        first, second = posterior.predict(candidates), frozen.predict(candidates)
        for key in ("completion", "objective"):
            for moment in ("mean", "variance"):
                np.testing.assert_array_equal(first[key][moment], second[key][moment])
        for moment in ("mean", "variance"):
            np.testing.assert_array_equal(first["constraints"]["g"][moment], second["constraints"]["g"][moment])
        self.assertEqual(first["constraints"]["g"]["scale"], second["constraints"]["g"]["scale"])
        self.problem.evaluate.assert_not_called()

    def test_averaged_posterior_preserves_top_level_predict_and_diagnostics(self):
        self.settings.continuous_grid = (.05, .15, .4, 1.)
        self.settings.discrete_grid = (.5, 1.5)
        records = [self.record(0, 0., -.2), self.record(1, 2., -.1)]
        posterior = self.fit(records)
        prediction = posterior.predict([record.candidate for record in records])
        self.assertEqual(set(prediction), {"completion", "constraints", "objective"})
        self.assertEqual(set(prediction["constraints"]), {"g"})
        self.assertEqual(prediction["constraints"]["g"]["scale"], posterior.constraints["g"].scale)
        diagnostics = posterior.diagnostics()
        self.assertEqual(set(diagnostics), {"feature_identity", "history_digest", "heads", "best_observed"})
        self.assertEqual(set(diagnostics["heads"]), {"completion", "g", "objective"})
        for forecast in (prediction["completion"], prediction["constraints"]["g"], prediction["objective"]):
            self.assertEqual(forecast["components"]["mean"].shape, (8, 2))
            self.assertTrue(np.isfinite(forecast["variance"]).all())
            self.assertTrue((forecast["variance"] >= 0).all())
        json.dumps(diagnostics, allow_nan=False)
        self.problem.evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
