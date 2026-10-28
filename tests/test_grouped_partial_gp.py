"""Synthetic posterior contracts; no evaluator or optimization runs."""

from copy import deepcopy
from dataclasses import replace
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import ndtr
from scipy.stats import norm

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.models import EvidenceGP, LearnedMixedHead
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.partial_bo.selector import PartialBOSettings, choose_acquisition
from ra_lsbo.partial_refinement import posterior
from ra_lsbo.partial_refinement.posterior import (
    GroupedMixedHead, RefinementPosterior, group_divisors, pointwise_acquisition)
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings
from ra_lsbo.risk_core.observations import OutputObservation, RunObservation


def structured_inputs():
    return np.array([
        [0., .2, 1., 1., 1., 0., 0.],
        [.4, .1, 1., 0., 0., 1., 0.],
        [.8, .7, 1., 1., 0., 0., 1.],
        [.2, .9, 0., 1., 1., 0., 0.],
        [.6, .4, 1., 1., 0., 1., 0.],
        [.9, 0., 1., 0., 0., 0., 1.],
    ])


def mixed_observations():
    return [
        OutputObservation("exact", "numeric-0", value=10.),
        OutputObservation("exact", "numeric-1", value=14.),
        OutputObservation("interval", "bounded", lower=9., upper=13.,
                          upper_inclusive=False),
        OutputObservation("interval", "lower-only", lower=12.,
                          lower_inclusive=False),
        OutputObservation("missing", "unobserved", reason="not_measured"),
        OutputObservation("interval", "upper-only", upper=11.),
    ]


def matern52(left, right, divisors):
    radius = np.sqrt(5.) * np.linalg.norm(
        (left[:, None, :] - right[None, :, :]) / divisors, axis=2)
    return (1. + radius + radius**2 / 3.) * np.exp(-radius)


def acquisition_prediction():
    return {
        "completion": {"mean": np.array([-1., .4, 2., -.3, 1.2]),
                       "variance": np.array([.1, .25, .01, 1., .02])},
        "constraints": {
            "stress": {"mean": np.array([-.3, .2, -1., .8, -.2]),
                       "variance": np.array([0., .16, .04, .04, .01]), "scale": 2.},
            "power": {"mean": np.array([-.2, -.1, .1, -.5, -.8]),
                      "variance": np.array([.09, .04, .01, .01, .04]), "scale": .5},
        },
        "objective": {"mean": np.array([.1, .7, -1., .5, .8]),
                      "variance": np.array([.09, .16, .01, .25, .04])},
    }


def prediction_rows(prediction, indices):
    def rows(forecast):
        return {name: value if name == "scale" else np.asarray(value)[indices].copy()
                for name, value in forecast.items()}
    return {"completion": rows(prediction["completion"]),
            "constraints": {name: rows(value)
                            for name, value in prediction["constraints"].items()},
            "objective": None if prediction["objective"] is None else rows(prediction["objective"])}


class GroupDivisorTests(unittest.TestCase):
    def test_normalized_blocks_define_a_positive_diagonal_metric(self):
        for dimension, count in ((3, 1), (7, 2), (11, 3)):
            with self.subTest(dimension=dimension, continuous_count=count):
                divisors = group_divisors(dimension, count, .3, 1.7)
                expected = np.concatenate((np.repeat(.3*np.sqrt(count), count),
                    np.repeat(np.sqrt(count), count),
                    np.repeat(1.7*np.sqrt(dimension-2*count), dimension-2*count)))
                assert_allclose(divisors, expected, rtol=1e-15)
                self.assertEqual(divisors.shape, (dimension,))
                self.assertTrue(np.all(np.isfinite(divisors)))
                self.assertTrue(np.all(np.linalg.eigvalsh(np.diag(divisors**-2)) > 0))
                displacement = np.arange(1., dimension+1)
                expected_distance = (
                    np.mean(displacement[:count]**2)/.3**2
                    + np.mean(displacement[count:2*count]**2)
                    + np.mean(displacement[2*count:]**2)/1.7**2)
                assert_allclose(np.sum((displacement/divisors)**2), expected_distance)

    def test_changing_one_length_changes_only_its_block(self):
        original = group_divisors(7, 2, .3, 1.7)
        continuous = group_divisors(7, 2, .6, 1.7)
        discrete = group_divisors(7, 2, .3, 3.4)
        assert_allclose(continuous/original, [2., 2., 1., 1., 1., 1., 1.])
        assert_allclose(discrete/original, [1., 1., 1., 1., 2., 2., 2.])

    def test_invalid_group_sizes_and_lengths_are_rejected(self):
        invalid = [(0, 1, 1., 1.), (4, 2, 1., 1.), (3, 2, 1., 1.),
                   (7, 0, 1., 1.), (7, -1, 1., 1.), (7., 2, 1., 1.),
                   (7, 2., 1., 1.), (True, 1, 1., 1.), (7, True, 1., 1.)]
        for length in (0., -1., np.nan, np.inf, -np.inf):
            invalid.extend(((7, 2, length, 1.), (7, 2, 1., length)))
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                group_divisors(*arguments)


class GroupedPosteriorMathTests(unittest.TestCase):
    def test_exact_objective_matches_independent_gaussian_conditioning(self):
        inputs = structured_inputs()[:4]
        values = np.array([8., 9., 12., 11.])
        observations = [OutputObservation("exact", str(index), value=float(value))
                        for index, value in enumerate(values)]
        head = GroupedMixedHead(inputs, observations, continuous_count=2,
            continuous_grid=(.35,), discrete_grid=(1.2,), objective=True, seed=11)
        divisors = np.array([.35*np.sqrt(2)]*2 + [np.sqrt(2)]*2 + [1.2*np.sqrt(3)]*3)
        query = structured_inputs()[[0, 4, 5]]
        covariance = matern52(inputs, inputs, divisors)
        assert_allclose(np.diag(covariance), 1.)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.)
        assert_allclose(head.head.model._kernel(inputs/divisors, inputs/divisors),
                        covariance, atol=1e-14)
        covariance += np.eye(len(inputs))*(1e-6+1e-8)
        center, scale = values.mean(), values.std()
        normalized = (values-center)/scale
        cross = matern52(query, inputs, divisors)
        expected_mean = center+scale*cross@np.linalg.solve(covariance, normalized)
        expected_variance = scale**2*(1.-np.sum(
            cross*np.linalg.solve(covariance, cross.T).T, axis=1))
        log_evidence = (-.5*normalized@np.linalg.solve(covariance, normalized)
                        -.5*np.linalg.slogdet(covariance)[1]
                        -.5*len(inputs)*np.log(2*np.pi))
        actual = head.predict(query)
        assert_allclose(actual["mean"], expected_mean, atol=1e-11)
        assert_allclose(actual["variance"], expected_variance, atol=1e-11)
        assert_allclose(head.selected["log_evidence"], log_evidence, atol=1e-11)

    def test_mixed_ep_matches_diagonal_ard_gp_for_every_grid_candidate(self):
        inputs, observations = structured_inputs(), mixed_observations()
        normalized = [replace(value, value=(value.value-12.)/2.) if value.kind == "exact"
            else replace(value,
                lower=None if value.lower is None else (value.lower-12.)/2.,
                upper=None if value.upper is None else (value.upper-12.)/2.)
            if value.kind == "interval" else value for value in observations]
        head = GroupedMixedHead(inputs, observations, continuous_count=2,
            continuous_grid=(.2, .7), discrete_grid=(.5, 1.5), objective=True,
            reference_scale=25., seed=29)
        references = []
        for candidate in head.candidates:
            with self.subTest(candidate=candidate):
                continuous, discrete = candidate["continuous_length"], candidate["discrete_length"]
                divisors = ([continuous*np.sqrt(2)]*2 + [np.sqrt(2)]*2
                            + [discrete*np.sqrt(3)]*3)
                reference = EvidenceGP(inputs, normalized, seed=29, settings=MixedGPSettings(
                    length_scale=divisors, exact_noise=1e-3, jitter=1e-8))
                self.assertEqual(candidate["status"], "ok")
                assert_allclose(candidate["log_evidence"], reference.log_evidence(), atol=1e-9)
                references.append(reference)
        evidence = [reference.log_evidence() for reference in references]
        selected = int(np.argmax(evidence))
        self.assertEqual(head.selected, head.candidates[selected])
        self.assertGreater(np.ptp(evidence), 1e-3)
        query = np.vstack((inputs[[1, 3]], np.mean(inputs, axis=0)))
        expected = references[selected].predict(query)
        actual = head.predict(query)
        assert_allclose(actual["mean"], 12.+2.*expected["mean"], atol=1e-8)
        assert_allclose(actual["variance"], 4.*expected["variance"], atol=1e-8)

    def test_equal_blocks_recover_frozen_single_scale_head(self):
        inputs = structured_inputs()[:, :6]
        observations = mixed_observations()
        grouped = GroupedMixedHead(inputs, observations, continuous_count=2,
            continuous_grid=(1.,), discrete_grid=(1.,), objective=True, seed=7)
        frozen = LearnedMixedHead(inputs, observations, objective=True,
                                  length_grid=(np.sqrt(2./6.),), seed=7)
        query = np.vstack((inputs[[0, 3]], inputs.mean(axis=0)))
        for name, actual in grouped.predict(query).items():
            assert_allclose(actual, frozen.predict(query)[name], atol=1e-8)
        assert_allclose(grouped.selected["log_evidence"], frozen.selected["log_evidence"], atol=1e-9)

    def test_exact_plus_one_bound_matches_conditional_truncated_normal(self):
        inputs = structured_inputs()[:2]
        observations = [OutputObservation("exact", "numeric", value=-1.),
                        OutputObservation("interval", "positive", lower=0., lower_inclusive=False)]
        head = GroupedMixedHead(inputs, observations, continuous_count=2,
                               continuous_grid=(.8,), discrete_grid=(1.2,))
        divisors = np.array([.8*np.sqrt(2)]*2 + [np.sqrt(2)]*2 + [1.2*np.sqrt(3)]*3)
        correlation = matern52(inputs, inputs, divisors)[0, 1]
        conditional_mean = -correlation/(1.+1e-6+1e-8)
        conditional_variance = 1.-correlation**2/(1.+1e-6+1e-8)
        alpha = -conditional_mean/np.sqrt(conditional_variance)
        ratio = norm.pdf(alpha)/ndtr(-alpha)
        expected_mean = conditional_mean+np.sqrt(conditional_variance)*ratio
        expected_variance = conditional_variance*(1.+alpha*ratio-ratio**2)
        actual = head.predict(inputs[[1]])
        assert_allclose(actual["mean"], [expected_mean], atol=5e-8)
        assert_allclose(actual["variance"], [expected_variance], atol=5e-8)
        exact_log = norm.logpdf(-1., scale=np.sqrt(1.+1e-6+1e-8))
        expected_evidence = exact_log+norm.logsf(
            -conditional_mean/np.sqrt(conditional_variance+1e-8))
        assert_allclose(head.selected["log_evidence"], expected_evidence, atol=1e-7)


class GroupedPosteriorContractTests(unittest.TestCase):
    def test_grid_fits_only_the_supplied_prefix_and_forwards_likelihood_options(self):
        inputs, observations = structured_inputs()[:3], mixed_observations()[:3]
        input_before, observations_before = inputs.copy(), deepcopy(observations)
        candidates = [SimpleNamespace(selected={"log_evidence": evidence},
            scale=3., identity="mock-head-"+str(index))
            for index, evidence in enumerate((-8., -2., -5., -9.))]
        with patch.object(posterior, "LearnedMixedHead", side_effect=candidates) as fit:
            head = GroupedMixedHead(inputs, observations, continuous_count=2,
                continuous_grid=(.2, .7), discrete_grid=(.5, 1.5),
                objective=True, completion=False, reference_scale=30., seed=19)
        self.assertEqual(fit.call_count, 4)
        self.assertIs(head.head, candidates[1])
        self.assertEqual(head.selected["log_evidence"], -2.)
        for call, (continuous, discrete) in zip(fit.call_args_list,
                ((.2, .5), (.2, 1.5), (.7, .5), (.7, 1.5))):
            expected_divisors = np.array([continuous*np.sqrt(2)]*2
                + [np.sqrt(2)]*2 + [discrete*np.sqrt(3)]*3)
            assert_allclose(call.args[0], input_before/expected_divisors)
            self.assertIs(call.args[1], observations)
            self.assertEqual(call.kwargs, {"seed": 19, "length_grid": (1./np.sqrt(7),),
                "objective": True, "completion": False, "reference_scale": 30.})
        assert_array_equal(inputs, input_before)
        self.assertEqual(observations, observations_before)

    def test_failed_kernel_is_excluded_without_replacing_its_evidence(self):
        valid = SimpleNamespace(selected={"log_evidence": -12.}, scale=1., identity="valid")
        with patch.object(posterior, "LearnedMixedHead", side_effect=[
                FloatingPointError("synthetic EP failure"), valid]):
            head = GroupedMixedHead(structured_inputs(), mixed_observations(),
                continuous_count=2, continuous_grid=(.2,), discrete_grid=(.5, 1.5))
        self.assertEqual(head.candidates[0]["status"], "FloatingPointError")
        self.assertIsNone(head.candidates[0]["log_evidence"])
        self.assertIs(head.head, valid)

    def test_all_failed_kernels_raise_instead_of_inventing_a_posterior(self):
        with patch.object(posterior, "LearnedMixedHead", side_effect=RuntimeError("failed EP")):
            with self.assertRaisesRegex(RuntimeError, "No grouped kernel"):
                GroupedMixedHead(structured_inputs(), mixed_observations(), continuous_count=2,
                                 continuous_grid=(.2, .7), discrete_grid=(.5,))

    def test_normalization_preserves_exact_interval_missing_evidence(self):
        inputs, observations = structured_inputs(), mixed_observations()
        before = deepcopy(observations)
        head = GroupedMixedHead(inputs, observations, continuous_count=2,
            continuous_grid=(.4,), discrete_grid=(1.5,), objective=True)
        self.assertEqual((head.head.center, head.scale), (12., 2.))
        normalized = head.head.model.observations
        self.assertEqual([value.kind for value in normalized],
                         ["exact", "exact", "interval", "interval", "missing", "interval"])
        self.assertEqual([normalized[index].value for index in (0, 1)], [-1., 1.])
        self.assertEqual((normalized[2].lower, normalized[2].upper), (-1.5, .5))
        self.assertEqual((normalized[3].lower, normalized[3].upper), (0., None))
        self.assertEqual((normalized[5].lower, normalized[5].upper), (None, -.5))
        self.assertEqual(normalized[4], observations[4])
        for original, transformed in zip(observations, normalized):
            self.assertEqual(transformed.evidence_id, original.evidence_id)
            self.assertEqual(transformed.lower_inclusive, original.lower_inclusive)
            self.assertEqual(transformed.upper_inclusive, original.upper_inclusive)
            if original.kind != "exact":
                self.assertIsNone(transformed.value)
        self.assertEqual(observations, before)
        counts = head.diagnostics()["counts"]
        self.assertEqual((counts["n_exact"], counts["n_interval"], counts["n_missing"]), (2, 3, 1))

    def test_missing_rows_cannot_change_evidence_selection_or_predictions(self):
        inputs, observations = structured_inputs(), mixed_observations()
        keep = [index for index, value in enumerate(observations) if value.kind != "missing"]
        settings = dict(continuous_count=2, continuous_grid=(.2, .7),
                        discrete_grid=(.5, 1.5), objective=True, seed=31)
        supported = GroupedMixedHead(inputs[keep], [observations[index] for index in keep], **settings)
        extended = GroupedMixedHead(np.vstack((inputs, np.full((1, 7), 100.))),
            observations+[OutputObservation("missing", "future", reason="not_executed")], **settings)
        self.assertEqual(supported.candidates, extended.candidates)
        self.assertEqual(supported.selected, extended.selected)
        for name, expected in supported.predict(inputs).items():
            assert_allclose(extended.predict(inputs)[name], expected, atol=1e-12)

    def test_all_missing_objective_retains_prior_and_does_not_fabricate_targets(self):
        observations = [OutputObservation("missing", str(index), reason="not_executed")
                        for index in range(6)]
        head = GroupedMixedHead(structured_inputs(), observations, continuous_count=2,
            continuous_grid=(.2, .7), discrete_grid=(.5,), objective=True, reference_scale=3.)
        actual = head.predict(structured_inputs()[[0, 3]])
        assert_allclose(actual["mean"], [0., 0.])
        assert_allclose(actual["variance"], [9., 9.])
        self.assertTrue(head.head.model.diagnostics["prior_only"])
        self.assertEqual(head.head.model.diagnostics["n_exact"], 0)
        self.assertTrue(all(value["log_evidence"] == 0. for value in head.candidates))

    def test_completion_uses_soft_sign_likelihood_and_unit_scale(self):
        inputs = structured_inputs()[:3]
        observations = [OutputObservation("interval", "success", upper=0.),
                        OutputObservation("interval", "failure", lower=0., lower_inclusive=False),
                        OutputObservation("missing", "unknown", reason="no_completion_label")]
        head = GroupedMixedHead(inputs, observations, continuous_count=2,
            continuous_grid=(.4,), discrete_grid=(1.5,), completion=True, reference_scale=20.)
        self.assertEqual((head.head.center, head.scale), (0., 1.))
        self.assertEqual(head.head.model.settings.interval_noise, 1.)
        self.assertEqual(head.head.model.observations, tuple(observations))
        self.assertEqual(head.head.model.diagnostics["n_exact"], 0)

    def test_queries_leave_all_training_and_ep_state_unchanged(self):
        inputs, observations = structured_inputs(), mixed_observations()
        original_inputs = inputs.copy()
        head = GroupedMixedHead(inputs, observations, continuous_count=2,
            continuous_grid=(.2, .7), discrete_grid=(.5, 1.5), objective=True, seed=41)
        before = pickle.dumps(head, protocol=pickle.HIGHEST_PROTOCOL)
        diagnostics = deepcopy(head.diagnostics())
        query = np.vstack((inputs[[0, 3]], np.full((1, 7), 20.)))
        query_before = query.copy()
        with patch.object(posterior, "LearnedMixedHead", side_effect=AssertionError("query refit")), \
                patch.object(EvidenceGP, "log_evidence", side_effect=AssertionError("query selection")):
            batch = head.predict(query)
            for index in range(len(query)):
                single = head.predict(query[[index]])
                for name in ("mean", "variance"):
                    self.assertEqual(batch[name].shape, (len(query),))
                    assert_allclose(single[name], batch[name][[index]], atol=1e-10)
                    self.assertTrue(np.isfinite(batch[name]).all())
            for name, value in head.predict(query[::-1]).items():
                assert_allclose(value, batch[name][::-1], atol=1e-10)
            self.assertTrue(np.all(batch["variance"] >= 0.))
        self.assertEqual(pickle.dumps(head, protocol=pickle.HIGHEST_PROTOCOL), before)
        self.assertEqual(head.diagnostics(), diagnostics)
        assert_array_equal(inputs, original_inputs)
        assert_array_equal(query, query_before)


class RefinementEvidenceContractTests(unittest.TestCase):
    def test_terminal_status_controls_labels_without_calling_a_problem(self):
        problem = Mock()
        problem.contract_id = "synthetic-contract"
        problem.space.continuous = ("first", "second")
        problem.required_constraints = ("physical", "analytical")
        problem.analytical_constraints = ("analytical",)
        problem.objective_reference_scale = 10.
        problem.evaluate.side_effect = AssertionError("No truth calls in posterior tests")
        records = []
        statuses = ("completed", "hidden_failure", "unknown", "infrastructure_error")
        for index, status in enumerate(statuses):
            candidate = {"index": index}
            objective = (OutputObservation("missing", "objective-1", reason="not_reached")
                         if status == "hidden_failure" else
                         OutputObservation("exact", "objective-"+str(index), value=5.-index))
            physical = (OutputObservation("interval", "physical-1", lower=0., lower_inclusive=False)
                        if status == "hidden_failure" else
                        OutputObservation("exact", "physical-"+str(index), value=-1.))
            packet = RunObservation(digest(candidate), problem.contract_id, "run-"+str(index),
                status, objective, (("physical", physical),
                    ("analytical", OutputObservation("exact", "analytic-"+str(index), value=-1.))),
                failure_cause="synthetic_failure" if status == "hidden_failure" else None)
            records.append(DesignRecord(candidate, packet, {}))
        settings = SimpleNamespace(latent=False, encoder_epochs=1, kernel="grouped",
                                   continuous_grid=(.4,), discrete_grid=(1.5,))
        features = Mock(identity="synthetic-features")
        features.transform.return_value = structured_inputs()[:4]
        with patch.object(posterior, "PrefixFeatures", return_value=features):
            model = RefinementPosterior(problem, None, records, 53, settings)
            prediction = model.predict([record.candidate for record in records])
        completion = model.completion.head.model.observations
        self.assertEqual([value.kind for value in completion], ["interval", "interval", "missing", "missing"])
        self.assertEqual((completion[0].lower, completion[0].upper), (None, 0.))
        self.assertEqual((completion[1].lower, completion[1].upper), (0., None))
        self.assertFalse(completion[1].lower_inclusive)
        self.assertEqual(set(model.constraints), {"physical"})
        physical = model.constraints["physical"].head.model.observations
        self.assertEqual([value.kind for value in physical], ["exact", "interval", "missing", "missing"])
        self.assertIsNone(physical[1].value)
        objective = model.objective.head.model.observations
        self.assertEqual([value.kind for value in objective], ["exact", "missing", "missing", "missing"])
        self.assertEqual(model.best, 5.)
        self.assertEqual(model.history_digest, digest([record.to_dict() for record in records]))
        self.assertEqual(prediction["constraints"]["physical"]["scale"], model.constraints["physical"].scale)
        problem.evaluate.assert_not_called()


class PointwiseAcquisitionTests(unittest.TestCase):
    modes = ("feasible_discovery", "balanced_feasible_improvement",
             "budgeted_boundary_information", "posterior_support_restoration")

    def test_scores_and_support_margins_are_invariant_to_query_batch(self):
        prediction = acquisition_prediction()
        before = pickle.dumps(prediction, protocol=pickle.HIGHEST_PROTOCOL)
        for mode in self.modes:
            with self.subTest(mode=mode):
                scores, margins = pointwise_acquisition(prediction, .75, mode, 1.25, .55)
                self.assertEqual(scores.shape, (5,))
                self.assertEqual(margins.shape, (5, 3))
                for indices in ([0], [1], [2], [3], [4], [4, 1, 0], [1, 3, 1, 2, 4, 0]):
                    actual_scores, actual_margins = pointwise_acquisition(
                        prediction_rows(prediction, indices), .75, mode, 1.25, .55)
                    assert_allclose(actual_scores, scores[indices], atol=1e-14, rtol=1e-13)
                    assert_allclose(actual_margins, margins[indices], atol=1e-14, rtol=1e-13)
        self.assertEqual(pickle.dumps(prediction, protocol=pickle.HIGHEST_PROTOCOL), before)

    def test_margins_and_all_scores_match_independent_pointwise_formulas(self):
        prediction = acquisition_prediction()
        completed = prediction["completion"]
        success = ndtr(-completed["mean"]/np.sqrt(1.+completed["variance"]))
        margins = [ndtr(-completed["mean"]+1.25*np.sqrt(completed["variance"]))-.55]
        balanced, joint, uncertainty = np.log(success), success.copy(), np.zeros(5)
        for forecast in prediction["constraints"].values():
            mean, variance = forecast["mean"], forecast["variance"]
            sd = np.sqrt(variance)
            margins.append((1.25*sd-mean)/forecast["scale"])
            positive = variance > 0
            ratio = np.divide(mean, sd, out=np.zeros_like(mean), where=positive)
            probability = np.where(positive, ndtr(-ratio), (mean <= 0).astype(float))
            band_mass = ndtr(1.96-ratio)-ndtr(-1.96-ratio)
            factor = np.where(positive, np.clip((1.+band_mass)*probability, 0., 1.), probability)
            balanced += np.log(np.maximum(factor, 1e-300))
            joint *= probability
            uncertainty += sd/forecast["scale"]
        expected_margins = np.column_stack(margins)
        objective = prediction["objective"]
        sd = np.sqrt(objective["variance"])
        standardized = (.75-objective["mean"])/sd
        improvement = (.75-objective["mean"])*ndtr(standardized)+sd*norm.pdf(standardized)
        probability = np.clip(joint, 1e-12, 1.-1e-12)
        entropy = -probability*np.log(probability)-(1.-probability)*np.log(1.-probability)
        expected = {"feasible_discovery": balanced,
                    "balanced_feasible_improvement": balanced+np.log(improvement),
                    "budgeted_boundary_information": entropy+.01*uncertainty/(1.+uncertainty),
                    "posterior_support_restoration": -np.maximum(-expected_margins, 0.).sum(axis=1)}
        for mode in self.modes:
            with self.subTest(mode=mode):
                scores, actual_margins = pointwise_acquisition(prediction, .75, mode, 1.25, .55)
                assert_allclose(scores, expected[mode], atol=1e-14)
                assert_allclose(actual_margins, expected_margins, atol=1e-14)

    def test_normal_modes_match_frozen_acquisition_on_eligible_candidates(self):
        prediction = acquisition_prediction()
        cases = ((None, 0., "feasible_discovery"),
                 (.75, 0., "balanced_feasible_improvement"),
                 (.75, .25, "budgeted_boundary_information"))
        for best, boundary_fraction, expected_mode in cases:
            with self.subTest(mode=expected_mode):
                settings = PartialBOSettings(horizon=9, boundary_fraction=boundary_fraction,
                    completion_threshold=.55, initial_kappa=1.75, final_kappa=.75)
                index, metadata = choose_acquisition(prediction, best, 3, settings)
                self.assertIsNone(metadata["fallback"])
                self.assertEqual(metadata["mode"], expected_mode)
                scores, margins = pointwise_acquisition(prediction, best, metadata["mode"],
                    metadata["kappa"], settings.completion_threshold)
                eligible = np.asarray(metadata["eligible"], dtype=bool)
                self.assertTrue(eligible.any())
                self.assertFalse(eligible.all())
                self.assertTrue(np.isfinite(scores).all())
                assert_array_equal(np.all(margins >= 0., axis=1), eligible)
                assert_allclose(scores[eligible], np.array(metadata["scores"], dtype=float)[eligible],
                                rtol=1e-13, atol=1e-14)
                self.assertTrue(all(value is None for value, supported in
                                    zip(metadata["scores"], eligible) if not supported))
                self.assertEqual(int(np.argmax(np.where(eligible, scores, -np.inf))), index)

    def test_no_constraints_has_one_completion_margin_and_correct_probit(self):
        prediction = {"completion": {"mean": np.array([0., 1., -1.]),
                                     "variance": np.array([0., 4., .25])},
                      "constraints": {}, "objective": None}
        expected = np.log(ndtr(-prediction["completion"]["mean"]/
                              np.sqrt(1.+prediction["completion"]["variance"])))
        scores, margins = pointwise_acquisition(prediction, None, "feasible_discovery", 2., .5)
        assert_allclose(scores, expected, atol=1e-14)
        assert_allclose(margins[:, 0], ndtr(-prediction["completion"]["mean"]
            + 2.*np.sqrt(prediction["completion"]["variance"]))-.5, atol=1e-14)
        self.assertEqual(margins.shape, (3, 1))

    def test_deterministic_constraints_include_zero_and_keep_log_scores_finite(self):
        prediction = {"completion": {"mean": np.full(3, -1.), "variance": np.zeros(3)},
                      "constraints": {"physical": {"mean": np.array([-1., 0., 1.]),
                                      "variance": np.zeros(3), "scale": 2.}},
                      "objective": None}
        scores, margins = pointwise_acquisition(prediction, None, "feasible_discovery", 1., .5)
        assert_allclose(scores, np.log(ndtr(1.))+np.array([0., 0., np.log(1e-300)]))
        assert_allclose(margins[:, 1], [.5, 0., -.5])
        self.assertTrue(np.isfinite(scores).all())
        assert_array_equal(np.all(margins >= 0., axis=1), [True, True, False])

    def test_missing_objective_or_incumbent_leaves_discovery_score(self):
        for best, with_objective in ((None, True), (.75, False), (None, False)):
            with self.subTest(best=best, with_objective=with_objective):
                prediction = acquisition_prediction()
                if not with_objective:
                    prediction["objective"] = None
                expected = pointwise_acquisition(prediction, best, "feasible_discovery", 1.25, .55)
                actual = pointwise_acquisition(prediction, best, "balanced_feasible_improvement", 1.25, .55)
                for result, reference in zip(actual, expected):
                    assert_allclose(result, reference, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
