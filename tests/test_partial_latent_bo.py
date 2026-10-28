from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_ndtr

from ra_lsbo.partial_bo.models import EvidenceGP, LearnedMixedHead, PrefixFeatures, log_interval_probability
from ra_lsbo.partial_bo.problems import (
    HierarchicalBranin, SimplifiedElectricAircraft, actuator_disk_power, candidate_pool)
from ra_lsbo.partial_bo.records import DesignRecord, without_intervals
from ra_lsbo.partial_bo.selector import PartialBOSettings, PartialLatentBO, choose_acquisition
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings
from ra_lsbo.risk_core.observations import OutputObservation

BASE = (Path(__file__).resolve().parents[1]/"runs/compact_mda_20261021T081006Z_20eb608d/"
        "qualification_recovered_20261025/frozen_base")


class EvidenceLikelihoodTests(unittest.TestCase):
    def test_single_interval_normalizer_matches_exact_gaussian_probability(self):
        for lower, upper in ((None, 0.), (1., None), (-.5, .8)):
            observed = OutputObservation("interval", "bound", lower=lower, upper=upper)
            model = EvidenceGP([[0.]], [observed])
            sd = np.sqrt(1+model.settings.jitter)
            expected = log_interval_probability(
                -np.inf if lower is None else lower/sd,
                np.inf if upper is None else upper/sd)
            assert_allclose(model.log_evidence(), expected, atol=1e-7)

    def test_numeric_evidence_matches_gaussian_log_density(self):
        values = np.asarray([.2, -.1, .5])
        inputs = np.asarray([[0.], [.4], [1.]])
        model = EvidenceGP(inputs, [OutputObservation("exact", str(index), value=float(value))
                                   for index, value in enumerate(values)])
        covariance = model._kernel(inputs, inputs)
        covariance.flat[::4] += model.settings.exact_noise**2+model.settings.jitter
        factor = cho_factor(covariance, lower=True)
        expected = (-.5*values@cho_solve(factor, values)-np.log(np.diag(factor[0])).sum()
                    -1.5*np.log(2*np.pi))
        assert_allclose(model.log_evidence(), expected, atol=1e-10)

    def test_exact_plus_one_sign_matches_conditional_probability(self):
        model = EvidenceGP([[0.], [.5]], [OutputObservation("exact", "numeric", value=.2),
                                           OutputObservation("interval", "sign", upper=0.)])
        mean = model._baseline_mean(np.asarray([[.5]]))[0]
        variance = model._baseline_covariance(np.asarray([[.5]]), np.asarray([[.5]]))[0, 0]
        variance += model.settings.jitter
        exact_model = EvidenceGP([[0.]], [OutputObservation("exact", "numeric", value=.2)])
        expected = exact_model.log_evidence()+log_ndtr(-mean/np.sqrt(variance))
        assert_allclose(model.log_evidence(), expected, atol=1e-7)

    def test_duplicate_hard_signs_do_not_multiply_evidence(self):
        sign = OutputObservation("interval", "sign", upper=0.)
        single = EvidenceGP([[0.]], [sign])
        repeated = EvidenceGP([[0.], [0.]], [sign, replace(sign, evidence_id="again")])
        assert_allclose(single.log_evidence(), repeated.log_evidence(), atol=1e-10)

    def test_kernel_learning_selects_only_by_prefix_evidence(self):
        inputs = np.linspace(0, 1, 7)[:, None]
        observed = [OutputObservation("exact", str(index), value=float(np.sin(value)))
                    for index, value in enumerate(inputs[:, 0])]
        head = LearnedMixedHead(inputs, observed, objective=True)
        self.assertEqual(head.selected["log_evidence"], max(
            candidate["log_evidence"] for candidate in head.candidates if candidate["status"] == "ok"))
        before = head.identity
        head.predict([[2.], [3.]])
        self.assertEqual(head.identity, before)


class AcquisitionAndRecordTests(unittest.TestCase):
    def predictions(self):
        return {"completion": {"mean": [2., -1.], "variance": [.01, .04]},
                "constraints": {"g": {"mean": [-10., -.1], "variance": [.01, .1], "scale": 1.}},
                "objective": {"mean": [-100., 0.], "variance": [1., 1.]}}

    def test_completion_support_blocks_confident_unusable_candidate(self):
        index, metadata = choose_acquisition(self.predictions(), None, 0, PartialBOSettings())
        self.assertEqual(index, 1)
        self.assertFalse(metadata["eligible"][0])
        self.assertFalse(metadata["probability_guarantee"])

    def test_uncertain_completion_is_not_rejected_by_mean_probability(self):
        forecasts = self.predictions()
        forecasts["completion"] = {"mean": [1., 1.], "variance": [4., 4.]}
        _, metadata = choose_acquisition(forecasts, None, 0, PartialBOSettings())
        self.assertTrue(all(metadata["eligible"]))

    def test_boundary_allocation_stays_within_declared_fraction(self):
        settings = PartialBOSettings(horizon=24, boundary_fraction=.2)
        slots = sum(choose_acquisition(self.predictions(), None, iteration, settings)[1]["boundary_slot"]
                    for iteration in range(settings.horizon))
        self.assertEqual(slots, int(np.floor(settings.horizon*settings.boundary_fraction)))

    def test_missing_physical_constraint_is_not_declared_infeasible(self):
        problem = HierarchicalBranin(BASE)
        candidate = problem.anchors()[0]
        record = problem.record(candidate, OutputObservation("missing", "objective", reason="not_executed"),
            {"region": OutputObservation("missing", "region", reason="not_executed"), "resource": -.2},
            {"termination_reason": "numerical_failure", "failure_cause": "residual_too_large"},
            status="hidden_failure")
        self.assertIsNone(record.physical_feasible)
        self.assertFalse(record.usable_feasible)
        self.assertEqual(DesignRecord.from_dict(record.to_dict()), record)

    def test_interval_ablation_does_not_turn_bounds_into_pseudo_values(self):
        problem = HierarchicalBranin(BASE)
        record = problem.record(problem.anchors()[0], 3., {
            "region": OutputObservation("interval", "region", lower=0., lower_inclusive=False),
            "resource": -.2}, {"termination_reason": "controlled_feedback"})
        self.assertFalse(record.physical_feasible)
        removed = without_intervals(record)
        self.assertIsNone(removed.physical_feasible)
        self.assertEqual(removed.observation.constraint("region").kind, "missing")
        self.assertIsNone(record.legacy_row(1)["risk"]["constraints"]["region"])


class PrefixAndAircraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem = HierarchicalBranin(BASE)
        candidates, _ = candidate_pool(cls.problem, 91, 6)
        cls.records = [cls.problem.record(candidate, 10.+index, {"region": -.2, "resource": -.1},
                                         {"termination_reason": "mock_unit_fixture"})
                       for index, candidate in enumerate(candidates)]

    def test_prefix_encoder_and_selector_replay_without_truth(self):
        candidates, _ = candidate_pool(self.problem, 92, 8, [record.observation.design_id for record in self.records])
        settings = PartialBOSettings(encoder_epochs=8, length_grid=(.5, 1.))
        left = PartialLatentBO(self.problem, BASE, 90, settings)
        right = PartialLatentBO(self.problem, BASE, 90, settings)
        proposed, metadata = left.propose(self.records, candidates, 0)
        self.assertEqual((proposed, metadata), right.propose(self.records, candidates, 0))
        with self.assertRaises(RuntimeError):
            left.propose(self.records, candidates, 1)
        wrong = self.records[0]
        with self.assertRaises(ValueError):
            left.accept(wrong)
        settled = self.problem.record(proposed, 1., {"region": -.2, "resource": -.1},
                                      {"termination_reason": "mock_unit_fixture"})
        left.accept(settled)
        self.assertIsNone(left.pending)

    def test_prefix_transform_rejects_encoder_mutation(self):
        features = PrefixFeatures(self.problem, BASE, self.records, 93, epochs=3)
        with torch.no_grad():
            next(features.encoder.model.parameters()).add_(1.)
        with self.assertRaises(RuntimeError):
            features.transform([self.records[0].candidate])

    def test_ideal_disk_obeys_power_and_disk_area_limits(self):
        small = actuator_disk_power(800., 50., 1.225, 2.)
        large = actuator_disk_power(800., 50., 1.225, 8.)
        self.assertGreater(small, large)
        self.assertGreater(large, 800*50)
        self.assertEqual(actuator_disk_power(0., 50., 1.225, 2.), 0.)

    def test_aircraft_conditional_geometry_and_reference_anchors(self):
        problem = SimplifiedElectricAircraft(BASE)
        for candidate in problem.anchors():
            self.assertTrue(all(value <= 0 for value in problem.precheck(candidate).values()))
        candidate = problem.anchors()[0]
        modified = {"continuous": dict(candidate["continuous"]), "discrete": dict(candidate["discrete"])}
        modified["continuous"]["dep_diameter"] = 1.1
        self.assertEqual(problem.canonical(candidate), problem.canonical(modified))


if __name__ == "__main__":
    unittest.main()
