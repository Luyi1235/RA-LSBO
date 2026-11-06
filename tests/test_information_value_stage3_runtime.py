"""Real EP-GP adapter and imports, NOT a mocked scientific module stack."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import runpy
import unittest
from types import SimpleNamespace
import numpy as np

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.models import EvidenceGP
from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import StableGroupedHead
from ra_lsbo.experiments.information_value_stage3.native import FrozenFantasyFactory, FantasyValueBO
from ra_lsbo.experiments.information_value_stage3.planner import Outcome, choose
from ra_lsbo.experiments.information_value_stage3.runner import NativeAdapter, _decision


class Features:
    def transform(self, candidates):
        # The real grouped kernel requires continuous, active-mask and discrete
        # blocks: with one continuous variable dimension must be greater than 2.
        return np.array([[c["x"], 1., 1.] for c in candidates], dtype=float)


def make_factory():
    features = Features()
    x = features.transform([{"x": value} for value in (0., .25, .7, 1.)])
    def head(values, name, objective=False):
        obs = [OutputObservation("exact", name+str(i), value=v) for i, v in enumerate(values)]
        return StableGroupedHead(x, obs, continuous_count=1, seed=19, name=name,
                                 objective=objective, reference_scale=1.)
    heads = {"region": head([.7, .3, -.4, -.2], "region"),
             "resource": head([-.4, -.2, -.3, -.1], "resource")}
    objective = head([8., 6., 3., 5.], "objective", True)
    candidates = [{"x": .4}, {"x": .55}, {"x": .8}, {"x": .95}]
    post = SimpleNamespace(constraints=heads, objective=objective,
                           risk_features=features, objective_features=features)
    forecasts = {"constraints": {name: h.predict(features.transform(candidates)) for name, h in heads.items()},
                 "objective": objective.predict(features.transform(candidates))}
    return FrozenFantasyFactory(post, candidates, forecasts), forecasts


class Stage3RuntimeTests(unittest.TestCase):
    def test_real_abort_is_interval_conditioning_not_exact_value(self):
        factory, original = make_factory()
        before = {n: digest(h.model.to_dict()) for n, h in factory.posterior.constraints.items()}
        result = factory.condition(0, Outcome(True, None, -.3, None))
        self.assertEqual(factory.started, 2)
        self.assertEqual(factory.completed, 2)
        self.assertIs(result["objective"], original["objective"])
        self.assertFalse(np.array_equal(result["constraints"]["region"]["mean"], original["constraints"]["region"]["mean"]))
        self.assertEqual(before, {n: digest(h.model.to_dict()) for n, h in factory.posterior.constraints.items()})

    def test_abort_ablation_retains_resource_but_not_interval_update(self):
        factory, original = make_factory()
        result = factory.condition(0, Outcome(True, None, -.3, None), use_abort_interval=False)
        self.assertEqual(factory.started, 1)
        self.assertIs(result["constraints"]["region"], original["constraints"]["region"])
        self.assertIs(result["objective"], original["objective"])

    def test_completed_infeasible_point_still_updates_observable_objective(self):
        factory, original = make_factory()
        result = factory.condition(0, Outcome(False, -.05, .4, 4.))
        self.assertEqual(factory.completed, 3)
        self.assertFalse(np.array_equal(result["objective"]["mean"], original["objective"]["mean"]))

    def test_frozen_normalization_matches_manual_real_GP_refit(self):
        factory, _ = make_factory()
        h = factory.posterior.objective
        value = 4.2
        result = factory.condition(0, Outcome(False, -.05, -.3, value))
        x = np.vstack((h.model.x, factory.objective_x[0]/h.divisor))
        obs = tuple(h.model.observations) + (OutputObservation("exact", "manual", value=(value-h.center)/h.scale),)
        fitted = EvidenceGP(x, obs, seed=h.model.seed, settings=h.model.settings)
        expected = fitted.predict(factory.objective_x/h.divisor)
        np.testing.assert_allclose(result["objective"]["mean"], h.center+h.scale*expected["mean"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result["objective"]["variance"], h.scale**2*expected["variance"], rtol=1e-10, atol=1e-10)

    def test_actual_two_step_refits_are_executed(self):
        factory, forecast = make_factory()
        _, decision = choose(forecast, 5., 2, factory, arm="cei_two_step", seed=123, iteration=0)
        self.assertGreater(factory.started, 0)
        self.assertEqual(factory.completed, factory.started)
        self.assertIsNone(decision["fallback"])
        self.assertGreater(len(decision["fantasies"]), 0)

    def test_entry_import_does_not_dispatch(self):
        root = Path(__file__).resolve().parents[1]
        namespace = runpy.run_path(str(root/"run_ra_lsbo_information_stage3_pycharm.py"), run_name="pr68_import_test")
        self.assertTrue(callable(namespace["main"]))
        self.assertTrue(callable(NativeAdapter))
        self.assertTrue(callable(FantasyValueBO))

    def test_receipt_hash_uses_native_digest_not_compact_audit_fingerprint(self):
        candidate = {"continuous": {"x": .2}, "discrete": {"branch": 1}}
        receipt = {"selected_candidate_hash": digest(candidate), "selected_predicted_failure": .2, "fantasy_fits": {}}
        self.assertTrue(_decision({"information_value_decision": receipt}, candidate)["forecast_available"])
        bad = deepcopy(receipt)
        bad["selected_candidate_hash"] = "0"*64
        with self.assertRaises(ValueError):
            _decision({"information_value_decision": bad}, candidate)

    def test_coverage_has_no_fake_forecast(self):
        decision = _decision({"mode": "model_failure_coverage"}, {"x": .2})
        self.assertFalse(decision["forecast_available"])
        self.assertIsNone(decision["predicted_failure"])


if __name__ == "__main__":
    unittest.main()
