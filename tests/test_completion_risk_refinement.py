from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr
import torch

from ra_lsbo.risk_core.completion_refinement import IntervalCompletionProbability, RiskWeightedRefinement
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings, MixedObservationGP
from ra_lsbo.risk_core.observations import OutputObservation


def test_interval_probability_matches_ep_and_supports_local_gradients():
    inputs = np.array([[0.1, 0.3], [0.2, 0.4], [0.8, 0.9]])
    evidence = [
        OutputObservation("interval", "complete", upper=0.),
        OutputObservation("interval", "aborted-late", lower=.08, upper=.2),
        OutputObservation("interval", "aborted-early", lower=.3, upper=.55),
    ]
    divisor = np.array([.5, 1.8])
    model = MixedObservationGP(inputs/divisor, evidence, settings=MixedGPSettings(interval_noise=0.))
    probability = IntervalCompletionProbability(SimpleNamespace(model=model, divisor=divisor, scale=1.))
    points = np.vstack((inputs, [[.3, .5], [.6, .7]]))
    prediction = model.predict(points/divisor)
    reference = ndtr(-prediction["mean"]/np.sqrt(np.maximum(prediction["variance"], 1e-14)))
    features = torch.tensor(points, requires_grad=True)
    actual = probability.log_probability(features).exp()
    np.testing.assert_allclose(actual.detach().numpy(), reference, rtol=1e-10, atol=1e-12)
    actual.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert torch.any(features.grad[-2:] != 0)


def test_refinement_replaces_the_legacy_completion_factor_once():
    native = SimpleNamespace(model=None, fit_diagnostics={},
                             log_acquisition=lambda x: torch.full_like(x[..., 0].squeeze(-1), -2.),
                             solver_probabilities=lambda x: {"domain": np.full(len(x), .8),
                                                             "success": np.full(len(x), .4)})
    completion = SimpleNamespace(log_probability=lambda x: torch.full_like(x[..., 0], np.log(.6)))
    surrogate = RiskWeightedRefinement(native, completion)
    points = torch.zeros(2, 4, dtype=torch.double)
    np.testing.assert_allclose(surrogate.score(points), -2.+np.log(.6))
    assert np.allclose(surrogate.solver_probabilities(points)["success"], .6)
    assert np.allclose(surrogate.solver_probabilities(points)["domain"], .8)
