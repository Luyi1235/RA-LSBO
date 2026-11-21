from types import SimpleNamespace

import numpy as np
from scipy.special import log_ndtr
import torch

from aircraft_censored_refinement import CensoredEnergyRefinement, LogEnergyImprovement
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings, MixedObservationGP
from ra_lsbo.risk_core.observations import OutputObservation
from run_aircraft_refinement_confirmation import registration, shard_plan
from run_aircraft_refinement_fork import CONTROL, PROFILE, TREATMENT, UsableAircraft, run_fork, verify_fork


def test_differentiable_improvement_matches_ep_and_preserves_native_risk():
    inputs = np.array([[0., .3], [1., .6], [.5, .3]])
    divisor = np.array([.5, 1.8])
    observations = [OutputObservation("exact", "best", value=0.),
                    OutputObservation("exact", "other", value=.3),
                    OutputObservation("interval", "abort", lower=.15)]
    settings = MixedGPSettings(stable_tail_moments=True)
    exact = MixedObservationGP(inputs[:2]/divisor, observations[:2], settings=settings)
    mixed = MixedObservationGP(inputs/divisor, observations, settings=settings)
    points = np.array([[.55, .34], [.9, .3]])
    features = torch.tensor(points, requires_grad=True)
    terms = []
    for model in (exact, mixed):
        term = LogEnergyImprovement(model, divisor)
        prediction = model.predict(points/divisor)
        expected = log_ndtr(-prediction["mean"]/np.sqrt(prediction["variance"].clip(min=1e-14)))
        np.testing.assert_allclose(term.log_probability(features).detach().numpy(), expected, atol=1e-9)
        terms.append(term)
    base = SimpleNamespace(model=None, fit_diagnostics={},
                           log_acquisition=lambda x: x[..., 0].squeeze(-1)*0.-2.,
                           solver_probabilities=lambda x: {"success": np.full(len(x), .7)})
    update = CensoredEnergyRefinement(base, *terms)
    score = update.log_acquisition(features.unsqueeze(-2))
    assert score[0] < -2.
    score.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert update.solver_probabilities(features)["success"].tolist() == [.7, .7]


def test_synthetic_aircraft_fork_replays_without_repaying_shared_history(tmp_path, monkeypatch):
    seed = registration(PROFILE)["scenarios"]["headwind"]["seeds"][0]
    spec = shard_plan(PROFILE, "headwind", seed)
    spec.update(total_budget=12, horizon=4, calls_per_seed=16, quality_budgets=[12], native_gp_steps=3)
    calls = []

    def evaluator(problem, candidate):
        calls.append(candidate)
        settings = problem.settings
        speed = candidate["continuous"]["cruise_airspeed"]
        duration = (settings["climb_height_m"]/2. + settings["climb_height_m"]/1.5 +
                    settings["nominal_cruise_distance_m"]/(speed-4.) + settings["reserve_duration_s"])
        constraints = {name: -.1 for name in problem.required_constraints}
        minimum_soc = settings["soc_reserve_requirement"]
        if len(calls) <= 8:
            reason = "physical_soc_reserve_abort"
            objective = OutputObservation("interval", "simulated_prefix", lower=42000.)
            return problem.record(candidate, objective, constraints, {
                "mission_completed": False, "termination_reason": reason, "failure_cause": reason,
                "trace": [{"elapsed_s": .75*duration, "soc": minimum_soc-.01}],
            }, status="hidden_failure")
        return problem.record(candidate, 43000.+10*len(calls), constraints, {
            "mission_completed": True, "termination_reason": "complete",
            "trace": [{"elapsed_s": duration, "stage": "reserve", "soc": minimum_soc+.01}],
        })

    monkeypatch.setattr(UsableAircraft, "evaluate", evaluator)
    folder = tmp_path/"sealed"
    run_fork(spec, seed, "synthetic", folder)
    verified = verify_fork(folder, spec, "synthetic")
    assert verified["cold"]
    assert verified["fork_evaluation"] == 11
    assert verified["paid_calls"] == len(calls) == 13
    assert verified["arms"][CONTROL]["energy_update_calls"] == 0
    assert verified["arms"][TREATMENT]["energy_update_calls"] == 1
