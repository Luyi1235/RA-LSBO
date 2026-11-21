from shutil import copyfile
from types import SimpleNamespace

import numpy as np
import pytest

from aircraft_structured_risk import (CONFIRMATION_PROFILE, EXTERNAL_64_PROFILE, EXTERNAL_PROFILE,
                                      LATENT_DECISION, PROFILE, STRUCTURED_ARMS,
                                      STRUCTURED_DECISION, STRUCTURED_SIGN_GREEDY,
                                      HierarchicalTimingGP, choose_completion_with_lookahead, with_configuration)
from ra_lsbo.risk_core.observations import OutputObservation
from run_aircraft_native_risk_lsbo import UsableAircraft, run, verify
from run_aircraft_refinement_confirmation import (CONFIRMATION_ARMS, ORIGINAL, PAPER_ARMS, registration,
                                                  shard_plan, target_contrast)


def test_external_budget_extension_reuses_only_the_registered_initialization():
    original = registration(EXTERNAL_PROFILE)
    extended = registration(EXTERNAL_64_PROFILE)
    assert extended["primary"] == {**original["primary"], "budget": 64}
    assert extended["energy_targets_Wh"] == original["energy_targets_Wh"]
    assert extended["arms"] == original["arms"]
    assert extended["quality_budgets"] == [16, 24, 32, 48, 64]
    assert (extended["initial_evaluations"], extended["horizon"], extended["total_budget"]) == (8, 56, 64)
    for scenario, section in extended["scenarios"].items():
        assert section["seeds"] == original["scenarios"][scenario]["seeds"]
        assert len(section["seeds"]) == 24
        assert section["calls_per_seed"] == 176
        seed = section["seeds"][0]
        spec = shard_plan(EXTERNAL_64_PROFILE, scenario, seed)
        baseline = shard_plan(EXTERNAL_PROFILE, scenario, seed)
        assert baseline["total_budget"] == 32 and baseline["calls_per_seed"] == 80
        assert spec["contrasts"] == baseline["contrasts"]
        assert spec["initial_sizes"] == baseline["initial_sizes"] == [8]


def test_energy_targets_reached_after_a_checkpoint_are_censored_at_that_checkpoint():
    rows = [{"arms": {"a": {"first_energy_target": 33}, "b": {"first_energy_target": 32}}}]
    early = target_contrast(rows, "a", "b", 32)
    late = target_contrast(rows, "a", "b", 64)
    assert early["attainment_rate_delta"]["mean"] == -1.
    assert early["capped_time_delta"]["mean"] == 1.
    assert late["attainment_rate_delta"]["mean"] == 0.
    assert late["capped_time_delta"]["mean"] == 1.


def test_hierarchical_kernel_preserves_variance_and_learns_from_other_layouts():
    candidates = [
        {"discrete": {"prop_layout": layout, "propeller_count": count, "parallel_strings": strings}}
        for layout, count, strings in (("wingtip", 2, 32), ("distributed", 4, 32), ("distributed", 4, 48))
    ]
    features = with_configuration(np.array([[0.], [3.], [3.]]), candidates)
    model = HierarchicalTimingGP(features[:1], [OutputObservation("interval", "stop", lower=.1)])
    covariance = model._kernel(features, features)
    np.testing.assert_allclose(covariance.diagonal(), 1.)
    assert np.linalg.eigvalsh(covariance).min() > 0.
    assert covariance[0, 1] > covariance[0, 2]
    assert model.predict(features[1:])["mean"][0] > model.predict(features[1:])["mean"][1] > 0.


def test_zero_correlation_lookahead_keeps_greedy_order():
    p = np.array([.55, .54, .30])
    from scipy.special import ndtri

    inputs = np.eye(3)
    model = SimpleNamespace(interval_x=[], _baseline_covariance=lambda a, b: a @ b.T)
    head = SimpleNamespace(divisor=np.ones(3), scale=1., model=model,
                           predict=lambda x: {"mean": -ndtri(p), "variance": np.ones(len(x))})
    candidates = [{"id": k} for k in range(3)]
    index, info = choose_completion_with_lookahead(head, inputs, candidates, [np.array([0., .3])]*3)
    assert index == 0
    np.testing.assert_allclose(info["adaptive_two_call_completion"], .55+(.45*.54))
    np.testing.assert_allclose(info["adaptive_two_call_completion"], info["fixed_second_completion"])


@pytest.mark.parametrize("profile,arms", [(PROFILE, PAPER_ARMS), (CONFIRMATION_PROFILE, CONFIRMATION_ARMS)])
def test_five_arm_run_accepts_only_shared_and_online_evaluation_evidence(tmp_path, monkeypatch, profile, arms):
    seed = registration(profile)["scenarios"]["headwind"]["seeds"][0]
    spec = shard_plan(profile, "headwind", seed)
    assert spec["arms"] == list(arms)
    assert spec["calls_per_seed"] == 144
    spec.update(total_budget=10, horizon=2, calls_per_seed=22, quality_budgets=[10],
                completion_audit_size=4, completion_audit_checkpoints=[8, 10])
    paid = 0

    def evaluator(problem, candidate):
        nonlocal paid
        paid += 1
        settings = problem.settings
        speed = candidate["continuous"]["cruise_airspeed"]
        duration = (settings["climb_height_m"]/2. + settings["climb_height_m"]/1.5 +
                    settings["nominal_cruise_distance_m"]/(speed-4.) + settings["reserve_duration_s"])
        minimum = settings["soc_reserve_requirement"]
        constraints = {name: -.1 for name in problem.required_constraints}
        if paid <= 8 or paid in (19, 21):
            reason = "physical_soc_reserve_abort"
            return problem.record(candidate, OutputObservation("interval", "simulated_prefix", lower=42000.),
                                  constraints, {"mission_completed": False, "termination_reason": reason,
                                                "failure_cause": reason,
                                                "trace": [{"elapsed_s": .75*duration, "soc": minimum-.01}]},
                                  status="hidden_failure")
        return problem.record(candidate, 43000., constraints, {
            "mission_completed": True, "termination_reason": "complete",
            "trace": [{"elapsed_s": duration, "stage": "reserve", "soc": minimum+.01}],
        })

    monkeypatch.setattr(UsableAircraft, "evaluate", evaluator)
    folder = tmp_path/"sealed"
    run(spec, seed, "test", folder)
    compact = tmp_path/"compact"
    compact.mkdir()
    for name in ("plan.json", "calls.jsonl", "seal.json", *(arm+".json" for arm in spec["arms"])):
        copyfile(folder/name, compact/name)
    report = verify(compact, spec, "test")
    assert paid == report["paid_calls"] == 22
    assert report["cold"]
    assert report["completion_audit"]["8"][STRUCTURED_ARMS[0]] == report["completion_audit"]["8"][STRUCTURED_DECISION]
    if profile == CONFIRMATION_PROFILE:
        assert ORIGINAL not in report["completion_audit"]["8"]
        assert report["completion_audit"]["8"][STRUCTURED_SIGN_GREEDY] != report["completion_audit"]["8"][STRUCTURED_DECISION]
    else:
        assert report["completion_audit"]["8"][LATENT_DECISION]["n"] == 4
    for arm in arms:
        assert report["arms"][arm]["cold_risk_calls"] == (0 if arm == ORIGINAL else 2)
        assert report["arms"][arm]["accumulation"]["capped_target_call"] == 11
