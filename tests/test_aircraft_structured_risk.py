from shutil import copyfile
from types import SimpleNamespace

import numpy as np

from aircraft_structured_risk import (LATENT_DECISION, PROFILE, STRUCTURED_ARMS, STRUCTURED_DECISION,
                                      HierarchicalTimingGP, choose_completion_with_lookahead, with_configuration)
from ra_lsbo.risk_core.observations import OutputObservation
from run_aircraft_native_risk_lsbo import UsableAircraft, run, verify
from run_aircraft_refinement_confirmation import PAPER_ARMS, registration, shard_plan


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


def test_five_arm_run_accepts_only_shared_and_online_evaluation_evidence(tmp_path, monkeypatch):
    seed = registration(PROFILE)["scenarios"]["headwind"]["seeds"][0]
    spec = shard_plan(PROFILE, "headwind", seed)
    assert spec["arms"] == list(PAPER_ARMS)
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
    assert report["completion_audit"]["8"][LATENT_DECISION]["n"] == 4
    for arm in PAPER_ARMS:
        assert report["arms"][arm]["cold_risk_calls"] == 2
        assert report["arms"][arm]["accumulation"]["capped_target_call"] == 11
