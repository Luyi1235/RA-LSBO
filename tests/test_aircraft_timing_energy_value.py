from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from aircraft_timing_energy_value import energy_evidence, target_choice, target_marks, two_call_target
from ra_lsbo.risk_core.observations import OutputObservation
from run_aircraft_coverage_factorial import FixedTimingHead
from run_aircraft_native_risk_lsbo import ENERGY_ARMS, ENERGY_CONTRASTS, UsableAircraft, choose_cold, plan
from run_aircraft_refinement_confirmation import registration, shard_plan


def test_unresolved_constraints_and_abort_energies_are_not_failure_labels():
    problem = UsableAircraft(plan("headwind", "full-pilot", "refinement"))
    point = problem.space.default().to_dict()
    unknown = OutputObservation("missing", "constraint", reason="not_observed")
    completed = problem.record(point, 40000., {"current": unknown}, {"mission_completed": True})
    wrong = problem.record(point, 40000., {"current": .1}, {"mission_completed": True})
    stopped = problem.record(point, OutputObservation("interval", "prefix", lower=10000.),
                             {"current": .1}, {"mission_completed": False})
    assert energy_evidence(completed) == (completed.observation.objective, None)
    assert energy_evidence(wrong) == (wrong.observation.objective, False)
    assert energy_evidence(stopped) is None
    near = np.zeros((1, 1))
    for label, expected in ((None, 1./4), (False, 1./6), (True, 1./3)):
        np.testing.assert_allclose(target_marks(near, near, [(completed.observation.objective, label)],
                                                 40000., 10), [expected])


def test_censored_energy_can_rank_candidates_with_no_exact_energy():
    low_bound = OutputObservation("interval", "early_stop", lower=34000.)
    high_bound = OutputObservation("interval", "late_stop", lower=42000.)
    references = np.array([[0.], [1.]])
    marks = target_marks(references, np.array([[.1], [.9], [10.]]),
                         [(low_bound, None), (high_bound, None)], 40000., 10)
    assert 0. < marks[1] < marks[0] < marks[2]
    np.testing.assert_allclose(marks[2], .25, atol=1e-12)


def test_adaptive_and_fixed_follow_ups_account_for_first_hit_only_once():
    cells = np.array([[.1, .1], [.35, .10], [.10, .35]])
    adaptive, static = two_call_target(.1, .5, np.array([.8, .8]), cells)
    np.testing.assert_allclose([adaptive, static], [.70, .50])
    independent = np.array([.2, .3, .5])[:, None] * np.array([.6, .8])
    np.testing.assert_allclose(two_call_target(.1, .5, np.array([.7, .5]), independent), [.478, .478])


def test_target_greedy_can_choose_below_the_completion_floor():
    head = SimpleNamespace(predict=lambda query: {"mean": np.array([-1., 0.]), "variance": np.ones(2)})
    candidates = [{"id": "high_completion"}, {"id": "high_target"}]
    index, receipt = target_choice(head, np.zeros((2, 1)), candidates, None, np.array([.1, .8]), lookahead=False)
    assert index == 1
    assert receipt == {"p_complete": .5, "target_probability": .4}


def test_lookahead_can_prefer_an_informative_query_over_the_target_greedy():
    factors = np.array([[1.5, -.2], [.3, -.6], [-.2, -.6], [1.4, .6]])
    covariance = factors @ factors.T + .2*np.eye(4)
    sd = np.sqrt(np.diag(covariance))
    covariance = covariance / sd[:, None] / sd
    model = SimpleNamespace(interval_x=[], _baseline_covariance=lambda left, right: covariance[np.ix_(
        left[:, 0].astype(int), right[:, 0].astype(int))])
    head = SimpleNamespace(model=model, divisor=1., scale=1., predict=lambda query: {
        "mean": np.array([.3, -.4, -.5, .2]), "variance": np.ones(len(query))})
    query = np.arange(4).reshape(-1, 1)
    candidates = [{"id": label} for label in ("A", "B", "timing_query", "C")]
    marks = np.array([.4, .5, .3, .6])
    edges = [np.array([0., .3, .6])]*4
    greedy, _ = target_choice(head, query, candidates, edges, marks, lookahead=False)
    active, receipt = target_choice(head, query, candidates, edges, marks, lookahead=True)
    assert greedy == 1
    assert active == 2
    assert receipt["predicted_two_call_target"] > receipt["predicted_fixed_follow_up_target"]


def test_interval_head_integration_with_zero_and_one_exact_energy_measurements():
    problem = UsableAircraft(plan("headwind", "full-pilot", "refinement"))
    point = problem.space.default().to_dict()
    speed = point["continuous"]["cruise_airspeed"]
    settings = problem.settings
    total = (settings["climb_height_m"]/2. + settings["climb_height_m"]/1.5
             + settings["nominal_cruise_distance_m"]/(speed-4.) + settings["reserve_duration_s"])
    stop = problem.record(point, OutputObservation("interval", "prefix", lower=41000.), {"current": .1}, {
        "mission_completed": False, "termination_reason": "physical_soc_reserve_abort",
        "failure_cause": "physical_soc_reserve_abort",
        "trace": [{"elapsed_s": .6*total, "soc": settings["soc_reserve_requirement"]-.01}],
    }, status="hidden_failure")
    stop_without_energy = replace(stop, observation=replace(stop.observation, objective=OutputObservation(
        "missing", "prefix", reason="not_returned")))
    complete = problem.record(point, 39800., {"current": .1}, {
        "mission_completed": True, "termination_reason": "complete",
        "trace": [{"elapsed_s": total, "stage": "reserve", "soc": settings["soc_reserve_requirement"]+.01}],
    })
    candidates = [{**point, "continuous": {**point["continuous"], "cruise_airspeed": speed+i}} for i in range(3)]
    inputs = np.zeros((2, 2*len(problem.space.continuous)+4))
    inputs[1, 0] = .5
    query = np.zeros((3, inputs.shape[1]))
    query[:, 0] = [.35, .45, .65]
    stopped_head = FixedTimingHead(problem, inputs[:1], [stop], 10)

    def choose(head, train_inputs, history, policy):
        return choose_cold(head, train_inputs, history, query, candidates, settings, policy, 40000., 10)

    old_index, baseline = choose(stopped_head, inputs[:1], [stop], "information")
    for policy in ("energy_greedy", "energy_lookahead"):
        index, receipt = choose(stopped_head, inputs[:1], [stop_without_energy], policy)
        assert index == old_index
        assert receipt == {**baseline, "energy_observations": 0, "energy_lower_bounds": 0,
                           "energy_fallback": "no_energy_evidence"}

    _, bounded_only = choose(stopped_head, inputs[:1], [stop], "energy_lookahead")
    assert bounded_only["energy_observations"] == bounded_only["energy_lower_bounds"] == 1
    assert "energy_fallback" not in bounded_only

    fitted_head = FixedTimingHead(problem, inputs, [stop, complete], 10)
    _, receipt = choose(fitted_head, inputs, [stop, complete], "energy_lookahead")
    assert receipt["energy_observations"] == 2
    assert receipt["energy_lower_bounds"] == 1
    assert 0. < receipt["target_probability"] < receipt["p_complete"] <= 1.
    assert receipt["target_probability"] <= receipt["predicted_fixed_follow_up_target"]
    assert receipt["predicted_fixed_follow_up_target"] <= receipt["predicted_two_call_target"] + 1e-12


def test_registered_pilot_is_paired_and_leaves_prior_protocol_intact():
    pilot = registration("energy-pilot")
    for scenario, section in pilot["scenarios"].items():
        assert len(section["seeds"]) == 6
        assert section["calls_per_seed"] == 168
        spec = shard_plan("energy-pilot", scenario, section["seeds"][0])
        assert spec["arms"] == list(ENERGY_ARMS)
        assert spec["contrasts"] == [list(pair) for pair in ENERGY_CONTRASTS]
        assert spec["total_budget"] == 48
        assert section["seeds"][0] not in registration("cold-pilot")["scenarios"][scenario]["seeds"]
        assert registration("cold-pilot")["scenarios"][scenario]["calls_per_seed"] == 128
