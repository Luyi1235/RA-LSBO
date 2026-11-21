from types import SimpleNamespace

from aircraft_multi_feasible import PROFILE, accumulation_metrics, acquisition_stage, compare_accumulation, score_audit
from ra_lsbo.risk_core.observations import OutputObservation
from run_aircraft_native_risk_lsbo import UsableAircraft, run, verify
from run_aircraft_refinement_confirmation import registration, shard_plan


def test_heldout_risk_uses_equal_configuration_weights_and_scores_ordering():
    points = [{"discrete": {"layout": n}} for n in (0, 0, 0, 1, 2)]
    result = score_audit(points, [0, 0, 0, 1, 1], [.2, .2, .2, .7, .2])
    assert abs(result["configuration_balanced_brier"] - (.04+.09+.64)/3.) < 1e-12
    assert result["comparable_configuration_pairs"] == 2
    assert result["configuration_rank_agreement"] == .75


def test_paired_accumulation_reaches_third_and_keeps_panel_out_of_optimization(tmp_path, monkeypatch):
    seed = registration(PROFILE)["scenarios"]["headwind"]["seeds"][0]
    spec = shard_plan(PROFILE, "headwind", seed)
    assert spec["calls_per_seed"] == 104
    spec.update(total_budget=13, horizon=5, calls_per_seed=22, quality_budgets=[13], native_gp_steps=3,
                completion_audit_size=4, completion_audit_checkpoints=[8, 12])
    count = 0

    def evaluator(problem, candidate):
        nonlocal count
        count += 1
        settings = problem.settings
        speed = candidate["continuous"]["cruise_airspeed"]
        duration = (settings["climb_height_m"]/2. + settings["climb_height_m"]/1.5 +
                    settings["nominal_cruise_distance_m"]/(speed-4.) + settings["reserve_duration_s"])
        constraints = {name: -.1-.00001*count for name in problem.required_constraints}
        minimum_soc = settings["soc_reserve_requirement"]
        # Initial eight and the first proposal of either arm abort.
        if count <= 8 or count in (9, 14, 19):
            reason = "physical_soc_reserve_abort"
            objective = OutputObservation("interval", "simulated_prefix", lower=42000.)
            return problem.record(candidate, objective, constraints, {
                "mission_completed": False, "termination_reason": reason, "failure_cause": reason,
                "trace": [{"elapsed_s": .75*duration, "soc": minimum_soc-.01}],
            }, status="hidden_failure")
        return problem.record(candidate, 43000.+10*count, constraints, {
            "mission_completed": True, "termination_reason": "complete",
            "trace": [{"elapsed_s": duration, "stage": "reserve", "soc": minimum_soc+.01}],
        })

    monkeypatch.setattr(UsableAircraft, "evaluate", evaluator)
    folder = tmp_path/"sealed"
    run(spec, seed, "synthetic", folder)
    result = verify(folder, spec, "synthetic")
    assert result["cold"]
    assert count == result["paid_calls"] == 22
    for arm, metrics in result["arms"].items():
        accumulation = metrics["accumulation"]
        assert accumulation["target_call"] == 12
        assert accumulation["nonusable_outcomes"] == {"abort": 1}
        assert metrics["cold_risk_calls"] == 4
        assert metrics["refinement_risk_calls"] == 1
        assert result["completion_audit"]["8"][arm]["n"] == 4
        assert result["completion_audit"]["12"][arm]["n"] == 4
    left, right = result["arms"]
    assert result["completion_audit"]["8"][left] == result["completion_audit"]["8"][right]
    contrast = compare_accumulation([result], left, right, [8, 12])
    assert contrast["target_hits"] == {left: 1, right: 1}
    assert contrast["capped_target_call_delta"]["ties"] == 1
    assert contrast["heldout_completion"]["8"]["configuration_balanced_brier_delta"]["mean"] == 0.


def test_unreached_target_censors_time_but_counts_all_online_nonusable(monkeypatch):
    rows = [SimpleNamespace(usable_feasible=i == 1, candidate={"discrete": {"layout": 1}},
                            to_dict=lambda: {}) for i in range(5)]
    monkeypatch.setattr("aircraft_multi_feasible.discovery_outcome", lambda _: "violation")
    assert acquisition_stage(rows, 3)
    assert not acquisition_stage(rows, 1)
    received = accumulation_metrics(rows, 2, 3)
    assert received["target_call"] is None
    assert received["capped_target_call"] == 6
    assert received["nonusable_outcomes"] == {"violation": 3}
