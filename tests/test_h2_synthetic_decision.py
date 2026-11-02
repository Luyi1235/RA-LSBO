"""Toy on-policy H2 decision proxy: deterministic and independent streams."""
from tools.run_h2_synthetic_decision import (
    run_trial, pool_for, choose, analytic_forecasts, selected_truth,
    ARMS,SEEDS,STEPS,SCENARIOS)
from ra_lsbo.experiments.simplified_aircraft.h2_risk_controller import CausalSOCController


def test_fixed_shared_pool_is_independent_of_arm_and_rerunnable():
    original=pool_for(SCENARIOS[0],SEEDS[0],0)
    assert original==pool_for(SCENARIOS[0],SEEDS[0],0)
    assert original!=pool_for(SCENARIOS[0],SEEDS[0],1)
    assert len(original)==64


def test_h2_online_trial_is_exactly_replay_deterministic_and_bounded():
    one,history=run_trial("conditional_hierarchy",SEEDS[0],"h2")
    same,records=run_trial("conditional_hierarchy",SEEDS[0],"h2")
    assert one==same and history==records
    assert len(history)==STEPS
    assert all(.1<=row["H2_weight_before"]<=.9 for row in history)
    assert all(.1<=row["H2_weight_after"]<=.9 for row in history)
    assert sum(row["observation_kind"]=="interval" for row in history)==one["censored_count"]
    assert one["synthetic_calls"]==STEPS


def test_control_arms_use_same_original_design_pool_and_synthetic_truth():
    for arm in ARMS:
        result,rows=run_trial("onehot_valid",SEEDS[0],arm)
        assert len(rows)==STEPS
        assert result["synthetic_calls"]==STEPS
        assert result["arm"]==arm
        assert all(row["arm"]==arm for row in rows)
