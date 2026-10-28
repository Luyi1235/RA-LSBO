from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import math

import pytest

from ra_lsbo.risk_core.online_calibration import OnlineResidualCalibrator


def seeded(values=(1., 2., 3., 4.), **kwargs):
    calibrator = OnlineResidualCalibrator(**kwargs)
    for i, value in enumerate(values):
        key = "seed-{}".format(i)
        calibrator.predict(0., 1., key)
        calibrator.observe(key, value)
    return calibrator


def test_cold_start_predict_before_update_and_strict_ties():
    c = OnlineResidualCalibrator()
    before = c.state_dict()
    with pytest.raises(KeyError, match="predict must precede"):
        c.observe("unknown", 1.)
    assert c.state_dict() == before
    prediction = c.predict(2., 3., "first")
    assert prediction["upper"] == prediction["quantile"] == math.inf
    assert prediction["alpha"] == .2
    assert prediction["n_residuals"] == 0
    assert prediction["quantile_index"] == 1
    update = c.observe("first", 8.)
    assert update["residual"] == 2.
    assert update["error"] == 0
    assert c.alpha == pytest.approx(.21)
    c = seeded((1.,) * 4)
    prediction = c.predict(2., 3., "tie")
    assert prediction["quantile"] == 1.
    assert prediction["upper"] == 5.
    assert c.observe("tie", 5.)["error"] == 0


def test_finite_sample_rank_and_unavailable_rank_not_interpolated():
    c = seeded()
    prediction = c.predict(0., 1., "available")
    assert prediction["alpha"] == pytest.approx(.24)
    assert prediction["quantile_index"] == math.ceil(5 * (1 - c.alpha)) == 4
    assert prediction["quantile"] == 4.
    c = seeded(target_alpha=.01, learning_rate=.001)
    prediction = c.predict(0., 1., "unavailable")
    assert prediction["quantile_index"] == 5
    assert prediction["quantile_status"] == "insufficient_history"
    assert prediction["upper"] == math.inf


def test_current_quantile_reads_candidate_pool_without_reserving_keys():
    c = seeded()
    before = c.state_dict()
    calibration = c.current_quantile()
    assert calibration["quantile"] == 4.
    uppers = [mean + calibration["quantile"] * .5 for mean in range(192)]
    assert uppers[0] == 2. and uppers[-1] == 193.
    assert c.current_quantile() == calibration
    assert c.state_dict() == before
    prediction = c.predict(0., .5, "selected")
    assert all(prediction[name] == value for name, value in calibration.items())
    assert prediction["upper"] == uppers[0]
    assert len(c.state_dict()["pending"]) == 1
    calibration["quantile"] = 999.
    assert c.current_quantile()["quantile"] == 4.
    c.observe("selected", None)


def test_current_quantile_cold_start_has_explicit_unavailable_status():
    c = OnlineResidualCalibrator()
    before = c.state_dict()
    calibration = c.current_quantile()
    assert calibration == {
        "quantile": math.inf, "alpha": .2, "quantile_index": 1,
        "quantile_status": "insufficient_history", "n_residuals": 0,
        "n_observed": 0,
    }
    assert c.state_dict() == before


def test_tokens_are_immutable_detached_and_future_data_cannot_change_error():
    c = seeded()
    old = c.predict(0., 1., "old")
    token = old["token"]
    with pytest.raises(FrozenInstanceError):
        token.alpha = .99
    with pytest.raises(FrozenInstanceError):
        token.quantile = 999.
    c.predict(100., 1., "later")
    c.observe("later", 200.)
    current_alpha = c.alpha
    old["upper"], old["quantile"], old["alpha"], old["token"] = 999., 999., .99, None
    update = c.observe("old", 4.5)
    assert token.upper == token.quantile == 4.
    assert token.n_observed == 4
    assert update["token"] == token
    assert update["error"] == 1
    assert update["residual"] == 4.5
    assert update["alpha_before"] == current_alpha
    assert c.alpha == pytest.approx(current_alpha + .05 * (.2 - 1))


def test_same_prediction_does_not_use_its_own_residual():
    c = seeded()
    c.predict(0., 1., "outlier")
    update = c.observe("outlier", 1000.)
    assert update["error"] == 1
    assert update["token"].upper == 4.
    assert c.predict(0., 1., "after")["quantile"] == 1000.


def test_missing_closes_once_without_changing_residual_alpha_or_exact_counts():
    c = seeded()
    before = c.state_dict()
    c.predict(0., 1., "missing")
    result = c.observe("missing", None)
    after = c.state_dict()
    assert result["observed"] is False
    assert result["residual"] is result["error"] is None
    assert before["residuals"] == after["residuals"]
    assert before["alpha"] == after["alpha"]
    assert before["n_observed"] == after["n_observed"]
    assert before["n_errors"] == after["n_errors"]
    assert after["n_missing"] == before["n_missing"] + 1
    with pytest.raises(ValueError, match="already observed"):
        c.observe("missing", -1.)


@pytest.mark.parametrize("sd", [0., 1e-15, 2.])
def test_scale_floor_produces_finite_residuals_and_zero_sd_infinite_upper(sd):
    c = OnlineResidualCalibrator()
    prediction = c.predict(0., sd, "a")
    assert prediction["upper"] == math.inf
    assert not math.isnan(prediction["upper"])
    result = c.observe("a", 1e-12)
    assert math.isfinite(result["residual"])
    assert result["residual"] == 1e-12 / max(sd, 1e-12)


def test_upper_uses_supplied_sd_even_below_epsilon_and_error_uses_upper():
    c = seeded((1.,) * 4)
    prediction = c.predict(0., 1e-15, "small")
    assert prediction["upper"] == 1e-15
    result = c.observe("small", 2e-15)
    assert result["residual"] == .002
    assert result["error"] == 1


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1", [1.],
                                  {"lower": 0.}])
def test_invalid_or_nonexact_feedback_does_not_consume_prediction(value):
    c = OnlineResidualCalibrator()
    c.predict(0., 1., "a")
    before = c.state_dict()
    with pytest.raises(ValueError):
        c.observe("a", value)
    assert c.state_dict() == before
    assert c.observe("a", -1.)["observed"]


def test_residual_overflow_is_rejected_atomically():
    c = OnlineResidualCalibrator()
    c.predict(-1e308, 0., "overflow")
    before = c.state_dict()
    with pytest.raises(ValueError, match="computed residual"):
        c.observe("overflow", 1e308)
    assert c.state_dict() == before
    c.observe("overflow", None)


def test_alpha_excursion_below_zero_is_not_clipped_and_recovers():
    c = seeded((0.,), target_alpha=.25, learning_rate=1.)
    assert c.alpha == .5
    c.predict(0., 1., "miss")
    c.observe("miss", 1.)
    assert c.alpha == -.25
    for i, expected_alpha in enumerate((0., .25)):
        prediction = c.predict(0., 0., "recovery-{}".format(i))
        assert prediction["alpha"] <= 0
        assert prediction["upper"] == prediction["quantile"] == math.inf
        assert prediction["quantile_status"] == "alpha_nonpositive"
        result = c.observe("recovery-{}".format(i), 0.)
        assert result["error"] == 0
        assert c.alpha == pytest.approx(expected_alpha)


def test_alpha_excursion_above_one_forces_empty_prediction_even_at_zero_sd():
    c = OnlineResidualCalibrator(learning_rate=10.)
    c.predict(0., 0., "first")
    c.observe("first", 0.)
    assert c.alpha == pytest.approx(2.2)
    prediction = c.predict(0., 0., "empty")
    assert prediction["quantile_status"] == "alpha_at_least_one"
    assert prediction["upper"] == prediction["quantile"] == -math.inf
    assert c.observe("empty", -100.)["error"] == 1
    assert c.alpha == pytest.approx(-5.8)


def test_alpha_exactly_one_uses_empty_prediction():
    c = OnlineResidualCalibrator(target_alpha=.5, learning_rate=1.)
    c.predict(0., 1., "a")
    c.observe("a", 0.)
    assert c.alpha == 1.
    assert c.predict(0., 1., "b")["upper"] == -math.inf
    assert c.observe("b", 0.)["error"] == 1
    assert c.alpha == .5


def test_window_is_bounded_and_duplicate_keys_remain_rejected_after_eviction():
    c = seeded((1., 2., 3., 4., 5.), window=2, target_alpha=.5)
    assert c.state_dict()["residuals"] == [4., 5.]
    assert c.predict(0., 1., "latest")["quantile"] == 5.
    before = c.state_dict()
    for key in ("seed-0", "latest"):
        with pytest.raises(ValueError, match="already predicted"):
            c.predict(0., 1., key)
    with pytest.raises(ValueError, match="already observed"):
        c.observe("seed-0", 0.)
    assert c.state_dict() == before


def test_upper_overflow_rejects_prediction_without_consuming_key():
    c = seeded((2.,) * 4)
    before = c.state_dict()
    with pytest.raises(ValueError, match="computed upper"):
        c.predict(1e308, 1e308, "overflow")
    assert c.state_dict() == before
    assert c.predict(0., 1., "overflow")["upper"] == 2.


def test_signed_margins_and_coverage_error_are_distinct_from_feasibility():
    c = seeded((-2.,) * 4)
    prediction = c.predict(1., 1., "negative-upper")
    assert prediction["quantile"] == -2.
    assert prediction["upper"] == -1.
    result = c.observe("negative-upper", -.5)
    assert result["error"] == 1  # Feasible g, but missed the frozen upper bound.
    c = seeded((2.,) * 4)
    assert c.predict(-1., 1., "positive-upper")["upper"] == 1.
    assert c.observe("positive-upper", .5)["error"] == 0  # Covered, but g > 0.
    c = seeded((0.,) * 4)
    assert c.predict(0., 1., "boundary")["upper"] == 0.
    assert c.observe("boundary", 0.)["error"] == 0


def test_state_restores_pending_frozen_predictions_and_duplicate_protection():
    c = seeded()
    c.predict(0., 0., "missing")
    c.observe("missing", None)
    c.predict(0., 1., "pending")
    c.predict(10., 2., "later")
    c.observe("later", 100.)
    state = c.state_dict()
    # Python's JSON defaults retain extended float values; strict JSON does not.
    restored = OnlineResidualCalibrator.from_state_dict(json.loads(json.dumps(state)))
    assert restored.state_dict() == state
    assert restored.observe("pending", 5.) == c.observe("pending", 5.)
    assert restored.predict(1., 2., "next") == c.predict(1., 2., "next")
    assert restored.state_dict() == c.state_dict()
    for key in ("seed-0", "missing", "pending", "next"):
        with pytest.raises(ValueError):
            restored.predict(0., 1., key)
    state["residuals"].clear()
    state["pending"][0]["upper"] = 999.
    assert restored.state_dict() == c.state_dict()


@pytest.mark.parametrize("learning_rate", [.05, 10.])
def test_state_restores_infinite_pending_bounds(learning_rate):
    c = OnlineResidualCalibrator(learning_rate=learning_rate)
    c.predict(0., 0., "cold")
    c.observe("cold", 0.)
    c.predict(0., 0., "pending")
    restored = OnlineResidualCalibrator.from_state_dict(c.state_dict())
    assert restored.observe("pending", 0.) == c.observe("pending", 0.)


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("alpha", math.nan), ("residuals", [math.inf]),
    ("n_errors", 5), ("closed_keys", []),
])
def test_invalid_checkpoints_are_rejected(field, value):
    state = seeded().state_dict()
    state[field] = value
    with pytest.raises(ValueError):
        OnlineResidualCalibrator.from_state_dict(state)


def test_future_history_or_changed_upper_in_pending_checkpoint_is_rejected():
    c = seeded()
    c.predict(0., 1., "pending")
    for field, value in (("n_observed", 100), ("upper", 99.), ("quantile", math.nan)):
        state = deepcopy(c.state_dict())
        state["pending"][0][field] = value
        with pytest.raises(ValueError):
            OnlineResidualCalibrator.from_state_dict(state)


@pytest.mark.parametrize("kwargs", [
    {"target_alpha": 0.}, {"target_alpha": 1.}, {"target_alpha": math.nan},
    {"target_alpha": True}, {"learning_rate": 0.}, {"learning_rate": math.inf},
    {"window": 0}, {"window": 1.5}, {"window": True}, {"epsilon": 0.},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        OnlineResidualCalibrator(**kwargs)


@pytest.mark.parametrize("mean,sd,key", [
    (math.nan, 1., "a"), (0., math.inf, "a"), (0., -1., "a"),
    (True, 1., "a"), (0., True, "a"), (0., 1., ""), (0., 1., []),
])
def test_invalid_prediction_is_atomic(mean, sd, key):
    c = OnlineResidualCalibrator()
    before = c.state_dict()
    with pytest.raises(ValueError):
        c.predict(mean, sd, key)
    assert c.state_dict() == before
