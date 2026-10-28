import io
import random

import numpy as np
import pytest
import torch

from ra_lsbo.risk_core.boundary_ensemble import (
    BoundaryEnsemble, BoundarySettings, boundary_acquisition, constraint_labels,
)
from ra_lsbo.risk_core.observations import OutputObservation


@pytest.fixture(scope="module")
def fitted():
    x = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
    return x, [0, 0, 0, 1, 1, 1], BoundaryEnsemble(x, [0, 0, 0, 1, 1, 1], seed=7)


def assert_same_predictions(first, second, x):
    for key, values in first.predict(x).items():
        np.testing.assert_array_equal(values, second.predict(x)[key])


def test_exact_and_interval_signs_preserve_unknown_and_zero_boundary():
    observations = [
        OutputObservation("exact", "negative", value=-1.0),
        OutputObservation("exact", "zero", value=0.0),
        OutputObservation("exact", "positive", value=1.0),
        OutputObservation("interval", "nonpositive", upper=0.0),
        OutputObservation("interval", "negative-open", upper=0.0, upper_inclusive=False),
        OutputObservation("interval", "positive-open", lower=0.0, lower_inclusive=False),
        OutputObservation("interval", "positive", lower=0.1),
        OutputObservation("interval", "zero-or-positive", lower=0.0),
        OutputObservation("interval", "crosses-zero", lower=-1.0, upper=1.0),
        OutputObservation("missing", "not-returned", reason="solver_failure"),
    ]
    assert constraint_labels(observations) == [1, 1, 0, 1, 1, 0, 0, None, None, None]
    with pytest.raises(ValueError):
        constraint_labels([-1.0])


def test_observed_labels_learn_orientation_and_finite_bands(fitted):
    x, _, model = fitted
    result = model.predict(x)
    assert set(result) == {"probability", "lower", "upper", "disagreement"}
    assert model.settings == BoundarySettings()
    assert model.diagnostics["status"] == "fitted_two_class"
    assert model.diagnostics["observed"] == 6
    assert result["probability"][0] < 0.2
    assert result["probability"][-1] > 0.8
    for key, value in result.items():
        assert value.shape == (len(x),)
        assert value.dtype == np.float64
        assert np.isfinite(value).all()
        assert (value >= 0).all() and (value <= 1).all()
    assert (result["lower"] <= result["probability"]).all()
    assert (result["upper"] >= result["probability"]).all()
    assert not model.diagnostics["calibrated"]


def test_masked_rows_never_change_training(fitted):
    x, labels, model = fitted
    masked_x = np.insert(x, 2, 1234.5, axis=0)
    masked_labels = labels[:2] + [None] + labels[2:]
    masked = BoundaryEnsemble(masked_x, masked_labels, seed=7)
    assert masked.observed_mask.tolist() == [True, True, False, True, True, True, True]
    assert masked.diagnostics["missing"] == 1
    assert_same_predictions(model, masked, np.array([[0.0], [0.5], [1.0]]))
    for key, value in model.nets.state_dict().items():
        assert torch.equal(value, masked.nets.state_dict()[key])


def test_disagreement_is_member_spread_not_bernoulli_variance(fitted):
    _, _, model = fitted
    query = torch.tensor([[0.3], [0.5], [0.7]], dtype=torch.float64)
    with torch.no_grad():
        members = torch.stack([
            torch.special.ndtr(net(query).squeeze(-1)) for net in model.nets
        ]).numpy()
    result = model.predict(query)
    np.testing.assert_allclose(result["probability"], members.mean(axis=0), atol=1e-15)
    np.testing.assert_allclose(result["disagreement"], members.std(axis=0), atol=1e-15)
    np.testing.assert_allclose(
        result["lower"], np.clip(members.mean(axis=0) - members.std(axis=0), 0, 1))
    np.testing.assert_allclose(
        result["upper"], np.clip(members.mean(axis=0) + members.std(axis=0), 0, 1))
    assert result["disagreement"].max() > 1e-5
    assert len({next(net.parameters()).data_ptr() for net in model.nets}) == 5

    copied = BoundaryEnsemble.from_state_dict(model.state_dict())
    for net in copied.nets[1:]:
        net.load_state_dict(copied.nets[0].state_dict())
    collapsed = copied.predict(query)
    np.testing.assert_array_equal(collapsed["disagreement"], np.zeros(len(query)))
    np.testing.assert_array_equal(collapsed["lower"], collapsed["upper"])


@pytest.mark.parametrize("label", [0, 1])
def test_single_observed_class_has_no_invented_counterexamples(label):
    model = BoundaryEnsemble([[0.2], [0.9]], [label, None], seed=3)
    assert model.diagnostics["status"] == "fitted_single_class"
    assert model.diagnostics["observed"] == 1
    assert model.diagnostics["positive"] == label
    assert model.diagnostics["negative"] == 1 - label
    probability = model.predict([[0.2]])["probability"][0]
    assert 0 < probability < 1
    assert (probability > 0.5) == bool(label)
    assert_same_predictions(
        model, BoundaryEnsemble([[0.2]], [label], seed=3), [[0.2], [0.5]])


@pytest.mark.parametrize("x, labels", [
    (np.empty((0, 2)), []),
    ([[0.2, 0.3], [0.9, 0.1]], [None, None]),
])
def test_no_labels_returns_explicit_uninformative_band(x, labels):
    model = BoundaryEnsemble(x, labels)
    assert model.diagnostics["status"] == "prior_only_no_labels"
    assert len(model.nets) == 0
    assert model.diagnostics["positive"] == model.diagnostics["negative"] == 0
    result = model.predict([[0.2, 0.3], [0.9, 0.1]])
    np.testing.assert_array_equal(result["probability"], [0.5, 0.5])
    np.testing.assert_array_equal(result["lower"], [0.0, 0.0])
    np.testing.assert_array_equal(result["upper"], [1.0, 1.0])
    np.testing.assert_array_equal(result["disagreement"], [0.0, 0.0])
    assert_same_predictions(model, BoundaryEnsemble.from_state_dict(model.state_dict()),
                            [[0.2, 0.3]])


def test_repeat_prediction_refit_and_checkpoint_are_exact(fitted):
    x, labels, model = fitted
    query = np.linspace(0, 1, 21).reshape(-1, 1)
    assert_same_predictions(model, model, query)
    assert_same_predictions(model, BoundaryEnsemble(x, labels, seed=7), query)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    loaded = BoundaryEnsemble.from_state_dict(torch.load(buffer, weights_only=True))
    assert loaded.diagnostics == model.diagnostics
    assert_same_predictions(model, loaded, query)
    assert not any(parameter.requires_grad for parameter in loaded.nets.parameters())
    detached_state = model.state_dict()
    detached_state["networks"]["0.0.weight"].zero_()
    assert not torch.equal(
        detached_state["networks"]["0.0.weight"], model.nets.state_dict()["0.0.weight"])


def test_training_and_restore_isolate_all_global_rng_and_settings():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    first = BoundaryEnsemble([[0.0], [1.0]], [0, 1], seed=101,
                             settings={"steps": 4})
    BoundaryEnsemble.from_state_dict(first.state_dict()).predict([[0.5]])
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert torch.get_default_dtype() == dtype
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    other = BoundaryEnsemble([[0.0], [1.0]], [0, 1], seed=102,
                             settings={"steps": 4})
    assert not np.array_equal(first.predict([[0.5]])["probability"],
                              other.predict([[0.5]])["probability"])


def test_nondefault_settings_numeric_scalars_and_tensor_inputs_round_trip():
    settings = BoundarySettings(
        n_networks=np.int64(3), hidden_width=np.int64(4), steps=np.int64(5),
        learning_rate=np.float64(0.02), weight_decay=0, band_scale=2)
    model = BoundaryEnsemble(torch.tensor([[0.0], [1.0]], dtype=torch.bfloat16),
                             torch.tensor([0, 1]), seed=np.int64(9), settings=settings)
    assert len(model.nets) == 3
    assert model.nets[0][0].weight.shape == (4, 1)
    assert model.diagnostics["settings"] == {
        "n_networks": 3, "hidden_width": 4, "steps": 5, "learning_rate": 0.02,
        "weight_decay": 0.0, "band_scale": 2.0,
    }
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = BoundaryEnsemble.from_state_dict(torch.load(buffer, weights_only=True))
    assert_same_predictions(model, restored, [[0.2], [0.5], [0.8]])
    assert restored.settings == settings
    assert restored.diagnostics == model.diagnostics
    result = model.predict([[0.2], [0.5], [0.8]])
    np.testing.assert_allclose(
        result["upper"], np.clip(result["probability"] + 2 * result["disagreement"], 0, 1))


@pytest.mark.parametrize("x, labels", [
    ([[float("nan")]], [1]), ([[float("inf")]], [None]),
    ([[1.0 + 2j]], [1]), ([["0.2"]], [1]),
    ([0.1, 0.2], [0, 1]), (np.empty((2, 0)), [0, 1]),
    ([[0.1]], []), ([[0.1]], [float("nan")]), ([[0.1]], [float("inf")]),
    ([[0.1]], [-1]), ([[0.1]], [2]), ([[0.1]], ["1"]),
    ([[0.1]], [[1]]), ([[0.1]], None),
])
def test_invalid_training_inputs_are_rejected(x, labels):
    with pytest.raises(ValueError):
        BoundaryEnsemble(x, labels, settings={"steps": 1})


@pytest.mark.parametrize("settings", [
    {"n_networks": 1}, {"hidden_width": 0}, {"steps": 0}, {"steps": 2.5},
    {"learning_rate": 0}, {"learning_rate": float("nan")},
    {"weight_decay": -1}, {"band_scale": 0}, {"band_scale": float("inf")},
    {"unknown_option": 2},
])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        BoundaryEnsemble([[0.0]], [None], settings=settings)


@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5])
def test_invalid_seeds_are_rejected(seed):
    with pytest.raises(ValueError):
        BoundaryEnsemble([[0.0]], [None], seed=seed)


def test_prediction_validation_and_empty_batches(fitted):
    _, _, model = fitted
    for x in ([[float("nan")]], [[float("inf")]], [[0.1, 0.2]], [0.2], [["0.2"]]):
        with pytest.raises(ValueError):
            model.predict(x)
    for values in model.predict(np.empty((0, 1))).values():
        assert values.shape == (0,)
    tensor = torch.tensor([[0.0], [1.0]], requires_grad=True)
    model.predict(tensor)
    assert tensor.grad is None
    with pytest.raises(ValueError, match="Nonfinite"):
        model.predict([[np.finfo(np.float64).max]])


def test_training_numerical_overflow_is_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        BoundaryEnsemble([[np.finfo(np.float64).max]], [0], settings={"steps": 1})


def test_eligibility_is_per_head_upper_band_with_inclusive_threshold():
    class FixedHead:
        def __init__(self, upper):
            self.upper = np.array(upper)

        def predict(self, x):
            return {"probability": self.upper - 0.1, "lower": self.upper - 0.2,
                    "upper": self.upper.copy(), "disagreement": np.full(len(x), 0.1)}

    result = boundary_acquisition(
        {"first": FixedHead([0.7, 0.5, 0.8]), "second": FixedHead([0.7, 0.8, 0.49])},
        [[0.0], [0.5], [1.0]])
    np.testing.assert_array_equal(result["eligible"], [True, True, False])
    np.testing.assert_allclose(result["heads"]["first"]["probability"], [0.6, 0.4, 0.7])
    # A product-of-upper >= 0.5 rule would wrongly exclude the first candidate.
    assert 0.7 * 0.7 < 0.5
    with pytest.raises(ValueError):
        boundary_acquisition({}, [[0.0]])


def test_checkpoint_rejects_corrupt_metadata_and_weights(fitted):
    _, _, model = fitted
    for field, value in (("version", 99), ("dimensions", 0), ("class_counts", (1, 0)),
                         ("observed_mask", torch.ones(6)), ("networks", {})):
        state = model.state_dict()
        state[field] = value
        with pytest.raises(ValueError):
            BoundaryEnsemble.from_state_dict(state)
    state = model.state_dict()
    state["networks"]["0.0.weight"].fill_(float("nan"))
    with pytest.raises(ValueError):
        BoundaryEnsemble.from_state_dict(state)
    for field in ("settings", "seed", "networks"):
        state = model.state_dict()
        del state[field]
        with pytest.raises(ValueError):
            BoundaryEnsemble.from_state_dict(state)
    state = model.state_dict()
    del state["settings"]["steps"]
    with pytest.raises(ValueError):
        BoundaryEnsemble.from_state_dict(state)
