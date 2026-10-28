"""Numerical tests need no simulation backend, GP fitting, or model calls."""

import importlib.util
from pathlib import Path
import warnings

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import norm

path = Path(__file__).resolve().parents[1]/"ra_lsbo/numerics.py"
spec = importlib.util.spec_from_file_location("ra_lsbo_numerics_under_test", path)
numerics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numerics)


@pytest.mark.parametrize("z,expected", [
    (0., -0.9189385332046727), (-1., -2.4851210257126413),
    (-10., -55.55312203612236), (-20., -206.9178385094251),
    (-30., -457.724653760598), (-50., -1258.744182868461),
    (-100., -5010.12957880025)])
def test_logei_against_high_precision_reference(z, expected):
    # Reference evaluated with 100-decimal arithmetic, not ordinary floating EI.
    value = numerics.log_expected_improvement(np.array([-z]), np.ones(1), 0.)[0]
    assert value == pytest.approx(expected, abs=1e-8)


def test_logei_does_not_flatten_underflow_tail():
    result = numerics.log_expected_improvement([40., 50., 100.], [1., 1., 1.], 0.)
    assert np.isfinite(result).all()
    assert np.all(np.diff(result) < 0)
    assert result[0] < np.log(1e-300)


def test_logei_agrees_with_ordinary_formula_in_well_conditioned_region():
    mean = np.linspace(-10, 3, 200)
    std, best = 2.3, 1.7
    z = (best-mean)/std
    expected = np.log((best-mean)*norm.cdf(z)+std*norm.pdf(z))
    np.testing.assert_allclose(numerics.log_expected_improvement(mean, std, best), expected,
                               atol=2e-13, rtol=2e-13)


def test_logei_tail_switch_is_continuous_and_monotone():
    mean = np.array([20.-1e-8, 20., 20.+1e-8])
    result = numerics.log_expected_improvement(mean, np.ones(3), 0.)
    assert np.all(np.diff(result) < 0)
    assert np.max(abs(np.diff(result))) < 3e-7


def test_logei_deterministic_limit():
    values = numerics.log_expected_improvement([1., 2., 3.], [0., 0., 0.], 2.)
    assert values[0] == 0.
    assert np.isneginf(values[1:]).all()


@pytest.mark.parametrize("mean,std,best", [([np.nan],[1.],0.),([0.],[-1.],0.),([0.],[np.inf],0.),([0.],[1.],np.nan)])
def test_logei_rejects_invalid_inputs(mean, std, best):
    with pytest.raises(ValueError):
        numerics.log_expected_improvement(mean, std, best)


def test_roundoff_projection_and_material_excursion():
    point, excess = numerics.bounded_point([-1e-14, 1.+1e-14], [0., 0.], [1., 1.])
    np.testing.assert_array_equal(point, [0., 1.])
    assert 0 < excess < 1e-10
    with pytest.raises(numerics.InvalidLocalPoint):
        numerics.bounded_point([1.001], [0.], [1.])


@pytest.mark.parametrize("x", [[np.nan], [np.inf], [-np.inf]])
def test_nonfinite_local_points_are_rejected(x):
    with pytest.raises(numerics.InvalidLocalPoint):
        numerics.bounded_point(x, [0.], [1.])


def test_warning_recording_is_scoped_and_preserves_other_warnings():
    old_filters = list(warnings.filters)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        old_show = warnings.showwarning
        with numerics.record_slsqp_clipping() as diagnostics:
            for _ in range(5):
                warnings.warn(numerics.SLSQP_CLIP_MESSAGE, RuntimeWarning)
            warnings.warn("unrelated numerical warning", RuntimeWarning)
        assert warnings.showwarning is old_show
        assert diagnostics["slsqp_clipping_warnings"] == 5
        assert [str(item.message) for item in seen] == ["unrelated numerical warning"]
    assert warnings.filters == old_filters


def test_warning_filter_restored_after_exception():
    old_show = warnings.showwarning
    with pytest.raises(RuntimeError):
        with numerics.record_slsqp_clipping():
            raise RuntimeError("intentional test")
    assert warnings.showwarning is old_show


def test_real_slsqp_respects_callback_bounds():
    def objective(x):
        bounded, _ = numerics.bounded_point(x, [0.], [1.])
        return float((bounded[0]-2)**2)
    with numerics.record_slsqp_clipping() as diagnostics:
        result = minimize(objective, [.5], method="SLSQP", bounds=[(0., 1.)])
    assert result.success
    assert result.x[0] == pytest.approx(1.)
    assert diagnostics["slsqp_clipping_warnings"] >= 0
