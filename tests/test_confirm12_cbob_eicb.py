"""Independent numerical contracts for the adapted EICB method, no evaluation."""
import numpy as np
import pytest

from confirm12.cbob_eicb_candidate import (
    balanced_probability, expected_improvement, candidate_eicb, FORMULA,
)


def test_balanced_pof_finite_bounds_and_diagnostics():
    mean=np.array([-2.,0.,2.])
    variance=np.ones(3)*.5
    p, info=balanced_probability(mean,variance)
    assert p.shape==(3,)
    assert np.isfinite(p).all()
    assert ((p>=0)&(p<=1)).all()
    assert p[0] > p[2]
    assert info["balance"] >= .001
    assert 0 < info["cdf_standard"][0] <= 1
    assert all(0 <= x <= 1 for x in info["cdf_standard"])


def test_eicb_recomputes_EI_times_constraint_probability():
    prediction={
        "completion":{"mean":np.array([-.5,-.2]),
                      "variance":np.array([.15,.15])},
        "constraints":{"soc":{"mean":np.array([-.1,.3]),
                              "variance":np.array([.1,.2])}},
        "objective":{"mean":np.array([11.,9.]),"variance":np.array([1.,2.])},
    }
    scores,meta=candidate_eicb(prediction,10.)
    assert np.allclose(scores,
        expected_improvement(np.array([11.,9.]),np.array([1.,2.]),10.)*
        np.array(meta["feasibility_joint"]))
    assert meta["formula"]==FORMULA
    assert meta["not_upstream_code"]
    cold,info=candidate_eicb(prediction,None)
    assert np.allclose(cold,info["feasibility_joint"])
    assert info["mode"]=="cold_balanced_feasibility"


@pytest.mark.parametrize("mean,var", [
    ([0.],[-1.]),([np.nan],[1.]),([0.],[np.inf]),
    ([0.,1.],[1.]),([],[]),
])
def test_eicb_refuses_untrusted_gp_arrays(mean,var):
    with pytest.raises(ValueError):
        balanced_probability(mean,var)


def test_expected_improvement_deterministic():
    low=expected_improvement(np.array([5.,10.]),np.array([1.,1.]),7.)
    assert low[0] > low[1] and np.all(low>=0)


def test_balance_uses_common_pool_no_hidden_objective_or_phys():
    first,_=balanced_probability(np.array([0.,1.]),np.array([1.,1.]))
    again,_=balanced_probability(np.array([0.,1.]),np.array([1.,1.]))
    assert np.array_equal(first,again)
