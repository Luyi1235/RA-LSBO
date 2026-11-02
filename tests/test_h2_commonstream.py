"""Common forecast-stream H2-B/H2-M scientific invariants; zero physical calls."""
import math

from tools import compare_h2_controllers_common_stream as c


def test_same_96_analytic_outcomes_per_seed_no_early_update():
    rows=c.run(c.SEEDS[0],c.SCENARIOS[0])
    assert len(rows)==c.STEPS
    assert rows[0]["H2B_prior_onehot_weight"]==.75
    assert rows[0]["H2M_prior_onehot_weight"]==.5
    for i,r in enumerate(rows):
        assert r["step"]==i
        assert set(r["NLL"])==set(c.ARMS)
        assert r["score_type"] in {"exact_density","interval_mass"}
        assert r["H2B_selected_view"] in {"onehot","latent"}
        if i:
            assert r["H2B_prior_onehot_weight"]==rows[i-1]["H2B_weight_after"]
            assert r["H2M_prior_onehot_weight"]==rows[i-1]["H2M_weight_after"]
        for value in r["NLL"].values():
            assert math.isfinite(value)
        for name,b in r["Brier"].items():
            assert b is None or 0<=b<=1


def test_mixture_true_density_is_bounded_by_component_densities():
    rows=c.run(c.SEEDS[1],c.SCENARIOS[1])
    for r in rows:
        one=r["NLL"]["onehot"]
        latent=r["NLL"]["latent"]
        mixture=r["NLL"]["H2M_log_mixture"]
        assert min(one,latent)-1e-12 <= mixture <= max(one,latent)+1e-12
        assert r["NLL"]["H2B_hard_Brier"] == r["NLL"][r["H2B_selected_view"]]
        assert r["NLL"]["min_guard"] == r["NLL"][r["min_guard_selected_view"]]


def test_trajectory_is_exactly_reproducible():
    assert c.run(c.SEEDS[0],c.SCENARIOS[2])==c.run(c.SEEDS[0],c.SCENARIOS[2])


def test_evidence_types_are_not_pooled_and_budget_is_exact():
    data=c.benchmark()
    assert data["plan"]["new_aircraft_calls"]==0
    assert data["plan"]["new_RCAIDE_calls"]==0
    assert data["plan"]["new_GP_fits"]==0
    assert data["plan"]["new_VAE_fits"]==0
    assert data["plan"]["synthetic_oracle_calls"]==len(c.SCENARIOS)*len(c.SEEDS)*c.STEPS==3072
    assert len(data["rows"])==3072
    assert len(data["summaries"])==len(c.SCENARIOS)*2*len(c.ARMS)==40
    assert {r["score_type"] for r in data["summaries"]}=={
        "exact_density","interval_mass"}
    assert all(r["n"]>0 for r in data["summaries"])
