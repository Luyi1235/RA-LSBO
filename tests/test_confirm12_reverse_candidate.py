"""Feature-routing test only; no truth, VAE fitting or GP fitting."""
from confirm12 import reverse_candidate as reverse


def test_reverse_candidate_routes_explicit_objective_latent_risk(monkeypatch):
    calls = []
    latent, explicit = object(), object()
    monkeypatch.setattr(reverse, 'feature_snapshot', lambda *args: latent)
    monkeypatch.setattr(reverse, 'PrefixFeatures', lambda *args: explicit)
    def posterior(problem, records, seed, settings, objective, risk, **kwargs):
        calls.append((problem, records, seed, settings, objective, risk, kwargs))
        return 'unfitted-test-stub'
    monkeypatch.setattr(reverse, 'HeterogeneousRiskPosterior', posterior)
    engine = object.__new__(reverse.ReverseRepresentationCandidate)
    engine.problem, engine.base, engine.seed, engine.settings, engine.session = object(), object(), 42, object(), object()
    records = [object()]
    assert engine._build_posterior(records) == 'unfitted-test-stub'
    assert calls[0][4] is explicit and calls[0][5] is latent
    assert calls[0][6] == {'dual_soc':False, 'view_policy':'explicit_objective_latent_constraints'}
