"""Unqualified reverse-view research candidate, never an approved adapter.

Uses the existing frozen corpus schedule and posterior implementation. It is
not registered in any real runner and cannot qualify itself for confirmation.
"""
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.structure_corpus import feature_snapshot
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import (
    HeterogeneousRiskPosterior, StableCorpusDiscoveryBO,
)


class ReverseRepresentationCandidate(StableCorpusDiscoveryBO):
    """Explicit objective features; latent completion and constraint features."""
    def __init__(self, problem, base, seed, settings, output):
        super().__init__(problem, base, seed, settings, output, allocation='latent_all')

    def _build_posterior(self, records):
        latent = feature_snapshot(self.problem, self.base, records, self.session)
        explicit = PrefixFeatures(self.problem, self.base, records, self.seed, False, 30)
        return HeterogeneousRiskPosterior(self.problem, records, self.seed, self.settings,
            explicit, latent, dual_soc=False, view_policy='explicit_objective_latent_constraints')
