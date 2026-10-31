"""Opt-in persistent structural corpus in the frozen phase-RA decision policy.

Only the feature lifecycle changes. GP likelihoods/grids, nominal physics,
coverage, local search, output semantics and incumbent selection are inherited.
No corpus draw is an evaluator call or a feasibility label.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from ...budget import digest
from ...features import load_base
from ...partial_bo.models import PrefixFeatures
from ...partial_refinement.posterior import RefinementPosterior, GroupedMixedHead
from ...risk_core.observations import OutputObservation
from ...soc_revision.posterior import AveragedGroupedHead
from ...structure_corpus import (CorpusRepresentation, native_observations,
                                 feature_snapshot, grammar_diagnostics)
from ...structured_encoder import EncoderSettings, hierarchy_space
from ..core.runner import append, save
from ..discovery_stages.optimizer import DiscoveryBO

UPDATE_EVERY = 8
CORPUS_COUNT = 2000


def encoder_settings():
    # Same training protocol as the accepted offline corpus experiment, also in
    # smoke. A two-step smoke does NOT exercise the later scheduled updates.
    return EncoderSettings(initial_epochs=30, update_epochs=20, joint_view='latent')


def scheduled_updates(horizon):
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Positive integer horizon required')
    return list(range(0, horizon, UPDATE_EVERY))


def feature_report(features):
    candidates = []
    for config in features.space.configurations():
        raw = features.space.default().to_dict()
        raw['discrete'] = config
        candidates.append(features.candidate(raw).to_dict())
    nc = len(features.space.continuous)
    return {'feature_identity': features.identity, 'encoder_state': features.state_hash,
            'offset': features.offset.tolist(), 'scale': features.scale.tolist(),
            'configurations': features.space.configurations(),
            'normalized_codebook': features.transform(candidates)[:, 2*nc:].tolist()}


class SuppliedFeaturePosterior(RefinementPosterior):
    """RiskPosterior's frozen likelihood/heads with an explicitly supplied view.

    Keeping the original class untouched preserves the exact old controls and
    historical source identities. Tests compare this adapter to RiskPosterior.
    """
    def __init__(self, problem, records, seed, settings, features):
        self.problem, self.features = problem, features
        inputs = features.transform([r.candidate for r in records])
        packets = [r.observation for r in records]

        def fit(observations, **kwargs):
            head = AveragedGroupedHead if settings.average_kernel else GroupedMixedHead
            return head(inputs, observations, continuous_count=len(problem.space.continuous),
                        continuous_grid=settings.continuous_grid,
                        discrete_grid=settings.discrete_grid, **kwargs)

        def missing(packet, name):
            return OutputObservation('missing', packet.evidence_id+':'+name,
                                     reason='untrusted_terminal_status')

        def supported(packet, value, name):
            return value if packet.status not in {'unknown', 'infrastructure_error'} else missing(packet, name)

        labels = []
        for packet in packets:
            label = packet.completion_label()
            labels.append(missing(packet, 'completion') if label is None else
                          OutputObservation('interval', packet.evidence_id+':completion', upper=0.) if label else
                          OutputObservation('interval', packet.evidence_id+':completion',
                                            lower=0., lower_inclusive=False))
        self.completion = fit(labels, seed=seed, completion=True)
        self.constraints = {name: fit([supported(p, p.constraint(name), name) for p in packets],
                            seed=seed+number+1)
                            for number, name in enumerate(problem.required_constraints)
                            if name not in problem.analytical_constraints}
        observations = [supported(p, p.objective, 'objective') for p in packets]
        self.objective = fit(observations, seed=seed+100, objective=True,
                             reference_scale=getattr(problem, 'objective_reference_scale', 1.)) if any(
                             v.kind != 'missing' for v in observations) else None
        self.best = min((r.observation.objective.value for r in records
                         if r.usable_feasible and r.observation.objective.kind == 'exact'), default=None)
        self.history_digest = digest([r.to_dict() for r in records])


class LoggedControlBO(DiscoveryBO):
    """Unchanged old30/old480/onehot policy, with truthful fit-cost diagnostics."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_fit = None

    def _build_posterior(self, records):
        started = time.perf_counter()
        features = PrefixFeatures(self.problem, self.base, records, self.seed,
                                  self.settings.latent, self.settings.encoder_epochs)
        feature_seconds = time.perf_counter()-started
        started = time.perf_counter()
        posterior = SuppliedFeaturePosterior(self.problem, records, self.seed, self.settings, features)
        gp_seconds = time.perf_counter()-started
        # No future label is available here. Diagnostic forward passes cannot
        # advance the encoder RNG or change the selected point.
        diagnostic = {**feature_report(features), 'feature_seconds': feature_seconds,
                      'gp_seconds': gp_seconds, 'encoder_fit_calls': int(features.latent),
                      'gradient_updates': (self.settings.encoder_epochs *
                          int(np.ceil(len(features.encoder.grammar)/features.encoder.settings.batch_size)))
                          if features.latent else 0,
                      'grammar': grammar_diagnostics(features.encoder) if features.latent else None}
        self.last_fit = diagnostic
        return posterior

    def propose(self, records, global_pool, iteration):
        self.last_fit = None
        candidate, metadata = super().propose(records, global_pool, iteration)
        return candidate, {**metadata, 'representation_fit': deepcopy(self.last_fit)}


class CorpusDiscoveryBO(DiscoveryBO):
    """Train before t=0; retain parameters; update at t=8,16,24, even on coverage.

    The inherited acquisition may skip GP fitting on scheduled coverage. VAE
    training still follows its own fixed schedule and is fully charged/logged.
    Initial and update fits use only the current settled prefix. No resume API.
    """
    def __init__(self, problem, base, seed, settings, output):
        super().__init__(problem, base, seed, settings, cold_global=True, protect=True,
                         feedback='interval', random_discovery=False, coverage_scope='cold')
        self.output = Path(output)
        self.modules, _ = load_base(base)
        space = hierarchy_space(self.modules, [asdict(v) for v in problem.space.variables])
        self.session = CorpusRepresentation(space, seed, count=CORPUS_COUNT, settings=encoder_settings())
        self.current_features = None
        self.last_checkpoint = None
        self.last_fit_index = None
        self._proposal_failed = False
        self.gp_seconds = None

    def _update_representation(self, records, iteration):
        relative = 'representation/checkpoints/t'+str(iteration).zfill(3)
        start = {'event': 'started', 'iteration': iteration, 'history_count': len(records),
                 'prefix_digest': digest([r.to_dict() for r in records]),
                 'checkpoint': relative, 'corpus_identity': self.session.corpus['identity']}
        append(self.output/'representation/events.jsonl', start)
        event = self.session.fit(native_observations(records, self.modules, self.session.space),
                                 incremental=bool(self.session.events))
        folder = self.output/relative
        identity = self.session.save_checkpoint(folder)
        diagnostic = grammar_diagnostics(self.session.representation)
        save(folder/'grammar.json', diagnostic)
        append(self.output/'representation/events.jsonl', {**start, 'event': 'settled',
            'metadata_sha256': identity, 'state_identity': self.session._sealed_state,
            'gradient_updates': event['gradient_updates'],
            'cumulative_gradient_updates': event['cumulative_gradient_updates'],
            'fit_seconds': self.session.fit_seconds[-1],
            'raw_mean_active_exact': diagnostic['raw_mean_active_exact']})
        self.last_checkpoint, self.last_fit_index = relative, iteration

    def _build_posterior(self, records):
        if self.current_features is None:
            raise RuntimeError('Scheduled representation must precede posterior fitting')
        started = time.perf_counter()
        posterior = SuppliedFeaturePosterior(self.problem, records, self.seed,
                                             self.settings, self.current_features)
        self.gp_seconds = time.perf_counter()-started
        return posterior

    def propose(self, records, global_pool, iteration):
        prefix = digest([r.to_dict() for r in records])
        if (self._proposal_failed or self.pending is not None or iteration != self.next_iteration
                or not 0 <= iteration < self.settings.horizon
                or (self.expected_prefix is not None and prefix != self.expected_prefix)):
            raise ValueError('Invalid or failed persistent proposal state; no retry')
        if any(r.observation.contract_id != self.problem.contract_id or
               r.observation.status in {'unknown', 'infrastructure_error'} or
               r.diagnostics.get('numerical_failure', False) for r in records):
            raise ValueError('Untrusted or wrong-contract representation training')
        self.gp_seconds = None
        try:
            if iteration == 0:
                save(self.output/'representation/corpus.json', self.session.corpus)
            if iteration % UPDATE_EVERY == 0:
                self._update_representation(records, iteration)
            started = time.perf_counter()
            self.current_features = feature_snapshot(self.problem, self.base, records, self.session)
            snapshot_seconds = time.perf_counter()-started
            candidate, metadata = super().propose(records, global_pool, iteration)
            report = feature_report(self.current_features)
            report.update(checkpoint=self.last_checkpoint, last_fit_iteration=self.last_fit_index,
                          snapshot_seconds=snapshot_seconds, gp_seconds=self.gp_seconds,
                          scheduled_fit=iteration % UPDATE_EVERY == 0,
                          corpus_identity=self.session.corpus['identity'],
                          encoder_history_count=len(self.session.history),
                          gp_history_count=len(records),
                          cumulative_gradient_updates=self.session.total_updates,
                          cumulative_fit_seconds=sum(self.session.fit_seconds),
                          encoder_fit_calls=len(self.session.events))
            return candidate, {**metadata, 'persistent_representation': report}
        except Exception:
            self._proposal_failed = True
            raise
