"""Opt-in corpus BO integration; historical optimizer implementations stay frozen.

This adapter changes feature construction only. The mixed likelihood, kernel
candidate grid, acquisition, global pools and phase policy are inherited.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ...budget import digest
from ...features import load_base
from ...structure_corpus import (
    CorpusRepresentation, feature_snapshot, grammar_diagnostics,
    native_observations, observation_rows, sampled_grammar,
)
from ...structured_encoder import EncoderSettings, hierarchy_space
from ...risk_core.observations import OutputObservation
from ...soc_revision.posterior import RiskPosterior, AveragedGroupedHead
from ...partial_refinement.posterior import GroupedMixedHead
from ..discovery_stages.optimizer import DiscoveryBO

SCHEMA = 'ra_lsbo_closedloop_corpus_state_v1'
COUNT = 2000
UPDATE_EVERY = 8
INITIAL_EPOCHS = 30
UPDATE_EPOCHS = 20


def encoder_settings():
    return EncoderSettings(initial_epochs=INITIAL_EPOCHS, update_epochs=UPDATE_EPOCHS,
                           joint_view='latent')


def training_iterations(horizon):
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Positive integer horizon required')
    return list(range(0, horizon, UPDATE_EVERY))


def write_new(path, value):
    """Never replace an existing checkpoint or journal entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SnapshotRiskPosterior(RiskPosterior):
    """Same response-head construction as RiskPosterior, injected sealed features.

    No global monkeypatch: simultaneous references and controls cannot acquire
    another trial's representation. The inherited predict/diagnostics methods
    use the exact same head/feature interface as the frozen implementation.
    """
    def __init__(self, problem, records, seed, settings, features):
        self.problem, self.features = problem, features
        if any(r.observation.contract_id != problem.contract_id for r in records):
            raise ValueError('Posterior history contract differs')
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
                OutputObservation('interval', packet.evidence_id+':completion', lower=0., lower_inclusive=False))
        self.completion = fit(labels, seed=seed, completion=True)
        self.constraints = {name: fit([supported(p, p.constraint(name), name) for p in packets],
            seed=seed+number+1) for number, name in enumerate(problem.required_constraints)
            if name not in problem.analytical_constraints}
        observations = [supported(p, p.objective, 'objective') for p in packets]
        self.objective = fit(observations, seed=seed+100, objective=True,
            reference_scale=getattr(problem, 'objective_reference_scale', 1.)) if any(
                v.kind != 'missing' for v in observations) else None
        self.best = min((r.observation.objective.value for r in records
                        if r.usable_feasible and r.observation.objective.kind == 'exact'), default=None)
        self.history_digest = digest([r.to_dict() for r in records])


class CorpusDiscoveryBO(DiscoveryBO):
    """Persistent representation with a fixed, outcome-independent fit schedule.

    Fit before proposals 0,8,16,24, including a scheduled coverage proposal.
    Between fits, the GP still uses all current observations, encoded by a
    current immutable snapshot. A checkpoint is written before its first real
    evaluation. A failed propose cannot silently reuse a partly advanced state.
    """
    def __init__(self, problem, base, seed, settings, output):
        if not settings.latent or settings.encoder_epochs != INITIAL_EPOCHS:
            raise ValueError('Corpus path requires latent=True and the fixed 30 initial epochs')
        super().__init__(problem, base, seed, settings, cold_global=True, protect=True,
                         feedback='interval', coverage_scope='cold')
        self.output = Path(output)
        self.session = None
        self.initial_count = None
        self.latest_training = None
        self.last_requested_iteration = -1
        self.failed = False

    def _update_representation(self, records, iteration):
        if self.session is None:
            if iteration != 0:
                raise ValueError('A new persistent trial must start at iteration zero')
            self.initial_count = len(records)
            self.modules, _ = load_base(self.base)
            space = hierarchy_space(self.modules, [asdict(v) for v in self.problem.space.variables])
            self.session = CorpusRepresentation(space, self.seed, count=COUNT, settings=encoder_settings())
        if len(records) != self.initial_count+iteration:
            raise ValueError('Representation schedule requires one settled observation per iteration')
        if iteration % UPDATE_EVERY != 0:
            return
        observations = native_observations(records, self.modules, self.session.space)
        event = self.session.fit(observations, incremental=iteration > 0)
        folder = self.output/'representation'/f'iteration_{iteration:03d}'
        self.session.save_checkpoint(folder)
        diagnostics = grammar_diagnostics(self.session.representation)
        write_new(folder/'diagnostics.json', diagnostics)
        self.latest_training = {
            'iteration': iteration, 'raw_training_prefix': digest([r.to_dict() for r in records]),
            'native_training_prefix': digest(observation_rows(observations)),
            'observed_rows': len(records), 'parameter_state': event['parameter_state_after'],
            'checkpoint': str(folder.relative_to(self.output)).replace('\\', '/'),
            'files': {name: file_sha(folder/name)
                      for name in ('metadata.json', 'state.npz', 'diagnostics.json')},
        }

    def _build_posterior(self, records):
        features = feature_snapshot(self.problem, self.base, records, self.session)
        return SnapshotRiskPosterior(self.problem, records, self.seed, self.settings, features)

    def propose(self, records, global_pool, iteration):
        prefix = digest([r.to_dict() for r in records])
        if (self.failed or self.pending is not None or iteration != self.next_iteration
                or iteration != self.last_requested_iteration+1
                or not 0 <= iteration < self.settings.horizon
                or (self.expected_prefix is not None and prefix != self.expected_prefix)):
            raise ValueError('Persistent proposal state/prefix changed; no implicit retry')
        if any(r.observation.contract_id != self.problem.contract_id or
               r.observation.status in {'unknown', 'infrastructure_error'} or
               r.diagnostics.get('numerical_failure', False) for r in records):
            raise ValueError('Untrusted or foreign representation training history')
        try:
            self._update_representation(records, iteration)
            candidate, meta = super().propose(records, global_pool, iteration)
        except Exception:
            self.failed = True
            raise
        self.last_requested_iteration = iteration
        meta['corpus_training'] = {
            'schema': SCHEMA, 'corpus_identity': self.session.corpus['identity'],
            'draws': COUNT, 'latest': deepcopy(self.latest_training),
            'fit_count': len(self.session.events),
            'cumulative_gradient_updates': self.session.total_updates,
            'cumulative_fit_seconds': float(sum(self.session.fit_seconds)),
            'current_gp_prefix': prefix,
            'representation_fit_on_this_proposal': iteration % UPDATE_EVERY == 0,
            'gp_fit_skipped': meta['posterior_fit_skipped'],
            'representation_fit_included_in_proposal_time': True,
            'reconstruction_controls_selection': False,
        }
        return candidate, meta


def audit_training(problem, base, seed, initial, proposals, online, output, settings):
    """Validate checkpoints and training prefixes using forward passes, no fits.

    Recomputes the deterministic codebook diagnostics from numeric checkpoints.
    Does not reproduce SGD, fit GP heads or replay candidate optimization.
    """
    if len(proposals) != len(online):
        raise ValueError('Incomplete persistent trial')
    output = Path(output)
    modules, _ = load_base(base)
    space = hierarchy_space(modules, [asdict(v) for v in problem.space.variables])
    _, corpus = sampled_grammar(space, COUNT, seed+62000000)
    latest, prior_session = None, None
    checked = []
    durations = []
    updates = 0
    for i, (prop, row) in enumerate(zip(proposals, online)):
        history = initial+online[:i]
        raw_prefix = digest([r.to_dict() for r in history])
        if prop['iteration'] != i or prop['candidate'] != row.candidate or prop['prefix_digest'] != raw_prefix:
            raise ValueError('Persistent proposal association changed')
        item = prop['metadata'].get('corpus_training')
        if not isinstance(item, dict):
            raise ValueError('Missing representation training evidence')
        if i % UPDATE_EVERY == 0:
            relative = f'representation/iteration_{i:03d}'
            folder = output/relative
            observations = native_observations(history, modules, space)
            session = CorpusRepresentation.load_checkpoint(folder, space, observations)
            if (session.seed != seed or asdict(session.settings) != asdict(encoder_settings())
                    or session.corpus != corpus or len(session.events) != i//UPDATE_EVERY+1):
                raise ValueError('Checkpoint settings, corpus or schedule changed')
            event = session.events[-1]
            epochs = INITIAL_EPOCHS if i == 0 else UPDATE_EPOCHS
            expected_updates = epochs*((COUNT+len(history)+127)//128)
            updates += expected_updates
            if (event['gradient_updates'] != expected_updates or session.total_updates != updates
                    or event['epochs'] != epochs or event['observed_rows'] != len(history)
                    or event['structural_rows'] != COUNT+len(history)
                    or event['observation_prefix_identity'] != digest(observation_rows(observations))
                    or event['learning_rate'] != (.001 if i == 0 else .0004)
                    or len(event['training_trace']) != epochs):
                raise ValueError('Training budget or event history changed')
            if prior_session is not None and (
                session.events[:-1] != prior_session.events or
                session.fit_seconds[:-1] != prior_session.fit_seconds or
                event['parameter_state_before'] != prior_session._sealed_state):
                raise ValueError('Persistent checkpoint chain changed')
            diagnostics = json.loads((folder/'diagnostics.json').read_bytes())
            if diagnostics != grammar_diagnostics(session.representation):
                raise ValueError('Saved structural diagnostics differ from checkpoint')
            latest = {'iteration': i, 'raw_training_prefix': raw_prefix,
                      'native_training_prefix': digest(observation_rows(observations)),
                      'observed_rows': len(history), 'parameter_state': session._sealed_state,
                      'checkpoint': relative,
                      'files': {n: file_sha(folder/n) for n in ('metadata.json','state.npz','diagnostics.json')}}
            durations = list(session.fit_seconds)
            checked.append({'iteration': i, 'gradient_updates': expected_updates,
                            'fit_seconds': durations[-1],
                            'mean_roundtrip_target_met': diagnostics['mean_roundtrip_target_met'],
                            'raw_sampled_active_exact': diagnostics['raw_sampled_active_exact']})
            prior_session = session
        expected = {'schema': SCHEMA, 'corpus_identity': corpus['identity'], 'draws': COUNT,
                    'latest': latest, 'fit_count': len(checked),
                    'cumulative_gradient_updates': updates, 'cumulative_fit_seconds': float(sum(durations)),
                    'current_gp_prefix': raw_prefix, 'representation_fit_on_this_proposal': i%UPDATE_EVERY==0,
                    'gp_fit_skipped': prop['metadata']['posterior_fit_skipped'],
                    'representation_fit_included_in_proposal_time': True,
                    'reconstruction_controls_selection': False}
        if item != expected:
            raise ValueError('Proposal representation identity or cost changed')
        if not prop['metadata']['posterior_fit_skipped']:
            features = feature_snapshot(problem, base, history, prior_session)
            if prop['metadata']['feature_identity'] != features.identity:
                raise ValueError('GP feature snapshot does not match the scheduled representation')
    actual = {p.name for p in (output/'representation').iterdir()}
    if actual != {f'iteration_{i:03d}' for i in training_iterations(len(online))}:
        raise ValueError('Missing or unregistered representation checkpoint')
    return {'fit_count': len(checked), 'gradient_updates': updates, 'fit_seconds': float(sum(durations)),
            'checkpoint_events': checked, 'corpus_identity': corpus['identity'],
            'draws': COUNT, 'distinct_configurations': corpus['distinct_configurations'],
            'training_replayed': False, 'GP_refitted': False, 'diagnostic_forward_passes': True}
