"""Opt-in simulation-free structural corpus; frozen BO paths are unchanged.

The user reports 2,000 randomly generated, possibly repeated configurations in
LSBO's X-57 case. Here the sampling law is explicitly uniform over the canonical
legal configurations, with replacement. This is a declared adaptation, not a
claim to reproduce unspecified historical X-57 seeds or distributions.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .budget import digest
from .structured_encoder import EncoderSettings, StructuredRepresentation, state_digest


CORPUS_SCHEMA = 'ra_lsbo_structure_corpus_2000_v1'


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def observation_rows(observations):
    """Identity only. These are native LSBO observations, not invented labels."""
    return _json([asdict(o) for o in observations])


def sampled_grammar(space, count=2000, seed=0):
    """No evaluator, precheck, objective, feasibility test, or rejection sampling."""
    _positive(count, 'sample count')
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer corpus seed required')
    configurations = space.configurations()
    if not configurations or len({digest(c) for c in configurations}) != len(configurations):
        raise ValueError('Require a nonempty unique canonical configuration list')
    # A separate RNG cannot change VAE or optimizer random streams.
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(configurations), size=count).tolist()
    draws = [deepcopy(configurations[i]) for i in indices]
    tensor = torch.tensor([space.indices(d) for d in draws], dtype=torch.long)
    frequencies = Counter(indices)
    packet = {'schema': CORPUS_SCHEMA, 'sampling': 'uniform_canonical_with_replacement',
              'seed': seed, 'draws': count, 'distinct_configurations': len(frequencies),
              'available_configurations': len(configurations), 'configurations': configurations,
              'draw_indices': indices, 'counts': [frequencies[i] for i in range(len(configurations))],
              'simulation_labels': False, 'model_evaluations': 0}
    packet['identity'] = digest(packet)
    return tensor, packet


def native_observations(records, modules, space):
    """Exact trusted targets only; structural rows need no target or feasibility."""
    from .risk_core.observations import validated_runs
    validated_runs(r.observation for r in records)
    observations = []
    for index, record in enumerate(records, 1):
        c = modules['space'].Candidate(**deepcopy(record.candidate))
        if space.canonical(c).to_dict() != record.candidate:
            raise ValueError('Training candidates must already be canonical')
        run = record.observation
        trusted = (run.status not in {'unknown', 'infrastructure_error'}
                   and not record.diagnostics.get('numerical_failure', False))
        exact = trusted and run.objective.kind == 'exact'
        evaluation = modules['records'].Evaluation(
            run.objective.value if exact else None, exact,
            {name: value.value if trusted and value.kind == 'exact' else None
             for name, value in run.constraints}, run.status)
        observations.append(modules['records'].Observation(index, c, evaluation, 'current_prefix'))
    return observations


class CorpusRepresentation:
    """Joint initial training, then parameter-preserving low-rate updates.

    Every fit uses the fixed 2,000 draws plus one structural row per observed
    design (including designs without targets). Only real trusted exact targets
    enter the existing performance loss. No deduplication of structural draws.
    Adam is recreated each fit, as in the frozen implementation; neural weights
    and the private stochastic state are retained. This is not Adam-state resume.
    """
    def __init__(self, space, seed=0, *, count=2000, settings=None):
        if type(seed) is not int or seed < 0:
            raise ValueError('Nonnegative integer representation seed required')
        self.space = space
        self.settings = settings or EncoderSettings(initial_epochs=30, update_epochs=20, joint_view='latent')
        self.seed = seed
        self.sampled_indices, self.corpus = sampled_grammar(space, count, seed + 62000000)
        self.representation = StructuredRepresentation(space, self.settings, seed)
        self.history = []
        self.events = []
        self.total_updates = 0
        self.fit_seconds = []
        self._sealed_state = None
        self._failed = False

    def fit(self, observations, *, incremental=False):
        observations = list(observations)
        rows = observation_rows(observations)
        if self._failed:
            raise RuntimeError('Failed fit cannot be silently reused')
        if self._sealed_state is not None and state_digest(self.representation) != self._sealed_state:
            raise RuntimeError('Representation changed outside its controlled update')
        if digest({k: v for k, v in self.corpus.items() if k != 'identity'}) != self.corpus['identity']:
            raise ValueError('Structural corpus identity changed')
        expected = torch.tensor([self.space.indices(self.corpus['configurations'][i])
                                 for i in self.corpus['draw_indices']], dtype=torch.long)
        if not torch.equal(expected, self.sampled_indices):
            raise ValueError('Structural corpus draws changed')
        ids = [o.index for o in observations]
        if (not rows or any(type(i) is not int or i < 1 for i in ids)
                or ids != sorted(set(ids))):
            raise ValueError('Nonempty, increasing unique observation indices required')
        if incremental:
            if (not self.events or len(rows) <= len(self.history)
                    or rows[:len(self.history)] != self.history):
                raise ValueError('Incremental update requires an unchanged append-only prefix')
        elif self.events:
            raise ValueError('Use a new session for reset training')
        for o in observations:
            if self.space.canonical(o.candidate).to_dict() != o.candidate.to_dict():
                raise ValueError('Noncanonical structural observation')
        before = state_digest(self.representation)
        observed_indices = self.representation._indices([o.candidate for o in observations])
        self.representation.grammar = torch.cat([self.sampled_indices, observed_indices], dim=0)
        epochs = self.settings.update_epochs if incremental else self.settings.initial_epochs
        batches = math.ceil(len(self.representation.grammar)/self.settings.batch_size)
        started = time.perf_counter()
        try:
            self.representation.fit(observations, epochs=epochs, incremental=incremental)
        except Exception:
            self._failed = True
            raise
        self.fit_seconds.append(time.perf_counter()-started)
        self.total_updates += epochs*batches
        self.history = rows
        self._sealed_state = state_digest(self.representation)
        event = {'stage': 'incremental' if incremental else 'initial_joint',
                 'observed_rows': len(rows), 'structural_rows': len(self.representation.grammar),
                 'random_structural_rows': len(self.sampled_indices),
                 'trusted_exact_target_rows': sum(o.result.has_objective() for o in observations),
                 'epochs': epochs, 'batches_per_epoch': batches,
                 'gradient_updates': epochs*batches, 'cumulative_gradient_updates': self.total_updates,
                 'learning_rate': self.settings.update_learning_rate if incremental else self.settings.learning_rate,
                 'parameter_state_before': before, 'parameter_state_after': self._sealed_state,
                 'observation_prefix_identity': digest(rows), 'corpus_identity': self.corpus['identity'],
                 'training_trace': deepcopy(self.representation.training_trace)}
        self.events.append(event)
        return deepcopy(event)

    def save_checkpoint(self, folder):
        """Non-pickle numeric checkpoint, bound to schema, corpus and observations."""
        folder = Path(folder)
        if self._failed or not self.events or state_digest(self.representation) != self._sealed_state:
            raise RuntimeError('Only a valid sealed trained session can be saved')
        if folder.exists():
            raise FileExistsError('Checkpoint directory must be new')
        folder.mkdir(parents=True)
        names = sorted(self.representation.model.state_dict())
        arrays = {str(i): self.representation.model.state_dict()[n].detach().cpu().numpy()
                  for i, n in enumerate(names)}
        arrays['generator'] = self.representation.generator.get_state().numpy()
        arrays['grammar'] = self.representation.grammar.numpy()
        arrays['sampled_indices'] = self.sampled_indices.numpy()
        np.savez_compressed(folder/'state.npz', **arrays)
        rep = self.representation
        meta = {'schema': CORPUS_SCHEMA, 'variables': [asdict(v) for v in self.space.variables],
                'settings': asdict(self.settings), 'seed': self.seed, 'corpus': self.corpus,
                'history': self.history, 'events': self.events, 'total_updates': self.total_updates,
                'fit_seconds': self.fit_seconds,
                'tensor_names': names, 'state_digest': self._sealed_state,
                'version': rep.version, 'y_mean': rep.y_mean, 'y_std': rep.y_std,
                'performance_fitted': rep.performance_fitted,
                'state_file_sha256': hashlib.sha256((folder/'state.npz').read_bytes()).hexdigest()}
        text = json.dumps(meta, sort_keys=True, indent=2, allow_nan=False)+'\n'
        (folder/'metadata.json').write_text(text, encoding='utf-8')
        return hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def load_checkpoint(cls, folder, space, observations):
        folder = Path(folder)
        m = json.loads((folder/'metadata.json').read_bytes())
        if (m['schema'] != CORPUS_SCHEMA or m['variables'] != _json([asdict(v) for v in space.variables])
                or m['history'] != observation_rows(observations)
                or hashlib.sha256((folder/'state.npz').read_bytes()).hexdigest() != m['state_file_sha256']):
            raise ValueError('Checkpoint schema, history or state file differs')
        obj = cls(space, m['seed'], count=m['corpus']['draws'], settings=EncoderSettings(**m['settings']))
        if obj.corpus != m['corpus']:
            raise ValueError('Checkpoint corpus differs from deterministic draws')
        with np.load(folder/'state.npz', allow_pickle=False) as a:
            names = sorted(obj.representation.model.state_dict())
            if names != m['tensor_names']:
                raise ValueError('Checkpoint model tensor names differ')
            obj.representation.model.load_state_dict({n: torch.from_numpy(a[str(i)].copy()) for i,n in enumerate(names)})
            obj.representation.generator.set_state(torch.from_numpy(a['generator'].copy()))
            obj.representation.grammar = torch.from_numpy(a['grammar'].copy())
            if not torch.equal(obj.sampled_indices, torch.from_numpy(a['sampled_indices'].copy())):
                raise ValueError('Checkpoint sampled indices differ')
        rep = obj.representation
        rep.version, rep.y_mean, rep.y_std = m['version'], m['y_mean'], m['y_std']
        rep.performance_fitted = m['performance_fitted']
        rep.training_trace = deepcopy(m['events'][-1]['training_trace'])
        rep.model.eval()
        obj.history, obj.events, obj.total_updates = m['history'], m['events'], m['total_updates']
        obj.fit_seconds = list(m['fit_seconds'])
        if len(obj.fit_seconds) != len(obj.events) or any(not math.isfinite(t) or t < 0 for t in obj.fit_seconds):
            raise ValueError('Invalid fit timing record')
        obj._sealed_state = state_digest(rep)
        if obj._sealed_state != m['state_digest']:
            raise ValueError('Checkpoint reconstructed identity differs')
        return obj


def feature_snapshot(problem, base, records, session):
    """Independent immutable-by-seal GP view; legacy normalization is unchanged.

    A current GP prefix may be longer than the last scheduled encoder-training
    prefix. That distinction is explicit in both identities. Later session fits
    cannot mutate an earlier GP feature snapshot.
    """
    from .partial_bo.models import PrefixFeatures
    from .features import load_base
    if session.settings.latent_dim != 4:
        raise ValueError('Legacy GP feature adapter requires four latent dimensions')
    if [asdict(v) for v in problem.space.variables] != [asdict(v) for v in session.space.variables]:
        raise ValueError('Representation schema differs from the GP schema')
    if session._failed or not session.events or state_digest(session.representation) != session._sealed_state:
        raise RuntimeError('Need valid sealed structural representation')
    if any(r.observation.contract_id != problem.contract_id for r in records):
        raise ValueError('Training contract differs')
    modules, _ = load_base(base)
    observations = native_observations(records, modules, session.space)
    rows = observation_rows(observations)
    if rows[:len(session.history)] != session.history:
        raise ValueError('GP prefix does not extend the encoder-training prefix')
    obj = PrefixFeatures.__new__(PrefixFeatures)
    obj.modules, obj.space = modules, session.space
    obj.problem, obj.latent = problem, True
    obj.encoder = deepcopy(session.representation)
    raw = obj._raw([r.candidate for r in records])
    obj.offset, obj.scale = np.zeros(raw.shape[1]), np.ones(raw.shape[1])
    obj.offset[-4:] = raw[:, -4:].min(0)-.5
    obj.scale[-4:] = raw[:, -4:].max(0)+.5-obj.offset[-4:]
    obj.state_hash = state_digest(obj.encoder)
    obj.identity = digest({'schema': CORPUS_SCHEMA, 'prefix': [r.to_dict() for r in records],
                          'corpus': session.corpus['identity'], 'settings': asdict(session.settings), 'encoder': obj.state_hash,
                          'encoder_prefix': digest(session.history),
                          'offset': obj.offset.tolist(), 'scale': obj.scale.tolist()})
    return obj


def grammar_diagnostics(representation, *, noise_draws=32):
    """Raw-head correctness, not repaired validity, generalization or GP quality."""
    _positive(noise_draws, 'noise draws')
    indices = torch.tensor([representation.space.indices(c) for c in representation.space.configurations()], dtype=torch.long)
    with torch.no_grad():
        mu, logvar = representation.model.encode(indices)
        mask = representation.model.discrete_mask(indices).bool()
        decoded = torch.stack([h(mu).argmax(-1) for h in representation.model.decoder], -1)
        good = (decoded == indices) | ~mask
        rng = torch.Generator().manual_seed(71000001)  # Does not advance training RNG.
        z = mu[None] + (logvar*.5).exp()[None]*torch.randn((noise_draws, *mu.shape), generator=rng, dtype=torch.double)
        sampled = torch.stack([h(z).argmax(-1) for h in representation.model.decoder], -1)
        sample_good = (sampled == indices[None]) | ~mask[None]
    return {'configurations': representation.space.configurations(),
            'raw_mean_active_exact': float(good.all(-1).double().mean()),
            'raw_sampled_active_exact': float(sample_good.all(-1).double().mean()),
            'active_field_accuracy': {v.name: (float(good[:,i][mask[:,i]].double().mean()) if mask[:,i].any() else None)
                                      for i,v in enumerate(representation.space.discrete)},
            'mu': mu.tolist(), 'logvar': logvar.tolist(), 'decoded_indices': decoded.tolist(),
            'noise_draws': noise_draws, 'training_grammar_not_holdout': True,
            'mean_roundtrip_target_met': bool(good.all()), 'automatic_BO_authorization': False}
