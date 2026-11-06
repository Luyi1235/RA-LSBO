"""Restore saved VAE weights and selected GP heads; NEVER train the VAE or reselect kernels."""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace
import time
from unittest.mock import patch
import numpy as np
from ra_lsbo.budget import digest
from ra_lsbo.features import load_base
from ra_lsbo.structured_encoder import hierarchy_space
from ra_lsbo.structure_corpus import CorpusRepresentation, native_observations, feature_snapshot
from ra_lsbo.partial_bo.models import PrefixFeatures, EvidenceGP
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings
from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.partial_refinement.posterior import group_divisors
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import HeterogeneousRiskPosterior
from ra_lsbo.experiments.information_value_stage1.io import append
from pr71_protocol import RTOL, ATOL


class GPLedger:
    def __init__(self, path):
        self.path, self.context = path, {}
        self.started = self.completed = self.errors = 0

    def fit(self, phase, *args, **kwargs):
        ticket = self.started
        append(self.path, {'ticket': ticket, 'event': 'started', 'phase': phase, 'context': dict(self.context)})
        self.started += 1
        t = time.perf_counter()
        try:
            model = EvidenceGP(*args, **kwargs)
        except BaseException as e:
            self.errors += 1
            append(self.path, {'ticket': ticket, 'event': 'error', 'phase': phase,
                'context': dict(self.context), 'error_type': type(e).__name__, 'error': str(e),
                'seconds': time.perf_counter()-t})
            raise
        self.completed += 1
        append(self.path, {'ticket': ticket, 'event': 'completed', 'phase': phase,
            'context': dict(self.context), 'seconds': time.perf_counter()-t})
        return model

    def summary(self):
        return {'started': self.started, 'completed': self.completed, 'errors': self.errors}


@contextmanager
def no_new_truth_or_training(ledger):
    from ra_lsbo.experiments.core.problems import MathematicalCase, AircraftCase
    from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft
    from ra_lsbo.experiments.information_value_stage3 import native
    def forbidden(*args, **kwargs):
        raise RuntimeError('PR71 forbids evaluator calls and VAE training')
    with ExitStack() as stack:
        for cls in (MathematicalCase, AircraftCase, SimplifiedElectricAircraft):
            stack.enter_context(patch.object(cls, 'evaluate', forbidden))
        stack.enter_context(patch.object(CorpusRepresentation, 'fit', forbidden))
        stack.enter_context(patch.object(native, 'EvidenceGP', lambda *a, **k: ledger.fit('fantasy', *a, **k)))
        yield


class RestoredHead:
    def __init__(self, inputs, observations, saved, nc, ledger):
        self.center, self.scale = float(saved['center']), float(saved['scale'])
        if not np.isfinite([self.center, self.scale]).all() or self.scale <= 0:
            raise ValueError('Invalid saved normalization')
        sel = saved['selected']
        self.divisor = group_divisors(inputs.shape[1], nc, sel['continuous_length'], sel['discrete_length'])
        normalized = []
        for o in observations:
            if o.kind == 'exact':
                o = replace(o, value=(o.value-self.center)/self.scale)
            elif o.kind == 'interval':
                o = replace(o, lower=None if o.lower is None else (o.lower-self.center)/self.scale,
                            upper=None if o.upper is None else (o.upper-self.center)/self.scale)
            normalized.append(o)
        counts = saved['counts']
        ledger.context['head'] = saved['name']
        self.model = ledger.fit('prefix_selected_kernel', inputs/self.divisor, normalized,
            seed=counts['seed'], settings=MixedGPSettings(**counts['settings']))
        ledger.context.pop('head', None)
        # Identity is the original selected-head provenance, not a new kernel search.
        self.identity = saved['identity']

    def predict(self, inputs):
        p = self.model.predict(np.asarray(inputs, float)/self.divisor)
        return {'mean': self.center+self.scale*p['mean'], 'variance': self.scale**2*p['variance']}


def restore(problem, base, seed, initial, history, folder, meta, ledger):
    modules, _ = load_base(base)
    space = hierarchy_space(modules, [asdict(v) for v in problem.space.variables])
    checkpoint = folder/'representation'/'iteration_000'
    if meta['corpus_training']['latest']['checkpoint'] != 'representation/iteration_000':
        raise ValueError('Unregistered representation update')
    session = CorpusRepresentation.load_checkpoint(checkpoint, space, native_observations(initial, modules, space))
    if session.seed != seed or session._sealed_state != meta['corpus_training']['latest']['parameter_state']:
        raise ValueError('Checkpoint state or seed changed')
    latent = feature_snapshot(problem, base, history, session)
    risk = PrefixFeatures(problem, base, history, seed, False, 30)
    if digest([r.to_dict() for r in history]) != meta['history_digest']:
        raise ValueError('Posterior training history differs')
    post = HeterogeneousRiskPosterior.__new__(HeterogeneousRiskPosterior)
    post.problem, post.features = problem, latent
    post.objective_features, post.risk_features = latent, risk
    post.dual_soc, post.dual_soc_head = False, None
    post.view_policy, post.history_digest = meta['view_policy'], meta['history_digest']
    packets = [r.observation for r in history]
    if any(p.contract_id != problem.contract_id or p.status != 'completed' for p in packets):
        raise ValueError('Foreign or untrusted prefix')
    nc = len(problem.space.continuous)
    xo = latent.transform([r.candidate for r in history])
    xr = risk.transform([r.candidate for r in history])
    labels = []
    for p in packets:
        label = p.completion_label()
        if label is None:
            raise ValueError('Unknown completion in source')
        labels.append(OutputObservation('interval', p.evidence_id+':completion', upper=0.) if label else
                      OutputObservation('interval', p.evidence_id+':completion', lower=0., lower_inclusive=False))
    saved = meta['heads']
    post.completion = RestoredHead(xr, labels, saved['completion'], nc, ledger)
    post.constraints = {name: RestoredHead(xr, [p.constraint(name) for p in packets], saved[name], nc, ledger)
                        for name in ('region', 'resource')}
    post.objective = (RestoredHead(xo, [p.objective for p in packets], saved['objective'], nc, ledger)
                      if 'objective' in saved else None)
    post.best = min((r.observation.objective.value for r in history if r.usable_feasible), default=None)
    if post.best != meta['best_observed']:
        raise ValueError('Incumbent changed')
    heads = {**post.constraints, 'completion': post.completion}
    if post.objective is not None:
        heads['objective'] = post.objective
    identity = digest({name: h.model.to_dict() for name, h in heads.items()})
    receipt = {'checkpoint_state': session._sealed_state,
        'history_digest': post.history_digest, 'selected_head_identities': {n: h.identity for n, h in heads.items()},
        'restored_model_hash': identity,
        'feature_identity_exact': latent.identity == meta['objective_feature_identity'],
        'risk_feature_identity_exact': risk.identity == meta['risk_feature_identity'],
        'kernel_reselection': False, 'VAE_training': False}
    return post, receipt


def model_hash(post):
    heads = {**post.constraints, 'completion': post.completion}
    if post.objective is not None:
        heads['objective'] = post.objective
    return digest({name: h.model.to_dict() for name, h in heads.items()})
