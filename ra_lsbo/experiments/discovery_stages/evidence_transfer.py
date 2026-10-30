"""Experimental cold-stage representation trust from prequential constraint evidence.

This is likelihood-weighted expert aggregation, NOT a posterior probability that
an encoder is correct. Predictions are sealed before evaluation. Missing/untrusted
outputs do not update weights; exact values are scored as densities and intervals
as probability masses. No interval midpoint or fabricated objective is used.
Model averaging and prequential weighting are established methods; novelty of this
particular hierarchical partial-feedback application is a hypothesis, not a claim.
"""
from copy import deepcopy
from dataclasses import replace

import numpy as np
from scipy.special import logsumexp

from ...budget import digest
from ...partial_bo.models import log_interval_probability
from ...soc_revision.posterior import RiskPosterior
from .optimizer import DiscoveryBO

EXPERTS = ('latent', 'hierarchical_onehot')
PREDICTIVE_VARIANCE_FLOOR = 1e-12  # relative to physical head scale squared
EXACT_NOISE_SD = 1e-3  # same normalized observation noise as LearnedMixedHead


def weights(log_weights):
    x = np.asarray(log_weights, float)
    if x.shape != (2,) or not np.isfinite(x).all():
        raise ValueError('Two finite expert log weights required')
    return np.exp(x-logsumexp(x))


def evidence_log_score(forecast, observation):
    """Score ONE actual constraint output using its pre-evaluation predictive law."""
    if observation.kind == 'missing':
        return None
    mean, variance, scale = (float(forecast[k]) for k in ('mean', 'variance', 'scale'))
    if not np.isfinite([mean, variance, scale]).all() or variance < 0 or scale <= 0:
        raise ValueError('Invalid expert forecast')
    variance = max(variance, PREDICTIVE_VARIANCE_FLOOR*scale**2)
    if observation.kind == 'exact':
        variance += (EXACT_NOISE_SD*scale)**2
        z = (observation.value-mean)/np.sqrt(variance)
        score = -.5*((np.log(2*np.pi)+np.log(variance))+z*z)
    elif observation.kind == 'interval':
        sd = np.sqrt(variance)
        lower = -np.inf if observation.lower is None else (observation.lower-mean)/sd
        upper = np.inf if observation.upper is None else (observation.upper-mean)/sd
        score = log_interval_probability(lower, upper)
    else:
        raise ValueError('Unsupported observation kind')
    if not np.isfinite(score):
        raise FloatingPointError('Nonfinite predictive evidence score; do not silently clip')
    return float(score)


def advance(log_weights, forecasts, record, mode):
    """Pure function used both after settlement and by the evidence auditor."""
    if mode not in {'equal', 'prequential'} or set(log_weights) != set(forecasts):
        raise ValueError('Expert policy or head identity changed')
    updated = deepcopy(log_weights)
    for name, pair in forecasts.items():
        weights(log_weights[name])
        if len(pair) != 2:
            raise ValueError('Missing expert prediction')
        if record.observation.status in {'unknown', 'infrastructure_error'}:
            continue
        value = record.observation.constraint(name)
        scores = [evidence_log_score(pred, value) for pred in pair]
        if scores[0] is None:
            continue
        if mode == 'prequential':
            current = np.asarray(log_weights[name], float)+np.asarray(scores)
            current -= logsumexp(current)
            if not np.isfinite(current).all():
                raise FloatingPointError('Expert log-weight update is nonfinite')
            updated[name] = current.tolist()
    return updated


class ConstraintMixture:
    """Mixture CDFs/factors, not probabilities from a moment-matched Gaussian.

    Objective and completion heads remain the latent expert. Only expensive
    constraint heads are mixed. There is no guarantee of joint calibration.
    """
    def __init__(self, experts, log_weights):
        self.experts = experts
        self.log_weights = deepcopy(log_weights)
        self.best = experts[0].best

    def predict(self, candidates):
        packets = [expert.predict(candidates) for expert in self.experts]
        result = dict(packets[0])
        result['constraints'] = {}
        for name in self.log_weights:
            pair = [packet['constraints'][name] for packet in packets]
            mean = np.asarray([p['mean'] for p in pair], float)
            variance = np.asarray([p['variance'] for p in pair], float)
            w = weights(self.log_weights[name])
            mixed_mean = w@mean
            mixed_var = np.sum(w[:, None]*(variance+(mean-mixed_mean)**2), axis=0)
            result['constraints'][name] = {'mean': mixed_mean, 'variance': mixed_var,
                'scale': float(w@np.asarray([p['scale'] for p in pair])),
                'components': {'weights': w, 'mean': mean, 'variance': variance}}
        return result

    def selected_forecasts(self, candidate):
        packets = [expert.predict([candidate]) for expert in self.experts]
        return {name: [{key: float(p['constraints'][name][key][0]) for key in ('mean', 'variance')}
                       | {'scale': float(p['constraints'][name]['scale'])} for p in packets]
                for name in self.log_weights}

    def diagnostics(self):
        return {**self.experts[0].diagnostics(), 'constraint_experts': [e.diagnostics() for e in self.experts],
                'constraint_mixture_weights': {n: weights(v).tolist() for n,v in self.log_weights.items()},
                'expert_weight_interpretation': 'prequential expert scores, not encoder posterior probabilities'}


class EvidenceTransferBO(DiscoveryBO):
    """Cold-only constraint mixture; after first raw feasible point use frozen phase RA.

    Equal and adaptive controls fit the same two experts, even on coverage steps,
    to score all selected cold evaluations without retrospective/in-sample scoring.
    No fitting or updating of the second expert occurs after the phase transition.
    """
    def __init__(self, problem, base, seed, settings, *, mode):
        if mode not in {'equal', 'prequential'} or not settings.latent or settings.average_kernel:
            raise ValueError('Registered transfer mode requires single-kernel latent baseline')
        super().__init__(problem, base, seed, settings, cold_global=True,
                         protect=True, feedback='interval', coverage_scope='cold')
        self.transfer_mode = mode
        self.log_weights = {name: [-np.log(2.), -np.log(2.)] for name in problem.required_constraints
                            if name not in problem.analytical_constraints}
        self._expert_cache = None
        self._expert_prefix = None
        self._pending_transfer = None

    def _build_posterior(self, records):
        if any(row.usable_feasible for row in records):
            return super()._build_posterior(records)
        prefix = digest([r.to_dict() for r in records])
        if self._expert_prefix != prefix:
            experts = [RiskPosterior(self.problem, self.base, records, self.seed,
                       replace(self.settings, latent=latent)) for latent in (True, False)]
            self._expert_cache = ConstraintMixture(experts, self.log_weights)
            self._expert_prefix = prefix
        return self._expert_cache

    def _finish(self, candidate, records, metadata, prefix):
        cold = not any(r.usable_feasible for r in records)
        state = {'mode': self.transfer_mode, 'active': cold, 'experts': list(EXPERTS),
                 'log_weights_before': deepcopy(self.log_weights), 'prefix_digest': prefix,
                 'design_id': digest(candidate), 'forecasts': None,
                 'objective_and_completion': 'latent_expert_unchanged'}
        if cold:
            posterior = self._build_posterior(records)
            state['forecasts'] = posterior.selected_forecasts(candidate)
            metadata['selection_posterior_fit_skipped'] = metadata['posterior_fit_skipped']
            metadata['posterior_fit_skipped'] = False
            metadata['weight_only_fit_on_coverage'] = metadata['selection_posterior_fit_skipped']
            metadata['constraint_mixture_weights'] = {n: weights(v).tolist() for n,v in self.log_weights.items()}
        state['forecast_digest'] = digest(state)
        self._pending_transfer = deepcopy(state)
        candidate, metadata = super()._finish(candidate, records, metadata, prefix)
        metadata['evidence_transfer'] = state
        return candidate, metadata

    def accept(self, record):
        state = self._pending_transfer
        if state is None or state['design_id'] != record.observation.design_id:
            raise ValueError('No matching pre-evaluation expert forecasts')
        proposed = (advance(self.log_weights, state['forecasts'], record, self.transfer_mode)
                    if state['active'] else deepcopy(self.log_weights))
        super().accept(record)  # validates pending candidate, contract and prefix before mutation below
        self.log_weights = proposed
        self._pending_transfer = self._expert_cache = self._expert_prefix = None


def audit_transfer(problem, spec, initial, proposals, online):
    """Replay only score arithmetic/phase rules; never refit either GP or evaluate."""
    if len(proposals) != len(online):
        raise ValueError('Expert trajectory lengths differ')
    mode = spec['transfer_mode']
    state = {name: [-np.log(2.), -np.log(2.)] for name in problem.required_constraints
             if name not in problem.analytical_constraints}
    history = list(initial)
    for proposal, record in zip(proposals, online):
        meta, packet = proposal['metadata'], deepcopy(proposal['metadata'].get('evidence_transfer'))
        if not isinstance(packet, dict):
            raise ValueError('Missing expert evidence')
        sealed = packet.pop('forecast_digest')
        cold = not any(r.usable_feasible for r in history)
        if (digest(packet) != sealed or packet['mode'] != mode or packet['active'] != cold
                or packet['experts'] != list(EXPERTS) or packet['log_weights_before'] != state
                or packet['prefix_digest'] != digest([r.to_dict() for r in history])
                or packet['design_id'] != record.observation.design_id
                or packet['objective_and_completion'] != 'latent_expert_unchanged'):
            raise ValueError('Expert forecast seal, phase or causal weight sequence changed')
        if cold:
            if meta['posterior_fit_skipped'] or packet['forecasts'] is None:
                raise ValueError('Cold forecasts missing or mislabeled')
            if meta.get('constraint_mixture_weights') != {n: weights(v).tolist() for n,v in state.items()}:
                raise ValueError('Constraint mixture weights changed')
            if 'forecasts' in meta:
                index = meta['selected_index']
                for name, pair in packet['forecasts'].items():
                    components = meta['forecasts']['constraints'][name]['components']
                    if components['weights'] != weights(state[name]).tolist():
                        raise ValueError('Selected component weights differ')
                    for m, pred in enumerate(pair):
                        for key in ('mean', 'variance'):
                            if not np.isclose(pred[key], components[key][m][index], rtol=1e-10, atol=1e-12):
                                raise ValueError('Selected expert forecast differs from selection pool')
            state = advance(state, packet['forecasts'], record, mode)
        elif packet['forecasts'] is not None or meta.get('weight_only_fit_on_coverage', False):
            raise ValueError('Expert adaptation escaped cold phase')
        history.append(record)
    return {'mode': mode, 'final_log_weights': state, 'score_replay_only': True,
            'cold_scored_calls': sum(p['metadata']['evidence_transfer']['active'] for p in proposals)}
