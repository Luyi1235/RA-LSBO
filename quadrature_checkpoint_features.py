"""Restore an exact archived feature identity without retraining or hash aliases.

Some CPU backends round VAE forward passes differently. The original checkpoint
already stores the complete finite configuration codebook in diagnostics.json.
Use it only if its original byte hash, tensor-state binding, native forward
agreement (UNCHANGED replay tolerance), and exact historical feature ID all pass.
Never set feature.identity to an expected string to make a comparison succeed.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.models import PrefixFeatures
from ra_lsbo.structure_corpus import feature_snapshot, observation_rows, native_observations, CORPUS_SCHEMA
from quadrature_audit_protocol import require, close, sha, RTOL, ATOL


class ArchivedCodebookFeatures(PrefixFeatures):
    def _raw(self, candidates):
        values = [self.candidate(candidate) for candidate in candidates]
        units = np.asarray([self.space.unit(candidate) for candidate in values])
        masks = np.asarray([self.space.mask(candidate.discrete) for candidate in values])
        latents = np.asarray([self._archived_codebook[digest(c.discrete)] for c in values], dtype=float)
        return np.hstack([units*masks, masks, latents])


def restore_features(problem, base, seed, history, session, checkpoint, metadata):
    native = feature_snapshot(problem, base, history, session)
    risk = PrefixFeatures(problem, base, history, seed, False, 30)
    require(risk.identity == metadata['risk_feature_identity'],
            'Explicit-risk feature identity differs: native='+risk.identity+' recorded='+metadata['risk_feature_identity'])
    proof = {'native_objective_identity': native.identity,
             'recorded_objective_identity': metadata['objective_feature_identity'],
             'risk_identity': risk.identity, 'original_rtol': RTOL, 'original_atol': ATOL,
             'encoder_state': native.state_hash, 'vae_training': False,
             'recorded_identity_assigned_without_reconstruction': False}
    if native.identity == metadata['objective_feature_identity']:
        proof.update(mode='native_forward_exact_identity', restored_objective_identity=native.identity)
        return native, risk, proof

    checkpoint = Path(checkpoint)
    saved_path = checkpoint/'diagnostics.json'
    recorded_sha = metadata['corpus_training']['latest']['files']['diagnostics.json']
    require(sha(saved_path) == recorded_sha, 'Archived codebook is not the original sealed diagnostics file')
    saved = json.loads(saved_path.read_text(encoding='utf-8'))
    configs = session.space.configurations()
    require(saved['configurations'] == configs, 'Archived codebook configuration roster differs')
    mu = np.asarray(saved['mu'], float)
    require(mu.shape == (len(configs), 4) and np.isfinite(mu).all(), 'Invalid archived latent codebook')
    require(len({digest(c) for c in configs}) == len(configs), 'Duplicate archived configurations')
    indices = torch.tensor([session.space.indices(c) for c in configs], dtype=torch.long)
    with torch.no_grad():
        current_mu, _ = session.representation.model.encode(indices)
    current_mu = current_mu.detach().cpu().numpy()
    close(current_mu.tolist(), mu.tolist(), 'sealed codebook/native VAE forward')

    restored = ArchivedCodebookFeatures.__new__(ArchivedCodebookFeatures)
    restored.__dict__.update(native.__dict__)
    restored._archived_codebook = {digest(c): row.tolist() for c, row in zip(configs, mu)}
    raw = restored._raw([r.candidate for r in history])
    restored.offset, restored.scale = np.zeros(raw.shape[1]), np.ones(raw.shape[1])
    restored.offset[-4:] = raw[:, -4:].min(0)-.5
    restored.scale[-4:] = raw[:, -4:].max(0)+.5-restored.offset[-4:]
    observations = native_observations(history, restored.modules, session.space)
    rows = observation_rows(observations)
    # Same complete identity payload as frozen structure_corpus.feature_snapshot.
    restored.identity = digest({'schema': CORPUS_SCHEMA, 'prefix': [r.to_dict() for r in history],
        'corpus': session.corpus['identity'], 'settings': asdict(session.settings),
        'encoder': restored.state_hash, 'encoder_prefix': digest(session.history),
        'offset': restored.offset.tolist(), 'scale': restored.scale.tolist()})
    proof.update(mode='sealed_archived_codebook', diagnostics_file_sha256=recorded_sha,
                 archived_codebook_digest=digest({'configurations': configs, 'mu': saved['mu']}),
                 max_abs_native_forward_difference=float(np.max(np.abs(current_mu-mu))),
                 native_offset=native.offset.tolist(), native_scale=native.scale.tolist(),
                 restored_offset=restored.offset.tolist(), restored_scale=restored.scale.tolist(),
                 restored_objective_identity=restored.identity)
    if restored.identity != metadata['objective_feature_identity']:
        error = ValueError('Archived codebook also fails EXACT historical feature identity: '+json.dumps(proof, sort_keys=True))
        error.feature_identity_proof = proof
        raise error
    close(restored.transform([r.candidate for r in history]).tolist(),
          native.transform([r.candidate for r in history]).tolist(), 'archived/native prefix features')
    require(rows[:len(session.history)] == session.history, 'Archived feature prefix does not extend checkpoint')
    return restored, risk, proof
