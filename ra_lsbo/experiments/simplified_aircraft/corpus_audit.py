"""Read-only checks for persistent representation evidence; never fits a model."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from ...budget import digest
from ...features import load_base
from ...structure_corpus import (CorpusRepresentation, native_observations,
                                 feature_snapshot, grammar_diagnostics)
from ...structured_encoder import hierarchy_space
from ..core.runner import read, lines, plain
from .corpus_optimizer import (CORPUS_COUNT, UPDATE_EVERY, encoder_settings,
                               scheduled_updates, feature_report)

OFFLINE_SHA = 'd7953cdcc91891eff8fa0ff7c1fc1f7103ccdb7d24a39cbd94651b4ef2a2628f'
OFFLINE_ROOT = 'RA-LSBO-structure-corpus-output-20261031_01/full/'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def offline_receipt():
    return {'archive_sha256': OFFLINE_SHA,
            'source_commit': '08eed076f54fc6ed155f285a9bb695ff6d857bf8',
            'result_commit': '06dd926297f98e049bd753a8aedc67572971ba8f',
            'counts': {'encoder_fit_calls':160,'onehot_transforms':32,'GP_head_fit_calls':320},
            'use': 'provenance only; no historical labels or trained weights supplied to optimizers'}


def verify_offline_archive(path):
    if file_sha(path) != OFFLINE_SHA:
        raise ValueError('Not the fixed reviewed 2000-corpus archive')
    with zipfile.ZipFile(path) as archive:
        names = [x.filename for x in archive.infolist() if not x.is_dir()]
        if len(names) != 317 or len(set(names)) != len(names):
            raise ValueError('Offline archive file inventory differs')
        audit = json.loads(archive.read(OFFLINE_ROOT+'audit.json'))
        plan = json.loads(archive.read(OFFLINE_ROOT+'plan.json'))
        if (audit.get('completed') is not True or audit['counts'] != offline_receipt()['counts']
                or plan['source_commit'] != offline_receipt()['source_commit']):
            raise ValueError('Offline provenance or counts differ')
        if len(audit['files']) != 294:
            raise ValueError('Offline seal inventory differs')
        for relative, identity in audit['files'].items():
            if hashlib.sha256(archive.read(OFFLINE_ROOT+relative)).hexdigest() != identity:
                raise ValueError('Changed sealed offline evidence')
    return offline_receipt()


def finite_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def same_numbers(actual, expected):
    """Cross-file floating serialization comparisons, not a model acceptance tolerance."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(same_numbers(actual[k], v) for k,v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(same_numbers(a,b) for a,b in zip(actual, expected)))
    if type(expected) is float:
        return type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
    return actual == expected


def audit_representation(problem, base, output, initial, online, proposals, seed, method):
    """Validate fixed schedule and checkpoint-derived per-step feature identities.

    Restores numeric VAE states and runs forward diagnostics, no training/GP fit
    or re-evaluation. This cannot prove a saved optimizer decision was replayed.
    """
    output = Path(output)
    if method != 'ra_corpus':
        if (output/'representation').exists():
            raise ValueError('Unexpected persistent training files in a control')
        for prop in proposals:
            metadata = prop['metadata']
            if 'persistent_representation' in metadata:
                raise ValueError('Persistent representation in a control')
            info = metadata.get('representation_fit')
            if method == 'random':
                if info is not None:
                    raise ValueError('Invented fitted representation for random search')
                continue
            if metadata['posterior_fit_skipped']:
                if info is not None:
                    raise ValueError('Invented control fit on coverage')
                continue
            latent = method != 'ra_onehot'
            epochs = 480 if method == 'ra_old480' else 30
            if (not isinstance(info, dict) or info['encoder_fit_calls'] != int(latent)
                    or info['gradient_updates'] != (epochs if latent else 0)
                    or info['feature_identity'] != metadata['feature_identity']
                    or (info['encoder_state'] is None) != (not latent)
                    or (info['grammar'] is None) != (not latent)
                    or info['configurations'] != problem.space.configurations()
                    or not finite_seconds(info['feature_seconds'])
                    or not finite_seconds(info['gp_seconds'])):
                raise ValueError('Control fit accounting or feature identity changed')
            code = np.asarray(info['normalized_codebook'], float)
            offset, scale = np.asarray(info['offset'], float), np.asarray(info['scale'], float)
            if (code.ndim != 2 or code.shape[0] != len(problem.space.configurations())
                    or offset.shape != scale.shape or scale.shape != (2*len(problem.space.continuous)+code.shape[1],)
                    or not np.isfinite(code).all() or not np.isfinite(offset).all()
                    or not np.isfinite(scale).all() or np.any(scale <= 0)):
                raise ValueError('Invalid control feature geometry')
        return None
    expected_steps = scheduled_updates(len(proposals))
    events = lines(output/'representation/events.jsonl')
    if len(events) != 2*len(expected_steps):
        raise ValueError('Incomplete or unregistered representation fit')
    expected_dirs = {'t'+str(t).zfill(3) for t in expected_steps}
    folders = output/'representation/checkpoints'
    if {p.name for p in folders.iterdir()} != expected_dirs:
        raise ValueError('Unexpected checkpoint inventory')
    modules, _ = load_base(base)
    space = hierarchy_space(modules, [asdict(v) for v in problem.space.variables])
    expected_session = CorpusRepresentation(space, seed, count=CORPUS_COUNT, settings=encoder_settings())
    if read(output/'representation/corpus.json') != expected_session.corpus:
        raise ValueError('Corpus draws, duplicates or random identity changed')
    sessions, previous = {}, None
    for t,(started,settled) in zip(expected_steps,zip(events[::2],events[1::2])):
        history = initial+online[:t]
        relative = 'representation/checkpoints/t'+str(t).zfill(3)
        expected = {'event':'started','iteration':t,'history_count':len(history),
                    'prefix_digest':digest([r.to_dict() for r in history]),
                    'checkpoint':relative,'corpus_identity':expected_session.corpus['identity']}
        if started != expected:
            raise ValueError('Representation fit used wrong prefix/schedule')
        folder = output/relative
        if {p.name for p in folder.iterdir()} != {'state.npz','metadata.json','grammar.json'}:
            raise ValueError('Unexpected numeric checkpoint files')
        session = CorpusRepresentation.load_checkpoint(folder,space,native_observations(history,modules,space))
        if session.settings != encoder_settings() or session.seed != seed or session.corpus != expected_session.corpus:
            raise ValueError('Persistent representation settings changed')
        event = session.events[-1]
        cumulative = 480 + 320*(t//UPDATE_EVERY)
        # Both registered profiles keep 2000+history <=2048 at every fit.
        if (event['observed_rows'] != len(history) or event['structural_rows'] != CORPUS_COUNT+len(history)
                or event['random_structural_rows'] != CORPUS_COUNT
                or event['epochs'] != (30 if t == 0 else 20) or event['batches_per_epoch'] != 16
                or event['gradient_updates'] != (480 if t == 0 else 320)
                or event['cumulative_gradient_updates'] != cumulative or session.total_updates != cumulative
                or event['learning_rate'] != (.001 if t == 0 else .0004)
                or event['stage'] != ('initial_joint' if t == 0 else 'incremental')
                or event['observation_prefix_identity'] != digest(session.history)
                or len(session.events) != t//UPDATE_EVERY+1):
            raise ValueError('Training accounting or declared protocol changed')
        if previous is not None and (session.events[:-1] != previous.events or
                session.fit_seconds[:-1] != previous.fit_seconds or
                event['parameter_state_before'] != previous._sealed_state):
            raise ValueError('Incremental state chain was reset or rewritten')
        if len(event['training_trace']) != event['epochs'] or not all(finite_seconds(s) for s in session.fit_seconds):
            raise ValueError('Incomplete trace or training timing')
        diagnostic = grammar_diagnostics(session.representation)
        if not same_numbers(read(folder/'grammar.json'),diagnostic):
            raise ValueError('Saved structure diagnostic differs from checkpoint')
        expected_settled = {**expected,'event':'settled','metadata_sha256':file_sha(folder/'metadata.json'),
            'state_identity':session._sealed_state,'gradient_updates':event['gradient_updates'],
            'cumulative_gradient_updates':cumulative,'fit_seconds':session.fit_seconds[-1],
            'raw_mean_active_exact':diagnostic['raw_mean_active_exact']}
        if not same_numbers(settled,expected_settled):
            raise ValueError('Settled representation ledger differs')
        sessions[t],previous = session,session
    for i,prop in enumerate(proposals):
        session = sessions[(i//UPDATE_EVERY)*UPDATE_EVERY]
        features = feature_snapshot(problem,base,initial+online[:i],session)
        expected = feature_report(features)
        info = prop['metadata']['persistent_representation']
        for k,v in expected.items():
            if not same_numbers(info.get(k),v):
                raise ValueError('Stored GP feature view differs from sealed encoder: '+k)
        if (info['last_fit_iteration'] != (i//UPDATE_EVERY)*UPDATE_EVERY
                or info['checkpoint'] != 'representation/checkpoints/t'+str(info['last_fit_iteration']).zfill(3)
                or info['scheduled_fit'] != (i%UPDATE_EVERY == 0)
                or info['encoder_history_count'] != len(session.history)
                or info['gp_history_count'] != len(initial)+i
                or info['encoder_fit_calls'] != len(session.events)
                or info['corpus_identity'] != session.corpus['identity']
                or info['cumulative_gradient_updates'] != session.total_updates
                or info['cumulative_fit_seconds'] != sum(session.fit_seconds)
                or not finite_seconds(info['snapshot_seconds'])):
            raise ValueError('Stale or future representation provenance')
        if prop['metadata']['posterior_fit_skipped']:
            if info['gp_seconds'] is not None:
                raise ValueError('Invented GP fit on scheduled coverage')
        elif (not finite_seconds(info['gp_seconds']) or
              prop['metadata']['feature_identity'] != features.identity):
            raise ValueError('Posterior not bound to current feature snapshot')
    return {'fits':len(expected_steps),'gradient_updates':previous.total_updates,
            'fit_seconds':sum(previous.fit_seconds),'checkpoint_restore':True,
            'training_or_optimizer_replay':False}
