"""Write-once, zero-truth P3 first-divergence replay; no VAE fitting."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import h2_p4_decision_sensitivity as p4
from ra_lsbo.budget import digest
from ra_lsbo.features import load_base
from ra_lsbo.partial_bo.models import PrefixFeatures, EvidenceGP
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.structure_corpus import CorpusRepresentation, feature_snapshot, native_observations
from ra_lsbo.structured_encoder import hierarchy_space
from ra_lsbo.experiments.core.problems import AircraftCase
from ra_lsbo.experiments.discovery_stages.optimizer import DiscoveryBO
from ra_lsbo.experiments.simplified_aircraft.h2_optimizer import H2RiskPosterior
from ra_lsbo.experiments.simplified_aircraft.h2_p3_pilot import BASE, pool_for, problem_for, model_settings

ROOT = Path(__file__).resolve().parents[1]
ABS_TOL, REL_TOL = 1e-10, 1e-10


def forbidden(*args, **kwargs):
    raise RuntimeError('Forbidden physical evaluation or VAE training attempted')


def differences(expected, actual, path='', limit=30):
    """Strict structures, exact IDs, fixed tolerance only for floating values."""
    out = []
    def walk(a, b, key):
        if len(out) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                out.append({'field': key, 'expected_keys': sorted(a), 'actual_keys': sorted(b)})
            for k in sorted(set(a) & set(b)):
                walk(a[k], b[k], key + '.' + k)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append({'field': key, 'expected_length': len(a), 'actual_length': len(b)})
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f'{key}[{i}]')
        elif isinstance(a, (float, int)) and not isinstance(a, bool) and isinstance(b, (float, int)) and not isinstance(b, bool):
            if not math.isfinite(float(a)) or not math.isfinite(float(b)) or not math.isclose(a, b, abs_tol=ABS_TOL, rel_tol=REL_TOL):
                out.append({'field': key, 'expected': a, 'actual': b})
        elif a != b or type(a) is not type(b):
            out.append({'field': key, 'expected': a, 'actual': b})
    walk(expected, actual, path)
    return out


def source_check(z, plan):
    checked = {}
    for name, expected in plan['sources'].items():
        current = ROOT / name
        raw = current.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError('Frozen P3 source differs: ' + name)
        checked[name] = actual
    return checked


def pair_inputs(z, plan, case, seed):
    dynamic = p4.trial(z, case, seed, 'h2_dynamic')
    fixed = p4.trial(z, case, seed, 'h2_fixed50')
    iteration = next(i for i, (a, b) in enumerate(zip(dynamic['proposals'], fixed['proposals']))
                     if a['candidate'] != b['candidate'])
    a, b = dynamic['proposals'][iteration], fixed['proposals'][iteration]
    ha = dynamic['manifest']['initial'] + dynamic['observations'][:iteration]
    hb = fixed['manifest']['initial'] + fixed['observations'][:iteration]
    if ha != hb or digest(ha) != a['prefix_digest'] or a['prefix_digest'] != b['prefix_digest']:
        raise ValueError('Matched history mismatch')
    if a['pool'] != b['pool']:
        raise ValueError('Global candidate pool mismatch')
    ma, mb = a['metadata'], b['metadata']
    for name in ('feature_identity', 'risk_feature_identity', 'branch_radii', 'branch_stalls'):
        if ma[name] != mb[name]:
            raise ValueError('Matched state differs: ' + name)
    checkpoint = 8 if iteration >= 8 else 0
    files = []
    for arm in p4.ARMS:
        prefix = f'optimization/{case}/{seed}_{arm}/representation/iteration_{checkpoint:03d}/'
        files.append({name: z.read(p4.ROOT + prefix + name) for name in ('metadata.json', 'state.npz')})
    left, right = (json.loads(f['metadata.json']) for f in files)
    left.pop('fit_seconds', None)
    right.pop('fit_seconds', None)
    if left != right or files[0]['state.npz'] != files[1]['state.npz']:
        raise ValueError('Matched checkpoint state or training history differs')
    return {'iteration': iteration, 'history': ha, 'arms': [a, b], 'checkpoint': checkpoint, 'files': files[0]}


class ReplayBO(DiscoveryBO):
    def _build_posterior(self, records):
        if digest([r.to_dict() for r in records]) != self.replay_prefix:
            raise ValueError('Posterior training prefix changed')
        return self.replay_posterior


def replay(problem, settings, seed, iteration, history, pool, saved, posterior, weight):
    engine = ReplayBO(problem, BASE, seed, settings, cold_global=True, protect=True,
                      feedback='interval', coverage_scope='cold')
    engine.next_iteration = iteration
    engine.radii = copy.deepcopy(saved['branch_radii'])
    engine.stalls = copy.deepcopy(saved['branch_stalls'])
    engine.replay_prefix = digest([r.to_dict() for r in history])
    engine.replay_posterior = posterior
    posterior.h2_weight = weight
    posterior.h2_snapshot = copy.deepcopy(saved['controller_snapshot_before_prediction'])
    return engine.propose(history, pool, iteration)


def verify_replay(proposal, candidate, metadata):
    saved = proposal['metadata']
    checks = {}
    checks['selected_id'] = [] if digest(proposal['candidate']) == digest(candidate) else [
        {'expected': digest(proposal['candidate']), 'actual': digest(candidate)}]
    for name in ('pool',):
        x = [digest(c) for c in saved[name]]
        y = [digest(c) for c in metadata[name]]
        checks['ordered_' + name + '_ids'] = differences(x, y, name)
    for name in ('local_reports', 'branch_radii', 'branch_stalls', 'mode', 'fallback',
                 'initial_pool_mode', 'initial_pool_fallback', 'coverage_reason',
                 'selected_index', 'model_selected_index', 'scores', 'forecasts', 'heads'):
        checks[name] = differences(saved.get(name), metadata.get(name), name)
    return {'passed': not any(checks.values()), 'differences': {k: v for k, v in checks.items() if v}}


def execute_pair(z, plan, case, seed):
    p = pair_inputs(z, plan, case, seed)
    a, b = p['arms']
    m = a['metadata']
    result = {'case': case, 'seed': seed, 'iteration': p['iteration'], 'prefix_digest': a['prefix_digest'],
              'dynamic_weight': m['SOC_view_weight_onehot'], 'checkpoint_iteration': p['checkpoint'],
              'original_candidate_ids': [digest(x['candidate']) for x in p['arms']],
              'gp_candidate_fits': 0, 'vae_fits': 0, 'aircraft_calls': 0, 'rcaide_calls': 0,
              'category': 'not_identified', 'native_replays': {}, 'intervention': None}
    problem = problem_for(case)
    history = [DesignRecord.from_dict(row) for row in p['history']]
    pool, _ = pool_for(problem, plan, case, seed, p['iteration'], history)
    if pool != a['pool']:
        raise ValueError('Regenerated global pool differs')
    modules, _ = load_base(BASE)
    space = hierarchy_space(modules, [asdict(v) for v in problem.space.variables])
    with tempfile.TemporaryDirectory(prefix='h2-p5-checkpoint-') as directory:
        for name, data in p['files'].items():
            (Path(directory) / name).write_bytes(data)
        encoder_history = history[:12 + p['checkpoint']]
        session = CorpusRepresentation.load_checkpoint(directory, space, native_observations(encoder_history, modules, space))
        latent = feature_snapshot(problem, BASE, history, session)
        risk = PrefixFeatures(problem, BASE, history, seed, False, 30)
    if latent.identity != m['feature_identity'] or risk.identity != m['risk_feature_identity']:
        raise ValueError('Restored feature identity differs')
    settings = model_settings('h2_dynamic', plan['profile'])
    original_init = EvidenceGP.__init__
    def counted(obj, *args, **kwargs):
        result['gp_candidate_fits'] += 1
        return original_init(obj, *args, **kwargs)
    start = time.perf_counter()
    try:
        with patch.object(EvidenceGP, '__init__', counted):
            posterior = H2RiskPosterior(problem, history, seed, settings, latent, risk,
                                        m['SOC_view_weight_onehot'], m['controller_snapshot_before_prediction'])
    finally:
        result['gp_fit_seconds'] = time.perf_counter() - start
    result['head_discrepancies'] = differences(m['heads'], posterior.diagnostics()['heads'], 'heads')
    if result['head_discrepancies']:
        result['reason'] = 'posterior_head_mismatch'
        return result
    for arm, proposal in zip(p4.ARMS, p['arms']):
        attempts = []
        for _ in range(2):
            candidate, metadata = replay(problem, settings, seed, p['iteration'], history, pool,
                                         proposal['metadata'], posterior, proposal['metadata']['SOC_view_weight_onehot'])
            attempts.append(verify_replay(proposal, candidate, metadata))
        result['native_replays'][arm] = attempts
    if not all(attempt['passed'] for attempts in result['native_replays'].values() for attempt in attempts):
        result['reason'] = 'native_replay_mismatch'
        return result
    interventions = []
    for weight in (m['SOC_view_weight_onehot'], .5):
        candidate, metadata = replay(problem, settings, seed, p['iteration'], history, pool, m, posterior, weight)
        scores = sorted((x for x in metadata['scores'] if x is not None), reverse=True)
        interventions.append({'weight': weight, 'candidate_id': digest(candidate),
            'selected_score': metadata['scores'][metadata['selected_index']],
            'score_margin': scores[0]-scores[1] if len(scores)>1 else None,
            'mode': metadata['mode'], 'coverage_reason': metadata['coverage_reason'],
            'selected_from_local_augmentation': metadata['selected_from_local_augmentation'],
            'local_reports': metadata['local_reports'], 'pool_ids': [digest(c) for c in metadata['pool']]})
    if any(x['score_margin'] is not None and abs(x['score_margin']) <= ABS_TOL for x in interventions):
        result['reason'] = 'numerical_near_tie'
        result['intervention'] = interventions
        return result
    result['category'] = 'identified'
    result['intervention'] = interventions
    result['weight_changes_choice'] = interventions[0]['candidate_id'] != interventions[1]['candidate_id']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    z, plan = p4.validate_archive(args.archive)
    sources = source_check(z, plan)
    (out/'source_hashes.json').write_text(json.dumps({'p3_commit': p4.SOURCE_SHA, 'archive_sha256': p4.ARCHIVE_SHA,
        'source_sha256': sources, 'absolute_tolerance': ABS_TOL, 'relative_tolerance': REL_TOL}, indent=2), encoding='utf8')
    results = []
    with patch.object(AircraftCase, 'evaluate', forbidden), patch.object(CorpusRepresentation, 'fit', forbidden):
        for case in p4.CASES:
            for seed in range(12071, 12079):
                print(f'{case} {seed}', flush=True)
                try:
                    row = execute_pair(z, plan, case, seed)
                except Exception as error:
                    if 'Forbidden physical evaluation or VAE training' in str(error):
                        raise
                    row = {'case': case, 'seed': seed, 'category': 'not_identified',
                           'reason': 'reconstruction_exception', 'error': repr(error), 'traceback': traceback.format_exc()}
                results.append(row)
                with (out/'pair_results.jsonl').open('a', encoding='utf8') as f:
                    f.write(json.dumps(row, allow_nan=False)+'\n')
    identified = [r for r in results if r['category'] == 'identified']
    payload = {'archive_sha256': p4.ARCHIVE_SHA, 'p3_source_commit': p4.SOURCE_SHA,
               'pairs': results, 'identified_count': len(identified),
               'confirmed_choice_flips': sum(r['weight_changes_choice'] for r in identified),
               'zero_new_aircraft_calls': True, 'zero_vae_fits': True, 'zero_rcaide_calls': True}
    (out/'full_path_interventions.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({'identified': len(identified), 'flips': payload['confirmed_choice_flips']}), flush=True)


if __name__ == '__main__':
    main()
