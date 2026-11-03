"""Offline P3 two-stage weight interventions, without physical evaluations."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import h2_p5_full_path as p5
from tools import h2_p4_decision_sensitivity as p4
from ra_lsbo.budget import digest
from ra_lsbo.experiments.discovery_stages import optimizer as discovery
from ra_lsbo.experiments.core.problems import AircraftCase
from ra_lsbo.experiments.simplified_aircraft.h2_optimizer import H2RiskPosterior
from ra_lsbo.features import load_base
from ra_lsbo.partial_bo.models import EvidenceGP, PrefixFeatures
from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.structured_encoder import hierarchy_space
from ra_lsbo.structure_corpus import CorpusRepresentation, feature_snapshot, native_observations

GRID = (1e-12, 1e-11, 1e-10, 1e-9, 1e-8)
PRIMARY = 1e-10


def two_stage_replay(problem, settings, seed, iteration, history, pool, saved, posterior, wg, ws):
    """Switch only after the original post-augmentation near-filter returns."""
    original = discovery.filter_near
    calls = []
    def filtered(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 2:
            posterior.h2_weight = ws
        return value
    with patch.object(discovery, 'filter_near', filtered):
        candidate, meta = p5.replay(problem, settings, seed, iteration, history, pool, saved, posterior, wg)
    if len(calls) != 2:
        raise ValueError('Expected original global and final near filters; coverage or code path differs')
    return candidate, meta


def cell_summary(candidate, meta, wg, ws):
    scores = [(float(x), i) for i, x in enumerate(meta['scores']) if x is not None]
    scores.sort(key=lambda x: (-x[0], x[1]))
    margin = scores[0][0] - scores[1][0] if len(scores) > 1 else None
    top = [{'candidate_id': digest(meta['pool'][i]), 'score': score} for score, i in scores[:2]]
    scale = max(1., abs(scores[0][0]), abs(scores[1][0])) if len(scores) > 1 else None
    reports = meta['local_reports']
    return {'wg': wg, 'ws': ws, 'candidate_id': digest(candidate),
            'pool_ids': [digest(x) for x in meta['pool']], 'selected_index': meta['selected_index'],
            'model_selected_index': meta['model_selected_index'], 'mode': meta['mode'],
            'scores': meta['scores'], 'eligible': meta['eligible'],
            'fallback': meta['fallback'], 'coverage_reason': meta['coverage_reason'],
            'selected_from_local_augmentation': meta['selected_from_local_augmentation'],
            'local_reports': reports, 'solver_failures': sum(not r['success'] for r in reports),
            'top_candidates': top, 'score_gap': margin, 'score_scale': scale,
            'normalized_gap': None if margin is None else margin/scale,
            'exact_tie': margin == 0 if margin is not None else None,
            'selected_soc_probability': meta['per_constraint_probability']['soc'][meta['selected_index']]}


def classify(cells, native_ok, stable):
    if not native_ok:
        return 'not_identified', 'native_replay_mismatch'
    if not stable:
        return 'not_identified', 'repeat_instability'
    if any(c['coverage_reason'] is not None for c in cells.values()):
        return 'not_identified', 'coverage_override'
    if any(c['score_gap'] is None or c['score_gap'] <= PRIMARY for c in cells.values()):
        return 'not_identified', 'numerical_near_tie_or_single_candidate'
    return 'identified', None


def prepare_pair(z, plan, case, seed):
    p = p5.pair_inputs(z, plan, case, seed)
    meta = p['arms'][0]['metadata']
    problem = p5.problem_for(case)
    history = [DesignRecord.from_dict(r) for r in p['history']]
    pool, _ = p5.pool_for(problem, plan, case, seed, p['iteration'], history)
    if pool != p['arms'][0]['pool']:
        raise ValueError('Global pool identity differs')
    modules, _ = load_base(p5.BASE)
    space = hierarchy_space(modules, [asdict(v) for v in problem.space.variables])
    with tempfile.TemporaryDirectory(prefix='h2-p6-checkpoint-') as folder:
        for name, data in p['files'].items():
            (Path(folder)/name).write_bytes(data)
        session = CorpusRepresentation.load_checkpoint(folder, space,
            native_observations(history[:12+p['checkpoint']], modules, space))
        latent = feature_snapshot(problem, p5.BASE, history, session)
        risk = PrefixFeatures(problem, p5.BASE, history, seed, False, 30)
    if latent.identity != meta['feature_identity'] or risk.identity != meta['risk_feature_identity']:
        raise ValueError('Frozen representation identity differs')
    return p, problem, history, pool, latent, risk


def execute_pair(plan, case, seed, prepared):
    p, problem, history, pool, latent, risk = prepared
    a, b = p['arms']
    ma = a['metadata']
    settings = p5.model_settings('h2_dynamic', plan['profile'])
    result = {'case': case, 'seed': seed, 'iteration': p['iteration'], 'prefix_digest': a['prefix_digest'],
              'dynamic_weight': ma['SOC_view_weight_onehot'], 'gp_candidate_fits': 0,
              'vae_fits': 0, 'aircraft_calls': 0, 'rcaide_calls': 0, 'native_replays': {}, 'cells': {}}
    original_init = EvidenceGP.__init__
    def counted(obj, *args, **kwargs):
        result['gp_candidate_fits'] += 1
        return original_init(obj, *args, **kwargs)
    start = time.perf_counter()
    with patch.object(EvidenceGP, '__init__', counted):
        posterior = H2RiskPosterior(problem, history, seed, settings, latent, risk,
                                    ma['SOC_view_weight_onehot'], ma['controller_snapshot_before_prediction'])
    result['gp_fit_seconds'] = time.perf_counter()-start
    result['head_discrepancies'] = p5.differences(ma['heads'], posterior.diagnostics()['heads'], 'heads')
    if result['head_discrepancies']:
        result.update(category='not_identified', reason='posterior_head_mismatch')
        return result
    wd = ma['SOC_view_weight_onehot']
    controls = [('DD', a, wd), ('FF', b, .5)]
    stored = {}
    for name, proposal, weight in controls:
        attempts, outputs = [], []
        for _ in range(2):
            candidate, meta = two_stage_replay(problem, settings, seed, p['iteration'], history, pool,
                                               proposal['metadata'], posterior, weight, weight)
            attempts.append(p5.verify_replay(proposal, candidate, meta))
            outputs.append(cell_summary(candidate, meta, weight, weight))
        result['native_replays'][name] = attempts
        stored[name] = outputs
    native_ok = all(a['passed'] for attempts in result['native_replays'].values() for a in attempts)
    if not native_ok:
        result.update(category='not_identified', reason='native_replay_mismatch')
        return result
    for name, wg, ws in [('DF', wd, .5), ('FD', .5, wd)]:
        outputs = []
        for _ in range(2):
            candidate, meta = two_stage_replay(problem, settings, seed, p['iteration'], history, pool,
                                               ma, posterior, wg, ws)
            outputs.append(cell_summary(candidate, meta, wg, ws))
        stored[name] = outputs
    stable = True
    for name, outputs in stored.items():
        diffs = p5.differences(outputs[0], outputs[1], name)
        result['cells'][name] = {**outputs[0], 'repeat_identity': not diffs, 'repeat_differences': diffs}
        stable &= not diffs
    cells = result['cells']
    result['category'], result['reason'] = classify(cells, native_ok, stable)
    result['contrasts'] = {
        'generation_at_dynamic': cells['DD']['candidate_id'] != cells['FD']['candidate_id'],
        'generation_at_fixed': cells['DF']['candidate_id'] != cells['FF']['candidate_id'],
        'ranking_on_dynamic_pool': cells['DD']['candidate_id'] != cells['DF']['candidate_id'],
        'ranking_on_fixed_pool': cells['FD']['candidate_id'] != cells['FF']['candidate_id']}
    aa, bb = cells['DD']['pool_ids'], cells['FF']['pool_ids']
    result['generated_pool_difference'] = {
        'first_ordered_difference': next((i for i,(x,y) in enumerate(zip(aa,bb)) if x!=y),
                                         min(len(aa),len(bb)) if len(aa)!=len(bb) else None),
        'dynamic_only': sorted(set(aa)-set(bb)), 'fixed_only': sorted(set(bb)-set(aa))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    z, plan = p4.validate_archive(args.archive)
    sources = p5.source_check(z, plan)
    previous = json.loads((p5.ROOT/'results/h2-p5-full-path-intervention-20261103-revised/full_path_interventions.json').read_text())
    if previous['archive_sha256'] != p4.ARCHIVE_SHA or len(previous['pairs']) != 16:
        raise ValueError('P5 evidence mismatch')
    p5pairs = {(r['case'],r['seed']):r for r in previous['pairs']}
    rows = []
    with patch.object(AircraftCase, 'evaluate', p5.forbidden), \
         patch.object(SimplifiedElectricAircraft, 'evaluate', p5.forbidden), \
         patch.object(CorpusRepresentation, 'fit', p5.forbidden), \
         patch.dict(sys.modules, {'RCAIDE': None}):
        prepared = {}
        for case in p4.CASES:
            for seed in range(12071,12079):
                item = prepare_pair(z, plan, case, seed)
                old = p5pairs[(case,seed)]
                if item[0]['arms'][0]['prefix_digest'] != old['prefix_digest'] or item[0]['iteration'] != old['iteration']:
                    raise ValueError('P5 frozen prefix mismatch')
                prepared[(case,seed)] = item
        for case in p4.CASES:
            for seed in range(12071,12079):
                print(case, seed, flush=True)
                row = execute_pair(plan, case, seed, prepared[(case,seed)])
                rows.append(row)
                with (out/'pair_results.jsonl').open('a', encoding='utf8') as f:
                    f.write(json.dumps(row, allow_nan=False)+'\n')
    result = {'archive_sha256': p4.ARCHIVE_SHA, 'p3_source_commit': p4.SOURCE_SHA,
              'source_sha256': sources, 'pairs': rows, 'aircraft_calls': 0, 'rcaide_calls': 0, 'vae_fits': 0}
    (out/'two_stage_cells.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('completed', len(rows), flush=True)


if __name__ == '__main__':
    main()
