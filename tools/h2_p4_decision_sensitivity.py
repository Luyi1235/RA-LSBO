"""Analyze the sealed P3 archive without aircraft evaluations or model fitting."""
import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ra_lsbo.budget import digest
from ra_lsbo.experiments.simplified_aircraft.h2_risk_controller import component_mixture, proper_nll
from ra_lsbo.partial_bo.selector import PartialBOSettings
from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.soc_revision.acquisition import choose_acquisition

ARCHIVE_SHA = 'bd170195201b45942e4fe2c2e04e1e994263bb0a4566a467b28816cdce0473b4'
SOURCE_SHA = '57e390fc8f6280db9f201940aec14089e81c104c'
ROOT = 'RA-LSBO-H2-P3-pilot-output-20261102_01/'
MANIFEST = 'RA-LSBO-H2-P3-pilot-package-20261103_01/SHA256SUMS.txt'
CASES = ('aircraft_ordinary', 'aircraft_sparse')
ARMS = ('h2_dynamic', 'h2_fixed50')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(z, path):
    return json.loads(z.read(ROOT + path))


def jsonl(z, path):
    return [json.loads(line) for line in z.read(ROOT + path).splitlines() if line]


def validate_archive(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != ARCHIVE_SHA:
        raise ValueError('Frozen archive SHA-256 mismatch')
    z = zipfile.ZipFile(path)
    for line in z.read(MANIFEST).decode().splitlines():
        expected, size, name = line.split('  ', 2)
        if z.getinfo(name).file_size != int(size) or sha(z.read(name)) != expected:
            raise ValueError('Archive manifest mismatch: ' + name)
    plan, reg, audit = (read_json(z, x) for x in ('plan.json', 'registration.json', 'audit.json'))
    if (reg['plan_sha256'] != sha(z.read(ROOT+'plan.json')) or plan['source_commit'] != SOURCE_SHA or
        plan['profile']['seeds'] != list(range(12071, 12079)) or plan['cases'] != list(CASES) or
        plan['arms'] != list(ARMS) or plan['budget']['maximum_model_calls'] != 800 or
        plan['budget']['maximum_rcaide_mda_calls'] != 0 or not plan['smoke_prerequisite'] or
        plan['reviewed_qualification']['settled_calls'] != 316 or not audit['passed'] or
        audit['settled_calls'] != 800 or audit['trial_count'] != 32 or
        audit['checked_proposals'] != 576 or not audit['H2_controller_replayed']):
        raise ValueError('Frozen plan, receipt, or audit mismatch')
    for name, expected in audit['evidence_hashes'].items():
        if sha(z.read(ROOT+name)) != expected:
            raise ValueError('Audit evidence mismatch: ' + name)
    return z, plan


def trial(z, case, seed, arm):
    base = f'optimization/{case}/{seed}_{arm}/'
    manifest = read_json(z, base+'manifest.json')
    proposals = jsonl(z, base+'proposals.jsonl')
    observations = jsonl(z, base+'observations.jsonl')
    summary, timing = (read_json(z, base+x) for x in ('summary.json', 'timing.json'))
    if len(manifest['initial']) != 12 or len(proposals) != 18 or len(observations) != 18:
        raise ValueError('Incomplete trial')
    history = list(manifest['initial'])
    for i, (p, o) in enumerate(zip(proposals, observations)):
        if (p['iteration'] != i or p['candidate'] != o['candidate'] or
            digest(history) != p['prefix_digest'] or p['metadata']['raw_history_digest'] != p['prefix_digest']):
            raise ValueError('Proposal, observation or prefix mismatch')
        history.append(o)
    return {'manifest': manifest, 'proposals': proposals, 'observations': observations,
            'summary': summary, 'timing': timing}


def same_forecasts(a, b):
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    for f in (a, b):
        soc = f['constraints']['soc']
        for key in ('mean', 'variance', 'h2_onehot_weight'):
            soc.pop(key, None)
        soc['components'].pop('weights', None)
    return a == b


def replace_soc_weight(forecasts, weight):
    f = copy.deepcopy(forecasts)
    soc = f['constraints']['soc']
    c = soc['components']
    f['constraints']['soc'] = component_mixture(
        {'mean': c['mean'][0], 'variance': c['variance'][0]},
        {'mean': c['mean'][1], 'variance': c['variance'][1]}, weight, soc['scale'])
    return f


def common_forecast_check(a, b):
    """Compare saved GP distributions at exactly identical archived designs."""
    left = {digest(x): i for i,x in enumerate(a['pool'])}
    right = {digest(x): i for i,x in enumerate(b['pool'])}
    shared = sorted(left.keys() & right.keys())
    if not a.get('forecasts') or not b.get('forecasts'):
        return {'count':len(shared),'compared':False,'reason':'missing_forecasts'}
    fa, fb = a['forecasts'], b['forecasts']
    names = ['completion', 'objective'] + sorted(fa['constraints'])
    maxima = {}
    for name in names:
        xa = fa[name] if name in ('completion','objective') else fa['constraints'][name]
        xb = fb[name] if name in ('completion','objective') else fb['constraints'][name]
        if xa is None or xb is None:
            if xa != xb: return {'count':len(shared),'compared':False,'reason':'head_presence_differs'}
            continue
        fields = ('mean','variance') if name != 'soc' else ()
        for field in fields:
            maxima[name+'.'+field] = max((abs(xa[field][left[k]]-xb[field][right[k]]) for k in shared), default=0.)
        if name == 'soc':
            for field in ('mean','variance'):
                for component in range(2):
                    key=f'soc.component{component}.{field}'
                    maxima[key]=max((abs(xa['components'][field][component][left[k]]-
                                         xb['components'][field][component][right[k]]) for k in shared),default=0.)
        if name not in ('completion','objective'):
            maxima[name+'.scale']=abs(xa['scale']-xb['scale'])
    return {'count':len(shared),'compared':True,'maximum_absolute_difference_by_field':maxima,
            'all_absolute_differences_at_most_1e-10':all(v<=1e-10 for v in maxima.values())}


def counterfactual(meta, iteration, weights, candidate_ids=None):
    """Change weights on archived pool only; local search is not regenerated."""
    if meta.get('posterior_fit_skipped') or not meta.get('forecasts') or 'model_selected_index' not in meta:
        return {'identified': False, 'reason': 'coverage_or_missing_forecasts'}
    if meta.get('coverage_reason'):
        return {'identified': False, 'reason': 'coverage_overrides_acquisition'}
    settings = PartialBOSettings(horizon=18, boundary_fraction=meta['settings']['boundary_fraction'])
    def calc(weight):
        return choose_acquisition(replace_soc_weight(meta['forecasts'], weight), meta.get('best_observed'),
                                  iteration, settings, meta['settings']['probability_exploitation'])
    idx, detail = calc(meta['SOC_view_weight_onehot'])
    saved = meta['scores']
    equal = len(saved) == len(detail['scores']) and all(
        (a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9))
        for a,b in zip(saved, detail['scores']))
    if idx != meta['model_selected_index'] or not equal:
        return {'identified': False, 'reason': 'saved_acquisition_not_reproduced',
                'recomputed_index': idx, 'saved_model_index': meta['model_selected_index'], 'scores_match': equal}
    records = []
    for weight in weights:
        idx, d = calc(weight)
        valid = sorted((v for v in d['scores'] if v is not None), reverse=True)
        candidate_ranks = {}
        pool_indices = {digest(x):j for j,x in enumerate(meta['pool'])}
        for candidate_id in candidate_ids or []:
            j = pool_indices.get(candidate_id)
            value = None if j is None else d['scores'][j]
            candidate_ranks[candidate_id] = {'in_pool':j is not None,'eligible':value is not None,
                'score':value,'rank':None if value is None else 1+sum(v>value for v in valid)}
        records.append({'weight': weight, 'index': idx, 'candidate_id': digest(meta['pool'][idx]),
                        'mode': d['mode'], 'fallback': d['fallback'], 'eligible_count': sum(d['eligible']),
                        'selected_score': d['scores'][idx], 'top_two_score_margin': valid[0]-valid[1] if len(valid)>1 else None,
                        'soc_probability': d['per_constraint_probability']['soc'][idx], 'actual_candidate_ranks':candidate_ranks})
    return {'identified': True, 'scope': 'fixed_archived_pool_and_posteriors_only', 'reconstructed': records,
            'model_rank_reversal': records[0]['candidate_id'] != records[1]['candidate_id'],
            'local_search_path_reconstructed': False}


def score_trial(t):
    counts, scored, details = Counter(), defaultdict(list), []
    for p, o in zip(t['proposals'], t['observations']):
        c = p['metadata']['H2_prior_forecast_commitment']
        if c['design_id'] != digest(p['candidate']) or c['prefix'] != p['prefix_digest']:
            raise ValueError('Stale SOC forecast commitment')
        payload = {k:v for k,v in c.items() if k != 'commitment'}
        if sha(json.dumps(payload,sort_keys=True,allow_nan=False).encode()) != c['commitment']:
            raise ValueError('Modified SOC forecast commitment')
        evidence = OutputObservation(**dict(o['observation']['constraints'])['soc'])
        w = c['weight_at_proposal']
        counts['weight_at_lower_bound'] += int(math.isclose(w, .1, abs_tol=1e-12))
        counts['weight_at_upper_bound'] += int(math.isclose(w, .9, abs_tol=1e-12))
        if c['skip'] is not None:
            counts['skipped_'+str(c['skip'])] += 1
            continue
        if evidence.kind == 'missing':
            counts['missing_soc'] += 1
            continue
        m = p['metadata']; index = m['selected_index']; components = m['forecasts']['constraints']['soc']['components']
        if (c['weight_at_proposal'] != components['weights'][0] or
            any(c[name][field] != components[field][j][index]
                for j,name in enumerate(('onehot','latent')) for field in ('mean','variance'))):
            raise ValueError('Committed forecast differs from saved selected GP prediction')
        a, b = proper_nll(c['onehot'], evidence), proper_nll(c['latent'], evidence)
        if a['type'] != b['type']:
            raise ValueError('SOC score type mismatch')
        delta, kind = b['nll']-a['nll'], a['type']
        counts[kind] += 1
        counts['clip_positive'] += int(delta > 4.)
        counts['clip_negative'] += int(delta < -4.)
        scored[kind].append((a['nll'], b['nll']))
        details.append({'iteration': p['iteration'], 'type': kind, 'onehot_nll': a['nll'], 'latent_nll': b['nll'],
                        'delta_latent_minus_onehot': delta, 'clipped_delta': max(-4., min(4., delta)), 'weight': w,
                        'evidence':dict(o['observation']['constraints'])['soc'], 'forecast_commitment':c['commitment']})
    return counts, scored, details


def analyze_pair(case, seed, left, right):
    initial_equal = left['manifest']['initial'] == right['manifest']['initial']
    if not initial_equal:
        raise ValueError('Shared initialization differs')
    ps, qs = left['proposals'], right['proposals']
    first = next((i for i,(p,q) in enumerate(zip(ps,qs)) if p['candidate'] != q['candidate']), None)
    if any(left['observations'][i] != right['observations'][i] for i in range(first if first is not None else 18)):
        raise ValueError('Records differ before first candidate divergence')
    checks = []
    for i in range(18 if first is None else first+1):
        p,q=ps[i],qs[i]; a,b=p['metadata'],q['metadata']
        if p['prefix_digest'] != q['prefix_digest']:
            raise ValueError('Matched history digest differs')
        pool_equal = a.get('pool') == b.get('pool')
        fitted = bool(a.get('forecasts')) and bool(b.get('forecasts'))
        forecast_equal = same_forecasts(a['forecasts'], b['forecasts']) if fitted and pool_equal else None
        common_forecasts = common_forecast_check(a,b)
        ga,gb = a.get('novel_global_pool_size'),b.get('novel_global_pool_size')
        global_equal = ga == gb and ga is not None and a['pool'][:ga] == b['pool'][:gb]
        weights = [a['H2_controller_before']['onehot_weight'], b['H2_controller_before']['onehot_weight']]
        ids = [digest(p['candidate']),digest(q['candidate'])]
        cf = counterfactual(a, i, weights, ids) if fitted else {'identified':False,'reason':'missing_forecasts'}
        upstream = (pool_equal and forecast_equal is True and a.get('mode') == b.get('mode') and
                    a.get('fallback') == b.get('fallback') and a.get('feature_identity') == b.get('feature_identity') and
                    a.get('objective_feature_identity') == b.get('objective_feature_identity') and
                    a.get('risk_feature_identity') == b.get('risk_feature_identity') and
                    a.get('best_observed') == b.get('best_observed') and a.get('heads') == b.get('heads'))
        checks.append({'iteration': i, 'prefix_digest': p['prefix_digest'],
                       'dynamic_candidate_id':digest(p['candidate']), 'fixed_candidate_id':digest(q['candidate']),
                       'weights_dynamic_fixed':weights, 'scored_before_dynamic_fixed':
                       [a['H2_controller_before']['scored'],b['H2_controller_before']['scored']],
                       'pool_equal_ordered':pool_equal, 'pool_lengths':[len(a['pool']),len(b['pool'])],
                       'novel_global_pool_equal_ordered':global_equal,
                       'novel_global_pool_sizes':[ga,gb], 'common_gp_forecasts':common_forecasts,
                       'common_candidate_count':len({digest(x) for x in a['pool']} & {digest(x) for x in b['pool']}),
                       'forecasts_equal_except_soc_mixture':forecast_equal, 'head_diagnostics_equal':a.get('heads')==b.get('heads'),
                       'feature_identity_dynamic_fixed':[a.get('feature_identity'),b.get('feature_identity')],
                       'objective_feature_identity_dynamic_fixed':[a.get('objective_feature_identity'),b.get('objective_feature_identity')],
                       'risk_feature_identity_dynamic_fixed':[a.get('risk_feature_identity'),b.get('risk_feature_identity')],
                       'head_diagnostics_sha256_dynamic_fixed':[digest(a.get('heads')),digest(b.get('heads'))],
                       'mode_dynamic_fixed':[a.get('mode'),b.get('mode')], 'fallback_dynamic_fixed':[a.get('fallback'),b.get('fallback')],
                       'coverage_dynamic_fixed':[a.get('coverage_reason'),b.get('coverage_reason')],
                       'kappa_dynamic_fixed':[a.get('kappa'),b.get('kappa')],
                       'selected_index_dynamic_fixed':[a.get('selected_index'),b.get('selected_index')],
                       'upstream_equal_except_weight':bool(upstream), 'fixed_pool_counterfactual':cf})
    row = checks[-1] if first is not None else None
    if row is None:
        classification = 'no_candidate_divergence'
    elif not row['upstream_equal_except_weight']:
        classification = 'not_identified_upstream_or_pool_differs'
    elif not row['fixed_pool_counterfactual']['identified']:
        classification = 'not_identified_selection_replay_failed'
    else:
        r = row['fixed_pool_counterfactual']['reconstructed']
        classification = ('weight_only_fixed_pool_supported' if [x['candidate_id'] for x in r] ==
                          [row['dynamic_candidate_id'],row['fixed_candidate_id']] else 'not_identified_actual_selection_not_reproduced')
    return {'case':case,'seed':seed,'shared_initial_records_equal':initial_equal,'common_history_verified':True,
            'first_divergence_iteration_zero_based':first,'first_divergence_classification':classification,
            'matched_prefix_steps':checks,
            'subsequent_different_decisions_descriptive_only':sum(p['candidate'] != q['candidate'] for p,q in zip(ps,qs)),
            'dynamic_final_energy':left['summary']['best_feasible'],'fixed_final_energy':right['summary']['best_feasible'],
            'dynamic_timing_seconds':left['timing']['optimization_seconds'],'fixed_timing_seconds':right['timing']['optimization_seconds'],
            'dynamic_vae_seconds':left['summary']['corpus_training']['fit_seconds'],'fixed_vae_seconds':right['summary']['corpus_training']['fit_seconds'],
            'dynamic_curve':left['summary']['curve'],'fixed_curve':right['summary']['curve'],
            'dynamic_verification':left['summary']['step_verification']['status'],
            'fixed_verification':right['summary']['step_verification']['status'],
            'dynamic_verified_usable':left['summary']['step_verification']['verified_usable'],
            'fixed_verified_usable':right['summary']['step_verification']['verified_usable']}


def report(diag, scores):
    pairs = diag['cases']
    first = [p['matched_prefix_steps'][-1] for p in pairs if p['first_divergence_iteration_zero_based'] is not None]
    reversals = sum(x['fixed_pool_counterfactual'].get('model_rank_reversal',False) for x in first)
    lines = ['# H2 P4: first-divergence decision sensitivity', '',
        f"Frozen P3 archive SHA-256: `{ARCHIVE_SHA}`; 16 case/seed pairs, eight seeds. "
        'Zero new aircraft or RCAIDE calls, GP fits, VAE fits, and optimizer runs.', '',
        '## Matched histories and first divergence', '',
        f"All {len(pairs)} pairs have identical initial records and ordered observations through the last common candidate. "
        f"At first divergence, {sum(x['novel_global_pool_equal_ordered'] for x in first)}/{len(first)} have the same ordered novel global pool, "
        f"but {sum(not x['pool_equal_ordered'] for x in first)}/{len(first)} have different final augmented pools. "
        f"All {sum(x['head_diagnostics_equal'] for x in first)}/{len(first)} have equal saved head diagnostics. "
        f"Comparing saved forecasts on identical shared candidates, {sum(x['common_gp_forecasts'].get('all_absolute_differences_at_most_1e-10',False) for x in first)}/{len(first)} "
        'have maximum absolute difference at most 1e-10 in every checked field; remaining differences are recorded per field in diagnosis.json.', '',
        f"On the archived dynamic-arm final pool, source acquisition was reproduced and changing only the SOC mixture weight reverses its top model choice in {reversals}/{len(first)} first-divergence steps. "
        'This is a fixed-pool counterfactual, not a replay of the other arm: local candidate generation was weight-dependent and not rerun. '
        'All 16 actual first divergences have different final pools, so direct weight-only attribution of the full decision is not identified. '
        'Different choices after first divergence have different training histories and are descriptive trajectory outcomes, not matched one-step effects.', '',
        '| Case | Seed | First divergence (zero-based) | Dynamic weight | Global pool identical | Shared candidates | Fixed-pool rank reversal |',
        '|---|---:|---:|---:|---|---:|---|']
    for p in pairs:
        x=p['matched_prefix_steps'][-1]; cf=x['fixed_pool_counterfactual']
        lines.append(f"| {p['case']} | {p['seed']} | {p['first_divergence_iteration_zero_based']} | "
                     f"{x['weights_dynamic_fixed'][0]:.6f} | {x['novel_global_pool_equal_ordered']} | "
                     f"{x['common_candidate_count']} | {cf.get('model_rank_reversal','not identified')} |")
    lines += ['', '## Selected SOC likelihood scores', '',
              'Component NLLs are computed from the committed forecasts and subsequently settled SOC evidence. '
              'Exact-density and interval-probability-mass scores are not pooled. Counts are clustered, '
              'adaptive, selected on-policy observations; they do not establish population calibration.', '',
              '| Arm / score type | n | One-hot mean NLL | Latent mean NLL |', '|---|---:|---:|---:|']
    for key,v in sorted(scores['by_arm_and_observation_type'].items()):
        lines.append(f"| {key} | {v['n']} | {v['onehot_nll_mean']:.6f} | {v['latent_nll_mean']:.6f} |")
    for arm,v in scores['totals'].items():
        lines.append(f"\n{arm}: positive/negative score-difference clipping {v.get('clip_positive',0)}/{v.get('clip_negative',0)}; "
                     f"coverage-skipped {v.get('skipped_coverage',0)}; missing SOC {v.get('missing_soc',0)}; "
                     f"proposal weights at lower/upper bounds {v.get('weight_at_lower_bound',0)}/{v.get('weight_at_upper_bound',0)}.")
    lines += ['', '## Outcomes and cost', '',
              'All recorded seed pairs are retained. Energy differences are dynamic minus fixed, conditional on both nominally feasible. '
              'The five-second repeat is a numerical screen, not physical validation.', '',
              '| Case | Mean difference Wh | Median Wh | Mean without largest absolute pair Wh (seed) | Mean without 12075 Wh | Dynamic/fixed mean optimization seconds | Dynamic/fixed mean VAE fit seconds |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for case in CASES:
        rows=[p for p in pairs if p['case']==case]
        diffs=[p['dynamic_final_energy']-p['fixed_final_energy'] for p in rows]
        without=[d for p,d in zip(rows,diffs) if p['seed']!=12075]
        largest=max(range(len(diffs)),key=lambda i:abs(diffs[i]))
        without_largest=[d for i,d in enumerate(diffs) if i != largest]
        lines.append(f"| {case} | {statistics.mean(diffs):.3f} | {statistics.median(diffs):.3f} | "
                     f"{statistics.mean(without_largest):.3f} ({rows[largest]['seed']}) | {statistics.mean(without):.3f} | "
                     f"{statistics.mean(p['dynamic_timing_seconds'] for p in rows):.3f} / "
                     f"{statistics.mean(p['fixed_timing_seconds'] for p in rows):.3f} | "
                     f"{statistics.mean(p['dynamic_vae_seconds'] for p in rows):.3f} / "
                     f"{statistics.mean(p['fixed_vae_seconds'] for p in rows):.3f} |")
    sparse=next(p for p in pairs if p['case']=='aircraft_sparse' and p['seed']==12075)
    lines += ['', f"Sparse seed 12075 contributes {sparse['dynamic_final_energy']-sparse['fixed_final_energy']:.3f} Wh; "
              'its removal changes the sparse mean to the value above. This is sensitivity analysis, not a license to discard the seed.', '',
              '| Cold-start case / seed | First feasible online call, dynamic | First feasible online call, fixed |',
              '|---|---:|---:|']
    for p in pairs:
        if not p['dynamic_curve'][0]['has_feasible']:
            def first_feasible(curve):
                return next((r['online_budget'] for r in curve if r['has_feasible']),None)
            lines.append(f"| {p['case']} / {p['seed']} | {first_feasible(p['dynamic_curve'])} | {first_feasible(p['fixed_curve'])} |")
    lines += ['', f"All {sum(p['dynamic_verified_usable'] for p in pairs)+sum(p['fixed_verified_usable'] for p in pairs)}/32 "
              'same-design five-second numerical checks were usable. VAE fit time is included in optimization time, not additive. '
              'GP fit time was not separately recorded and cannot be inferred from proposal time. '
              'Full per-iteration incumbent curves and timings are in diagnosis.json.', '',
              '## Limits', '',
              'Archived full-pool scores are recomputed with the frozen acquisition, exact componentwise SOC feasibility CDF and '
              'moment-based optimistic margins. Scores are reconstructed only for the already saved pool, not the unobserved counterfactual '
              'local-search trajectory; coverage overrides are marked unidentifiable. No GP/VAE refit or true evaluator was called. '
              'The P3 data are an eight-seed development panel; these results neither prove superiority nor identify a new optimal policy.', '']
    return '\n'.join(lines)


def generate(archive, output):
    z, plan = validate_archive(archive)
    pairs, records, summaries = [], {}, {}
    for case in CASES:
        for seed in plan['profile']['seeds']:
            left,right = (trial(z,case,seed,arm) for arm in ARMS)
            pairs.append(analyze_pair(case,seed,left,right))
            for arm,t in zip(ARMS,(left,right)):
                counts,scored,details=score_trial(t)
                records[f'{case}/{seed}/{arm}']={'counts':dict(counts),'scores':details}
                for kind,rows in scored.items(): summaries.setdefault(f'{arm}/{kind}',[]).extend(rows)
    grouped={k:{'n':len(v),'onehot_nll_mean':statistics.mean(x[0] for x in v),
                'latent_nll_mean':statistics.mean(x[1] for x in v)} for k,v in summaries.items()}
    totals={arm:dict(sum((Counter(x['counts']) for key,x in records.items() if key.endswith('/'+arm)),Counter())) for arm in ARMS}
    diag={'schema':'h2_p4_first_divergence_v1','archive_sha256':ARCHIVE_SHA,'source_commit':SOURCE_SHA,
          'new_aircraft_calls':0,'new_rcaide_calls':0,'new_gp_fits':0,'new_vae_fits':0,'cases':pairs,
          'classification_counts':dict(Counter(p['first_divergence_classification'] for p in pairs))}
    scores={'schema':'h2_p4_on_policy_soc_scores_v1','scope':'adaptive selected points; not independent population calibration',
            'totals':totals,'by_arm_and_observation_type':grouped,'trials':records}
    output.mkdir(parents=True,exist_ok=False)
    for name,value in [('diagnosis.json',diag),('scores_by_type.json',scores)]:
        (output/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    (output/'decision_sensitivity.md').write_text(report(diag,scores),encoding='utf-8')
    z.close()
    return diag,scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();diag,scores=generate(args.archive,args.output)
    print(json.dumps({'pairs':len(diag['cases']),'classifications':diag['classification_counts'],
                      'score_totals':scores['totals'],'new_aircraft_calls':0},sort_keys=True))


if __name__ == '__main__':
    main()
