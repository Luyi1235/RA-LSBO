"""Summarize an existing P5 execution without fitting or evaluating models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import h2_p4_decision_sensitivity as p4
from tools.h2_p5_full_path import differences
from ra_lsbo.budget import digest


def first_local_change(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        d = differences(a, b, 'local')
        if d:
            return {'report_index': i, 'first_difference': d[0]}
    return None if len(left) == len(right) else {'report_index': min(len(left), len(right)), 'reason': 'report_count'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    data = json.loads((run/'full_path_interventions.json').read_text(encoding='utf8'))
    z, plan = p4.validate_archive(args.archive)
    if data['archive_sha256'] != p4.ARCHIVE_SHA or data['p3_source_commit'] != p4.SOURCE_SHA or len(data['pairs']) != 16:
        raise ValueError('Execution evidence identity mismatch')
    p4data = json.loads((Path(__file__).resolve().parents[1]/'results/h2-p4-decision-sensitivity-20261103/diagnosis.json').read_text())
    p4rows = {(r['case'], r['seed']): r for r in p4data['cases']}
    records = []
    for row in data['pairs']:
        arms = [p4.trial(z, row['case'], row['seed'], arm)['proposals'][row['iteration']] for arm in p4.ARMS]
        all_pass = len(row['native_replays']) == 2 and all(a['passed'] for attempts in row['native_replays'].values() for a in attempts)
        record = {'case': row['case'], 'seed': row['seed'], 'iteration': row['iteration'],
                  'native_replays_passed': all_pass, 'category': row['category'], 'reason': row.get('reason'),
                  'original_candidate_ids': row['original_candidate_ids'], 'dynamic_weight': row['dynamic_weight'],
                  'gp_candidate_fits': row['gp_candidate_fits'], 'gp_fit_seconds': row['gp_fit_seconds'],
                  'first_local_change': None, 'global_candidate_scores': []}
        if row.get('intervention'):
            record['first_local_change'] = first_local_change(*[x['local_reports'] for x in row['intervention']])
            for intervention, proposal in zip(row['intervention'], arms):
                meta = proposal['metadata']
                if (intervention['candidate_id'] != digest(proposal['candidate']) or
                    intervention['pool_ids'] != [digest(x) for x in meta['pool']] or
                    differences(intervention['local_reports'], meta['local_reports'])):
                    raise ValueError('Intervention does not match validated native reference')
                indexes = {digest(candidate): i for i, candidate in enumerate(meta['pool'])}
                # Archived forecasts refer to precisely these validated native runs.
                values = []
                for candidate in proposal['pool']:
                    identity = digest(candidate)
                    i = indexes.get(identity)
                    values.append({'candidate_id': identity,
                        'score': None if i is None else meta['scores'][i],
                        'soc_feasibility_probability': None if i is None else meta['per_constraint_probability']['soc'][i]})
                record['global_candidate_scores'].append(values)
            record['intervention_ids'] = [x['candidate_id'] for x in row['intervention']]
            record['score_margins'] = [x['score_margin'] for x in row['intervention']]
            record['weight_changes_choice'] = record['intervention_ids'][0] != record['intervention_ids'][1]
        p4first = p4rows[(row['case'], row['seed'])]['matched_prefix_steps'][-1]
        record['p4_fixed_pool_analysis'] = p4first['fixed_pool_counterfactual']
        record['p4_common_gp_forecast_deviations'] = p4first['common_gp_forecasts']
        records.append(record)
    native = sum(r['native_replays_passed'] for r in records)
    identified = [r for r in records if r['category'] == 'identified']
    flips = sum(r['weight_changes_choice'] for r in identified)
    summary = {'native_replay_pairs': native, 'total_pairs': len(records), 'identified_pairs': len(identified),
               'confirmed_choice_flips': flips, 'pairs': records}
    (out/'decision_evidence.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf8')
    shutil.copyfile(run/'full_path_interventions.json', out/'full_path_interventions.json')
    shutil.copyfile(run/'source_hashes.json', out/'source_hashes.json')
    lines = [
        '# H2 P5: frozen-prefix full-path weight intervention', '',
        f'Native two-arm replay passed {native}/16 pairs, twice per arm. Full-path identified pairs: {len(identified)}/16; confirmed changes in final choice: {flips}/{len(identified)}.',
        'A further 14 pairs have a top-two acquisition gap <= 1e-10 and are conservatively not identified; no failed or near-tied pair is counted as zero effect.',
        'The threshold and absolute/relative comparison tolerances were fixed before fitting. Each prefix was fitted once; 1,152 GP kernel candidates were attempted, with no VAE fit or physical/RCAIDE call.',
        '', 'The original checkpoints and all 186 P3 source identities matched. The replay uses the original DiscoveryBO.propose and RiskRefiningBO._augment, including SLSQP, near filtering and coverage. Archived records contain solver starts, endpoints and status, not every internal SLSQP query; internal per-query path identity cannot be independently verified.',
        '', 'The fixed-pool P4 reversal (8/16) is a distinct estimand. P5 does not predict downstream outcomes of the unchosen candidate, establish safety or demonstrate performance superiority.',
        'P3 sparse mean energy difference was -262.908 Wh (dynamic minus fixed), but +9.150 Wh without seed 12075; this post-divergence sensitivity is descriptive and no seed is removed from P5.',
        '', '| Case | Seed | Iteration | Native replay | Category | Weight change | Margins |', '|---|---:|---:|---|---|---|---|']
    for r in records:
        lines.append(f"| {r['case']} | {r['seed']} | {r['iteration']} | {r['native_replays_passed']} | {r['category']} | {r.get('weight_changes_choice')} | {r.get('score_margins')} |")
    (out/'full_path_replay_review.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    (out/'SHA256.json').write_text(json.dumps(hashes, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: summary[k] for k in ('native_replay_pairs', 'identified_pairs', 'confirmed_choice_flips')}))


if __name__ == '__main__':
    main()
