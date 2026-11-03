"""Summarize completed P6 evidence without additional model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.h2_p6_two_stage import GRID, PRIMARY

CONTRASTS = ('generation_at_dynamic', 'generation_at_fixed',
             'ranking_on_dynamic_pool', 'ranking_on_fixed_pool')


def valid_at(row, threshold):
    cells = row.get('cells', {})
    return (set(cells) == {'DD', 'FF', 'DF', 'FD'} and
            len(row.get('native_replays', {})) == 2 and
            all(len(v) == 2 and all(x['passed'] for x in v) for v in row['native_replays'].values()) and
            all(c['repeat_identity'] and c['coverage_reason'] is None and
                c['score_gap'] is not None and c['score_gap'] > threshold for c in cells.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.output)
    data = json.loads((run/'two_stage_cells.json').read_text(encoding='utf8'))
    rows = data['pairs']
    if len(rows) != 16 or len({(r['case'],r['seed']) for r in rows}) != 16:
        raise ValueError('Expected 16 distinct matched histories')
    for row in rows:
        if (row['category'] == 'identified') != valid_at(row, PRIMARY):
            raise ValueError('Primary identification gate differs')
    sensitivity = []
    for t in GRID:
        valid = [r for r in rows if valid_at(r,t)]
        sensitivity.append({'threshold':t,'identified':len(valid),
            'conditional_choice_changes':{key:sum(r['contrasts'][key] for r in valid) for key in CONTRASTS}})
    pair_details = []
    for row in rows:
        pair_details.append({'case':row['case'], 'seed':row['seed'], 'category':row['category'],
            'reason':row['reason'], 'cells':{name:{key:c[key] for key in
                ('score_gap','score_scale','normalized_gap','exact_tie','top_candidates',
                 'repeat_identity','solver_failures','coverage_reason')}
                for name,c in row.get('cells',{}).items()}})
    tie = {'primary_threshold':PRIMARY, 'sensitivity':sensitivity, 'pairs':pair_details}
    identified = [r for r in rows if valid_at(r,PRIMARY)]
    counts = {key:sum(r['contrasts'][key] for r in identified) for key in CONTRASTS}
    if any(data[k] != 0 for k in ('aircraft_calls','rcaide_calls','vae_fits')):
        raise ValueError('Nonzero forbidden call count')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(run/'two_stage_cells.json', out/'two_stage_cells.json')
    (out/'tie_sensitivity.json').write_text(json.dumps(tie,indent=2,allow_nan=False)+'\n',encoding='utf8')
    lines = ['# H2 P6: two-stage SOC-weight attribution', '',
        f'Native DD/FF controls passed twice for {sum(len(r.get("native_replays",{}))==2 and all(len(v)==2 and all(x["passed"] for x in v) for v in r["native_replays"].values()) for r in rows)}/16 histories.',
        f'At the preregistered {PRIMARY:g} gap threshold, {len(identified)}/16 matched histories are identified; all four cells, repeated-run identity and coverage must pass.', '',
        '| Conditional contrast | Changed final candidate |', '|---|---:|']
    lines += [f'| {key} | {counts[key]}/{len(identified)} |' for key in CONTRASTS]
    lines += ['', 'These are categorical one-step contrasts on matched historical states, not additive energy effects. Unidentified histories are retained in the JSON; threshold sensitivity is descriptive and is not used to select a favorable cutoff.',
        f'Offline GP kernel-candidate fits: {sum(r["gp_candidate_fits"] for r in rows)}; total measured fit seconds: {sum(r["gp_fit_seconds"] for r in rows):.3f}. New aircraft calls, RCAIDE calls, and VAE fits: zero.',
        'Historical SLSQP did not store individual query trajectories; archived starts, endpoints, status, final pools and decisions were checked, while the unrecorded queries cannot be verified.',
        'The P3 140 km mean dynamic-minus-fixed energy was -262.91 Wh, versus +9.15 Wh excluding seed 12075 in a sensitivity analysis; no seed is excluded here. The data do not establish an energy, safety or calibration advantage.',
        'Decision gate: no algorithm promotion or new algorithm design is justified by these exploratory matched-step comparisons alone.', '',
        '| Case | Seed | Category | Reason | DD/FF/DF/FD score gaps |', '|---|---:|---|---|---|']
    for r in rows:
        gaps = {k:v['score_gap'] for k,v in r.get('cells',{}).items()}
        lines.append(f"| {r['case']} | {r['seed']} | {r['category']} | {r['reason']} | {gaps} |")
    (out/'two_stage_attribution.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'identified':len(identified),'conditional_changes':counts}))


if __name__ == '__main__':
    main()
