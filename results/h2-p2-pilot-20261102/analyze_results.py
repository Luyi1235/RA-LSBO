"""Read-only summary of completed H2 pilot; no model or optimizer calls."""
import json
from collections import Counter
from pathlib import Path

root = Path(r'D:\LSBO\RA-LSBO-H2-P2-pilot-20261102_02')
data = json.loads((root / 'results.json').read_text(encoding='utf-8'))
audit = json.loads((root / 'audit.json').read_text(encoding='utf-8'))
assert audit['passed'] and audit['settled_calls'] == data['charged_model_calls']
assert audit['trial_count'] == 48 and audit['checked_proposals'] == 864
summaries = data['summaries']
assert len(summaries) == 48
trials = []
for s in summaries:
    verification = s['step_verification']
    row = {k: s[k] for k in ('case', 'seed', 'method', 'initial_kind',
           'has_feasible', 'best_feasible', 'initial_best', 'online_calls',
           'first_feasible_online_call', 'model_failure_coverage_calls',
           'model_failure_heads', 'soc_prequential', 'corpus_training')}
    row['verification_status'] = verification['status']
    row['verification_usable'] = verification['verified_usable']
    row['verification_screen'] = (verification.get('screen') or {}).get('status')
    row['controller_replay'] = s['H2_controller_replay']
    if s['method'] == 'h2_causal':
        folder = root / 'optimization' / s['case'] / f"{s['seed']}_h2_causal"
        proposals = [json.loads(line) for line in (folder / 'proposals.jsonl').read_text(encoding='utf-8').splitlines()]
        weights = [p['metadata']['H2_controller_before']['onehot_weight'] for p in proposals]
        weights.append(s['H2_controller_replay']['final_state']['onehot_weight'])
        row['effective_weight_changes'] = sum(abs(b-a) > 1e-14 for a, b in zip(weights, weights[1:]))
        row['weight_sequence_before_each_proposal_and_after_last'] = weights
    trials.append(row)

heads = Counter()
for s in summaries:
    heads.update(s['model_failure_heads'])
result = {
    'scope': 'Read-only summaries of frozen pilot output; no model or optimizer calls',
    'source_commit': '7ca420613ccb50e138dc3c9d4292ac8e863c4ecf',
    'audit_passed': audit['passed'],
    'calls': data['charged_model_calls'],
    'calls_by_role': data['charged_by_role'],
    'rcaide_calls': data['new_rcaide_mda_calls'],
    'trial_count': len(trials),
    'h2_effective_weight_changes': sum(r.get('effective_weight_changes', 0) for r in trials),
    'h2_scored_events': sum(r['controller_replay']['scored_events'] for r in trials if r['controller_replay']),
    'h2_exact_scores': sum(r['controller_replay']['exact_scores'] for r in trials if r['controller_replay']),
    'h2_interval_scores': sum(r['controller_replay']['interval_scores'] for r in trials if r['controller_replay']),
    'model_failure_coverage_by_head': dict(heads),
    'verification_status_counts': dict(Counter(r['verification_status'] for r in trials)),
    'verification_screen_counts': dict(Counter(str(r['verification_screen']) for r in trials)),
    'effective_weight_change_definition': 'successive audited before-proposal weights and final replayed weight differ by more than 1e-14; not the same as number of scored observations',
    'trials': trials,
}
print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
