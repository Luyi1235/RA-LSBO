"""Read-only PR68 postmortem. Standard library only; never fit/evaluate designs.

Recombine saved quadrature contributions; do not call this a convergence test.
Retained next-pool actions are a consistency diagnostic, not causal regret.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

PARENT = '21a1267ee907774e6958a12539da98c0840cc1d9'
EVIDENCE = '2823c6204de7da0ffa537548e0dbbe411e41e22a'
RESULT_PATH = 'results/information-value-stage3-gp-fantasy-20261106'
ARMS = ('soft_gp_eligible', 'cei_myopic', 'cei_two_step', 'cei_two_step_no_abort_update')
CASES = ('branin_ordinary', 'branin_sparse', 'rosenbrock_ordinary', 'rosenbrock_sparse')


def require(test, message):
    if not test:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def token(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def raw_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(x):
    require(type(x) in (int, float) and math.isfinite(x), 'Nonfinite/non-numeric evidence')
    return float(x)


def close(a, b):
    return math.isclose(number(a), number(b), rel_tol=1e-9, abs_tol=1e-10)


def winner(values, candidates=None):
    candidates = list(range(len(values))) if candidates is None else list(candidates)
    require(bool(candidates), 'Empty decision support')
    return max(candidates, key=lambda i: (number(values[i]), -i))


def describe(values):
    values = [number(x) for x in values if x is not None]
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def risk_summary(rows):
    """p is ALWAYS probability of failure. No cross-policy calibration ranking."""
    pairs = [(number(r['predicted_failure']), int(r['nonfeasible'])) for r in rows
             if r['forecast_available']]
    require(all(0 <= p <= 1 and y in (0, 1) for p, y in pairs), 'Invalid risk evidence')
    result = {'predicted_rows': len(pairs), 'unpredicted_rows': len(rows)-len(pairs)}
    if pairs:
        result.update(mean_failure_probability=statistics.mean(p for p, _ in pairs),
                      observed_failure_fraction=statistics.mean(y for _, y in pairs),
                      brier=statistics.mean((p-y)**2 for p, y in pairs))
    for name, keep in (('predicted_failure_le_0_2', lambda p: p <= .2),
                       ('predicted_failure_ge_0_8', lambda p: p >= .8)):
        subset = [(p, y) for p, y in pairs if keep(p)]
        result[name] = {'n': len(subset), 'failures': sum(y for _, y in subset),
                        'feasible': sum(1-y for _, y in subset)}
    return result


def branch_rows(receipt, index):
    groups = defaultdict(list)
    for row in receipt['fantasies']:
        if row['first_index'] == index:
            groups[row['observation']['aborted']].append(row)
    require(bool(groups), 'Missing fantasy outcomes')
    return groups


def check_outcomes(receipt, n):
    for row in receipt['fantasies']:
        o = row['observation']
        require(type(o['aborted']) is bool and 0 <= number(row['weight']) <= 1,
                'Invalid fantasy probability or abort event')
        require(o['resource']['kind'] == 'exact', 'Observable resource lost')
        number(o['resource']['value'])
        if o['aborted']:
            require(o['region']['kind'] == 'interval' and o['region'].get('lower') == 0
                    and o['region'].get('lower_inclusive') is False
                    and o['objective']['kind'] == 'missing', 'Hidden abort response leaked')
            require('value' not in o['region'] and 'value' not in o['objective'], 'Hidden numeric abort output')
        else:
            require(o['region']['kind'] == 'exact' and number(o['region']['value']) <= 0,
                    'Invalid completed region')
        require(number(row['continuation']) >= 0, 'Negative continuation')
        second = row['second_index']
        require(second is None or (type(second) is int and 0 <= second < n
                                    and second != row['first_index']), 'Invalid/repeated second action')


def delete_one_node_choices(receipt):
    """3-of-4 saved-node sensitivity, NOT error bars or fresh integration."""
    candidates = receipt['first_action_indices']
    base = receipt['myopic_scores']
    groups = {i: branch_rows(receipt, i) for i in candidates}
    require(all(len(rows) == 4 for g in groups.values() for rows in g.values()), 'Expected 4 nodes per branch')
    chosen = []
    for removed in range(4):
        q = {}
        for i in candidates:
            q[i] = base[i] + sum(row['weight'] * row['continuation'] * (4./3.)
                                 for rows in groups[i].values()
                                 for j, row in enumerate(rows) if j != removed)
        chosen.append(winner(q, candidates))
    return chosen


def decision_metrics(proposal, curve, next_proposal=None):
    r = proposal['metadata'].get('information_value_decision')
    if r is None:
        return None
    pool = proposal['metadata']['pool']
    base, scores = r['myopic_scores'], r['scores']
    n = len(pool)
    require(n == len(base) == len(scores), 'Decision array cardinality mismatch')
    selected, myopic = r['selected_index'], r['myopic_index']
    require(type(selected) is int and 0 <= selected < n, 'Invalid selected index')
    require(myopic == winner(base) and proposal['candidate'] == pool[selected], 'Decision binding/argmax mismatch')
    require(all(number(v) >= 0 for v in base), 'Negative one-step value')
    result = {'iteration': proposal['iteration'], 'depth': r['depth'], 'criterion': r['criterion'],
              'selected_index': selected, 'myopic_index': myopic, 'deviated_from_own_myopic': selected != myopic,
              'immediate_sacrifice': base[myopic]-base[selected],
              'immediate_sacrifice_fraction': (base[myopic]-base[selected])/base[myopic] if base[myopic] > 0 else None,
              'predicted_two_step_advantage': None, 'node_deletion_changes': None,
              'next_pool_retention_mass': None, 'continuation_mass': None,
              'cold_success_mass_error_max': None, 'cold_score_above_one': False}
    if r['depth'] == 1:
        require(selected == myopic and not r['fantasies'] and not r['first_action_indices'], 'Last-step/myopic contract broken')
        return result
    require(r['depth'] == 2, 'Unregistered depth')
    if r.get('fallback'):
        result['fallback'] = r['fallback']
        return result
    candidates = r['first_action_indices']
    require(myopic in candidates and len(set(candidates)) == len(candidates)
            and all(type(i) is int and 0 <= i < n for i in candidates), 'Invalid first-action shortlist')
    require(all(row['first_index'] in candidates for row in r['fantasies']), 'Unlisted first action')
    check_outcomes(r, n)
    mass_errors = []
    for i in candidates:
        rows = [row for group in branch_rows(r, i).values() for row in group]
        require(close(sum(x['weight'] for x in rows), 1.), 'Fantasy mass not one')
        q = base[i] + sum(x['weight']*x['continuation'] for x in rows)
        require(close(q, scores[i]), 'Saved Q2 cannot be reconstructed from fantasies')
        if r['criterion'] == 'first_feasible_probability':
            success = sum(x['weight'] for x in rows if not x['observation']['aborted']
                          and x['observation']['resource']['value'] <= 0)
            mass_errors.append(abs(success-base[i]))
    require(selected == winner(scores, candidates), 'Recorded Q2 maximizer mismatch')
    result['predicted_two_step_advantage'] = scores[selected]-scores[myopic]
    result['cold_success_mass_error_max'] = max(mass_errors) if mass_errors else None
    result['cold_score_above_one'] = bool(mass_errors and max(scores[i] for i in candidates) > 1+1e-9)
    choices = delete_one_node_choices(r)
    result['node_deletion_selected_indices'] = choices
    result['node_deletion_changes'] = sum(i != selected for i in choices)
    selected_rows = [x for x in r['fantasies'] if x['first_index'] == selected and x['second_index'] is not None]
    result['continuation_mass'] = sum(x['weight'] for x in selected_rows)
    if next_proposal is not None:
        next_ids = {token(c) for c in next_proposal['metadata']['pool']}
        result['next_pool_retention_mass'] = sum(x['weight'] for x in selected_rows
            if token(pool[x['second_index']]) in next_ids)
        # Descriptive realized window, not a counterfactual value estimator.
        t = proposal['iteration']
        before, after = curve[t], curve[t+2]
        result['realized_next_two_calls_delivery_or_gain'] = (int(after is not None) if before is None
            else max(0., before-after))
    return result


def same_prefix(left, right):
    return (left['history_hash'] == right['history_hash'] and
            token(left['metadata']['pool']) == token(right['metadata']['pool']) and
            token(left['metadata'].get('forecasts')) == token(right['metadata'].get('forecasts')))


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, encoding='utf-8').strip()


def lock(root, expected):
    require(len(expected) == 40 and all(c in '0123456789abcdef' for c in expected), 'Exact SHA required')
    require(git(root, 'rev-parse', 'HEAD') == expected, 'Wrong worktree SHA')
    require(not git(root, 'status', '--porcelain'), 'Use a clean worktree')


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def audit(evidence_root, output, expected_sha):
    code = Path(__file__).resolve().parent
    evidence_root, output = Path(evidence_root).resolve(), Path(output).resolve()
    lock(code, expected_sha)
    lock(evidence_root, EVIDENCE)
    require(output != code and code not in output.parents and output != evidence_root
            and evidence_root not in output.parents and not output.exists(), 'Output must be NEW and external')
    git(code, 'merge-base', '--is-ancestor', PARENT, expected_sha)
    changes = git(code, 'diff', '--name-status', PARENT, expected_sha).splitlines()
    require(all(row.startswith('A\t') for row in changes), 'PR68 parent files must remain unchanged')
    # These modules and verifier use stdlib only; no GP/optimizer imports.
    from ra_lsbo.experiments.information_value_stage3.runner import verify
    source = evidence_root/RESULT_PATH/'full'
    verified = verify(source)
    report = read(source/'report.json')
    require(report['code_sha'] == PARENT and report['profile'] == 'full', 'Not the pinned full campaign')
    require(verified['actual_calls_settled'] == 576 and verified['trials'] == 64, 'Wrong PR68 bill')
    decisions, matched, trials, by_key, pair_details = [], [], {}, {}, []
    for case in CASES:
        for seed in range(998801, 998805):
            for arm in ARMS:
                folder = source/'trials'/case/str(seed)/arm
                trial, proposals = read(folder/'result.json'), lines(folder/'proposals.jsonl')
                trials[case, seed, arm] = trial
                by_key[case, seed, arm] = proposals
                for i, p in enumerate(proposals):
                    row = decision_metrics(p, trial['curve'], proposals[i+1] if i+1 < len(proposals) else None)
                    if row is not None:
                        row.update(case=case, seed=seed, arm=arm)
                        decisions.append(row)
            for other in ('cei_myopic', 'cei_two_step_no_abort_update'):
                a, b = trials[case, seed, 'cei_two_step'], trials[case, seed, other]
                x, y = a['final_best'], b['final_best']
                outcome = (0 if x is None and y is None else -1 if x is None else 1 if y is None
                           else 1 if x < y-1e-10 else -1 if x > y+1e-10 else 0)
                pair_details.append({'case': case, 'seed': seed, 'left': 'cei_two_step', 'right': other,
                    'left_final': x, 'right_final': y, 'win_loss_tie': outcome,
                    'objective_delta': x-y if x is not None and y is not None else None})
                for i, (p, q) in enumerate(zip(by_key[case, seed, 'cei_two_step'], by_key[case, seed, other])):
                    if same_prefix(p, q):
                        matched.append({'case': case, 'seed': seed, 'other': other, 'iteration': i,
                            'different_action': p['candidate'] != q['candidate'],
                            'history_hash': p['history_hash']})
    summary = {}
    for case in CASES:
        summary[case] = {}
        for arm in ARMS:
            ts = [trials[case, seed, arm] for seed in range(998801, 998805)]
            rs = [r for t in ts for r in t['decisions']]
            ds = [d for d in decisions if d['case'] == case and d['arm'] == arm and d['depth'] == 2]
            summary[case][arm] = {
                'final_feasible': sum(t['final_best'] is not None for t in ts),
                'online_failures': sum(t['nonfeasible'] for t in ts),
                'risk_on_own_selected_points': risk_summary(rs), 'depth2': len(ds),
                'own_myopic_overrides': sum(d['deviated_from_own_myopic'] for d in ds),
                'immediate_sacrifice_fraction': describe(d['immediate_sacrifice_fraction'] for d in ds),
                'node_deletion_sensitive_decisions': sum((d['node_deletion_changes'] or 0) > 0 for d in ds),
                'cold_scores_above_one': sum(d['cold_score_above_one'] for d in ds),
                'cold_success_mass_error': describe(d['cold_success_mass_error_max'] for d in ds),
                'continuation_mass': sum(d['continuation_mass'] or 0 for d in ds),
                'next_pool_retention_mass': sum(d['next_pool_retention_mass'] or 0 for d in ds)}
    checked_again = verify(source)
    require(checked_again == verified, 'Input changed during read-only audit')
    lock(code, expected_sha)
    lock(evidence_root, EVIDENCE)
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'COMPLETE_READ_ONLY_PR68_DECISION_AUDIT', 'audit_code_sha': expected_sha,
              'input_result_sha': EVIDENCE, 'input_code_sha': PARENT, 'verified_input': verified,
              'new_mathematical_evaluations': 0, 'new_GP_fits': 0, 'new_VAE_fits': 0,
              'aircraft_calls': 0, 'rcaide_calls': 0, 'by_case_arm': summary,
              'paired_final_results': pair_details, 'same_prefix_comparisons': matched,
              'limits': ['Saved-node deletion is sensitivity, not convergence or uncertainty bounds.',
                         'Exact next-pool retention is not geometric coverage or a causal performance explanation.',
                         'Different on-policy samples do not identify comparative model calibration.',
                         'Observed next-two-call gain is not a counterfactual acquisition-value estimator.',
                         'No new online results; no algorithm improvement or safety claim.']}
    write(output/'report.json', result)
    write(output/'decisions.json', decisions)
    write(output/'manifest.json', {p.name: raw_sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-sha', required=True)
    args = p.parse_args()
    result = audit(args.evidence_root, args.output, args.expected_sha)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
