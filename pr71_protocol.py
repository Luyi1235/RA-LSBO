"""Frozen-state numerical diagnostic; never an online performance study."""
import math
from pathlib import Path
import json
from pr69_evidence_verify import require

PARENT = '275aac2c5d5d6a8c8f6278c0d556de3de5867f74'
EVIDENCE = 'f859cca5adc56dd101d2f59665f7829a2b0331c4'
RESULT = 'results/pr70-next-pool-20261106/full'
ARM = 'cei_two_step_next_pool'
CASES = ('branin_ordinary', 'branin_sparse', 'rosenbrock_ordinary', 'rosenbrock_sparse')
RULES = ((4, 0), (16, 0), (64, 0), (64, 1))
RTOL, ATOL = 1e-6, 1e-8


def plan(profile):
    require(profile in ('ci', 'smoke', 'full'), 'Unregistered diagnostic profile')
    if profile == 'ci':
        states = [('rosenbrock_sparse', s, 0) for s in (998901, 998904)]
    else:
        seeds = (998901,) if profile == 'smoke' else range(998901, 998905)
        steps = (0,) if profile == 'smoke' else (0, 4)
        states = [(c, s, i) for c in CASES for s in seeds for i in steps]
    return {'schema': 'pr71_fixed_state_quadrature_v1', 'profile': profile,
        'states': [{'case': c, 'seed': s, 'iteration': i} for c, s, i in states],
        'arm': ARM, 'rules': [list(r) for r in RULES], 'n_states': len(states),
        'parent_code_sha': PARENT, 'input_result_sha': EVIDENCE,
        'state_selection': 'fixed case/seed/iteration roster; no selection by realized improvement',
        'same_history_kernels_features_roots_and_future_support': True,
        'first_actions': 4, 'scramble_offset': 104729,
        'comparison_rtol': RTOL, 'comparison_atol': ATOL,
        'new_mathematical_evaluations': 0, 'new_VAE_fits': 0,
        'aircraft_calls': 0, 'rcaide_calls': 0,
        'new_GP_refits': 'nonzero; prefix restoration and fantasy refits separately ledgered',
        'smoke_is_subset_of_full_not_independent_evidence': True,
        'limits': ['Posthoc numerical diagnostic, not new optimization performance.',
                   '64 nodes and two scrambles are not truth, confidence intervals or convergence proof.',
                   'Four-root screen, EP approximation and observation model remain unchanged.',
                   'No threshold is fitted and no policy is promoted from these states.']}


def rule_id(n, replicate):
    return 'n%d_r%d' % (n, replicate)


def summarize(rows):
    pairs = (('n4_r0', 'n16_r0'), ('n16_r0', 'n64_r0'), ('n64_r0', 'n64_r1'))
    output = {}
    for a, b in pairs:
        valid = [(r['rules'][a], r['rules'][b]) for r in rows
                 if r['rules'][a]['status'] == r['rules'][b]['status'] == 'ok']
        output[a + '_vs_' + b] = {'states_total': len(rows), 'comparable_states': len(valid),
            'unavailable_states': len(rows)-len(valid),
            'different_winner': sum(x['selected_index'] != y['selected_index'] for x, y in valid)}
    return output


def verify_output(folder):
    from ra_lsbo.experiments.information_value_stage1.io import verify as hashes
    root = Path(folder)
    status = hashes(root)
    read = lambda p: json.loads((root/p).read_text(encoding='utf-8'))
    report, spec = read('report.json'), read('plan.json')
    require(spec == plan(spec['profile']), 'Diagnostic plan mismatch')
    rows = [read('states/%s_%s_%s.json' % (s['case'], s['seed'], s['iteration'])) for s in spec['states']]
    require(report['states'] == rows and report['summary'] == summarize(rows), 'Diagnostic summary mismatch')
    require(report['input_result_sha'] == EVIDENCE and report['parent_code_sha'] == PARENT,
            'Wrong evidence identity')
    for key in ('new_mathematical_evaluations', 'new_VAE_fits', 'aircraft_calls', 'rcaide_calls'):
        require(report[key] == 0, 'Forbidden new work')
    events = [json.loads(s) for s in (root/'gp_refits.jsonl').read_text(encoding='utf-8').splitlines() if s]
    require(len(events) % 2 == 0, 'Unsettled GP refit')
    done = errors = 0
    for k in range(len(events)//2):
        a, b = events[2*k:2*k+2]
        require(a['ticket'] == b['ticket'] == k and a['event'] == 'started'
                and b['event'] in ('completed', 'error'), 'Broken GP ledger')
        done += b['event'] == 'completed'
        errors += b['event'] == 'error'
    require(report['gp_refits'] == {'started': done+errors, 'completed': done, 'errors': errors}, 'GP bill mismatch')
    for row, expected in zip(rows, spec['states']):
        require(row['state'] == expected and row['baseline_reproduced'], 'Invalid state or replay')
        require(set(row['rules']) == {rule_id(*r) for r in RULES}, 'Missing numerical rule')
        firsts = row['first_action_indices']
        for rid, r in row['rules'].items():
            if r['status'] != 'ok':
                require(r['status'] == 'numerical_failure' and r['selected_index'] is None, 'Improper failed-rule selection')
                continue
            n = r['n']
            require(rid == rule_id(n, r['replicate']), 'Wrong rule identity')
            rebuilt = {}
            for i in firsts:
                fs = [f for f in r['fantasies'] if f['first_index'] == i]
                require(len(fs) in (n, 2*n) and abs(sum(f['weight'] for f in fs)-1.) < 1e-10, 'Incomplete quadrature')
                for f in fs:
                    require(f['weight'] > 0 and math.isfinite(f['continuation']) and f['continuation'] >= 0, 'Invalid continuation')
                    o = f['observation']
                    if o['aborted']:
                        require(o['objective']['kind'] == 'missing' and o['region'] == {'kind': 'interval', 'lower': 0., 'lower_inclusive': False}, 'Hidden abort data')
                rebuilt[str(i)] = row['myopic_scores'][i] + sum(f['weight']*f['continuation'] for f in fs)
                require(math.isclose(rebuilt[str(i)], r['scores'][str(i)], rel_tol=1e-10, abs_tol=1e-12), 'Q2 reconstruction failed')
            winner = max(firsts, key=lambda i: (rebuilt[str(i)], -i))
            require(winner == r['selected_index'], 'Argmax differs')
    return {**status, 'states': len(rows), 'gp_refits_verified': done+errors,
            'new_GP_fits_in_verifier': 0, 'new_evaluations_in_verifier': 0}
