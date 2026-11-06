"""Frozen same-state numerical diagnostic; no new online policy or evaluator."""
import hashlib
import json
import math
from pathlib import Path

BASE = '275aac2c5d5d6a8c8f6278c0d556de3de5867f74'
EVIDENCE = 'f859cca5adc56dd101d2f59665f7829a2b0331c4'
RESULT = 'results/pr70-next-pool-20261106/full'
ARM = 'cei_two_step_next_pool'
CASES = ('branin_ordinary', 'branin_sparse', 'rosenbrock_ordinary', 'rosenbrock_sparse')
RULES = ((4, 0), (16, 0), (64, 0), (64, 1))
RTOL = 1e-7
ATOL = 1e-9


def require(test, message):
    if not test:
        raise ValueError(message)


def key(n, replicate):
    return 'n%d_r%d' % (n, replicate)


def plan(profile):
    require(profile in ('smoke', 'full'), 'Use smoke or full')
    seeds = [998901] if profile == 'smoke' else list(range(998901, 998905))
    iterations = [0] if profile == 'smoke' else [0, 3]
    states = [{'case': c, 'seed': s, 'iteration': i} for c in CASES for s in seeds for i in iterations]
    return {'schema': 'pr70_same_state_quadrature_diagnostic_v1', 'profile': profile,
            'base_code_sha': BASE, 'input_result_sha': EVIDENCE, 'input_arm': ARM,
            'states': states, 'rules': [list(x) for x in RULES],
            'root_candidates': 4, 'root_atol': ATOL, 'root_rtol': RTOL,
            'new_evaluation_budget': 0, 'new_vae_training_budget': 0,
            'gp_construction_ceiling': len(states) * (32 + 4*2*3*sum(n for n, _ in RULES)),
            'smoke_is_subset_of_full': True, 'future_outcomes_used_for_training': False,
            'independent_scrambles_at_64': 2, 'model_error_not_tested': True,
            'claim': 'Numerical ranking diagnostic on fixed PR70 prefixes, NOT new online performance'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(x):
    require(type(x) in (int, float) and math.isfinite(x), 'Nonfinite numeric evidence')
    return float(x)


def close(a, b, where='value'):
    if a is None or b is None:
        require(a is None and b is None, 'Missingness changed at '+where)
    elif isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), 'Keys changed at '+where)
        for k in a:
            close(a[k], b[k], where+'.'+k)
    elif isinstance(a, (list, tuple)):
        require(isinstance(b, (list, tuple)) and len(a) == len(b), 'Shape changed at '+where)
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, where+'[%d]' % i)
    elif type(a) in (float, int) and type(b) in (float, int):
        require(math.isclose(finite(a), finite(b), rel_tol=RTOL, abs_tol=ATOL), 'Replay mismatch at '+where)
    else:
        require(a == b, 'Replay mismatch at '+where)


def summarize(states):
    output = {}
    for case in CASES:
        group = [s for s in states if s['case'] == case]
        comparisons = {}
        for left, right in ((key(4, 0), key(16, 0)), (key(16, 0), key(64, 0)),
                            (key(4, 0), key(64, 0)), (key(64, 0), key(64, 1))):
            valid = [(s['rules'][left], s['rules'][right]) for s in group
                     if s['rules'][left]['status'] == s['rules'][right]['status'] == 'OK']
            comparisons[left+'_vs_'+right] = {
                'valid_pairs': len(valid), 'invalid_pairs': len(group)-len(valid),
                'winner_changes': sum(a['selected_index'] != b['selected_index'] for a,b in valid),
                'normalized_max_score_drift': [max(abs(a['scores'][str(i)]-b['scores'][str(i)])
                    for i in a['first_indices']) / max(1e-12, max(abs(v) for v in b['scores'].values())) for a,b in valid],
                'right_rule_loss_of_left_choice': [b['scores'][str(b['selected_index'])]-b['scores'][str(a['selected_index'])]
                                                   for a,b in valid]}
        output[case] = {'states': len(group), 'cold_states': sum(s['best'] is None for s in group),
                        'comparisons': comparisons}
    return output
