"""Read-only, hash-pinned review of the 2026-10-30 transfer pilot.

Usage: python tools/review_transfer_pilot.py ORIGINAL_PILOT_DIR NEW_REVIEW_DIR
No model fitting, evaluator, network, or historical source execution. The cold
selector check reconstructs saved arithmetic, NOT the optimization trajectory.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.experiments.discovery_stages.reviewed_phase import _read
from ra_lsbo.experiments.discovery_stages.decision_effect import (
    cold_model_choice, compare_saved_choices, normalized_weights,
    reweight_constraints, update_saved_weights)

PINNED = {
    'plan.json': '373fbee198add3471cce6db462b8c3926b0a0ac5b167c94abbb3daffbeab98a1',
    'registration.json': 'f1901aaf8a1cfcb35b5c9b4033cbf0e03c4188a739665b3aa9e5628e7956f154',
    'audit.json': 'd7bfa8463967181b28f6b93862dcce0389e8d2889e51e558bd6db90454e7d4de',
    'results.json': '412b9b3b9cadfc8380ff0dc1e6c31dd2276218a9c47ced406ea214eb46c8badf',
    'calls.jsonl': 'cae5360754fb77aca4ff02ed9427e7dab69b3594c7a36fd303337b5faa08569c',
}
SOURCE_COMMIT = '21505bba5f9485287a3e34f4ce50dc849edf4e04'
DATA_COMMIT = '497f810260fe8828189974f540ff1e5c5000db7e'
ARCHIVE_SHA256 = '90d13623d1175374d442b19c431493393617016a1267baa65db2cc2203f2fca1'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def satisfaction(value):
    if value['kind'] == 'exact':
        return value['value'] <= 0
    if value['kind'] == 'interval':
        if value['upper'] is not None and value['upper'] <= 0:
            return True
        if value['lower'] is not None and (value['lower'] > 0 or
                (value['lower'] == 0 and not value['lower_inclusive'])):
            return False
    return None


def feasible(row):
    o = row['observation']
    return (o['status'] == 'completed' and o['objective']['kind'] == 'exact'
            and row['diagnostics'].get('certified_physical_infeasible') is not True
            and bool(o['constraints']) and all(satisfaction(v) is True for _, v in o['constraints']))


def verify_bytes(root):
    for path, expected in PINNED.items():
        require(hashlib.sha256(_read(root, path)).hexdigest() == expected, 'Unreviewed/modified header: '+path)
    plan, audit = [json.loads(_read(root, p+'.json')) for p in ('plan', 'audit')]
    require(plan['source_commit'] == SOURCE_COMMIT and plan['stage'] == 'transfer'
            and plan['profile']['name'] == 'pilot' and audit['passed'], 'Wrong campaign identity')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p.suffix in {'.json', '.jsonl'} and p.name != 'audit.json'}
    require(actual == set(audit['evidence_hashes']), 'Sealed evidence file set differs')
    for path, expected in audit['evidence_hashes'].items():
        require(hashlib.sha256(_read(root, path)).hexdigest() == expected, 'Changed evidence: '+path)
    for expected in plan['sources'].values():
        require(len(expected) == 64 and set(expected) <= set('0123456789abcdef'), 'Invalid source hash')
        require(hashlib.sha256(_read(root, 'source_snapshot/'+expected+'.source')).hexdigest() == expected,
                'Changed source snapshot')
    return plan, audit


def choice_matches(actual, metadata):
    require(actual['model_index'] == metadata['model_selected_index'], 'Saved model argmax differs')
    require(actual['eligible'] == metadata['eligible'], 'Saved eligibility differs')
    require(actual['boundary_slot'] == metadata['boundary_slot'], 'Saved boundary schedule differs')
    for a, b in zip(actual['scores'], metadata['scores']):
        require((a is None and b is None) or (a is not None and b is not None and
                math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)), 'Saved score arithmetic differs')


def paired_outcome(left, right):
    if left['has_feasible'] != right['has_feasible']:
        return 'better' if left['has_feasible'] else 'worse'
    if not left['has_feasible'] or math.isclose(left['best_feasible'], right['best_feasible'],
                                               rel_tol=1e-10, abs_tol=1e-12):
        return 'tie'
    return 'better' if left['best_feasible'] < right['best_feasible'] else 'worse'


def median(values):
    values = list(values)
    return statistics.median(values) if values else None


def review(root, output):
    root, output = Path(root).absolute(), Path(output).resolve()
    require(output != root.resolve() and root.resolve() not in output.parents,
            'Output must be outside immutable evidence')
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    plan, seal = verify_bytes(root)
    read = lambda p: json.loads(_read(root, p))
    lines = lambda p: [json.loads(x) for x in _read(root, p).splitlines() if x.strip()]
    result, registration = read('results.json'), read('registration.json')
    require(registration == {'maximum_model_calls': 3108, 'new_model_calls': 0,
                             'plan_sha256': PINNED['plan.json']}, 'Wrong registration')
    events, ledger, ids, roles = lines('calls.jsonl'), {}, set(), Counter()
    require(len(events) == 2*3108, 'Wrong ledger event count')
    for number, (start, end) in enumerate(zip(events[::2], events[1::2]), 1):
        payload = {k: start[k] for k in ('number', 'tag', 'role', 'candidate', 'contract', 'new_rcaide_mda')}
        record = end['record']
        require(start['event'] == 'started' and end['event'] == 'settled' and start['number'] == number
                and start['call_id'] == digest(payload) == end['call_id'] and start['call_id'] not in ids
                and start['tag'] not in ledger and start['tag'] == end['tag']
                and record['candidate'] == start['candidate'] and digest(record) == end['record_digest']
                and digest(record['candidate']) == record['observation']['design_id']
                and record['observation']['contract_id'] == start['contract']
                and start['new_rcaide_mda'] is False, 'Ledger identity/linkage mismatch')
        ids.add(start['call_id']); roles[start['role']] += 1
        ledger[start['tag']] = (record, start['role'])
    require(dict(roles) == result['charged_by_role'] and result['charged_model_calls'] == 3108,
            'Wrong role counts')
    claimed = set()
    def claim(tag, role, candidate):
        require(tag in ledger and tag not in claimed, 'Missing or duplicated call tag')
        row, actual_role = ledger[tag]
        require(actual_role == role and row['candidate'] == candidate, 'Wrong call role/candidate')
        claimed.add(tag)
        return row
    for case, task in plan['tasks'].items():
        for group, tag, role in [('anchors', 'anchor', 'qualification_anchors'),
                                 ('panel', 'panel', 'independent_panel')]:
            for i, candidate in enumerate(task[group]):
                claim(f'{case}/{tag}/{i}', role, candidate)
    summaries = {(s['case'], s['seed'], s['method']): s for s in result['summaries']}
    require(len(summaries) == len(result['summaries']) == 96, 'Wrong trial count')
    traces, trial_rows, weight_rows, checked_scores, output_kinds = {}, [], [], 0, Counter()
    for record, _ in ledger.values():
        output_kinds[record['observation']['objective']['kind']] += 1
    for case in plan['profile']['cases']:
        for seed in plan['profile']['seeds']:
            initial = [claim(f'{case}/{seed}/random/initial/{i}', 'shared_initial', c)
                for i, c in enumerate(plan['tasks'][case]['initial'][str(seed)]['random']['random_candidates'])]
            for method in plan['arms']:
                directory = f'optimization/{case}/{seed}_{method}/'
                manifest, s = read(directory+'manifest.json'), summaries[case, seed, method]
                require(manifest['initial'] == initial and manifest['initial_digest'] == digest(initial),
                        'Initial observations are not shared')
                require(s == read(directory+'summary.json'), 'Trial summary differs from result packet')
                proposals, observations = lines(directory+'proposals.jsonl'), lines(directory+'observations.jsonl')
                require(len(proposals) == len(observations) == 30, 'Incomplete trial')
                history, known = list(initial), [(i+1, r['observation']['objective']['value'])
                                                 for i, r in enumerate(initial) if feasible(r)]
                initial_best = min((v for _, v in known), default=None)
                initial_kind = 'warm' if known else 'cold'
                state = {name: [-math.log(2)]*2 for name, _ in initial[0]['observation']['constraints']}
                cold_channels, cold_steps, score_sum = Counter(), 0, {name: np.zeros(2) for name in state}
                discovery_channel, model_cold_count, numerical_near = None, 0, 0
                for i, (p, row) in enumerate(zip(proposals, observations)):
                    require(p['iteration'] == i and p['prefix_digest'] == digest(history)
                            and p['candidate'] == row['candidate'], 'Proposal/history identity mismatch')
                    require(row == claim(f'{case}/{seed}/random/{method}/online/{i}', 'online', p['candidate']),
                            'Observation differs from settled record')
                    m, cold = p['metadata'], not bool(known)
                    channel = 'random' if method == 'random' else m.get('coverage_reason') or ('model_boundary' if m.get('boundary_slot') else 'model')
                    if cold:
                        cold_channels[channel] += 1; cold_steps += 1
                        model_cold_count += int(channel.startswith('model'))
                    candidate = row['candidate']; active = ('x', 'y', 'trim') if candidate['discrete']['architecture'] == 'augmented' else ('x', 'y')
                    distances = [max(abs(candidate['continuous'][k]-r['candidate']['continuous'][k]) for k in active)
                        for r in history if candidate['discrete'] == r['candidate']['discrete']]
                    numerical_near += int(bool(distances) and min(distances) <= 1e-8)
                    if method in {'ev_equal', 'ev_prequential'}:
                        packet = deepcopy(m['evidence_transfer']); forecast_digest = packet.pop('forecast_digest')
                        require(digest(packet) == forecast_digest and packet['active'] == cold
                                and packet['prefix_digest'] == p['prefix_digest']
                                and packet['design_id'] == digest(candidate)
                                and packet['mode'] == ('equal' if method == 'ev_equal' else 'prequential'),
                                'Expert pre-evaluation forecast seal/phase mismatch')
                        for name in state:
                            require(np.allclose(packet['log_weights_before'][name], state[name],
                                                rtol=1e-10, atol=1e-12), 'Causal weight sequence differs')
                        if cold:
                            state, scores = update_saved_weights(state, packet['forecasts'], row, packet['mode'])
                            for name, value in scores.items():
                                if value is not None:
                                    score_sum[name] += value
                            checked_scores += 1
                            if 'forecasts' in m:
                                choice_matches(cold_model_choice(m['forecasts'], i), m)
                        else:
                            require(packet['forecasts'] is None, 'Warm model unexpectedly scored')
                    if feasible(row):
                        if cold:
                            discovery_channel = channel
                        known.append((12+i+1, row['observation']['objective']['value']))
                    history.append(row)
                best, first = min((v for _, v in known), default=None), known[0][0] if known else None
                require(s['initial_kind'] == initial_kind and s['initial_best'] == initial_best
                        and s['has_feasible'] == bool(known) and s['best_feasible'] == best
                        and s['first_feasible_total_call'] == first, 'Reconstructed outcome differs')
                require(s['decision_diagnostics']['near_repeat_calls'] == numerical_near, 'Near repeat count differs')
                item = {'case': case, 'seed': seed, 'method': method, 'initial_kind': initial_kind,
                    'initial_best': initial_best, 'best_feasible': best, 'has_feasible': bool(known),
                    'first_feasible_online_call': first-12 if first and initial_kind == 'cold' else 0 if first else None,
                    'cold_wait_truncated_30': min(first-12, 30) if first and initial_kind == 'cold' else 30 if not first else None,
                    'optimization_seconds': s['optimization_seconds'], 'cold_steps': cold_steps,
                    'cold_channels': dict(cold_channels), 'model_cold_steps': model_cold_count,
                    'first_feasible_channel': discovery_channel, 'near_repeat_calls': numerical_near}
                trial_rows.append(item)
                if method in {'ev_equal', 'ev_prequential'}:
                    for name in state:
                        weight_rows.append({'case': case, 'seed': seed, 'method': method, 'constraint': name,
                            'cold_scored_steps': cold_steps, 'final_latent_weight': float(normalized_weights(state[name])[0]),
                            'latent_score_minus_onehot': float(score_sum[name][0]-score_sum[name][1])})
                    traces[case, seed, method] = (proposals, observations)
    require(claimed == set(ledger), 'Unconsumed ledger calls')
    comparisons, matched_candidates, matched_observations, cold_candidates = [], 0, 0, 0
    for case in plan['profile']['cases']:
        for seed in plan['profile']['seeds']:
            left, lo = traces[case, seed, 'ev_prequential']; right, ro = traces[case, seed, 'ev_equal']
            for p, e, actual, other in zip(left, right, lo, ro):
                effect = compare_saved_choices(p, e)
                matched_candidates += int(not effect['actual_candidate_changed'])
                matched_observations += int(actual == other)
                if not p['metadata']['evidence_transfer']['active']:
                    continue
                cold_candidates += 1
                m, em = p['metadata'], e['metadata']
                if 'forecasts' in m:
                    scales = {n: [expert['heads'][n]['scale'] for expert in m['constraint_experts']]
                              for n in m['forecasts']['constraints']}
                    # Both independently recorded expert predictions must match before reweighting.
                    for n in scales:
                        for field in ('mean', 'variance'):
                            require(np.allclose(m['forecasts']['constraints'][n]['components'][field],
                                                em['forecasts']['constraints'][n]['components'][field],
                                                rtol=1e-10, atol=1e-12), 'Expert predictions differ across paired histories')
                    counter = reweight_constraints(m['forecasts'], {n: [.5, .5] for n in scales}, scales)
                    choice_matches(cold_model_choice(counter, p['iteration']), em)
                comparisons.append({'case': case, 'seed': seed, 'iteration': p['iteration'], **effect})
    groups, pairs = [], []
    lookup = {(r['case'], r['seed'], r['method']): r for r in trial_rows}
    for case in plan['profile']['cases']:
        for method in plan['arms']:
            rows = [r for r in trial_rows if r['case'] == case and r['method'] == method]
            cold = [r for r in rows if r['initial_kind'] == 'cold']
            good = [r for r in rows if r['has_feasible']]
            groups.append({'case': case, 'method': method, 'runs': len(rows), 'feasible_runs':len(good),
                'cold_n':len(cold), 'cold_successes':sum(r['has_feasible'] for r in cold),
                'mean_truncated_cold_wait':statistics.mean(r['cold_wait_truncated_30'] for r in cold) if cold else None,
                'median_feasible_objective':median(r['best_feasible'] for r in good),
                'median_optimization_seconds':median(r['optimization_seconds'] for r in rows),
                'median_cold_trial_seconds':median(r['optimization_seconds'] for r in cold)})
            counts = Counter(paired_outcome(lookup[case,r['seed'],'ev_prequential'], r) for r in rows)
            pairs.append({'case':case, 'left':'ev_prequential', 'right':method, 'n':len(rows),
                          **{k:counts[k] for k in ('better','tie','worse')}})
    verify_bytes(root)  # no original bytes may have changed while analyzing
    facts = {'data_commit':DATA_COMMIT, 'source_commit':SOURCE_COMMIT, 'archive_sha256':ARCHIVE_SHA256,
        'pinned_input_files':PINNED, 'evidence_files_checked':len(seal['evidence_hashes']),
        'source_bindings_checked':len(plan['sources']), 'settled_calls':3108, 'roles':dict(roles),
        'trial_count':96, 'proposals_linked':2880, 'cold_scoring_settlements_checked':checked_scores,
        'objective_observation_kinds':dict(output_kinds), 'paired_candidate_count':480,
        'equal_adaptive_same_candidates':matched_candidates, 'equal_adaptive_same_observations':matched_observations,
        'equal_adaptive_cold_paired_steps':cold_candidates,
        'cold_model_comparable_steps':sum(r['model_comparable'] for r in comparisons),
        'cold_gate_changes':sum(r['gate_changed'] for r in comparisons),
        'cold_model_choice_changes':sum(r['model_candidate_changed'] for r in comparisons),
        'model_changes_overridden_by_coverage':sum(r['model_change_overridden_by_coverage'] for r in comparisons),
        'realized_cold_candidate_changes':sum(r['actual_candidate_changed'] for r in comparisons),
        'cold_channels_adaptive':dict(Counter(r['left_channel'] for r in comparisons)),
        'groups':groups, 'paired_final_objectives':pairs, 'new_model_calls':0, 'new_rcaide_mda_calls':0,
        'algorithm_replay':False, 'inference':'post-result descriptive development review; no superiority/equivalence test',
        'limits':['Only saved score arithmetic is reconstructed; no GP refit or optimizer replay.',
                  'Same realized history/pool is required; no counterfactual outcomes are evaluated.',
                  'Per-trial wall time is not controlled microbenchmarking; no per-step timing was recorded.',
                  'Current math intervals are mainly signs; no general finite-interval benefit is identified.']}
    output.mkdir(parents=True)
    for name, data in [('facts',facts), ('trials',trial_rows), ('weights',weight_rows), ('decision_effects',comparisons)]:
        (output/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name, rows in [('groups',groups), ('paired_outcomes',pairs), ('weights',weight_rows), ('decision_effects',comparisons)]:
        with (output/(name+'.csv')).open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(facts,indent=2,allow_nan=False))
    return facts


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    review(args.input,args.output)
