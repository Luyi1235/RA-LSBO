"""Reconstruct the pinned phase pilot, without evaluations or optimizer replay.

Usage: python tools/review_phase_pilot.py OLD_PILOT_DIR NEW_REVIEW_DIR
Writes derived facts only to a new directory outside the immutable input tree.
Python standard library only. The independently pinned import validates the bytes.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.experiments.discovery_stages.reviewed_phase import verify_reviewed_phase


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def satisfaction(value):
    if value['kind'] == 'exact':
        return value['value'] <= 0
    if value['kind'] == 'interval':
        lower, upper = value['lower'], value['upper']
        if upper is not None and upper <= 0:
            return True
        if lower is not None and (lower > 0 or (lower == 0 and not value['lower_inclusive'])):
            return False
    return None


def feasible(row):
    observation = row['observation']
    return (observation['status'] == 'completed'
            and observation['objective']['kind'] == 'exact'
            and row['diagnostics'].get('certified_physical_infeasible') is not True
            and bool(observation['constraints'])
            and all(satisfaction(value) is True for _, value in observation['constraints']))


def review(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError('Derived output must be outside the original evidence directory')
    if output.exists():
        raise FileExistsError('Choose a new derived output directory')
    receipt = verify_reviewed_phase(root)
    read = lambda p: json.loads(p.read_bytes())
    lines = lambda p: [json.loads(line) for line in p.read_bytes().splitlines() if line.strip()]
    plan, result = read(root/'plan.json'), read(root/'results.json')
    events = lines(root/'calls.jsonl')
    ledger = {start['tag']: (end['record'], start['role'])
              for start, end in zip(events[::2], events[1::2])}
    claimed = set()
    def claim(tag, role, candidate=None):
        require(tag in ledger and tag not in claimed, 'Missing/repeated call tag')
        row, actual_role = ledger[tag]
        require(actual_role == role, 'Wrong ledger role')
        require(candidate is None or candidate == row['candidate'], 'Wrong evaluated candidate')
        claimed.add(tag)
        return row
    for case, task in plan['tasks'].items():
        for group, role in [('anchors', 'qualification_anchors'), ('panel', 'independent_panel')]:
            for i, candidate in enumerate(task[group]):
                claim(f'{case}/{"anchor" if group == "anchors" else "panel"}/{i}', role, candidate)
    shared = {}
    for case in plan['profile']['cases']:
        for seed in plan['profile']['seeds']:
            shared[case, seed] = [claim(f'{case}/{seed}/random/initial/{i}', 'shared_initial', candidate)
                for i, candidate in enumerate(plan['tasks'][case]['initial'][str(seed)]['random']['random_candidates'])]
    keyed = {(s['case'], s['seed'], s['method']): s for s in result['summaries']}
    require(len(keyed) == len(result['summaries']) == 280, 'Wrong trial identities')
    trials, proposals_checked = [], 0
    for case in plan['profile']['cases']:
        for seed in plan['profile']['seeds']:
            for method in plan['arms']:
                summary = keyed[case, seed, method]
                directory = root/'optimization'/case/f'{seed}_{method}'
                manifest = read(directory/'manifest.json')
                initial = shared[case, seed]
                require(manifest['initial'] == initial and manifest['initial_digest'] == digest(initial),
                        'Initial observations are not shared')
                require(summary == read(directory/'summary.json'), 'Run summary identity differs')
                proposals, observations = lines(directory/'proposals.jsonl'), lines(directory/'observations.jsonl')
                require(len(proposals) == len(observations) == 30, 'Incomplete trial horizon')
                history = list(initial)
                known = [(i+1, r['observation']['objective']['value'])
                         for i, r in enumerate(initial) if feasible(r)]
                initial_best = min((v for _, v in known), default=None)
                initial_kind = 'warm' if known else 'cold'
                channels, cold_channels, discovery = Counter(), Counter(), None
                corners, near = 0, 0
                for i, (proposal, row) in enumerate(zip(proposals, observations)):
                    require(proposal['iteration'] == i and proposal['prefix_digest'] == digest(history)
                            and proposal['candidate'] == row['candidate'], 'Wrong proposal/history linkage')
                    require(claim(f'{case}/{seed}/random/{method}/online/{i}', 'online', row['candidate']) == row,
                            'Observation differs from settled ledger')
                    info = proposal['metadata']
                    channel = 'random' if method == 'random' else info.get('coverage_reason') or ('model_boundary' if info.get('boundary_slot') else 'model')
                    channels[channel] += 1
                    if not known:
                        cold_channels[channel] += 1
                    raw = row['candidate']
                    xy = raw['continuous']
                    corners += int(all(min(abs(xy[k]), abs(1-xy[k])) <= 1e-6 for k in ('x', 'y')))
                    # Current math domains are normalized x/y/trim in [0,1].
                    active = ('x', 'y', 'trim') if raw['discrete']['architecture'] == 'augmented' else ('x', 'y')
                    distances = [max(abs(xy[k]-r['candidate']['continuous'][k]) for k in active)
                        for r in history if raw['discrete'] == r['candidate']['discrete']]
                    near += int(bool(distances) and min(distances) <= 1e-8)
                    if feasible(row):
                        if not known:
                            discovery = channel
                        known.append((len(initial)+i+1, row['observation']['objective']['value']))
                    history.append(row)
                    proposals_checked += 1
                best = min((v for _, v in known), default=None)
                first = known[0][0] if known else None
                require(summary['initial_kind'] == initial_kind and summary['initial_best'] == initial_best
                        and summary['best_feasible'] == best and summary['first_feasible_total_call'] == first
                        and summary['has_feasible'] == bool(known), 'Recomputed summary differs')
                require(summary['decision_diagnostics']['near_repeat_calls'] == near,
                        'Recomputed near-repeat count differs')
                trials.append({'case': case, 'seed': seed, 'method': method,
                    'initial_kind': initial_kind, 'initial_best': initial_best, 'best_feasible': best,
                    'has_feasible': bool(known), 'first_feasible_total_call': first,
                    'cold_wait_truncated_30': min(30, first-12) if initial_kind == 'cold' and first else
                                              30 if initial_kind == 'cold' else None,
                    'first_feasible_channel': discovery, 'near_repeat_calls': near, 'box_corners': corners,
                    'channels': dict(channels), 'cold_channels': dict(cold_channels)})
    require(claimed == set(ledger), 'Unconsumed ledger calls')
    groups, pairs = [], []
    lookup = {(r['case'], r['seed'], r['method']): r for r in trials}
    for case in plan['profile']['cases']:
        for method in plan['arms']:
            rows = [r for r in trials if r['case'] == case and r['method'] == method]
            cold, warm = [r for r in rows if r['initial_kind'] == 'cold'], [r for r in rows if r['initial_kind'] == 'warm']
            good = [r['best_feasible'] for r in rows if r['has_feasible']]
            groups.append({'case': case, 'method': method, 'runs': len(rows),
                'feasible_runs': len(good), 'cold_discovered': sum(r['has_feasible'] for r in cold),
                'cold_n': len(cold), 'warm_n': len(warm),
                'warm_improved': sum(r['best_feasible'] < r['initial_best'] for r in warm),
                'median_feasible_objective': statistics.median(good) if good else None,
                'median_warm_objective': statistics.median(r['best_feasible'] for r in warm),
                'mean_cold_wait_truncated_30': statistics.mean(r['cold_wait_truncated_30'] for r in cold),
                'near_repeat_calls': sum(r['near_repeat_calls'] for r in rows),
                'box_corners': sum(r['box_corners'] for r in rows),
                'cold_first_feasible_channels': dict(Counter(r['first_feasible_channel'] for r in cold if r['has_feasible']))})
            counts, differences = Counter(), []
            for a in warm:
                b = lookup[case, a['seed'], 'ra_v2']
                x, y = a['best_feasible'], b['best_feasible']
                counts['tie' if math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-12) else
                       'better' if x < y else 'worse'] += 1
                differences.append(x-y)
            pairs.append({'case': case, 'method': method, 'reference': 'ra_v2', 'n': len(warm),
                **{k: counts[k] for k in ('better', 'tie', 'worse')},
                'median_paired_difference': statistics.median(differences), 'rtol': 1e-10, 'atol': 1e-12})
    require(verify_reviewed_phase(root) == receipt, 'Evidence changed during review')
    facts = {'reviewed_input': receipt, 'evidence_files':len(read(root/'audit.json')['evidence_hashes']), 'source_bindings':len(plan['sources']),
             'trials': len(trials), 'online_proposals_checked':proposals_checked,
             'groups':groups, 'warm_pairs_vs_ra_v2':pairs,
             'new_model_calls':0, 'algorithm_replay':False,
             'decision':'freeze phase control; test representation and evidence-dependent transfer',
             'limits':['already inspected development seeds', 'recorded channels are not causal ablations',
                       'no unpublished-candidate oracle evaluation', 'not the original Windows runtime audit']}
    output.mkdir(parents=True)
    (output/'facts.json').write_text(json.dumps(facts, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'trials.json').write_text(json.dumps(trials, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for name, rows in [('groups',groups), ('warm_pairs',pairs)]:
        with (output/(name+'.csv')).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(facts, indent=2))
    return facts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    review(args.input, args.output)
