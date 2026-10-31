"""Read-only reconstruction of the fixed nominal aircraft pilot, no model imports."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import zipfile

ARCHIVE_SHA = '35306153ceaf9f0d17d3586455d140931985f384e28b8dba93f1902095ef6bfd'
MANIFEST_SHA = '6abee3f22e39b31f75bdd1ed40b61dcbd40b37e5fa230301bf37600e5c2633b1'
ROOT = 'aircraft_nominal_20261030_01/'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def satisfaction(value):
    if value['kind'] == 'exact':
        return value['value'] <= 0
    if value['kind'] == 'interval':
        lo, hi = value['lower'], value['upper']
        if hi is not None and hi <= 0:
            return True
        if lo is not None and (lo > 0 or (lo == 0 and not value['lower_inclusive'])):
            return False
    return None


def usable(record):
    ob = record['observation']; cons = ob['constraints']
    return (ob['status'] == 'completed' and ob['objective']['kind'] == 'exact'
            and not record['diagnostics'].get('certified_physical_infeasible', False)
            and bool(cons) and all(satisfaction(v) is True for _, v in cons))


def pairing(left, right):
    require(set(left) == set(right), 'Unpaired seeds')
    ds = [left[s]-right[s] for s in sorted(left)]
    ties = [math.isclose(left[s], right[s], rel_tol=1e-10, abs_tol=1e-12) for s in sorted(left)]
    return {'n': len(ds), 'left_better': sum(d < 0 and not t for d, t in zip(ds, ties)),
            'ties': sum(ties), 'left_worse': sum(d > 0 and not t for d, t in zip(ds, ties)),
            'median_difference_Wh': median(ds),
            'median_relative_percent': median([100*(left[s]-right[s])/right[s] for s in sorted(left)]),
            'differences_Wh': ds, 'seeds': sorted(left), 'inference': 'exploratory, not confirmation'}


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def review(archive, manifest):
    require(hash_file(archive) == ARCHIVE_SHA, 'Wrong raw archive')
    require(hash_file(manifest) == MANIFEST_SHA, 'Wrong manifest')
    supplied = json.loads(Path(manifest).read_bytes())
    facts = {'archive_sha256': ARCHIVE_SHA, 'new_model_calls': 0, 'new_rcaide_mda_calls': 0,
             'optimizer_replay': False, 'physical_validation': False, 'runs': {}}
    trials, steps = [], []
    with zipfile.ZipFile(archive) as z:
        files = {item.filename for item in z.infolist() if not item.is_dir()}
        require(files == {ROOT+r['path'] for r in supplied['files']}, 'Archive/manifest member mismatch')
        hashes = {}
        for item in supplied['files']:
            name = ROOT+item['path']; raw = z.read(name)
            h = hashlib.sha256(raw).hexdigest(); hashes[name] = h
            require(len(raw) == item['bytes'] and h == item['sha256'], 'Archive member changed: '+name)
        facts['manifest_files'] = len(hashes)
        for run in ('smoke', 'pilot'):
            prefix = ROOT+run+'/'
            load = lambda rel: json.loads(z.read(prefix+rel))
            lines = lambda rel: [json.loads(line) for line in z.read(prefix+rel).splitlines() if line.strip()]
            plan, result, audit = load('plan.json'), load('results.json'), load('audit.json')
            require(plan['source_commit'] == 'e4fbcd37ca4d096bbdac03fc57b0f122f006aea3', 'Wrong executed source')
            require(audit['passed'] is True, 'Missing passing historical audit')
            for rel, h in audit['evidence_hashes'].items():
                require(hashes[prefix+rel] == h, 'Sealed evidence mismatch')
            for h in plan['sources'].values():
                require(hashes[prefix+'source_snapshot/'+h+'.source'] == h, 'Source snapshot mismatch')
            require(load('registration.json')['plan_sha256'] == hashes[prefix+'plan.json'], 'Plan registration mismatch')
            events = lines('calls.jsonl'); require(len(events) % 2 == 0, 'Unsettled call')
            settled = {}; ids = set(); roles = Counter()
            for number, (a, b) in enumerate(zip(events[::2], events[1::2]), 1):
                call = {k: a[k] for k in ('number', 'tag', 'role', 'candidate', 'contract', 'new_rcaide_mda')}
                row = b['record']; ob = row['observation']
                require(a['event'] == 'started' and b['event'] == 'settled' and a['number'] == number, 'Ledger order mismatch')
                require(a['call_id'] == b['call_id'] == digest(call) and a['call_id'] not in ids, 'Call identity mismatch')
                require(a['tag'] == b['tag'] and a['tag'] not in settled, 'Repeated or mismatched tag')
                require(a['candidate'] == row['candidate'] and ob['design_id'] == digest(a['candidate']), 'Candidate mismatch')
                require(ob['contract_id'] == a['contract'] and b['record_digest'] == digest(row), 'Record identity mismatch')
                require(a['new_rcaide_mda'] is False, 'Unexpected MDA call')
                require(math.isfinite(b['seconds']) and b['seconds'] >= 0, 'Invalid duration')
                ids.add(a['call_id']); settled[a['tag']] = (row, a['role']); roles[a['role']] += 1
            claimed = set(); count = 0; verified = 0
            lookup = {(r['case'], r['seed'], r['method']): r for r in result['summaries']}
            def claim(tag, role):
                require(tag not in claimed and tag in settled, 'Missing or repeated expected call')
                row, actual = settled[tag]; require(actual == role, 'Wrong role'); claimed.add(tag)
                return row
            for case in plan['cases']:
                for seed in plan['profile']['seeds']:
                    initial = [claim(f'{case}/{seed}/initial/{i}', 'shared_initial') for i in range(plan['profile']['initial'])]
                    require([r['candidate'] for r in initial] == plan['tasks'][case]['initial'][str(seed)]['candidates'], 'Shared initial candidates mismatch')
                    for method in plan['arms']:
                        key = f'optimization/{case}/{seed}_{method}/'; man = load(key+'manifest.json')
                        require(man['initial'] == initial and man['initial_digest'] == digest(initial), 'Different shared initial labels')
                        require(man['specification'] == plan['methods'][method], 'Method specification changed')
                        proposals, obs = lines(key+'proposals.jsonl'), lines(key+'observations.jsonl')
                        require(len(proposals) == len(obs) == plan['profile']['horizon'], 'Incomplete trial')
                        history = list(initial)
                        for i, (prop, row) in enumerate(zip(proposals, obs)):
                            require(prop['iteration'] == i and prop['prefix_digest'] == digest(history), 'History prefix mismatch')
                            actual = claim(f'{case}/{seed}/{method}/online/{i}', 'online')
                            require(actual == row and prop['candidate'] == row['candidate'], 'Proposal/ledger mismatch')
                            require(row['observation']['design_id'] not in {r['observation']['design_id'] for r in history}, 'Exact duplicate evaluated')
                            meta = prop['metadata']; idx = meta.get('selected_index')
                            if 'pool' in meta and idx is not None:
                                require(meta['pool'][idx] == row['candidate'], 'Selected candidate mismatch')
                            if run == 'pilot':
                                step = {'case': case, 'seed': seed, 'method': method, 'iteration': i,
                                    'cold': not any(usable(r) for r in history), 'usable': usable(row),
                                    'termination': row['diagnostics'].get('termination_reason'),
                                    'local': meta.get('selected_from_local_augmentation', False),
                                    'coverage': meta.get('coverage_reason'), 'mode': meta.get('mode'),
                                    'numerical_failure': bool(row['diagnostics'].get('numerical_failure')),
                                    'unknown': row['observation']['status'] in {'unknown', 'infrastructure_error'}}
                                if idx is not None and meta.get('forecasts', {}).get('objective') is not None:
                                    pr = meta['forecasts']['objective']; ob = row['observation']['objective']
                                    step.update(objective_mean=pr['mean'][idx], objective_variance=pr['variance'][idx])
                                    if ob['kind'] == 'exact':
                                        step['objective_error'] = pr['mean'][idx]-ob['value']
                                step['kernel_lengths'] = {n: h['selected'] for n, h in meta.get('heads', {}).items() if h}
                                steps.append(step)
                            history.append(row); count += 1
                        feasible = [r for r in history if usable(r)]
                        require(bool(feasible), 'Fixed review expects recorded incumbents, do not impute one')
                        best = min(feasible, key=lambda r: r['observation']['objective']['value'])
                        final = claim(f'{case}/{seed}/{method}/verification/5', 'incumbent_step_verification')
                        require(final == load(key+'verification.json') and final['candidate'] == best['candidate'], 'Verification design changed')
                        require(final['observation']['contract_id'] == plan['tasks'][case]['contracts']['5.0'], 'Wrong fine-step contract')
                        verified += usable(final)
                        summary = load(key+'summary.json'); require(summary == lookup[(case, seed, method)], 'Summary mismatch')
                        first = next(i for i, r in enumerate(history, 1) if usable(r))
                        require(summary['best_feasible'] == best['observation']['objective']['value'], 'Final objective mismatch')
                        require(summary['first_feasible_total_call'] == first, 'First feasible mismatch')
                        require(summary['step_verification']['record'] == final, 'Verification summary mismatch')
                        require(summary['step_verification']['verified_usable'] == usable(final), 'Verification label mismatch')
                        if run == 'pilot':
                            timing = load(key+'timing.json')
                            trials.append({'case': case, 'seed': seed, 'method': method,
                                'initial_kind': 'warm' if any(usable(r) for r in initial) else 'cold',
                                'first_feasible_online': max(0, first-len(initial)),
                                'best_feasible_Wh': summary['best_feasible'], 'online_usable': sum(usable(r) for r in obs),
                                'soc_aborts': sum(r['diagnostics'].get('termination_reason') == 'physical_soc_reserve_abort' for r in obs),
                                'best_candidate': best['candidate'], 'verified_usable': usable(final),
                                'optimization_seconds': timing['optimization_seconds'],
                                'proposal_seconds': sum(timing['proposal_seconds'])})
            require(set(settled) == claimed, 'Unclaimed calls')
            require(len(ids) == result['charged_model_calls'] == plan['budget']['maximum_model_calls'], 'Campaign count mismatch')
            require(dict(roles) == result['charged_by_role'], 'Role count mismatch')
            facts['runs'][run] = {'settled': len(ids), 'roles': dict(roles), 'trials': len(lookup), 'proposals': count,
                'verified_usable': verified, 'evidence_files': len(audit['evidence_hashes']), 'source_bindings': len(plan['sources'])}
    groups = defaultdict(list)
    for r in trials:
        groups[(r['case'], r['method'])].append(r)
    facts['groups'] = [{'case': c, 'method': m, 'n': len(g), 'cold_n': sum(r['initial_kind'] == 'cold' for r in g),
        'median_Wh': median(r['best_feasible_Wh'] for r in g), 'online_usable': sum(r['online_usable'] for r in g),
        'soc_aborts': sum(r['soc_aborts'] for r in g), 'median_seconds': median(r['optimization_seconds'] for r in g)}
        for (c, m), g in sorted(groups.items())]
    pairs = []
    for case in sorted({r['case'] for r in trials}):
        left = {r['seed']: r['best_feasible_Wh'] for r in groups[(case, 'ev_latent')]}
        for (c, m), g in sorted(groups.items()):
            if c == case and m != 'ev_latent':
                pairs.append({'case': case, 'left': 'ev_latent', 'right': m,
                    **pairing(left, {r['seed']: r['best_feasible_Wh'] for r in g})})
    facts['paired'] = pairs
    return facts, trials, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path); parser.add_argument('manifest', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); require(not args.output.exists(), 'Use a new derived directory')
    facts, trials, steps = review(args.archive, args.manifest); args.output.mkdir(parents=True)
    for name, value in [('facts', facts), ('trials', trials), ('steps', steps)]:
        (args.output/(name+'.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                                        allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
