"""PR71 numerical replay on PINNED PR70 states. No new online trajectories."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from pr69_evidence_verify import fingerprint, require, native_digest as digest
from pr71_protocol import PARENT, EVIDENCE, RESULT, ARM, RULES, plan, rule_id, summarize, verify_output
from pr71_source_identity import historical_source_bytes
from ra_lsbo.experiments.information_value_stage1.io import write_new, manifest


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, encoding='utf-8').strip()


def lock(root, sha):
    require(isinstance(sha, str) and len(sha) == 40 and all(c in '0123456789abcdef' for c in sha), 'Exact SHA required')
    require(git(root, 'rev-parse', 'HEAD') == sha and not git(root, 'status', '--porcelain'), 'Worktree identity/cleanliness failure')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s]


def run(evidence_root, output, expected_sha, profile):
    root = Path(__file__).resolve().parent
    evidence, out = Path(evidence_root).resolve(), Path(output).resolve()
    lock(root, expected_sha)
    lock(evidence, EVIDENCE)
    require(not out.exists() and all(out != p and p not in out.parents for p in (root, evidence)), 'NEW external output required')
    git(root, 'merge-base', '--is-ancestor', PARENT, expected_sha)
    require(all(s.startswith('A\t') for s in git(root, 'diff', '--name-status', PARENT, expected_sha).splitlines()), 'PR70 parent blobs changed')
    from pr70_protocol import verify as verify_pr70
    source = evidence/RESULT
    verified = verify_pr70(source)
    original_source = read(source/'source_receipt.json')
    saved_report = read(source/'report.json')
    require(saved_report['code_sha'] == PARENT and saved_report['profile'] == 'full', 'Wrong input campaign')
    spec = plan(profile)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'plan.json', spec)
    write_new(out/'source_receipt.json', {'audit_code_sha': expected_sha, 'input_result_sha': EVIDENCE,
        'parent_code_sha': PARENT, 'input_verification': verified, 'all_parent_blobs_unchanged': True})
    completed = []
    ledger = None
    started = time.perf_counter()
    try:
        from pr71_restore import GPLedger, restore, model_hash, no_new_truth_or_training
        from pr71_quadrature import score, parity, compare_forecasts
        from pr70_future_pool import NextPoolFactory
        from ra_lsbo.experiments.risk_budget63_runner import _case
        from ra_lsbo.experiments.core.protocol import BASE
        from ra_lsbo.partial_bo.records import DesignRecord
        from ra_lsbo.experiments.information_value_stage3.planner import one_step, shortlist
        ledger = GPLedger(out/'gp_refits.jsonl')
        with no_new_truth_or_training(ledger):
            for state in spec['states']:
                case, seed, i = state['case'], state['seed'], state['iteration']
                folder = source/'trials'/case/str(seed)/ARM
                initial = [DesignRecord.from_dict(r) for r in read(source/'shared_initial'/case/(str(seed)+'.json'))['records']]
                # Only the prefix before this proposal is supplied to inference.
                observed = lines(folder/'observations.jsonl')
                history = initial + [DesignRecord.from_dict(o['record']) for o in observed[:i]]
                p = lines(folder/'proposals.jsonl')[i]
                require(p['history_hash'] == fingerprint([r.to_dict() for r in history]), 'Prefix leakage/mismatch')
                meta, forecast = p['metadata'], p['metadata']['forecasts']
                saved = meta['information_value_decision']
                require(saved['depth'] == 2 and saved['fallback'] is None, 'Selected prefix has no complete source lookahead')
                # Exact original raw-byte hashes, NOT a blind contract-id override.
                with historical_source_bytes(root, original_source) as source_bytes:
                    problem = _case(case)
                ledger.context = {**state, 'rule': 'restore'}
                posterior, restoration = restore(problem, BASE, seed, initial, history, folder, meta, ledger)
                candidates = meta['pool']
                forecast_errors = compare_forecasts(posterior.predict(candidates), forecast)
                base = one_step(forecast, posterior.best)
                firsts = shortlist(forecast, base)
                require(firsts == saved['first_action_indices'], 'First-action screen changed')
                before = model_hash(posterior)
                row = {'state': state, 'history_hash': p['history_hash'], 'candidate_pool_hash': digest(candidates),
                    'best': posterior.best, 'restoration': restoration, 'forecast_abs_errors': forecast_errors,
                    'historical_source_bytes': source_bytes,
                    'first_action_indices': firsts, 'myopic_scores': base.tolist(),
                    'future_supports': saved['future_supports'], 'baseline_reproduced': False, 'rules': {}}
                for n, replicate in RULES:
                    rid = rule_id(n, replicate)
                    ledger.context = {**state, 'rule': rid}
                    factory = NextPoolFactory(posterior, candidates, problem, case, seed, i, history)
                    for index in firsts:
                        require(factory.support(index) == saved['future_supports'][str(index)], 'Future support changed')
                    r = score(forecast, posterior.best, factory, seed=seed, iteration=i, n=n, replicate=replicate)
                    if (n, replicate) == (4, 0):
                        row['baseline_reproduced'] = parity(r, saved)
                    require(model_hash(posterior) == before, 'Fantasy mutated current posterior')
                    row['rules'][rid] = r
                write_new(out/'states'/('%s_%s_%s.json' % (case, seed, i)), row)
                completed.append(row)
                print('PR71', case, seed, i, {k: (v['status'], v['selected_index']) for k,v in row['rules'].items()}, flush=True)
        require(verify_pr70(source) == verified, 'Source evidence changed during numerical replay')
        lock(root, expected_sha)
        lock(evidence, EVIDENCE)
        report = {'status': 'COMPLETE_FIXED_STATE_QUADRATURE_DIAGNOSTIC', 'audit_code_sha': expected_sha,
            'input_result_sha': EVIDENCE, 'parent_code_sha': PARENT, 'input_verification': verified,
            'profile': profile, 'states': completed, 'summary': summarize(completed),
            'gp_refits': ledger.summary(), 'new_GP_fits': ledger.started,
            'new_mathematical_evaluations': 0, 'new_VAE_fits': 0, 'aircraft_calls': 0, 'rcaide_calls': 0,
            'numerically_failed_rules': sum(v['status'] != 'ok' for r in completed for v in r['rules'].values()),
            'elapsed_seconds': time.perf_counter()-started, 'limits': spec['limits']}
        write_new(out/'report.json', report)
        manifest(out)
        verify_output(out)
        return report
    except BaseException as e:
        write_new(out/'failure.json', {'error_type': type(e).__name__, 'message': str(e),
            'completed_states': len(completed), 'gp_refits': None if ledger is None else ledger.summary(),
            'audit_code_sha': expected_sha, 'input_result_sha': EVIDENCE})
        if not (out/'manifest.json').exists():
            manifest(out)
        raise


def main():
    p = argparse.ArgumentParser(description='PR71 frozen-state quadrature diagnostic, zero evaluator calls')
    p.add_argument('--task', choices=('run', 'verify'), default='run')
    p.add_argument('--profile', choices=('ci', 'smoke', 'full'), default='smoke')
    p.add_argument('--expected-sha')
    p.add_argument('--evidence-root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--input', type=Path)
    a = p.parse_args()
    if a.task == 'verify':
        if a.input is None:
            p.error('--input required')
        result = verify_output(a.input)
    else:
        if a.output is None or a.expected_sha is None or a.evidence_root is None:
            p.error('--output, --expected-sha and --evidence-root required')
        r = run(a.evidence_root, a.output, a.expected_sha, a.profile)
        result = {k: r[k] for k in ('status', 'profile', 'gp_refits', 'numerically_failed_rules', 'new_mathematical_evaluations', 'new_VAE_fits')}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
