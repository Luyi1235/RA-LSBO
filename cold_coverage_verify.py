"""Standard-library raw ledger, cold-phase and matched-path reconciliation."""
import json
import math
from collections import Counter
from pathlib import Path
from cold_coverage_core import (CONTROL, TREATMENT, LEGACY, ARMS, PERIOD,
    plan, require, scheduled, first_feasible, summarize)


def check(meta, candidate, history, iteration, steps, arm):
    from recourse_value_protocol import check_receipt
    from pr69_evidence_verify import best, native_digest
    cold = best(history) is None
    require(meta['cold_start'] is cold, 'Wrong pre-outcome feasibility phase')
    decision = check_receipt(meta, candidate, iteration, steps,
                             CONTROL if arm == TREATMENT else arm)
    reason = meta.get('coverage_reason')
    if arm == TREATMENT:
        require(meta.get('cold_coverage_ablation') == {'arm': TREATMENT, 'cold_before': cold,
            'scope': 'cold', 'period': PERIOD, 'underlying_model_selector': CONTROL,
            'all_real_feedback_preserved': True, 'new_failure_threshold': False}, 'Wrong intervention receipt')
    else:
        require('cold_coverage_ablation' not in meta, 'Intervention leaked into control')
    if arm in (CONTROL, TREATMENT):
        due = arm == TREATMENT and scheduled(cold, iteration)
        if reason != 'model_failure':
            require(meta['coverage_scope'] == ('cold' if arm == TREATMENT else 'none'), 'Wrong coverage scope')
            require(meta['coverage_scheduled'] is due, 'Wrong inherited schedule')
        if due:
            require(meta['mode'] == reason == 'scheduled_coverage' and meta['posterior_fit_skipped'] is True,
                    'Scheduled cold action did not use the frozen coverage branch')
        elif reason != 'model_failure':
            require(reason is None or (arm == TREATMENT and cold and reason == 'eligible_at_most_one'
                    and meta['model_eligible_count'] <= 1), 'Unregistered coverage action')
        if reason:
            require('information_value_decision' not in meta and not decision['forecast_available'],
                    'Coverage retained a false selected-model receipt')
        else:
            iv = meta['information_value_decision']
            require(iv['depth'] == 1 and not iv['fantasies'] and iv['selected_index'] == iv['myopic_index'],
                    'Primary arms must use single-step scores')
    require(decision['fantasy_head_fits_started'] == decision['fantasy_head_fits_completed'] == 0,
            'No fantasy computations in this experiment')
    if reason:
        counts = Counter(native_digest(r['candidate']['discrete']) for r in history)
        pool = meta['pool']; groups = {native_digest(c['discrete']) for c in pool}
        chosen = native_digest(candidate['discrete'])
        least = min(counts[g] for g in groups)
        rule = meta['coverage_rule']
        require(chosen in groups and counts[chosen] == least and rule['available_branches'] == len(groups)
                and rule['selected_branch_visits'] == least
                and rule['rule'] == 'least_visited_available_branch_then_uniform_pool', 'Coverage branch rule differs')
    decision.update(cold_before=cold, coverage_reason=reason,
        scheduled_coverage=meta['mode'] == 'scheduled_coverage',
        degenerate_coverage=meta['mode'] == 'degenerate_gate_coverage',
        model_failure=reason == 'model_failure')
    return decision


def verify(output):
    from pr69_evidence_verify import fingerprint, feasible, best
    from ra_lsbo.experiments.information_value_stage1.io import verify as verify_manifest
    root = Path(output); checked = verify_manifest(root)
    def read(p): return json.loads((root/p).read_text(encoding='utf8'))
    def lines(p): return [json.loads(s) for s in (root/p).read_text(encoding='utf8').splitlines() if s.strip()]
    spec, report = read('plan.json'), read('report.json')
    require(spec == plan(spec['profile']) and report['code_sha'] == read('source_receipt.json')['head_sha'], 'Protocol/source mismatch')
    status = 'COMPLETE_COLD_COVERAGE_PILOT' if spec['profile'] == 'full' else 'COMPLETE_'+spec['profile'].upper()+'_TECHNICAL_ONLY'
    require(report['status'] == status, 'Incomplete campaign')
    events, count = lines('evaluations.jsonl'), spec['total_calls']
    require(len(events) == 2*count, 'Incomplete ledger')
    settled = {}
    for i in range(count):
        a, b = events[2*i:2*i+2]
        require(a['event'] == 'started' and b['event'] == 'settled' and a['ticket'] == b['ticket'] == i, 'Ticket mismatch')
        feasible(b['record'])
        require(a['candidate'] == b['record']['candidate'] and a['candidate_id'] == b['candidate_id'] == fingerprint(a['candidate'])
                and a['purpose'] == b['purpose'], 'Ledger association mismatch')
        settled[i] = b
    used, rows = set(), []
    for case in spec['cases']:
        for seed in spec['seeds']:
            init = read(Path('shared_initial')/case/(str(seed)+'.json'))
            require(len(init['records']) == len(init['tickets']) == 12 and init['hash'] == fingerprint(init['records']), 'Initial mismatch')
            for j, (t, rec) in enumerate(zip(init['tickets'], init['records'])):
                require(t not in used and settled[t]['record'] == rec and settled[t]['purpose'] ==
                        {'phase':'initial','case':case,'seed':seed,'index':j}, 'Initial ledger differs')
                used.add(t)
            for arm in ARMS:
                folder = Path('trials')/case/str(seed)/arm
                props, obs, row = lines(folder/'proposals.jsonl'), lines(folder/'observations.jsonl'), read(folder/'result.json')
                require(len(props) == len(obs) == spec['steps'], 'Online count mismatch')
                history = list(init['records']); curve = [best(history)]; path = []; ds = []
                for i, (p, o) in enumerate(zip(props, obs)):
                    rec, t = o['record'], o['ticket']
                    require(t not in used and settled[t]['record'] == rec and settled[t]['purpose'] ==
                            {'phase':'online','case':case,'seed':seed,'arm':arm,'iteration':i}, 'Online ledger differs')
                    used.add(t)
                    require(p['iteration'] == o['iteration'] == i and p['candidate'] == rec['candidate']
                        and p['candidate_hash'] == o['candidate_hash'] == fingerprint(rec['candidate'])
                        and p['history_hash'] == fingerprint(history), 'History/outcome mismatch')
                    require(len(p['raw_pool']) == 128 and len({fingerprint(c) for c in p['raw_pool']}) == 128
                        and p['pool_hash'] == fingerprint(p['raw_pool']) and rec['candidate'] in p['raw_pool']
                        and not any(rec['candidate'] == r['candidate'] for r in history), 'Pool/repeat mismatch')
                    require(p['metadata']['pool'] == p['protected_pool'] and
                        all(c in p['raw_pool'] for c in p['protected_pool']) and
                        not any(c == r['candidate'] for c in p['protected_pool'] for r in history), 'Protected pool differs')
                    d = check(p['metadata'], rec['candidate'], history, i, spec['steps'], arm)
                    d.update(iteration=i, candidate_hash=fingerprint(rec['candidate']), nonfeasible=int(not feasible(rec)))
                    ds.append(d); history.append(rec); curve.append(best(history)); path.append(fingerprint(rec['candidate']))
                require(row['case'] == case and row['seed'] == seed and row['arm'] == arm
                    and row['initial_hash'] == init['hash'] and row['initial_best'] == curve[0]
                    and row['final_best'] == curve[-1] and row['curve'] == curve
                    and row['first_feasible_call'] == first_feasible(curve)
                    and row['path_hash'] == fingerprint(path) and row['decisions'] == ds
                    and row['online_calls'] == spec['steps'], 'Trial summary mismatch')
                for field, term in (('nonfeasible','nonfeasible'),('scheduled_coverage_calls','scheduled_coverage'),
                    ('degenerate_coverage_calls','degenerate_coverage'),('model_failure_calls','model_failure'),('cold_decisions','cold_before')):
                    require(row[field] == sum(d[term] for d in ds), 'Counter differs: '+field)
                require(row['model_free_calls'] == sum(not d['forecast_available'] for d in ds)
                    and math.isclose(row['proposal_seconds'], sum(p['proposal_seconds'] for p in props), abs_tol=1e-8), 'Timing/coverage mismatch')
                rows.append(row)
    require(used == set(range(count)) and report['trials'] == rows and report['summary'] == summarize(rows, spec), 'Campaign mismatch')
    require(report['actual_calls_started'] == report['actual_calls_settled'] == count
        and report['aircraft_calls'] == report['rcaide_calls'] == report['fantasy_fits'] == 0, 'Bill differs')
    return {**checked, 'status':'VERIFIED_COLD_COVERAGE', 'trials':len(rows),
        'actual_calls_started':count, 'actual_calls_settled':count,
        'initially_cold_groups':sum(r['initial_best'] is None for r in rows if r['arm'] == CONTROL),
        'initially_warm_matched_paths_verified':sum(r['initial_best'] is not None for r in rows if r['arm'] == CONTROL),
        'new_evaluations_in_this_verification':0, 'gp_fits_in_this_verification':0}
