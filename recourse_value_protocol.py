"""Frozen prospective recourse-scoring pilot and model-free evidence verifier."""
import json
import math
import statistics
from pathlib import Path
from recourse_value_core import ADAPTIVE, FIXED, decompose, number, require, winner

PARENT = '932f7202cf81890a108e37266bbb639affba0ae7'
CASES = ('branin_ordinary', 'branin_sparse', 'rosenbrock_ordinary', 'rosenbrock_sparse')
ARMS = ('soft_gp_eligible', 'cei_myopic', ADAPTIVE, FIXED)
PAIRS = ((ADAPTIVE, FIXED), (ADAPTIVE, 'cei_myopic'), (FIXED, 'cei_myopic'),
         (ADAPTIVE, 'soft_gp_eligible'), ('cei_myopic', 'soft_gp_eligible'))


def plan(profile):
    require(profile in ('ci', 'smoke', 'full'), 'Unregistered profile')
    cases = (CASES[0], CASES[3]) if profile == 'ci' else CASES
    seeds = (999090,) if profile == 'ci' else ((999091,) if profile == 'smoke' else tuple(range(999101, 999109)))
    steps = 6 if profile == 'full' else 2
    groups = len(cases)*len(seeds)
    return {'schema': 'observable_recourse_scoring_pilot_v1', 'profile': profile,
            'cases': list(cases), 'seeds': list(seeds), 'arms': list(ARMS),
            'initial': 12, 'steps': steps, 'pool': 128,
            'initial_calls': groups*12, 'online_calls': groups*len(ARMS)*steps,
            'total_calls': groups*(12+len(ARMS)*steps),
            'first_actions': 4, 'nodes_per_region_branch': 4,
            'primary_pair': list(PAIRS[0]),
            'isolated_change': 'E[max second value] versus max E[second value] in first-action scoring',
            'same_fantasy_updates_for_both_two_step_arms': True,
            'actual_next_call_replans_in_both_arms': True,
            'soft_gp_legacy_internal_horizon': 12,
            'real_feedback_always_full': True, 'equal_true_calls_not_equal_wall_time': True,
            'early_stopping': False, 'exploratory_not_confirmatory': True,
            'aircraft_calls': 0, 'rcaide_calls': 0}


def check_receipt(meta, candidate, iteration, steps, arm):
    from ra_lsbo.experiments.information_value_stage3.runner import _decision
    from pr69_evidence_verify import native_digest as digest
    d = _decision(meta, candidate)
    d.update(scoring_disagreement=None, own_myopic_override=None, selected_recourse_premium=None)
    iv = meta.get('information_value_decision')
    if arm == 'soft_gp_eligible':
        require(iv is None and d['fantasy_head_fits_started'] == 0, 'Legacy arm changed')
        return d
    if iv is None:
        require(meta.get('coverage_reason') is not None, 'No forecast without registered coverage')
        return d
    pool = meta['pool']; idx = iv['selected_index']
    require(type(idx) is int and 0 <= idx < len(pool) and pool[idx] == candidate
            and iv['candidate_pool_hash'] == digest(pool), 'Actual candidate binding differs')
    require(iv['arm'] == arm and iv['remaining_actual_calls'] == steps-iteration,
            'Wrong arm/remaining budget')
    require(iv['quadrature_nodes_per_region_branch'] == 4, 'Node rule changed')
    d['own_myopic_override'] = idx != iv['myopic_index']
    if arm == 'cei_myopic' or iteration == steps-1:
        require(iv['depth'] == 1 and idx == iv['myopic_index'] and not iv['fantasies']
                and d['fantasy_head_fits_started'] == 0, 'Final/myopic decision not single-step')
        return d
    require(iv['depth'] == 2 and iv['condition_on_abort_interval'] is True
            and iv['actual_feedback_always_full'] is True
            and iv['fixed_second_is_scoring_only'] is True, 'Feedback/commitment contract changed')
    firsts, supports = iv['first_action_indices'], iv['future_supports']
    require(set(supports) == {str(i) for i in firsts}, 'Future root roster mismatch')
    require(iv['selected_future_support'] == supports[str(idx)], 'Selected future support mismatch')
    for i in firsts:
        s = supports[str(i)]; points = s['model_candidates']
        require(s['first_candidate_sha256'] == digest(pool[i]) and s['iteration'] == iteration+1
                and s['generation_uses_observations'] is False and points
                and s['model_pool_sha256'] == digest(points)
                and pool[i] not in points and len({digest(c) for c in points}) == len(points), 'Invalid next support')
    cold = iv['criterion'] == 'first_feasible_probability'
    require(cold or iv['criterion'] == 'expected_feasible_objective_improvement', 'Unknown criterion')
    for row in iv['fantasies']:
        i, obs = row['first_index'], row['observation']
        require(i in firsts and type(obs['aborted']) is bool, 'Invalid outcome identity')
        require(obs['resource']['kind'] == 'exact', 'Missing observable resource')
        resource = number(obs['resource']['value'])
        if obs['aborted']:
            require(obs['region'] == {'kind': 'interval', 'lower': 0., 'lower_inclusive': False}
                    and obs['objective'] == {'kind': 'missing'}, 'Hidden abort information')
        else:
            require(obs['region']['kind'] == 'exact' and number(obs['region']['value']) <= 0, 'Invalid completion')
            require(obs['objective'] == {'kind': 'missing'} if cold else
                    obs['objective']['kind'] == 'exact' and math.isfinite(number(obs['objective']['value'])), 'Objective contract differs')
        terminal = cold and not obs['aborted'] and resource <= 0
        values = row['future_values']
        require(type(values) is list and len(values) == len(supports[str(i)]['model_candidates'])
                and all(number(v) >= 0 for v in values), 'Invalid full continuation vector')
        require(row['conditioned'] is (not terminal) and row['abort_interval_used'] == obs['aborted'], 'Update masking differs')
        if terminal:
            require(all(v == 0 for v in values) and row['second_index'] is None
                    and row['second_candidate_sha256'] is None, 'Cold terminal must have no second action')
        else:
            second = winner(dict(enumerate(values)))
            require(row['second_index'] == second and row['second_candidate_sha256'] ==
                    digest(supports[str(i)]['model_candidates'][second]), 'Adaptive per-outcome action differs')
        require(row['continuation'] == max(values), 'Scenario maximum differs')
    if iv['fallback']:
        require(iv['fallback'] == 'fantasy_numerical_failure_full_myopic'
                and idx == iv['myopic_index'] and iv['scores'] == iv['myopic_scores']
                and iv['recourse_decomposition'] is None, 'Partial numerical fantasies used')
        return d
    computed = decompose(iv['myopic_scores'], firsts, iv['fantasies'],
                         {i: len(supports[str(i)]['model_candidates']) for i in firsts})
    require(computed == iv['recourse_decomposition'], 'Recourse decomposition does not reconstruct')
    which = 'adaptive' if arm == ADAPTIVE else 'fixed'
    require(idx == computed[which+'_index'], 'Wrong root winner')
    scores = [None]*len(pool)
    for i in firsts:
        scores[i] = computed['roots'][str(i)][which+'_score']
    require(iv['scores'] == scores and meta['scores'] == scores, 'Scoring ablation differs')
    require(d['fantasy_head_fits_started'] == d['fantasy_head_fits_completed'], 'Unreported fantasy fit failure')
    d.update(scoring_disagreement=computed['same_state_root_choice_differs'],
             selected_recourse_premium=computed['roots'][str(idx)]['recourse_premium'])
    return d


def summarize(rows, spec):
    from ra_lsbo.experiments.information_value_stage3.protocol import compare
    indexed = {(r['case'], r['seed'], r['arm']): r for r in rows}
    expected = {(c,s,a) for c in spec['cases'] for s in spec['seeds'] for a in ARMS}
    require(len(rows) == len(indexed) and set(indexed) == expected, 'Duplicate/incomplete roster')
    result = {}
    for c in spec['cases']:
        for s in spec['seeds']:
            require(len({indexed[c,s,a]['initial_hash'] for a in ARMS}) == 1, 'Initial evidence mismatch')
        item = {'arms': {}, 'pairs': {}}
        for a in ARMS:
            group = [indexed[c,s,a] for s in spec['seeds']]
            vals = [r['final_best'] for r in group if r['final_best'] is not None]
            ds = [d for r in group for d in r['decisions']]
            item['arms'][a] = {'trials': len(group), 'final_feasible': len(vals),
                'cold_discoveries': sum(r['initial_best'] is None and r['final_best'] is not None for r in group),
                'conditional_objective_mean': statistics.mean(vals) if vals else None,
                'own_myopic_overrides': sum(d['own_myopic_override'] is True for d in ds),
                'same_state_scoring_disagreements': sum(d['scoring_disagreement'] is True for d in ds),
                **{k: sum(r[k] for r in group) for k in ('online_calls','nonfeasible','proposal_seconds',
                    'fantasy_head_fits_started','fantasy_head_fits_completed','fantasy_fallbacks',
                    'model_free_calls','future_pool_checks')}}
        for left,right in PAIRS:
            wlt, details, changed = [0,0,0], [], 0
            for s in spec['seeds']:
                a,b = indexed[c,s,left],indexed[c,s,right]
                cmp = compare(a['final_best'],b['final_best'])
                wlt[0 if cmp == 1 else 1 if cmp == -1 else 2] += 1
                delta = a['final_best']-b['final_best'] if a['final_best'] is not None and b['final_best'] is not None else None
                details.append({'seed':s,'outcome':cmp,'objective_delta':delta})
                changed += a['path_hash'] != b['path_hash']
            item['pairs'][left+'_vs_'+right] = {'wins_losses_ties':wlt,'changed_paths':changed,'per_seed':details}
        result[c] = item
    return result


def verify(output):
    from pr69_evidence_verify import fingerprint, feasible, best
    from pr70_protocol import check_transition
    from ra_lsbo.experiments.information_value_stage1.io import verify as verify_manifest
    root=Path(output); checked=verify_manifest(root)
    def read(p): return json.loads((root/p).read_text(encoding='utf8'))
    def lines(p): return [json.loads(s) for s in (root/p).read_text(encoding='utf8').splitlines() if s.strip()]
    spec,report=read('plan.json'),read('report.json')
    require(spec==plan(spec['profile']), 'Plan changed')
    require(report['code_sha']==read('source_receipt.json')['head_sha'], 'Source identity differs')
    expected_status = 'COMPLETE_RECOURSE_SCORING_PILOT' if spec['profile']=='full' else 'COMPLETE_'+spec['profile'].upper()+'_TECHNICAL_ONLY'
    require(report['status']==expected_status, 'Not a complete campaign')
    events,count=lines('evaluations.jsonl'),spec['total_calls']
    require(len(events)==2*count, 'Incomplete ledger')
    settled={}
    for i in range(count):
        a,b=events[2*i:2*i+2]
        require(a['ticket']==b['ticket']==i and a['event']=='started' and b['event']=='settled', 'Ticket mismatch')
        feasible(b['record'])
        require(a['candidate']==b['record']['candidate'] and a['candidate_id']==b['candidate_id']==fingerprint(a['candidate'])
                and a['purpose']==b['purpose'], 'Ledger binding mismatch')
        settled[i]=b
    used,rows=set(),[]
    for case in spec['cases']:
        for seed in spec['seeds']:
            init=read(Path('shared_initial')/case/(str(seed)+'.json'))
            require(len(init['records'])==len(init['tickets'])==12 and init['hash']==fingerprint(init['records']), 'Initial mismatch')
            for j,(t,rec) in enumerate(zip(init['tickets'],init['records'])):
                require(t not in used and settled[t]['record']==rec and settled[t]['purpose']==
                        {'phase':'initial','case':case,'seed':seed,'index':j}, 'Initial ledger mismatch')
                used.add(t)
            for arm in ARMS:
                folder=Path('trials')/case/str(seed)/arm
                props,obs,row=lines(folder/'proposals.jsonl'),lines(folder/'observations.jsonl'),read(folder/'result.json')
                require(len(props)==len(obs)==spec['steps'], 'Wrong online count')
                history=list(init['records']);curve=[best(history)];path=[];ds=[];prev=None;checks=0
                for i,(p,o) in enumerate(zip(props,obs)):
                    rec,t=o['record'],o['ticket']
                    require(t not in used and settled[t]['record']==rec and settled[t]['purpose']==
                            {'phase':'online','case':case,'seed':seed,'arm':arm,'iteration':i}, 'Online ledger mismatch')
                    used.add(t)
                    require(p['iteration']==o['iteration']==i and p['candidate']==rec['candidate']
                            and p['candidate_hash']==o['candidate_hash']==fingerprint(rec['candidate'])
                            and p['history_hash']==fingerprint(history), 'Prefix/candidate binding differs')
                    require(len(p['raw_pool'])==128 and len({fingerprint(c) for c in p['raw_pool']})==128
                            and fingerprint(p['raw_pool'])==p['pool_hash'] and rec['candidate'] in p['raw_pool']
                            and not any(rec['candidate']==r['candidate'] for r in history), 'Pool/repeat mismatch')
                    require(p['metadata']['pool']==p['protected_pool'], 'Model support mismatch')
                    require(all(c in p['raw_pool'] for c in p['protected_pool'])
                            and not any(c==r['candidate'] for c in p['protected_pool'] for r in history), 'Protected support differs')
                    checks+=check_transition(prev,p['raw_pool'],p['protected_pool'],i)
                    d=check_receipt(p['metadata'],rec['candidate'],i,spec['steps'],arm)
                    d.update(iteration=i,candidate_hash=fingerprint(rec['candidate']),nonfeasible=int(not feasible(rec)))
                    ds.append(d);prev=p['metadata'].get('information_value_decision',{}).get('selected_future_support')
                    history.append(rec);curve.append(best(history));path.append(fingerprint(rec['candidate']))
                require(row['case']==case and row['seed']==seed and row['arm']==arm and row['initial_hash']==init['hash']
                        and row['curve']==curve and row['initial_best']==curve[0] and row['final_best']==curve[-1]
                        and row['decisions']==ds and row['path_hash']==fingerprint(path)
                        and row['online_calls']==spec['steps'] and row['future_pool_checks']==checks, 'Trial summary differs')
                for field, term in (('nonfeasible','nonfeasible'),('fantasy_fallbacks','fantasy_fallback'),
                                    ('fantasy_head_fits_started','fantasy_head_fits_started'),
                                    ('fantasy_head_fits_completed','fantasy_head_fits_completed')):
                    require(row[field]==sum(d[term] for d in ds), 'Trial counter differs: '+field)
                require(row['model_free_calls']==sum(not d['forecast_available'] for d in ds)
                        and math.isclose(row['proposal_seconds'],sum(p['proposal_seconds'] for p in props),abs_tol=1e-8), 'Time/coverage differs')
                rows.append(row)
    require(used==set(range(count)) and report['trials']==rows and report['summary']==summarize(rows,spec), 'Campaign does not reconcile')
    require(report['actual_calls_started']==report['actual_calls_settled']==count
            and report['aircraft_calls']==report['rcaide_calls']==0, 'Bill mismatch')
    return {**checked,'trials':len(rows),'actual_calls_started':count,'actual_calls_settled':count,
            'initial_calls':spec['initial_calls'],'online_calls':spec['online_calls'],
            'future_pool_checks':sum(r['future_pool_checks'] for r in rows),
            'new_evaluations_in_this_verification':0,'gp_fits_in_this_verification':0}
