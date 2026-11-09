"""Phase C1: equal-call 2x2 risk/update pilot; standard library only."""
import hashlib
import json
import math

SCHEMA = 'c1_matched_online_learning_v1'
PARENT = '27e9724150961b44268bd6672dc41abe32b8455b'
TASKS = tuple('tradeoff_s%d' % i for i in range(4))
ARMS = ('ei_frozen', 'risk_ei_frozen', 'ei_updated', 'risk_ei_updated',
        'risk_only_updated', 'uniform_hash')
INITIAL, POOL = 48, 256
PROFILES = {
    'ci': dict(tasks=(TASKS[0], TASKS[3]), repeats=1, steps=2, block=810000),
    'smoke': dict(tasks=TASKS, repeats=1, steps=3, block=820000),
    'full': dict(tasks=TASKS, repeats=8, steps=24, block=830000),
}
CONTRASTS = (
    ('risk_ei_updated', 'risk_ei_frozen'),
    ('ei_updated', 'ei_frozen'),
    ('risk_ei_updated', 'ei_updated'),
    ('risk_ei_frozen', 'ei_frozen'),
    ('risk_ei_updated', 'risk_only_updated'),
    ('risk_ei_updated', 'uniform_hash'),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def plan(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown profile')
    c = PROFILES[profile]
    units = [dict(task=t, repeat=r, seed=c['block']+1000*TASKS.index(t)+32*r)
             for t in c['tasks'] for r in range(c['repeats'])]
    n, h = len(units), c['steps']
    return dict(schema=SCHEMA, profile=profile, units=units, arms=list(ARMS),
        initial=INITIAL, pool=POOL, steps=h,
        checkpoints=[k for k in (6, 12, 24) if k <= h] or [h],
        initial_calls=n*INITIAL, online_calls=n*len(ARMS)*h,
        total_math_calls=n*(INITIAL+len(ARMS)*h), trials=n*len(ARMS),
        max_gp_heads=n*(3+8*(h-1)), max_gp_kernels=8*n*(3+8*(h-1)),
        primary=dict(contrast=list(CONTRASTS[0]), stratum='initial_warm',
                     endpoint='paired final best feasible objective per task',
                     full_online_budget=24),
        contrasts=[list(p) for p in CONTRASTS],
        no_resampling_for_labels=True, retained_task_s4_s7=False,
        frozen='posterior fixed at initial data; live feasible incumbent still updates',
        cold='EI arms use same outcome-blind hash until incumbent/objective available',
        post_failure='registered model failure uses explicit hash fallback; no same-prefix fit retry',
        total_cost_per_arm=INITIAL+h, online_cost_per_arm=h,
        training_shared_once=True, same_design_in_two_arms_charged_twice=True,
        all_unselected_candidates_unevaluated=True,
        new_vae_fits=0, new_fantasy_fits=0, aircraft_calls=0, rcaide_calls=0)


def bill():
    return {k: 0 for k in ('math_started', 'math_settled', 'math_errors',
        'gp_head_started', 'gp_head_completed', 'gp_head_errors',
        'gp_kernel_started', 'gp_kernel_completed', 'gp_kernel_errors',
        'kernel_candidates_rejected', 'prediction_started', 'prediction_completed',
        'prediction_errors', 'vae_fits', 'fantasy_fits', 'aircraft_calls', 'rcaide_calls')}


def feasible(row):
    return all(row[k]['kind'] == 'exact' and row[k]['value'] <= 0
               for k in ('region', 'resource'))


def best(rows):
    vals = [r['objective']['value'] for r in rows
            if feasible(r) and r['objective']['kind'] == 'exact']
    return min(vals) if vals else None


def information(rows):
    return dict(rows=len(rows), exact_objectives=sum(r['objective']['kind']=='exact' for r in rows),
        feasible=sum(feasible(r) for r in rows),
        exact_region=sum(r['region']['kind']=='exact' for r in rows),
        interval_region=sum(r['region']['kind']=='interval' for r in rows),
        branches={str(b):sum(r['candidate']['branch']==b for r in rows) for b in range(4)})


def ei(m, v, incumbent):
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (m, v, incumbent)) or v < 0:
        raise ValueError('Invalid EI moments')
    d = incumbent-m
    if not v:
        return max(0., d)
    sd = math.sqrt(v); z = d/sd
    return max(0., d*.5*math.erfc(-z/math.sqrt(2.))
               + sd*math.exp(-.5*z*z)/math.sqrt(2.*math.pi))


def is_updated(arm):
    return arm in ('ei_updated', 'risk_ei_updated', 'risk_only_updated')


def select(arm, ids, available, prediction, incumbent, salt):
    """Only public posterior moments and current incumbent; no task or truth."""
    if arm not in ARMS or not ids or len(ids)!=len(set(ids)) or not available:
        raise ValueError('Invalid arm/pool')
    if any(type(i) is not int or not 0 <= i < len(ids) for i in available) or len(set(available))!=len(available):
        raise ValueError('Bad available indices')
    if any(not isinstance(i, str) or len(i)!=64 for i in ids):
        raise ValueError('Canonical candidate IDs required')
    if incumbent is not None and (type(incumbent) not in (int,float) or not math.isfinite(incumbent)):
        raise ValueError('Invalid incumbent')
    scores = [0.]*len(ids)
    mode = 'uniform_hash'
    if arm != 'uniform_hash':
        if prediction is None:
            mode = 'model_failure_hash'
        else:
            if prediction['candidate_ids'] != ids:
                raise ValueError('Misaligned predictions')
            p = prediction['probabilities']['joint']
            if len(p)!=len(ids) or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in p):
                raise ValueError('Invalid probabilities')
            if arm == 'risk_only_updated':
                scores = list(p); mode = 'risk_only'
            elif incumbent is None:
                mode = 'cold_shared_hash'
            elif prediction['objective'] is None:
                mode = 'no_objective_model_hash'
            else:
                o = prediction['objective']
                if len(o['mean'])!=len(ids) or len(o['variance'])!=len(ids):
                    raise ValueError('Objective shape mismatch')
                scores = [ei(m,v,incumbent) for m,v in zip(o['mean'],o['variance'])]
                if arm.startswith('risk_ei'):
                    scores = [a*b for a,b in zip(scores,p)]; mode='risk_ei'
                else:
                    mode='ei'
    tie = [digest(['c1_outcome_blind_tie_v1',salt,i]) for i in ids]
    selected = min(available, key=lambda i: (-scores[i],tie[i],ids[i]))
    return dict(mode=mode, scores=scores, selected_index=selected,
                selected_candidate_id=ids[selected], incumbent=incumbent)


def trial_result(task, repeat, arm, initial, observations, proposals):
    h = list(initial); curve=[best(h)]; labels=[information(h)]
    first = 0 if curve[0] is not None else None
    for j, r in enumerate(observations):
        h.append(r); curve.append(best(h)); labels.append(information(h))
        if first is None and curve[-1] is not None:
            first=j+1
    return dict(task=task,repeat=repeat,arm=arm,initial_best=curve[0],initial_warm=curve[0] is not None,
        best_curve=curve,information_curve=labels,first_feasible_call=first,
        candidate_path=[r['candidate_id'] for r in observations],
        modes=[p['decision']['mode'] for p in proposals],
        model_failure_calls=sum(p['decision']['mode']=='model_failure_hash' for p in proposals),
        online_calls=len(observations),final_best=curve[-1])


def outcome(a,b):
    if a is None or b is None:
        return 0 if a is None and b is None else (1 if b is None else -1)
    tol=1e-10*max(1.,abs(a),abs(b))
    return 1 if a < b-tol else (-1 if a>b+tol else 0)


def summarize(rows, spec):
    keyed={(r['task'],r['repeat'],r['arm']):r for r in rows}
    expected={(u['task'],u['repeat'],a) for u in spec['units'] for a in ARMS}
    if len(keyed)!=len(rows) or set(keyed)!=expected:
        raise ValueError('Duplicate/missing trial')
    result={}
    for task in dict.fromkeys(u['task'] for u in spec['units']):
        repeats=[u['repeat'] for u in spec['units'] if u['task']==task]
        for k in spec['checkpoints']:
            for stratum in ('all','initial_warm','initial_cold'):
                selected=[r for r in repeats if stratum=='all' or
                    keyed[task,r,ARMS[0]]['initial_warm']==(stratum=='initial_warm')]
                arms={}
                for arm in ARMS:
                    rr=[keyed[task,r,arm] for r in selected]
                    arms[arm]=dict(n=len(rr),final_feasible=sum(t['best_curve'][k] is not None for t in rr),
                        improved=sum(t['initial_best'] is not None and outcome(t['best_curve'][k],t['initial_best'])>0 for t in rr),
                        cold_discoveries=sum(not t['initial_warm'] and t['best_curve'][k] is not None for t in rr),
                        new_exact_objectives=sum(t['information_curve'][k]['exact_objectives']-t['information_curve'][0]['exact_objectives'] for t in rr),
                        online_nonfeasible=sum(k-(t['information_curve'][k]['feasible']-t['information_curve'][0]['feasible']) for t in rr),
                        model_failure_calls=sum(sum(m=='model_failure_hash' for m in t['modes'][:k]) for t in rr))
                pairs={}
                for a,b in CONTRASTS:
                    values=[];wlt=[0,0,0]
                    for r in selected:
                        x,y=keyed[task,r,a],keyed[task,r,b];v,w=x['best_curve'][k],y['best_curve'][k]
                        z=outcome(v,w); wlt[0 if z>0 else 1 if z<0 else 2]+=1
                        values.append(dict(repeat=r,a=v,b=w,delta=None if v is None or w is None else v-w,
                            new_exact_difference=x['information_curve'][k]['exact_objectives']-y['information_curve'][k]['exact_objectives'],
                            different_path=x['candidate_path'][:k]!=y['candidate_path'][:k]))
                    ds=[v['delta'] for v in values if v['delta'] is not None]
                    pairs[a+'_vs_'+b]=dict(wins_losses_ties=wlt,paired=values,
                        mean_delta=math.fsum(ds)/len(ds) if ds else None)
                result['%s/k%d/%s'%(task,k,stratum)]=dict(repeats=selected,arms=arms,comparisons=pairs)
    return result
