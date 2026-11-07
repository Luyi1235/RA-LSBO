"""Frozen Phase B1: same-data objective decisions, not sequential BO."""
import hashlib
import json
import math

SCHEMA = 'risk_objective_decision_value_b1_v1'
PARENT = '634ff13c07d11e4df4748199806726a3eb14fd70'
TASKS = ('legacy_branin_sparse', 'legacy_rosenbrock_sparse',
         'tradeoff_s0', 'tradeoff_s1', 'tradeoff_s2', 'tradeoff_s3')
POLICIES = ('uniform_hash', 'risk_only', 'objective_ei', 'risk_ei')
BUDGETS = (1, 2, 4, 6, 12)
PROFILES = {
    'ci': dict(tasks=(TASKS[0], TASKS[2]), repeats=1, sizes=(12, 48), pool=32, block=781000),
    'smoke': dict(tasks=TASKS, repeats=1, sizes=(12, 48), pool=64, block=782000),
    'full': dict(tasks=TASKS, repeats=8, sizes=(12, 48, 192), pool=256, block=783000),
}


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def plan(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown frozen profile')
    c = PROFILES[profile]
    units = [dict(task=t, repeat=r, train_seed=c['block']+100*TASKS.index(t)+2*r,
                  pool_seed=c['block']+100*TASKS.index(t)+2*r+1)
             for t in c['tasks'] for r in range(c['repeats'])]
    n = len(units)
    return dict(schema=SCHEMA, profile=profile, units=units, sizes=list(c['sizes']),
                pool_count=c['pool'], budgets=list(BUDGETS), policies=list(POLICIES),
                train_calls=n*max(c['sizes']), audit_pool_calls=n*c['pool'],
                total_math_calls=n*(max(c['sizes'])+c['pool']),
                model_states=n*len(c['sizes']), scored_rows=n*len(c['sizes'])*len(POLICIES)*len(BUDGETS),
                maximum_gp_heads=3*n*len(c['sizes']),
                primary='risk_ei vs objective_ei in initially-feasible matched states',
                primary_n=48, primary_k=6, primary_tasks=list(TASKS[2:]),
                primary_endpoint='paired final best feasible objective, per task; W/L/T and raw paired differences',
                sampling='IID outcome-blind; no rejection based on labels or incumbent',
                model_update_after_selection=False, pool_truth_after_all_unit_rankings=True,
                online_BO=False, reserved_tasks_evaluated=False,
                costs='actual=train_max+full_audit_pool; nominal_per_policy=n+k (not extra physical calls)')


def finite(values, *, nonnegative=False, probability=False):
    for x in values:
        if type(x) not in (int, float) or not math.isfinite(x):
            raise ValueError('Finite numeric values required')
        if (nonnegative and x < 0) or (probability and not 0 <= x <= 1):
            raise ValueError('Value out of domain')


def expected_improvement(mean, variance, incumbent):
    finite([mean, variance, incumbent])
    if variance < 0:
        raise ValueError('Negative variance')
    delta = incumbent-mean
    if variance == 0:
        return max(0., delta)
    sd = math.sqrt(variance)
    z = delta/sd
    value = delta*.5*math.erfc(-z/math.sqrt(2.)) + sd*math.exp(-.5*z*z)/math.sqrt(2.*math.pi)
    if not math.isfinite(value):
        raise ValueError('Nonfinite EI')
    return max(0., value)


def incumbent_from_observed(rows):
    """Use only observable exact objectives and both known feasible margins."""
    vals = [r['objective']['value'] for r in rows
            if r['objective']['kind']=='exact'
            and all(r[h]['kind']=='exact' and r[h]['value']<=0 for h in ('region','resource'))]
    finite(vals)
    return min(vals) if vals else None


def objective_payload(rows):
    return [dict(candidate=r['candidate'], candidate_id=r['candidate_id'], objective=r['objective'])
            for r in rows]


def decisions(ids, probabilities, objective, incumbent, salt):
    """No observations, target labels or task formula accepted by this interface."""
    if not ids or len(set(ids))!=len(ids) or any(not isinstance(i,str) or len(i)!=64
            or any(c not in '0123456789abcdef' for c in i) for i in ids):
        raise ValueError('Unique canonical SHA256 query IDs required')
    if len(probabilities)!=len(ids):
        raise ValueError('Risk shape mismatch')
    finite(probabilities, probability=True)
    tie = {i:digest(['phase_b1_tie_v1',salt,i]) for i in ids}
    ei = [0.]*len(ids)
    mode = 'shared_uniform_no_feasible_incumbent'
    if incumbent is not None:
        if objective is None or len(objective['mean'])!=len(ids) or len(objective['variance'])!=len(ids):
            raise ValueError('Warm decision needs aligned objective moments')
        ei = [expected_improvement(m,v,incumbent) for m,v in zip(objective['mean'],objective['variance'])]
        mode = 'shared_expected_improvement'
    scores = dict(uniform_hash=[0.]*len(ids), risk_only=list(probabilities),
                  objective_ei=ei, risk_ei=[e*p for e,p in zip(ei,probabilities)])
    rankings = {p:sorted(range(len(ids)), key=lambda i:(-scores[p][i],tie[ids[i]],ids[i])) for p in POLICIES}
    return dict(mode=mode, incumbent=incumbent, candidate_ids=ids, scores=scores, rankings=rankings,
                positive_ei_candidates=sum(e>0 for e in ei),
                objective_proxy_shared=True, risk_proxy_shared=True,
                no_truth_or_target_input=True)


def objective_tolerance(value):
    return 1e-10*max(1.,abs(value))


def assess(pool, choice, k, incumbent, scale):
    """Post-seal scoring only. Pool truth is never passed to decisions()."""
    if (type(k) is not int or not 1<=k<=len(pool) or len(choice)!=len(pool)
            or any(type(i) is not int for i in choice) or set(choice)!=set(range(len(pool)))):
        raise ValueError('Not a full permutation or invalid budget')
    finite([scale], nonnegative=True)
    if scale==0:
        raise ValueError('Positive training-only objective scale required')
    selected = [pool[i] for i in choice[:k]]
    vals = [r['truth']['objective'] for r in selected if r['truth']['feasible']]
    available = [r['truth']['objective'] for r in pool if r['truth']['feasible']]
    best_selected = min(vals) if vals else None
    candidates = vals+([] if incumbent is None else [incumbent])
    best = min(candidates) if candidates else None
    oracle_vals = available+([] if incumbent is None else [incumbent])
    oracle = min(oracle_vals) if oracle_vals else None
    gain = None if incumbent is None else max(0.,incumbent-best)
    opportunity = None if incumbent is None else max(0.,incumbent-oracle)
    useful = opportunity is not None and opportunity>objective_tolerance(incumbent)
    return dict(k=k, selected_ids=[r['candidate_id'] for r in selected],
                selected_feasible=len(vals), selected_nonfeasible=k-len(vals),
                best_selected=best_selected, best_after=best, initial_best=incumbent,
                initial_warm=incumbent is not None, delivered_feasible=best is not None,
                new_discovery=incumbent is None and best is not None,
                improvement=gain, standardized_improvement=None if gain is None else gain/scale,
                improved=None if gain is None else gain>objective_tolerance(incumbent),
                pool_best=oracle, pool_opportunity=opportunity,
                opportunity_fraction=gain/opportunity if useful else None,
                oracle_is_posthoc_audit=True)


def compare(a,b):
    x,y=a['best_after'],b['best_after']
    if x is None or y is None:
        return 0 if x is None and y is None else (1 if y is None else -1)
    tol=1e-10*max(1.,abs(x),abs(y))
    return 1 if x<y-tol else (-1 if x>y+tol else 0)


def summarize(rows):
    """Matched task/n/k strata. No raw objective average across tasks."""
    groups={}
    for row in rows:
        for k in BUDGETS:
            key='%s/n%d/k%d'%(row['task'],row['n'],k)
            groups.setdefault(key,[]).append(row)
    result={}
    for key,items in groups.items():
        k=int(key.rsplit('k',1)[1]); out={}
        for stratum in ('all','initial_warm','initial_cold'):
            valid=[r for r in items if r['status']=='OK' and
                   (stratum=='all' or (r['decision']['incumbent'] is not None)==(stratum=='initial_warm'))]
            wlt=[0,0,0]; improvements={p:0 for p in POLICIES}; selected={p:0 for p in POLICIES}
            deltas=[]; pairs=[]; opportunities=0; changes=0
            for r in valid:
                a,b=(r['outcomes'][p][str(k)] for p in ('risk_ei','objective_ei'))
                c=compare(a,b); wlt[0 if c>0 else 1 if c<0 else 2]+=1
                diff=None
                if a['best_after'] is not None and b['best_after'] is not None:
                    diff=a['best_after']-b['best_after']; deltas.append(diff)
                opportunities+=int(a['opportunity_fraction'] is not None)
                changes+=int(a['selected_ids']!=b['selected_ids'])
                pairs.append(dict(repeat=r['repeat'],risk_minus_no_risk=diff,
                                  risk_best=a['best_after'],objective_best=b['best_after']))
                for p in POLICIES:
                    v=r['outcomes'][p][str(k)]
                    improvements[p]+=int(v['improved'] is True)
                    selected[p]+=v['selected_feasible']
            out[stratum]=dict(valid_pairs=len(valid),risk_vs_objective_wlt=wlt,
                improved_runs=improvements,selected_feasible=selected,
                opportunity_positive_pairs=opportunities,changed_ordered_prefixes=changes,
                paired_objectives=pairs,
                objective_delta_risk_minus_no_risk=math.fsum(deltas)/len(deltas) if deltas else None)
        out['failed_model_states']=sum(r['status']!='OK' for r in items)
        out['planned_repeats']=len(items)
        result[key]=out
    return result
