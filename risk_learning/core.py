"""Dependency-free, frozen design, feedback contracts and scoring for Phase A."""
import hashlib
import json
import math
import random

SCHEMA = 'risk_learnability_phase_a_v1'
PARENT = '491dbf6f1fd340864f5b5332bc2b75d75e9a6a9e'
SIZES = (12, 24, 48, 96, 192)
TASKS = ('legacy_branin_sparse', 'legacy_rosenbrock_sparse',
         'tradeoff_s0', 'tradeoff_s1', 'tradeoff_s2', 'tradeoff_s3')
MODELS = ('joint_prevalence', 'native_censored', 'native_sign', 'privileged_exact')
HEADS = ('region', 'resource')
PROFILES = {
    'ci': dict(tasks=(TASKS[0], TASKS[2]), repeats=1, sizes=(12, 24), test=64, block=751000),
    'smoke': dict(tasks=TASKS, repeats=1, sizes=(12, 24), test=128, block=752000),
    'full': dict(tasks=TASKS, repeats=3, sizes=SIZES, test=512, block=753000),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def plan(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown frozen profile')
    cfg = PROFILES[profile]
    units = []
    for task in cfg['tasks']:
        for repeat in range(cfg['repeats']):
            seed = cfg['block'] + 100 * TASKS.index(task) + 2 * repeat
            units.append(dict(task=task, repeat=repeat, train_seed=seed, test_seed=seed + 1))
    total = len(units) * (max(cfg['sizes']) + cfg['test'])
    return dict(schema=SCHEMA, profile=profile, units=units, sizes=list(cfg['sizes']),
                test_count=cfg['test'], models=list(MODELS), total_math_calls=total,
                train_math_calls=len(units)*max(cfg['sizes']),
                test_math_calls=len(units)*cfg['test'],
                model_rows=len(units)*len(cfg['sizes'])*len(MODELS),
                gp_head_attempts=len(units)*len(cfg['sizes'])*3*2,
                reserved_task_ids=['tradeoff_s4', 'tradeoff_s5', 'tradeoff_s6', 'tradeoff_s7'],
                sampling='IID uniform over four configurations and active continuous coordinates',
                online_optimization=False, training_data_pooled_across_tasks=False,
                test_truth_for_training=False, aircraft_calls=0, rcaide_calls=0, vae_fits=0)


def canonical(candidate):
    if set(candidate) != {'branch', 'x', 'y', 'trim'}:
        raise ValueError('Candidate schema changed')
    b = candidate['branch']
    if type(b) is not int or b not in range(4):
        raise ValueError('Invalid branch')
    result = {'branch': b}
    for key in ('x', 'y', 'trim'):
        v = candidate[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError('Nonfinite/out-of-domain coordinate')
        result[key] = float(v)
    if b == 0 and result['trim'] != .5:
        raise ValueError('Inactive trim must be canonical 0.5')
    return result


def candidates(seed, count):
    if type(seed) is not int or type(count) is not int or count < 1:
        raise ValueError('Integer seed and positive count required')
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        b = rng.randrange(4)
        out.append(dict(branch=b, x=rng.random(), y=rng.random(),
                        trim=rng.random() if b else .5))
    if len({digest(c) for c in out}) != count:
        raise ValueError('Duplicate random design; do not resample silently')
    return out


def native_candidate(candidate):
    c = canonical(candidate)
    return dict(continuous={k: c[k] for k in ('x', 'y', 'trim')},
                discrete=dict(architecture='augmented' if c['branch'] else 'basic',
                              module=c['branch'] or 1))


def features(candidate):
    """Exact existing PrefixFeatures(latent=False) hierarchy layout, unit bounds."""
    c = canonical(candidate)
    b = c['branch']
    active = float(b != 0)
    return [c['x'], c['y'], c['trim']*active, 1., 1., active,
            1.-active, active, 1.,
            float(b == 1), float(b == 2), float(b == 3), active]


def tradeoff_geometry(scenario, branch, trim):
    centers = ((.24, .24), (.72, .24), (.26, .73), (.73, .71))
    radii = ((.24, .20), (.18, .16), (.20, .18), (.16, .14))
    cx, cy = centers[branch]
    cx += .015*math.sin(scenario+branch) + (.05*(trim-.5) if branch else 0.)
    cy += .015*math.cos(2*scenario+branch)
    return cx, cy, radii[branch][0], radii[branch][1], .25*scenario+.35*branch


def truth(task, candidate):
    """Analytic truth. ONLY dataset generator/tests may call; never a learner."""
    if task not in TASKS:
        raise ValueError('Task not released for Phase A (reserved tasks stay sealed)')
    c = canonical(candidate)
    b, x, y, t = (c[k] for k in ('branch', 'x', 'y', 'trim'))
    if task.startswith('legacy_'):
        qx = x+.035*(b-2) if b else x
        cx, cy = (.68, .25) if task == TASKS[0] else (.72, .70)
        region = ((qx-cx)/.16)**2 + ((y-cy)/.20)**2 - 1.
        resource = t+.12*b-1.05 if b else -.2
        if task == TASKS[0]:
            X, Y = 15*qx-5, 15*y
            f = (Y-5.1/(4*math.pi**2)*X*X+5/math.pi*X-6)**2
            f += 10*(1-1/(8*math.pi))*math.cos(X)+10
            if b:
                f += 2*(t-.25*b)**2
        else:
            v = [4*qx-2, 4*y-2] + ([4*t-2] if b else [])
            f = sum(100*(v[i+1]-v[i]**2)**2+(1-v[i])**2 for i in range(len(v)-1))
    else:
        s = int(task[-1])
        cx, cy, rx, ry, angle = tradeoff_geometry(s, b, t)
        dx, dy = x-cx, y-cy
        u = (math.cos(angle)*dx+math.sin(angle)*dy)/rx
        v = (-math.sin(angle)*dx+math.cos(angle)*dy)/ry
        region = u*u+v*v-1.
        resource = t+.08*b+.06*(x+y)-.90 if b else -.2
        f = .7*(b-s)**2 + (u-.8)**2+.7*v*v + (.1*(t-.35)**2 if b else 0.)
    return dict(region=region, resource=resource, objective=f,
                feasible=region <= 0 and resource <= 0)


def optimum_witness(task, branch):
    """Transparent synthetic construction witness; inaccessible to fit() interface."""
    if task not in TASKS[2:]:
        raise ValueError('Witness only for tradeoff tasks')
    t = .35 if branch else .5
    cx, cy, rx, _, a = tradeoff_geometry(int(task[-1]), branch, t)
    return canonical(dict(branch=branch, x=cx+.8*rx*math.cos(a),
                          y=cy+.8*rx*math.sin(a), trim=t))


def observed(candidate, target):
    """Observable packet: region>0 loses its magnitude and objective, not resource."""
    c = canonical(candidate)
    region = (dict(kind='exact', value=target['region']) if target['region'] <= 0
              else dict(kind='interval', lower=0., lower_inclusive=False))
    objective = (dict(kind='exact', value=target['objective']) if target['region'] <= 0
                 else dict(kind='missing', reason='controlled_region_abort'))
    return dict(candidate=c, candidate_id=digest(c), region=region,
                resource=dict(kind='exact', value=target['resource']), objective=objective)


def known_label(row, head):
    o = row[head]
    if o['kind'] == 'exact':
        return o['value'] <= 0
    if o['kind'] == 'interval' and o.get('lower') == 0 and o.get('lower_inclusive') is False:
        return False
    raise ValueError('Unsupported observable label; no imputation')


def label_counts(rows):
    result = {}
    for head in (*HEADS, 'objective'):
        counts = {k: sum(r[head]['kind'] == k for r in rows) for k in ('exact', 'interval', 'missing')}
        if head != 'objective':
            positives = sum(known_label(r, head) for r in rows)
            counts.update(feasible=positives, violated=len(rows)-positives,
                          both_classes=0 < positives < len(rows))
        result[head] = counts
    result['joint_feasible'] = sum(all(known_label(r, h) for h in HEADS) for r in rows)
    result['branches'] = {str(b): sum(r['candidate']['branch'] == b for r in rows) for b in range(4)}
    return result


def binary_metrics(labels, probabilities):
    if len(labels) != len(probabilities):
        raise ValueError('Prediction/label alignment differs')
    if any(type(y) is not bool for y in labels):
        raise ValueError('Boolean labels required')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid probability')
    n = len(labels)
    if n == 0:
        return dict(n=0, positives=0, negatives=0, brier=None, log_loss=None,
                    auroc=None, precision_at_10pct=None, recall_at_10pct=None,
                    false_feasible_rate=None, false_infeasible_rate=None, calibration_bins=[])
    positives = sum(labels)
    order = sorted(range(n), key=lambda i: (-probabilities[i], i))
    k = max(1, int(math.ceil(.1*n)))
    top = sum(labels[i] for i in order[:k])
    pos = [p for y, p in zip(labels, probabilities) if y]
    neg = [p for y, p in zip(labels, probabilities) if not y]
    auc = (sum(float(p > q)+.5*float(p == q) for p in pos for q in neg)/(len(pos)*len(neg))
           if pos and neg else None)
    loss = sum(-math.log(max(min(p if y else 1-p, 1-1e-12), 1e-12))
               for y, p in zip(labels, probabilities))/n
    bins = []
    for j in range(5):
        ids = [i for i,p in enumerate(probabilities) if min(int(p*5),4) == j]
        bins.append(dict(bin=j, n=len(ids), mean_probability=sum(probabilities[i] for i in ids)/len(ids) if ids else None,
                         observed_fraction=sum(labels[i] for i in ids)/len(ids) if ids else None))
    return dict(n=n, positives=positives, negatives=n-positives,
                brier=sum((p-int(y))**2 for y,p in zip(labels,probabilities))/n,
                log_loss=loss, auroc=auc, precision_at_10pct=top/k,
                recall_at_10pct=top/positives if positives else None,
                false_feasible_rate=sum(p>=.5 and not y for y,p in zip(labels,probabilities))/(n-positives) if n>positives else None,
                false_infeasible_rate=sum(p<.5 and y for y,p in zip(labels,probabilities))/positives if positives else None,
                calibration_bins=bins)


def score(test, predictions):
    if [r['candidate_id'] for r in test] != predictions['candidate_ids']:
        raise ValueError('Heldout identity/order changed')
    n = len(test)
    strata = {'all': list(range(n)),
              'region_boundary': [i for i,r in enumerate(test) if abs(r['truth']['region']) <= .25]}
    strata.update({'branch_'+str(b): [i for i,r in enumerate(test) if r['candidate']['branch'] == b]
                   for b in range(4)})
    result = {}
    for name, ids in strata.items():
        part = {}
        for head in (*HEADS, 'joint'):
            labels = [(test[i]['truth']['feasible'] if head == 'joint'
                       else test[i]['truth'][head] <= 0) for i in ids]
            part[head] = binary_metrics(labels, [predictions['probabilities'][head][i] for i in ids])
            means = predictions.get('margin_mean', {}).get(head)
            if means is not None:
                if len(means) != n or any(not math.isfinite(x) for x in means):
                    raise ValueError('Bad margin mean')
                part[head]['margin_rmse'] = (math.sqrt(sum((means[i]-test[i]['truth'][head])**2 for i in ids)/len(ids))
                                              if ids else None)
        result[name] = part
    return result
