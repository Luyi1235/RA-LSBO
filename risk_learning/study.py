"""Offline supervised learning curves. No sequential proposals or aircraft path."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import platform
import math
import time
import traceback

from .core import HEADS, MODELS, SCHEMA, candidates, digest, label_counts, observed, plan, score, truth
from .io import append, read, write, seal, check_manifest
from .validation import metric_equal


def new_bill():
    return dict(math_started=0, math_settled=0, gp_head_started=0, gp_head_completed=0,
                gp_head_errors=0, gp_kernel_started=0, gp_kernel_completed=0,
                gp_kernel_errors=0, kernel_candidates_rejected=0,
                aircraft_calls=0, rcaide_calls=0, vae_fits=0, fantasy_fits=0)


def collect(task, seed, count, role, unit, root, bill):
    result = []
    for i, candidate in enumerate(candidates(seed, count)):
        call_id = '%s/%s/%04d' % (unit, role, i)
        before = dict(event='started', call_id=call_id, task=task, role=role,
                      candidate_id=digest(candidate), candidate=candidate)
        append(root/'evaluations.jsonl', before)
        bill['math_started'] += 1
        target = truth(task, candidate)
        packet = observed(candidate, target)
        result.append(dict(candidate=candidate, candidate_id=digest(candidate),
                           truth=target, observed=packet, call_id=call_id))
        append(root/'evaluations.jsonl', dict(event='settled', call_id=call_id, record_sha256=digest(result[-1])))
        bill['math_settled'] += 1
    return result


def summary(rows):
    grouped = {}
    for row in rows:
        key = '%s/n%s/%s' % (row['task'], row['n'], row['method'])
        grouped.setdefault(key, []).append(row)
    out = {}
    for key, items in grouped.items():
        ok = [r for r in items if r['status'] == 'OK']
        out[key] = dict(planned_repeats=len(items), valid_repeats=len(ok),
                        failed_repeats=len(items)-len(ok),
                        joint_brier_per_repeat=[dict(repeat=r['repeat'], value=r['metrics']['all']['joint']['brier']) for r in ok],
                        no_pooled_cross_task_mean=True)
    return out


def run(root, profile, source, predictor=None):
    from .native import predict
    predictor = predictor or predict
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError('New write-once output folder required; do not repeat finished runs')
    root.mkdir(parents=True)
    bill, rows = new_bill(), []
    start = time.perf_counter()
    spec = plan(profile)
    write(root/'plan.json', spec)
    write(root/'source_receipt.json', source)
    write(root/'environment.json', dict(python=platform.python_version(), platform=platform.platform()))
    try:
        for unit in spec['units']:
            name = '%s/r%d' % (unit['task'], unit['repeat'])
            folder = root/'datasets'/name
            train = collect(unit['task'], unit['train_seed'], max(spec['sizes']), 'train', name, root, bill)
            test = collect(unit['task'], unit['test_seed'], spec['test_count'], 'test', name, root, bill)
            if {r['candidate_id'] for r in train} & {r['candidate_id'] for r in test}:
                raise ValueError('Training/test design overlap')
            write(folder/'train.json', train)
            write(folder/'test.json', test)
            for n in spec['sizes']:
                # Holdout outcomes never enter the fit interface. Privileged training
                # margins enter ONLY the explicitly named diagnostic comparator.
                public = [deepcopy(r['observed']) for r in train[:n]]
                coverage = label_counts(public)
                for method in MODELS:
                    filename = '%s/n%d/%s.json' % (name, n, method)
                    append(root/'progress.jsonl', dict(event='fit_started', file=filename))
                    kwargs = dict(privileged=[deepcopy(r['truth']) for r in train[:n]]) if method == 'privileged_exact' else {}
                    before = deepcopy(bill)
                    guesses, receipt = predictor(public, [deepcopy(r['candidate']) for r in test],
                                                  method, unit['train_seed'], bill, **kwargs)
                    row = dict(schema=SCHEMA, task=unit['task'], repeat=unit['repeat'], n=n, method=method,
                               train_prefix_sha256=digest(train[:n]), test_set_sha256=digest(test),
                               labels=coverage, receipt=receipt, predictions=guesses,
                               status='OK' if guesses is not None else 'MODEL_NUMERICAL_FAILURE',
                               bill_delta={k: bill[k]-before[k] for k in bill},
                               metrics=score(test, guesses) if guesses is not None else None)
                    write(root/'fits'/filename, row)
                    rows.append(row)
                    append(root/'progress.jsonl', dict(event='fit_settled', file=filename, status=row['status']))
        expected = spec['total_math_calls']
        if bill['math_started'] != expected or bill['math_settled'] != expected:
            raise ValueError('Mathematical call bill changed')
        failures = sum(r['status'] != 'OK' for r in rows)
        report = dict(schema=SCHEMA, code_sha=source['code_sha'], profile=profile,
                      status='COMPLETE_LEARNABILITY' if failures == 0 else 'COMPLETE_WITH_MODEL_FAILURES',
                      bill=bill, model_rows=len(rows), failed_model_rows=failures,
                      summary=summary(rows), elapsed_seconds=time.perf_counter()-start,
                      limits=['Phase A supervised diagnostic, not online optimization or a new algorithm.',
                              'Full-margin comparator has privileged training information, not a deployable baseline.',
                              'Heldout evaluation calls and training calls are separately charged.',
                              'All tasks in this phase are development instances; no cross-task transfer claim.',
                              'No safety guarantee, no automatic model promotion, no target-dependent resampling.'])
        write(root/'report.json', report)
        seal(root)
        return report
    except Exception as error:
        write(root/'failure.json', dict(status='STOPPED_ERROR', bill=bill,
              error=type(error).__name__+': '+str(error), traceback=traceback.format_exc(),
              completed_model_rows=len(rows)))
        seal(root)
        raise


def verify(root, expected_sha=None):
    from .native import training_payload  # stdlib at module level; no native model import/fits
    root = Path(root)
    files = check_manifest(root)
    if (root/'failure.json').exists():
        raise ValueError('Failed run retained, not a completed experiment')
    spec, report = read(root/'plan.json'), read(root/'report.json')
    source = read(root/'source_receipt.json')
    if spec != plan(spec['profile']) or report['profile'] != spec['profile']:
        raise ValueError('Frozen study plan differs')
    if report['code_sha'] != source['code_sha'] or (expected_sha and source['code_sha'] != expected_sha):
        raise ValueError('Evidence code identity differs')
    records, rows, expected_fit_files = {}, [], set()
    for unit in spec['units']:
        name = '%s/r%d' % (unit['task'], unit['repeat'])
        train, test = (read(root/'datasets'/name/(role+'.json')) for role in ('train', 'test'))
        for role, data, seed, count in [('train',train,unit['train_seed'],max(spec['sizes'])),
                                        ('test',test,unit['test_seed'],spec['test_count'])]:
            # Regenerate input designs only, never call the truth evaluator.
            if [r['candidate'] for r in data] != candidates(seed, count):
                raise ValueError('Outcome-blind dataset design stream changed')
            for i,r in enumerate(data):
                cid = '%s/%s/%04d' % (name,role,i)
                if (r['candidate_id'] != digest(r['candidate']) or r['call_id'] != cid
                        or r['observed'] != observed(r['candidate'],r['truth']) or cid in records):
                    raise ValueError('Observation mask, candidate identity or call ID changed')
                if r['truth']['feasible'] != all(r['truth'][h] <= 0 for h in HEADS):
                    raise ValueError('Feasibility label differs from archived margins')
                records[cid] = r
        if {r['candidate_id'] for r in train} & {r['candidate_id'] for r in test}:
            raise ValueError('Train/test overlap')
        for n in spec['sizes']:
            public = [r['observed'] for r in train[:n]]
            for method in MODELS:
                filename = '%s/n%d/%s.json' % (name,n,method)
                expected_fit_files.add('fits/'+filename)
                row = read(root/'fits'/filename)
                if any(row[k] != v for k,v in dict(schema=SCHEMA,task=unit['task'],repeat=unit['repeat'],n=n,method=method,
                        train_prefix_sha256=digest(train[:n]),test_set_sha256=digest(test),labels=label_counts(public)).items()):
                    raise ValueError('Fit contract/learning-curve pairing changed')
                privileged = [r['truth'] for r in train[:n]] if method == 'privileged_exact' else None
                payload = training_payload(public,method,privileged)
                receipt = row['receipt']
                if (receipt['training_input_sha256'] != digest(payload) or receipt['training_rows'] != n
                    or receipt['privileged_training'] != (method=='privileged_exact') or receipt['objective_labels_used']):
                    raise ValueError('Training information contract differs')
                if receipt['query_ids'] != [r['candidate_id'] for r in test]:
                    raise ValueError('Query receipt changed')
                if row['status']=='OK':
                    if not metric_equal(score(test,row['predictions']), row['metrics']):
                        raise ValueError('Prediction metrics cannot be reproduced')
                    if method=='joint_prevalence':
                        from .native import predict
                        guess,_ = predict(public,[r['candidate'] for r in test],method,0,new_bill())
                        if guess != row['predictions']:
                            raise ValueError('Training-only prevalence baseline changed')
                elif row['status']!='MODEL_NUMERICAL_FAILURE' or row['predictions'] is not None or row['metrics'] is not None:
                    raise ValueError('Unregistered fit outcome')
                delta = row['bill_delta']
                if method == 'joint_prevalence':
                    if any(delta.values()) or receipt['heads']:
                        raise ValueError('Constant baseline has fabricated fits')
                else:
                    h = delta['gp_head_started']
                    if h not in (1,2) or delta['gp_kernel_started'] != 8*h:
                        raise ValueError('Native 4x2 grid construction count changed')
                    if row['status'] == 'OK':
                        if h != 2 or delta['gp_head_completed'] != 2 or delta['gp_head_errors'] != 0 or set(receipt['heads']) != set(HEADS):
                            raise ValueError('Successful native head accounting changed')
                        gaussian = row['predictions'].get('gaussian', {})
                        if set(gaussian) != set(HEADS):
                            raise ValueError('Missing native predictive moments')
                        for head,moments in gaussian.items():
                            noise = 1. if method == 'native_sign' else 0.
                            if moments['observation_noise_variance'] != noise or len(moments['mean']) != len(test) or len(moments['variance']) != len(test):
                                raise ValueError('Predictive noise/shape changed')
                            for m,v,p in zip(moments['mean'],moments['variance'],row['predictions']['probabilities'][head]):
                                if not math.isfinite(m) or not math.isfinite(v) or v<0:
                                    raise ValueError('Invalid Gaussian moments')
                                q = .5*math.erfc(m/math.sqrt(2*(v+noise))) if v+noise>0 else float(m<=0)
                                if abs(q-p)>1e-12:
                                    raise ValueError('Probabilities do not match native Gaussian moments')
                        pa = row['predictions']['probabilities']
                        if any(abs(a*b-j)>1e-12 for a,b,j in zip(pa['region'],pa['resource'],pa['joint'])):
                            raise ValueError('Joint-probability contract changed')
                        for diag in receipt['heads'].values():
                            if len(diag['candidates']) != 8 or diag['selected']['status'] != 'ok':
                                raise ValueError('Native kernel selection evidence missing')
                    elif delta['gp_head_errors'] != 1 or delta['gp_head_completed'] != h-1:
                        raise ValueError('Failed native head accounting changed')
                rows.append(row)
    if {p.relative_to(root).as_posix() for p in (root/'fits').rglob('*.json')} != expected_fit_files:
        raise ValueError('Unexpected/missing model rows')
    events = [__import__('json').loads(s) for s in (root/'evaluations.jsonl').read_text(encoding='utf-8').splitlines()]
    started, settled = {}, {}
    for event in events:
        cid = event['call_id']
        if cid not in records:
            raise ValueError('Unknown mathematical call')
        if event['event']=='started':
            if cid in started or event['candidate']!=records[cid]['candidate'] or event['candidate_id']!=records[cid]['candidate_id']:
                raise ValueError('Duplicate or mismatched started call')
            started[cid] = event
        elif event['event']=='settled':
            if cid not in started or cid in settled or event['record_sha256']!=digest(records[cid]):
                raise ValueError('Unstarted, duplicate or mismatched settlement')
            settled[cid]=event
        else:
            raise ValueError('Unknown call event')
    if set(started)!=set(records) or set(settled)!=set(records) or len(records)!=spec['total_math_calls']:
        raise ValueError('Unsettled call bill')
    reconstructed = new_bill()
    for row in rows:
        if set(row['bill_delta']) != set(reconstructed) or any(type(v) is not int or v<0 for v in row['bill_delta'].values()):
            raise ValueError('Invalid GP bill')
        for k,v in row['bill_delta'].items():
            reconstructed[k]+=v
    reconstructed.update(math_started=len(records),math_settled=len(records))
    if (reconstructed!=report['bill'] or reconstructed['gp_kernel_started']!=reconstructed['gp_kernel_completed']+reconstructed['gp_kernel_errors']
        or reconstructed['gp_head_started']!=reconstructed['gp_head_completed']+reconstructed['gp_head_errors']
        or any(reconstructed[k] for k in ('aircraft_calls','rcaide_calls','vae_fits','fantasy_fits'))):
        raise ValueError('Calculation ledger mismatch')
    failures=sum(r['status']!='OK' for r in rows)
    status='COMPLETE_LEARNABILITY' if not failures else 'COMPLETE_WITH_MODEL_FAILURES'
    if (report['model_rows']!=len(rows) or len(rows)!=spec['model_rows'] or report['failed_model_rows']!=failures
        or report['status']!=status or report['summary']!=summary(rows)):
        raise ValueError('Incomplete or changed summary')
    return dict(status='VERIFIED_LEARNABILITY', files=files, code_sha=report['code_sha'],
                model_rows=len(rows), failed_model_rows=failures, historical_bill=reconstructed,
                new_math_calls=0,new_gp_fits=0,new_vae_fits=0)
