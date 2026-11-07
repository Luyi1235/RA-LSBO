"""New-data fixed scoring experiment; all rankings sealed before pool truth."""
from copy import deepcopy
from pathlib import Path
import time
import traceback

from b1_core import (SCHEMA, POLICIES, BUDGETS, plan, digest, decisions,
                     incumbent_from_observed, assess, summarize)
from risk_learning.core import candidates, label_counts, score
from risk_learning.io import write, read, append, seal, file_sha
from risk_learning.study import new_bill, collect


def evaluate_states(root, task, repeat, sizes, pool):
    rows=[]
    name='%s/r%d'%(task,repeat)
    for n in sizes:
        row=read(root/'states'/name/('n%d.json'%n))
        result=dict(task=task,repeat=repeat,n=n,status=row['status'],outcomes={})
        if row['status']=='OK':
            d=row['decision']
            result['decision']=d
            for p in POLICIES:
                result['outcomes'][p]={str(k):dict(assess(pool,d['rankings'][p],k,d['incumbent'],row['objective_scale']),
                                           nominal_total_evaluations=n+k) for k in BUDGETS}
            result['pool_risk_metrics']=score(pool,row['predictions']['risk'])
        write(root/'scores'/name/('n%d.json'%n),result)
        rows.append(result)
    return rows


def run(root, profile, source, fitter=None):
    if fitter is None:
        from b1_native import fit_shared
        fitter=fit_shared
    root=Path(root).resolve()
    if root.exists():
        raise FileExistsError('Write-once output; do not repeat an existing experiment')
    root.mkdir(parents=True)
    spec=plan(profile); bill=new_bill(); rows=[]; started=time.perf_counter()
    write(root/'plan.json',spec); write(root/'source_receipt.json',source)
    import platform
    write(root/'environment.json',dict(python=platform.python_version(),platform=platform.platform()))
    timeline=root/'timeline.jsonl'
    try:
        for u in spec['units']:
            name='%s/r%d'%(u['task'],u['repeat']); data=root/'datasets'/name
            append(timeline,dict(event='train_started',unit=name))
            train=collect(u['task'],u['train_seed'],max(spec['sizes']),'train',name,root,bill)
            write(data/'train.json',train)
            queries=candidates(u['pool_seed'],spec['pool_count'])
            if {digest(c) for c in queries}&{r['candidate_id'] for r in train}:
                raise ValueError('Training and decision pool overlap')
            write(data/'queries.json',queries)
            for n in spec['sizes']:
                public=[deepcopy(r['observed']) for r in train[:n]]
                snapshot=digest(public); before=dict(bill)
                append(timeline,dict(event='fit_started',unit=name,n=n))
                predictions,receipt=fitter(public,deepcopy(queries),u['train_seed'],bill)
                if snapshot!=digest(public):
                    raise ValueError('Learner mutated observable training data')
                best=incumbent_from_observed(public)
                decision=None; scale=1.
                if predictions is not None:
                    ids=[digest(q) for q in queries]
                    if predictions['risk']['candidate_ids']!=ids:
                        raise ValueError('Prediction query identity mismatch')
                    decision=decisions(ids,predictions['risk']['probabilities']['joint'],
                                       predictions['objective'],best,name)
                    diag=receipt['objective']['head']
                    scale=diag['scale'] if diag else 1.
                row=dict(schema=SCHEMA,task=u['task'],repeat=u['repeat'],n=n,
                         status='OK' if predictions is not None else 'MODEL_NUMERICAL_FAILURE',
                         train_prefix_sha256=digest(train[:n]),observable_prefix_sha256=snapshot,
                         query_sha256=digest(queries),labels=label_counts(public),
                         predictions=predictions,receipt=receipt,decision=decision,objective_scale=scale,
                         bill_delta={k:bill[k]-before[k] for k in bill})
                path=root/'states'/name/('n%d.json'%n)
                write(path,row)
                append(timeline,dict(event='decision_sealed',unit=name,n=n,
                                      path=path.relative_to(root).as_posix(),sha256=file_sha(path)))
            # Every policy at every n is fixed before any pool target is evaluated.
            append(timeline,dict(event='pool_truth_started',unit=name))
            pool=collect(u['task'],u['pool_seed'],spec['pool_count'],'pool',name,root,bill)
            if [r['candidate'] for r in pool]!=queries:
                raise ValueError('Evaluated pool differs from sealed query designs')
            write(data/'pool.json',pool)
            rows.extend(evaluate_states(root,u['task'],u['repeat'],spec['sizes'],pool))
            append(timeline,dict(event='unit_scored',unit=name))
        if bill['math_started']!=spec['total_math_calls'] or bill['math_settled']!=spec['total_math_calls']:
            raise ValueError('True evaluation bill mismatch')
        failures=sum(r['status']!='OK' for r in rows)
        report=dict(schema=SCHEMA,code_sha=source['code_sha'],profile=profile,
                    status='COMPLETE_FIXED_OBJECTIVE_DECISIONS' if failures==0 else 'COMPLETE_WITH_MODEL_FAILURES',
                    model_states=len(rows),failed_states=failures,bill=bill,summary=summarize(rows),
                    elapsed_seconds=time.perf_counter()-started,
                    limits=['Fresh data on known development tasks, not independent task generalization.',
                            'Fixed marginal EI rankings, not adaptive sequential or joint batch EI.',
                            'No risk vs learned risk is matched only in initially-feasible states.',
                            'Cold objective policies share uniform fallback and are reported separately.',
                            'Full-pool truth is post-seal audit information; all calls are charged.',
                            'n+k is nominal deployment cost, not additional calls on top of the paid pool.',
                            'No aircraft, VAE, fantasy or safety guarantee.'])
        write(root/'report.json',report); seal(root)
        return report
    except Exception as error:
        write(root/'failure.json',dict(status='STOPPED_ERROR',error=str(error),
                    traceback=traceback.format_exc(),bill=bill,completed_states=len(rows)))
        seal(root)
        raise
