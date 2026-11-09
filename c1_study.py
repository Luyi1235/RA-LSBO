"""C1 selected-point-only online evaluator with causal evidence journals."""
from copy import deepcopy
from pathlib import Path
import platform
import time
import traceback
from unittest.mock import patch
from c1_protocol import (SCHEMA,ARMS,INITIAL,bill as new_bill,plan,digest,best,
                         select,is_updated,information,trial_result,summarize)
from risk_learning.core import candidates,observed,canonical
from risk_learning.io import write,append,seal


def run(root,profile,source,fitter=None,evaluator=None):
    if fitter is None:
        from c1_native import fit
        fitter=fit
    if evaluator is None:
        from risk_learning.core import truth
        evaluator=truth
    root=Path(root).resolve()
    if root.exists():
        raise FileExistsError('Write-once output; do not rerun an existing attempt')
    root.mkdir(parents=True)
    spec=plan(profile);bill=new_bill();all_trials=[];start=time.perf_counter()
    write(root/'plan.json',spec);write(root/'source_receipt.json',source)
    write(root/'environment.json',dict(python=platform.python_version(),platform=platform.platform()))
    events=root/'events.jsonl'; evaluations=root/'evaluations.jsonl'
    math_forbidden=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError('Truth is forbidden inside model/selector'))
    def save(rel,value,event):
        # Immutable append records keep full evidence without thousands of tiny files.
        unit='/'.join(rel.split('/')[:3]); journal=root/unit/'records.jsonl'
        journal.parent.mkdir(parents=True,exist_ok=True)
        append(journal,dict(ref=rel,value=value))
        append(events,dict(event=event,ref=rel,record_sha256=digest(value)))
    def evaluate(task,candidate,call_id,proposal=None):
        c=canonical(candidate);cid=digest(c)
        ticket=dict(event='started',call_id=call_id,task=task,candidate_id=cid,candidate=c,proposal=proposal)
        append(evaluations,ticket);append(events,dict(event='evaluation_started',call_id=call_id))
        bill['math_started']+=1
        try:
            target=evaluator(task,deepcopy(c))
            if (set(target)!={'region','resource','objective','feasible'} or type(target['feasible'])is not bool
                    or target['feasible']!=(target['region']<=0 and target['resource']<=0)):
                raise ValueError('Untrusted evaluator packet')
            r=dict(call_id=call_id,candidate=c,candidate_id=cid,truth=target,observed=observed(c,target))
            digest(r)
        except Exception as error:
            bill['math_errors']+=1
            append(evaluations,dict(event='error',call_id=call_id,kind=type(error).__name__,message=str(error)))
            raise
        bill['math_settled']+=1
        append(evaluations,dict(event='settled',call_id=call_id,record_sha256=digest(r)))
        append(events,dict(event='evaluation_settled',call_id=call_id,record_sha256=digest(r)))
        return r
    def fit_at(public,seed,risk_only,rel):
        before=dict(bill);snapshot=digest(public)
        append(events,dict(event='fit_started',ref=rel,training_sha256=snapshot))
        payload=deepcopy(public)
        with patch('risk_learning.core.truth',math_forbidden):
            model,receipt=fitter(payload,seed,bill,risk_only=risk_only)
        if digest(payload)!=snapshot or digest(public)!=snapshot or receipt['training_sha256']!=snapshot:
            raise ValueError('Model training prefix differs')
        row=dict(receipt=receipt,bill_delta={k:bill[k]-before[k] for k in bill})
        save(rel,row,'fit_sealed')
        return model,receipt
    def predict_at(model,queries,model_ref,rel):
        before=dict(bill);started=time.perf_counter()
        if model is None:
            prediction,error=None,dict(kind='FIT_FAILED',message='No posterior available')
        else:
            query_copy=deepcopy(queries)
            with patch('risk_learning.core.truth',math_forbidden):
                prediction,error=model.predict(query_copy,bill)
            if query_copy!=queries:
                raise ValueError('Prediction mutated query coordinates')
        row=dict(model_ref=model_ref,query_sha256=digest(queries),prediction=prediction,error=error,
            elapsed_seconds=time.perf_counter()-started,bill_delta={k:bill[k]-before[k] for k in bill})
        save(rel,row,'prediction_sealed')
        return prediction
    try:
        for u in spec['units']:
            name='%s/r%d'%(u['task'],u['repeat']);prefix='units/'+name
            initial=[evaluate(u['task'],c,name+'/initial/%03d'%i)
                     for i,c in enumerate(candidates(u['seed'],INITIAL))]
            save(prefix+'/initial.json',initial,'initial_sealed')
            init_public=[r['observed'] for r in initial]
            initial_ref=prefix+'/models/initial.json'
            frozen,_=fit_at(init_public,u['seed'],False,initial_ref)
            histories={a:deepcopy(init_public) for a in ARMS}
            online={a:[] for a in ARMS};proposals={a:[] for a in ARMS}
            for t in range(spec['steps']):
                queries=candidates(u['seed']+1+t,spec['pool'])
                pool_ref=prefix+'/pools/t%03d.json'%t
                save(pool_ref,queries,'pool_sealed')
                ids=[digest(q) for q in queries]
                fp_ref=prefix+'/predictions/frozen/t%03d.json'%t
                fp=predict_at(frozen,queries,initial_ref,fp_ref)
                for a in ARMS:
                    public=histories[a];model_ref=None;pred_ref=None;pred=None
                    if a!='uniform_hash':
                        if not is_updated(a) or t==0:
                            pred,model_ref,pred_ref=fp,initial_ref,fp_ref
                        else:
                            model_ref=prefix+'/models/%s/t%03d.json'%(a,t)
                            model,_=fit_at(public,u['seed'],a=='risk_only_updated',model_ref)
                            pred_ref=prefix+'/predictions/%s/t%03d.json'%(a,t)
                            pred=predict_at(model,queries,model_ref,pred_ref)
                    seen={r['candidate_id'] for r in public}
                    available=[i for i,cid in enumerate(ids) if cid not in seen]
                    with patch('risk_learning.core.truth',math_forbidden):
                        decision=select(a,ids,available,pred,best(public),[name,t])
                    rel=prefix+'/trials/%s/proposal_%03d.json'%(a,t)
                    p=dict(schema=SCHEMA,task=u['task'],repeat=u['repeat'],arm=a,iteration=t,
                        history_sha256=digest(public),history_information=information(public),
                        pool_ref=pool_ref,pool_sha256=digest(queries),available_indices=available,
                        model_ref=model_ref,prediction_ref=pred_ref,decision=decision,
                        model_training='initial_only' if a.endswith('frozen') else 'own_prefix' if a!='uniform_hash' else 'none')
                    save(rel,p,'proposal_sealed');proposals[a].append(p)
                    r=evaluate(u['task'],queries[decision['selected_index']],name+'/'+a+'/%03d'%t,rel)
                    save(prefix+'/trials/%s/observation_%03d.json'%(a,t),r,'observation_sealed')
                    online[a].append(r['observed']);histories[a].append(r['observed'])
            for a in ARMS:
                r=trial_result(u['task'],u['repeat'],a,init_public,online[a],proposals[a])
                save(prefix+'/trials/'+a+'/result.json',r,'trial_sealed');all_trials.append(r)
            append(root/'progress.jsonl',dict(unit=name,completed_trials=len(all_trials),math_settled=bill['math_settled']))
        if bill['math_started']!=spec['total_math_calls'] or bill['math_settled']!=spec['total_math_calls'] or bill['math_errors']:
            raise ValueError('Evaluation bill does not close')
        fallbacks=sum(r['model_failure_calls'] for r in all_trials)
        report=dict(schema=SCHEMA,code_sha=source['code_sha'],profile=profile,
            status='COMPLETE_C1_ONLINE_PILOT' if not fallbacks else 'COMPLETE_WITH_MODEL_FALLBACKS',
            bill=bill,trial_count=len(all_trials),model_failure_calls=fallbacks,
            summary=summarize(all_trials,spec),elapsed_seconds=time.perf_counter()-start,
            limits=['Known development tasks and new seeds; not held-out-task confirmation.',
              'Frozen means frozen posterior, not frozen incumbent or candidate schedule.',
              'Equal actual calls, not equal model-computation cost.',
              'More exact objectives is a diagnostic, not causal mediation proof.',
              'No unselected candidate truth, no test pool, no offline target imputation.',
              'Fixed initial48: not a comparison of alternative initial/online allocations.',
              'Common initial calls counted once; overlapping online choices counted per arm.'])
        write(root/'report.json',report);seal(root)
        return report
    except Exception as error:
        write(root/'failure.json',dict(status='STOPPED_ERROR',kind=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),bill=bill,completed_trials=len(all_trials)))
        seal(root)
        raise
