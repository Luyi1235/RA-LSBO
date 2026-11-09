"""Model-free causal-prefix, score, feedback, call and cost verification for C1."""
import json
import math
from pathlib import Path
from c1_protocol import (SCHEMA,ARMS,INITIAL,bill as new_bill,plan,digest,information,
                         best,is_updated,select,trial_result,summarize)
from risk_learning.core import candidates,canonical,observed
from risk_learning.io import read,check_manifest
from risk_learning.validation import metric_equal


def require(condition,message):
    if not condition:
        raise ValueError(message)


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def delta_valid(d,allowed):
    require(set(d)==set(new_bill()) and all(type(v)is int and v>=0 for v in d.values()),'Invalid computation bill')
    require(all(v==0 for k,v in d.items() if k not in allowed),'Unexpected work in computation bill')
    require(d['gp_head_started']==d['gp_head_completed']+d['gp_head_errors'],'Unsettled heads')
    require(d['gp_kernel_started']==d['gp_kernel_completed']+d['gp_kernel_errors'],'Unsettled kernels')
    require(d['prediction_started']==d['prediction_completed']+d['prediction_errors'],'Unsettled predictions')


def check_fit(row,public,seed,risk_only):
    r=row['receipt'];d=row['bill_delta']
    delta_valid(d,{'gp_head_started','gp_head_completed','gp_head_errors','gp_kernel_started',
                   'gp_kernel_completed','gp_kernel_errors','kernel_candidates_rejected'})
    rp=[dict(candidate=x['candidate'],candidate_id=x['candidate_id'],region=x['region'],resource=x['resource']) for x in public]
    op=[dict(candidate=x['candidate'],candidate_id=x['candidate_id'],objective=x['objective']) for x in public]
    require(r['schema']=='c1_native_fit_v1' and r['seed']==seed and r['risk_only'] is risk_only
            and r['training_sha256']==digest(public) and r['training_information']==information(public)
            and r['risk_input_sha256']==digest(rp) and r['objective_input_sha256']==digest(op)
            and r['privileged_training'] is False,'Fit prefix/information contract changed')
    exact=[x['objective']['value'] for x in public if x['objective']['kind']=='exact']
    names=['region','resource']+(['objective'] if exact and not risk_only else [])
    h=len(r['heads']); failed=r['status']=='MODEL_NUMERICAL_FAILURE'
    require(r['status'] in ('OK','MODEL_NUMERICAL_FAILURE') and set(r['heads'])==set(names[:h]),'Unregistered fitted heads')
    require(d['gp_head_started']==h+int(failed) and d['gp_head_completed']==h
            and d['gp_head_errors']==int(failed) and d['gp_kernel_started']==8*(h+int(failed)),
            'Native head/kernel count differs')
    rejected=0
    for name,diag in r['heads'].items():
        grid=diag['candidates'];good=[v for v in grid if v['status']=='ok']
        require([(v['continuous_length'],v['discrete_length']) for v in grid]==
                [(c,e) for c in (.05,.15,.4,1.) for e in (.5,1.5)],'Changed kernel grid')
        require(good and diag['selected']==max(good,key=lambda v:v['log_evidence']), 'Kernel is not evidence winner')
        require(diag['name']==name,'Head name mismatch')
        vals=[x[name]['value'] for x in public if x[name]['kind']=='exact']
        center=math.fsum(vals)/len(vals) if name=='objective' and vals else 0.
        if name=='objective' and len(vals)>1:
            scale=max(math.sqrt(math.fsum((v-center)**2 for v in vals)/len(vals)),.01)
        elif vals:
            v=sorted(abs(v) for v in vals);n=len(v)
            scale=max(v[n//2] if n%2 else .5*(v[n//2-1]+v[n//2]),.05)
        else:
            scale=1.
        require(math.isclose(center,diag['center'],rel_tol=1e-12,abs_tol=1e-12)
                and math.isclose(scale,diag['scale'],rel_tol=1e-12,abs_tol=1e-12),
                'Normalization not derived from observable prefix')
        rejected+=sum(v['status']!='ok' for v in grid)
    require(d['kernel_candidates_rejected']==rejected,'Kernel rejection count differs')
    if failed:
        require(h<len(names) and r['failure']['head']==names[h] and r['model_id'] is None,
                'Failure did not stop at failing head')
        require(len(r['failure']['details'])==8 and all(v['status']!='ok' for v in r['failure']['details']),
                'Unproven all-kernel failure')
    else:
        require(h==len(names) and r['failure'] is None,'Incomplete successful fit')
        expected=digest(dict(training_sha256=digest(public),seed=seed,risk_only=risk_only,
                            heads={k:v['identity'] for k,v in r['heads'].items()}))
        require(r['model_id']==expected,'Model identity differs')
    return r


def check_prediction(row,model_ref,fit,queries):
    d=row['bill_delta'];p=row['prediction']
    delta_valid(d,{'prediction_started','prediction_completed','prediction_errors'})
    require(row['model_ref']==model_ref and row['query_sha256']==digest(queries),'Prediction source differs')
    if fit['status']!='OK':
        require(p is None and row['error']['kind']=='FIT_FAILED' and d['prediction_started']==0,'Failed fit fabricated prediction')
        return None
    require(d['prediction_started']==1,'Prediction call count differs')
    if p is None:
        require(row['error']['kind']=='ModelNumericalFailure' and d['prediction_errors']==1,'Unknown prediction failure')
        return None
    require(row['error'] is None and d['prediction_completed']==1,'Successful prediction status differs')
    ids=[digest(c) for c in queries];n=len(ids)
    require(p['candidate_ids']==ids and p['model_id']==fit['model_id'],'Prediction identity differs')
    require(set(p['gaussian'])=={'region','resource'} and set(p['probabilities'])=={'region','resource','joint'},'Risk head set differs')
    for head in ('region','resource'):
        g=p['gaussian'][head];prob=p['probabilities'][head]
        require(len(g['mean'])==len(g['variance'])==len(prob)==n,'Risk moment shape differs')
        for m,v,q in zip(g['mean'],g['variance'],prob):
            require(math.isfinite(m) and math.isfinite(v) and v>=0 and math.isfinite(q) and 0<=q<=1,'Invalid risk forecast')
            expected=.5*math.erfc(m/math.sqrt(2*v)) if v else float(m<=0)
            require(abs(q-expected)<=1e-12,'Risk probability not implied by saved Gaussian')
    pp=p['probabilities']
    require(len(pp['joint'])==n and all(math.isfinite(q) and abs(a*b-q)<=1e-12 for a,b,q in
            zip(pp['region'],pp['resource'],pp['joint'])),'Joint risk probability differs')
    if 'objective' not in fit['heads']:
        require(p['objective'] is None,'Missing objective head fabricated')
    else:
        o=p['objective'];require(isinstance(o,dict),'Missing objective forecast')
        require(len(o['mean'])==len(o['variance'])==n and all(math.isfinite(x) for x in o['mean'])
                and all(math.isfinite(x) and x>=0 for x in o['variance']),'Invalid objective forecast')
    return p


def verify(root,expected_sha=None):
    root=Path(root).resolve();file_count=check_manifest(root)
    require(not (root/'failure.json').exists(),'Stopped evidence is not a completed run')
    spec=read(root/'plan.json');report=read(root/'report.json');source=read(root/'source_receipt.json')
    require(spec==plan(spec['profile']),'Frozen plan changed')
    require(report['code_sha']==source['code_sha'] and (expected_sha is None or source['code_sha']==expected_sha),'Code SHA differs')
    event_expect=[];eval_expect=[];progress_expect=[];all_trials=[];total=new_bill()
    paths={'plan.json','source_receipt.json','environment.json','report.json','events.jsonl','evaluations.jsonl','progress.jsonl'}
    def evaluate_expected(task,r,call_id,candidate,proposal=None):
        require(r['candidate']==canonical(candidate) and r['candidate_id']==digest(candidate) and r['call_id']==call_id,
                'Evaluation candidate/ticket differs')
        target=r['truth']
        require(set(target)=={'region','resource','objective','feasible'} and type(target['feasible'])is bool
                and all(type(target[k])in(int,float) and math.isfinite(target[k]) for k in ('region','resource','objective')),
                'Unknown/invalid evaluator output')
        require(target['feasible']==(target['region']<=0 and target['resource']<=0)
                and r['observed']==observed(candidate,target),'Observable feedback/feasibility changed')
        eval_expect.extend([dict(event='started',call_id=call_id,task=task,candidate_id=digest(candidate),candidate=candidate,proposal=proposal),
                            dict(event='settled',call_id=call_id,record_sha256=digest(r))])
        event_expect.extend([dict(event='evaluation_started',call_id=call_id),
                             dict(event='evaluation_settled',call_id=call_id,record_sha256=digest(r))])
        total['math_started']+=1;total['math_settled']+=1
    def add_delta(row):
        for k,v in row['bill_delta'].items():total[k]+=v
    for u in spec['units']:
        name='%s/r%d'%(u['task'],u['repeat']);prefix='units/'+name
        path=prefix+'/records.jsonl';paths.add(path);records=lines(root/path)
        mapping={r['ref']:r['value'] for r in records};order=[]
        require(len(mapping)==len(records),'Duplicate evidence references')
        def saved(ref,event):
            require(ref in mapping,'Missing evidence reference '+ref)
            value=mapping[ref];order.append(ref)
            event_expect.append(dict(event=event,ref=ref,record_sha256=digest(value)))
            return value
        def fit_saved(ref,public,risk_only):
            event_expect.append(dict(event='fit_started',ref=ref,training_sha256=digest(public)))
            r=saved(ref,'fit_sealed');check_fit(r,public,u['seed'],risk_only);add_delta(r)
            return r['receipt']
        initial=mapping[prefix+'/initial.json']
        require(len(initial)==INITIAL,'Initial count changed')
        for i,(r,c) in enumerate(zip(initial,candidates(u['seed'],INITIAL))):
            evaluate_expected(u['task'],r,name+'/initial/%03d'%i,c)
        saved(prefix+'/initial.json','initial_sealed')
        init_public=[r['observed'] for r in initial];initial_ref=prefix+'/models/initial.json'
        frozen=fit_saved(initial_ref,init_public,False)
        histories={a:list(init_public) for a in ARMS};obs={a:[] for a in ARMS};props={a:[] for a in ARMS}
        for t in range(spec['steps']):
            pool_ref=prefix+'/pools/t%03d.json'%t
            queries=saved(pool_ref,'pool_sealed')
            require(queries==candidates(u['seed']+1+t,spec['pool']),'Candidate stream changed')
            ids=[digest(c) for c in queries];fp_ref=prefix+'/predictions/frozen/t%03d.json'%t
            fp_row=saved(fp_ref,'prediction_sealed');fp=check_prediction(fp_row,initial_ref,frozen,queries);add_delta(fp_row)
            for a in ARMS:
                public=histories[a];model_ref=None;pred_ref=None;pred=None
                if a!='uniform_hash':
                    if not is_updated(a) or t==0:
                        pred,model_ref,pred_ref=fp,initial_ref,fp_ref
                    else:
                        model_ref=prefix+'/models/%s/t%03d.json'%(a,t)
                        fit_r=fit_saved(model_ref,public,a=='risk_only_updated')
                        pred_ref=prefix+'/predictions/%s/t%03d.json'%(a,t)
                        pr=saved(pred_ref,'prediction_sealed');pred=check_prediction(pr,model_ref,fit_r,queries);add_delta(pr)
                seen={r['candidate_id'] for r in public};available=[i for i,cid in enumerate(ids) if cid not in seen]
                d=select(a,ids,available,pred,best(public),[name,t])
                rel=prefix+'/trials/%s/proposal_%03d.json'%(a,t)
                p=saved(rel,'proposal_sealed')
                expected=dict(schema=SCHEMA,task=u['task'],repeat=u['repeat'],arm=a,iteration=t,
                    history_sha256=digest(public),history_information=information(public),
                    pool_ref=pool_ref,pool_sha256=digest(queries),available_indices=available,
                    model_ref=model_ref,prediction_ref=pred_ref,decision=d,
                    model_training='initial_only' if a.endswith('frozen') else 'own_prefix' if a!='uniform_hash' else 'none')
                require(metric_equal(p,expected),'Proposal has future/cross-arm data or score mismatch')
                props[a].append(p)
                or_ref=prefix+'/trials/%s/observation_%03d.json'%(a,t);r=mapping[or_ref]
                evaluate_expected(u['task'],r,name+'/'+a+'/%03d'%t,queries[d['selected_index']],rel)
                saved(or_ref,'observation_sealed');obs[a].append(r['observed']);public.append(r['observed'])
        for a,b in (('ei_frozen','ei_updated'),('risk_ei_frozen','risk_ei_updated')):
            require(obs[a][0]['candidate_id']==obs[b][0]['candidate_id'],'Matched first actions differ')
        for a in ARMS:
            r=trial_result(u['task'],u['repeat'],a,init_public,obs[a],props[a])
            got=saved(prefix+'/trials/'+a+'/result.json','trial_sealed')
            require(metric_equal(got,r),'Trial curve/label yield differs')
            all_trials.append(got)
        require(order==[r['ref'] for r in records],'Unexpected/missing/reordered unit record')
        progress_expect.append(dict(unit=name,completed_trials=len(all_trials),math_settled=total['math_settled']))
    require(lines(root/'events.jsonl')==event_expect,'Causal event ordering/seals changed')
    require(lines(root/'evaluations.jsonl')==eval_expect,'Actual started/settled ledger differs')
    require(lines(root/'progress.jsonl')==progress_expect,'Progress differs')
    require(set(read(root/'manifest.json')['files'])==paths,'Missing/extra evidence files')
    require(total==report['bill'] and total['math_started']==total['math_settled']==spec['total_math_calls'], 'Global bill mismatch')
    require(total['gp_head_started']<=spec['max_gp_heads'] and total['gp_kernel_started']<=spec['max_gp_kernels'],'Extra fits')
    fallback=sum(r['model_failure_calls'] for r in all_trials)
    status='COMPLETE_C1_ONLINE_PILOT' if not fallback else 'COMPLETE_WITH_MODEL_FALLBACKS'
    require(report['schema']==SCHEMA and report['profile']==spec['profile'] and report['trial_count']==len(all_trials)==spec['trials']
            and report['model_failure_calls']==fallback and report['status']==status
            and metric_equal(report['summary'],summarize(all_trials,spec)),'Summary changed')
    require(all(total[k]==0 for k in ('vae_fits','fantasy_fits','aircraft_calls','rcaide_calls','math_errors')),'Forbidden or failed evaluations')
    return dict(status='VERIFIED_C1_ONLINE',code_sha=source['code_sha'],files=file_count,
        trials=len(all_trials),model_failure_calls=fallback,historical_bill=total,
        prefix_isolation=True,selected_only_evaluation=True,new_math_calls=0,new_gp_fits=0)
