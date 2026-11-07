"""Standard-library verification; reconstruct decisions and costs, never refit."""
import json
import math
from pathlib import Path
from b1_core import (SCHEMA,PARENT,POLICIES,BUDGETS,plan,digest,decisions,
    incumbent_from_observed,objective_payload,assess,summarize)
from risk_learning.core import HEADS,candidates,observed,label_counts,score
from risk_learning.io import read,check_manifest,file_sha
from risk_learning.native import training_payload
from risk_learning.study import new_bill
from risk_learning.validation import metric_equal


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_head(diag):
    grid=diag['candidates']
    require(len(grid)==8,'Changed kernel grid')
    valid=[r for r in grid if r['status']=='ok']
    require(bool(valid),'No successful kernel')
    require(diag['selected']==max(valid,key=lambda r:r['log_evidence']),'Selected kernel not evidence winner')
    require([(r['continuous_length'],r['discrete_length']) for r in grid]==
            [(c,d) for c in (.05,.15,.4,1.) for d in (.5,1.5)],'Changed native grid values')


def check_prediction(row, public, queries):
    risk=row['receipt']['risk']; obj=row['receipt']['objective']
    ids=[digest(c) for c in queries]
    require(risk['training_input_sha256']==digest(training_payload(public,'native_censored')),
            'Risk training information differs')
    require(risk['training_rows']==len(public) and risk['method']=='native_censored'
            and risk['query_ids']==ids and risk['privileged_training'] is False
            and risk['objective_labels_used'] is False,'Risk receipt differs')
    exact=[r['objective']['value'] for r in public if r['objective']['kind']=='exact']
    require(obj['training_input_sha256']==digest(objective_payload(public))
            and obj['query_ids']==ids and obj['training_rows']==len(public)
            and obj['exact_objective_rows']==len(exact) and obj['privileged_training'] is False,
            'Objective training information differs')
    require(row['receipt']['same_objective_for_all_policies'] is True
            and row['receipt']['no_objective_truth_imputation'] is True,'Shared fit contract differs')
    delta=row['bill_delta']
    require(set(delta)==set(new_bill()) and all(type(v)is int and v>=0 for v in delta.values()),'Invalid computation bill')
    h=delta['gp_head_started']
    require(h in (1,2,3) and delta['gp_kernel_started']==8*h,'Head/kernel count mismatch')
    if row['status']!='OK':
        require(row['status']=='MODEL_NUMERICAL_FAILURE' and row['predictions'] is None
                and row['decision'] is None and row['receipt']['status']=='MODEL_NUMERICAL_FAILURE'
                and delta['gp_head_errors']==1 and delta['gp_head_completed']==h-1,'Unregistered failed state')
        return
    require(row['receipt']['status']=='OK' and h==2+int(bool(exact))
            and delta['gp_head_completed']==h and delta['gp_head_errors']==0,'Successful fit count mismatch')
    pred=row['predictions']; rp=pred['risk']
    require(rp['candidate_ids']==ids and set(risk['heads'])==set(HEADS),'Query/head mismatch')
    for head in HEADS:
        check_head(risk['heads'][head])
        g=rp['gaussian'][head]
        require(g['observation_noise_variance']==0. and len(g['mean'])==len(ids)
                and len(g['variance'])==len(ids) and len(rp['probabilities'][head])==len(ids),
                'Gaussian shape/noise mismatch')
        for m,v,p in zip(g['mean'],g['variance'],rp['probabilities'][head]):
            require(math.isfinite(m) and math.isfinite(v) and v>=0,'Invalid risk moments')
            q=.5*math.erfc(m/math.sqrt(2*v)) if v>0 else float(m<=0)
            require(math.isfinite(p) and abs(q-p)<=1e-12,'Probability not derived from Gaussian')
        require(metric_equal(rp['margin_mean'][head],g['mean']),'Risk margin differs')
    pa=rp['probabilities']
    require(len(pa['joint'])==len(ids) and all(abs(a*b-j)<=1e-12 for a,b,j in
            zip(pa['region'],pa['resource'],pa['joint'])),'Joint probability differs')
    if exact:
        check_head(obj['head'])
        center=math.fsum(exact)/len(exact)
        ordered=sorted(abs(v) for v in exact); m=len(exact)
        med=ordered[m//2] if m%2 else .5*(ordered[m//2-1]+ordered[m//2])
        scale=max(math.sqrt(math.fsum((v-center)**2 for v in exact)/len(exact)),.01) if len(exact)>1 else max(med,.05)
        require(math.isclose(obj['head']['center'],center,rel_tol=1e-12,abs_tol=1e-12)
                and math.isclose(obj['head']['scale'],scale,rel_tol=1e-12,abs_tol=1e-12),
                'Objective normalization not from observable training values')
        require(row['objective_scale']==obj['head']['scale'] and obj['status']=='OK','Objective scale/status differs')
        require(pred['objective'] is not None,'Objective prediction missing')
        for key in ('mean','variance'):
            require(len(pred['objective'][key])==len(ids) and all(math.isfinite(x) for x in pred['objective'][key]),'Bad objective moments')
        require(all(v>=0 for v in pred['objective']['variance']),'Negative objective variance')
    else:
        require(pred['objective'] is None and obj['head'] is None and row['objective_scale']==1.
                and obj['status']=='NO_OBSERVED_OBJECTIVE_LABELS','Objective fabricated from missing labels')


def verify(root, expected_sha=None):
    root=Path(root).resolve(); files=check_manifest(root)
    require(not (root/'failure.json').exists(),'Stopped run is not completed evidence')
    spec=read(root/'plan.json'); report=read(root/'report.json'); source=read(root/'source_receipt.json')
    require(spec==plan(spec['profile']),'Frozen plan differs')
    require(report['code_sha']==source['code_sha'] and (not expected_sha or source['code_sha']==expected_sha),'Code SHA differs')
    require(source.get('phase')=='B1' and source.get('phase_parent_sha')==PARENT
            and source.get('historical_risk_learning_unchanged') is True,'Phase source receipt differs')
    expected_events=[]; expected_timeline=[]; rows=[]; reconstructed=new_bill()
    expected_paths={'plan.json','source_receipt.json','environment.json','report.json','evaluations.jsonl','timeline.jsonl'}
    for u in spec['units']:
        name='%s/r%d'%(u['task'],u['repeat']); folder=root/'datasets'/name
        train=read(folder/'train.json'); pool=read(folder/'pool.json'); queries=read(folder/'queries.json')
        expected_paths.update('datasets/'+name+'/'+p for p in ('train.json','pool.json','queries.json'))
        require(queries==candidates(u['pool_seed'],spec['pool_count']),'Query stream changed')
        for role,data,seed,count in [('train',train,u['train_seed'],max(spec['sizes'])),('pool',pool,u['pool_seed'],spec['pool_count'])]:
            require([r['candidate'] for r in data]==candidates(seed,count),'Dataset stream changed')
            for i,r in enumerate(data):
                cid='%s/%s/%04d'%(name,role,i)
                require(r['candidate_id']==digest(r['candidate']) and r['call_id']==cid
                        and r['observed']==observed(r['candidate'],r['truth'])
                        and type(r['truth']['feasible']) is bool
                        and r['truth']['feasible']==all(r['truth'][h]<=0 for h in HEADS)
                        and all(type(r['truth'][h]) in (int,float) and math.isfinite(r['truth'][h])
                                for h in ('region','resource','objective')),'Feedback or truth identity differs')
                expected_events.extend([dict(event='started',call_id=cid,task=u['task'],role=role,
                    candidate_id=r['candidate_id'],candidate=r['candidate']),
                    dict(event='settled',call_id=cid,record_sha256=digest(r))])
        require(not ({r['candidate_id'] for r in train}&{r['candidate_id'] for r in pool}),'Train/pool overlap')
        expected_timeline.append(dict(event='train_started',unit=name))
        for n in spec['sizes']:
            rel='states/'+name+'/n%d.json'%n; row=read(root/rel)
            public=[r['observed'] for r in train[:n]]
            for key,value in dict(schema=SCHEMA,task=u['task'],repeat=u['repeat'],n=n,
                    train_prefix_sha256=digest(train[:n]),observable_prefix_sha256=digest(public),
                    query_sha256=digest(queries),labels=label_counts(public)).items():
                require(row[key]==value,'State identity differs: '+key)
            check_prediction(row,public,queries)
            result=dict(task=u['task'],repeat=u['repeat'],n=n,status=row['status'],outcomes={})
            if row['status']=='OK':
                d=decisions([digest(c) for c in queries],row['predictions']['risk']['probabilities']['joint'],
                            row['predictions']['objective'],incumbent_from_observed(public),name)
                require(metric_equal(row['decision'],d),'EI/risk or sealed rankings differ')
                result['decision']=row['decision']
                for p in POLICIES:
                    result['outcomes'][p]={str(k):dict(assess(pool,d['rankings'][p],k,d['incumbent'],row['objective_scale']),
                                              nominal_total_evaluations=n+k) for k in BUDGETS}
                result['pool_risk_metrics']=score(pool,row['predictions']['risk'])
            sr='scores/'+name+'/n%d.json'%n
            require(metric_equal(read(root/sr),result),'Post-seal result cannot be reconstructed')
            rows.append(read(root/sr)); expected_paths.update((rel,sr))
            for key,v in row['bill_delta'].items(): reconstructed[key]+=v
            expected_timeline.extend([dict(event='fit_started',unit=name,n=n),
                dict(event='decision_sealed',unit=name,n=n,path=rel,sha256=file_sha(root/rel))])
        expected_timeline.extend([dict(event='pool_truth_started',unit=name),dict(event='unit_scored',unit=name)])
    events=[json.loads(s) for s in (root/'evaluations.jsonl').read_text(encoding='utf-8').splitlines()]
    timeline=[json.loads(s) for s in (root/'timeline.jsonl').read_text(encoding='utf-8').splitlines()]
    require(events==expected_events,'Evaluation ordering, start/settlement or cost mismatch')
    require(timeline==expected_timeline,'Rankings not sealed before pool targets')
    require(set(read(root/'manifest.json')['files'])==expected_paths,'Unexpected or missing files')
    require(reconstructed['math_started']==reconstructed['math_settled']==0,'Fitting used evaluator')
    reconstructed.update(math_started=spec['total_math_calls'],math_settled=spec['total_math_calls'])
    require(reconstructed==report['bill'],'Global bill differs')
    require(reconstructed['gp_head_started']==reconstructed['gp_head_completed']+reconstructed['gp_head_errors']
            and reconstructed['gp_kernel_started']==reconstructed['gp_kernel_completed']+reconstructed['gp_kernel_errors'],
            'Unsettled GP work')
    require(all(reconstructed[k]==0 for k in ('aircraft_calls','rcaide_calls','vae_fits','fantasy_fits')),'Forbidden work')
    failures=sum(r['status']!='OK' for r in rows)
    expected_status='COMPLETE_FIXED_OBJECTIVE_DECISIONS' if failures==0 else 'COMPLETE_WITH_MODEL_FAILURES'
    require(report['schema']==SCHEMA and report['profile']==spec['profile']
            and report['model_states']==len(rows)==spec['model_states'] and report['failed_states']==failures
            and report['status']==expected_status and metric_equal(report['summary'],summarize(rows)),'Summary differs')
    return dict(status='VERIFIED_FIXED_OBJECTIVE_DECISIONS',code_sha=source['code_sha'],
                files=files,model_states=len(rows),failed_states=failures,
                rankings_sealed_before_truth=True,historical_bill=reconstructed,
                new_math_calls=0,new_gp_fits=0,new_vae_fits=0)
