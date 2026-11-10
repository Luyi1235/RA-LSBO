"""PR84/PR85: standard-library, observed-feedback-only C2 archive audit.

No evaluator, optimizer or model is imported. Historical predictions are replayed,
not refit. Public entry points pin both the code and the input Git commit.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess

RESULT_HEAD = 'ade354de8b35d20a2bb32b469b04bc7da92aa004'
CODE_SHA = '975d96caac53d675ebe1b790a0647479a75a2ed5'
REL = 'results/phase-c2-head-factorial-20261109'
MANIFESTS = {
    'smoke': '2ddf0fd06679a78597f09f31ca1c1406296264810b1ebacfd4d7b6b11998a8c5',
    'full': 'a13d0cc8cda75c64688d1b68c60bcf6c2970071bb23b098c4e4d44689070e824',
}
ARMS = ('risk_frozen_objective_frozen', 'risk_updated_objective_frozen',
        'risk_frozen_objective_updated', 'risk_updated_objective_updated',
        'risk_only_updated', 'uniform_hash')
TASKS = tuple('tradeoff_s%d' % i for i in range(4))
BILL_KEYS = ('math_started','math_settled','math_errors','gp_head_started',
    'gp_head_completed','gp_head_errors','gp_kernel_started','gp_kernel_completed',
    'gp_kernel_errors','kernel_candidates_rejected','prediction_started',
    'prediction_completed','prediction_errors','vae_fits','fantasy_fits',
    'aircraft_calls','rcaide_calls')
ZERO = dict(new_math_calls=0, new_gp_fits=0, new_gp_predictions=0,
            new_vae_fits=0, new_fantasy_fits=0, aircraft_calls=0, rcaide_calls=0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def same(a, b):
    if type(a) in (int, float) and type(b) in (int, float):
        return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def safe_path(root, rel):
    p = PurePosixPath(rel)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in rel
            and ':' not in rel and str(p) == rel, 'Unsafe evidence path: ' + rel)
    result = root / rel
    require(root.resolve() in result.resolve().parents, 'Evidence path escape')
    require(not any(q.is_symlink() for q in [result] + list(result.parents)
                    if q == root or root in q.parents), 'Symlink evidence forbidden')
    return result


def verify_manifest(root, pinned=None):
    root = Path(root).resolve()
    if pinned:
        require(sha(root/'manifest.json') == pinned, 'Pinned manifest differs')
    m = read(root/'manifest.json')
    require(m['schema'] == 'sha256_manifest_v1', 'Manifest schema differs')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != root/'manifest.json'}
    require(actual == set(m['files']), 'Manifest membership differs')
    for rel, h in m['files'].items():
        require(re.fullmatch('[0-9a-f]{64}', h) is not None, 'Invalid SHA256')
        require(sha(safe_path(root, rel)) == h, 'File SHA256 differs: ' + rel)
    return len(actual)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args],
                                   encoding='utf-8').strip()


def identity(code_root, expected_sha, evidence_root):
    require(re.fullmatch('[0-9a-f]{40}', expected_sha or '') is not None,
            'Exact audit code SHA required')
    require(git(code_root, 'rev-parse', 'HEAD') == expected_sha, 'Audit SHA differs')
    require(not git(code_root, 'status', '--porcelain'), 'Audit worktree dirty')
    require(git(evidence_root, 'rev-parse', 'HEAD') == RESULT_HEAD,
            'Input result HEAD differs; do not use floating branch')
    require(not git(evidence_root, 'status', '--porcelain'), 'Evidence worktree dirty')
    root = Path(evidence_root).resolve()/REL
    require(root.is_dir(), 'Pass evidence worktree root, not an invented local path')
    # Bind every archived file to its committed Git blob, including the outer manifest.
    raw = subprocess.check_output(['git','-C',str(evidence_root),'ls-tree','-r','-z',
                                   'HEAD','--',REL])
    committed = {}
    for item in raw.split(b'\0'):
        if not item:
            continue
        meta, path = item.decode().split('\t', 1)
        mode, kind, h = meta.split()
        require(mode == '100644' and kind == 'blob', 'Nonregular archive file')
        committed[path[len(REL)+1:]] = h
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(committed), 'Archive differs from committed file set')
    for rel, h in committed.items():
        p = safe_path(root, rel)
        state = hashlib.sha1(('blob %d\0' % p.stat().st_size).encode())
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                state.update(chunk)
        require(state.hexdigest() == h, 'Committed archive bytes differ: ' + rel)
    outer = {}
    for text in (root/'sha256_manifest.txt').read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64}) [ *](?:\./)?(.+)', text)
        require(match is not None, 'Bad outer manifest row')
        h, rel = match.groups()
        require(rel not in outer, 'Duplicate outer manifest path')
        outer[rel] = h
    require(set(outer) == actual-{'sha256_manifest.txt'}, 'Outer manifest membership')
    for rel, h in outer.items():
        require(sha(safe_path(root, rel)) == h, 'Outer manifest SHA256: ' + rel)
    return root, dict(result_head=RESULT_HEAD, audit_code_sha=expected_sha,
                     archived_code_sha=CODE_SHA, git_files_verified=len(actual),
                     outer_sha256_records=len(outer), outer_manifest_sha256=sha(root/'sha256_manifest.txt'))


def label(packet):
    kind = packet['kind']
    if kind == 'exact':
        require(finite(packet['value']), 'Nonfinite observed constraint')
        return packet['value'] <= 0
    require(kind == 'interval' and packet.get('lower') == 0
            and packet.get('lower_inclusive') is False, 'Unknown constraint label')
    return False


def observed_check(row):
    require(row['candidate_id'] == digest(row['candidate']), 'Observed candidate ID')
    r, s = label(row['region']), label(row['resource'])
    require(row['resource']['kind'] == 'exact', 'C2 resource feedback contract')
    obj = row['objective']
    require(obj['kind'] == ('exact' if r else 'missing'), 'C2 objective censoring contract')
    if obj['kind'] == 'exact':
        require(finite(obj['value']), 'Nonfinite observed objective')
    return r, s, r and s


def best(rows):
    ys = [x['objective']['value'] for x in rows
          if label(x['region']) and label(x['resource']) and x['objective']['kind']=='exact']
    return min(ys) if ys else None


def information(rows):
    return dict(rows=len(rows), exact_objectives=sum(x['objective']['kind']=='exact' for x in rows),
        feasible=sum(label(x['region']) and label(x['resource']) for x in rows),
        exact_region=sum(x['region']['kind']=='exact' for x in rows),
        interval_region=sum(x['region']['kind']=='interval' for x in rows),
        branches={str(b):sum(x['candidate']['branch']==b for x in rows) for b in range(4)})


def trial_replay(task, repeat, arm, initial, observations, proposals):
    history = list(initial)
    curve, counts = [best(history)], [information(history)]
    first = 0 if curve[0] is not None else None
    for i, row in enumerate(observations):
        observed_check(row)
        history.append(row)
        curve.append(best(history)); counts.append(information(history))
        if first is None and curve[-1] is not None:
            first = i+1
    return dict(task=task, repeat=repeat, arm=arm, initial_best=curve[0],
        initial_warm=curve[0] is not None, best_curve=curve, information_curve=counts,
        first_feasible_call=first, candidate_path=[r['candidate_id'] for r in observations],
        modes=[p['decision']['mode'] for p in proposals],
        model_failure_calls=sum(p['decision']['mode']=='model_failure_hash' for p in proposals),
        online_calls=len(observations), final_best=curve[-1])


def auc(labels, probs):
    positive, n = sum(labels), len(labels)
    if not positive or positive == n:
        return None
    pairs = sorted(zip(probs, labels)); i=0; ranks=0.
    while i < n:
        j=i+1
        while j<n and pairs[j][0]==pairs[i][0]:
            j+=1
        ranks += sum(y for _,y in pairs[i:j]) * ((i+1+j)/2.)
        i=j
    return (ranks-positive*(positive+1)/2.)/(positive*(n-positive))


def binary(labels, probs):
    require(len(labels)==len(probs), 'Probability/label alignment')
    require(all(type(x) is bool for x in labels), 'Boolean labels required')
    require(all(finite(p) and 0<=p<=1 for p in probs), 'Invalid probability')
    n=len(labels); bins=[]
    for b in range(10):
        ix=[i for i,p in enumerate(probs) if min(9,int(p*10))==b]
        bins.append(dict(bin=b,n=len(ix),mean_probability=math.fsum(probs[i] for i in ix)/len(ix) if ix else None,
                         observed_rate=sum(labels[i] for i in ix)/len(ix) if ix else None))
    return dict(n=n,positives=sum(labels),negatives=n-sum(labels),
        mean_probability=math.fsum(probs)/n if n else None,
        observed_rate=sum(labels)/n if n else None,
        brier=math.fsum((p-y)**2 for p,y in zip(probs,labels))/n if n else None,
        log_loss=-math.fsum(math.log(max(1e-15,p if y else 1-p)) for p,y in zip(probs,labels))/n if n else None,
        log_loss_clip=1e-15, clipped_events=sum((p if y else 1-p)<1e-15 for p,y in zip(probs,labels)),
        auroc=auc(labels,probs),reliability_bins=bins)


def diagnostics(rows):
    risk={}
    for h in ('region','resource','joint'):
        rr=[r for r in rows if r[h+'_p'] is not None]
        risk[h]=binary([r[h+'_label'] for r in rr],[r[h+'_p'] for r in rr])
    oo=[r for r in rows if r['objective_y'] is not None and r['objective_mu'] is not None]
    errors=[r['objective_mu']-r['objective_y'] for r in oo]
    zz=[abs(r['objective_mu']-r['objective_y'])/math.sqrt(r['objective_var']) for r in oo if r['objective_var']>0]
    return dict(n=len(rows), risk=risk,
        exact_objective_labels=sum(r['objective_y'] is not None for r in rows),
        exact_objective_rate=sum(r['objective_y'] is not None for r in rows)/len(rows) if rows else None,
        objective=dict(n=len(oo),mae=math.fsum(map(abs,errors))/len(oo) if oo else None,
            rmse=math.sqrt(math.fsum(e*e for e in errors)/len(oo)) if oo else None,
            standardized_n=len(zz),zero_variance_n=len(oo)-len(zz),
            mean_abs_standardized_residual=math.fsum(zz)/len(zz) if zz else None,
            coverage95=sum(z<=1.959963984540054 for z in zz)/len(zz) if zz else None))


def ei(m,v,b):
    require(all(finite(x) for x in (m,v,b)) and v>=0,'Invalid EI input')
    d=b-m
    if v==0: return max(0.,d)
    sd=math.sqrt(v); z=d/sd
    return max(0.,d*.5*math.erfc(-z/math.sqrt(2))+sd*math.exp(-.5*z*z)/math.sqrt(2*math.pi))


def selection(arm, ids, available, pred, incumbent, salt):
    scores=[0.]*len(ids); mode='uniform_hash'
    if arm!='uniform_hash':
        if pred is None: mode='model_failure_hash'
        elif arm=='risk_only_updated': scores=pred['probabilities']['joint']; mode='risk_only'
        elif incumbent is None: mode='cold_shared_hash'
        elif pred['objective'] is None: mode='no_objective_model_hash'
        else:
            o=pred['objective']; mode='risk_ei'
            scores=[ei(m,v,incumbent)*p for m,v,p in zip(o['mean'],o['variance'],pred['probabilities']['joint'])]
    require(len(scores)==len(ids) and all(finite(v) for v in scores),'Invalid scores')
    index=min(available,key=lambda i:(-scores[i],digest(['c2_outcome_blind_tie_v1',salt,ids[i]]),ids[i]))
    return dict(mode=mode,scores=scores,selected_index=index,selected_candidate_id=ids[index],incumbent=incumbent)


def quantile_rank(values, j, available):
    if not values: return None
    return (sum(values[i]<values[j] for i in available)+.5*sum(values[i]==values[j] for i in available))/len(available)


def audit_unit(root, spec, u):
    """Replay archived observations and predictions; never calculate new truth."""
    task,repeat=u['task'],u['repeat']; name='%s/r%d'%(task,repeat); prefix='units/'+name
    journal=list(lines(root/(prefix+'/records.jsonl')))
    mp={r['ref']:r for r in journal}
    require(len(mp)==len(journal),'Duplicate journal reference')
    order=[]; events=[]; calls=[]; totals={k:0 for k in BILL_KEYS}; fits={a:{k:0 for k in BILL_KEYS} for a in ARMS}; shared={k:0 for k in BILL_KEYS}
    def get(ref):
        require(ref in mp,'Missing journal ref: '+ref)
        row=mp[ref]; order.append(ref)
        events.append(dict(event='record_sealed',record_type=row['event_type'],ref=ref,record_sha256=digest(row['value'])))
        return row['value']
    def charge(row, bucket):
        d=row['bill_delta']; require(set(d)==set(BILL_KEYS),'Bill keys differ')
        for k,v in d.items():
            require(type(v) is int and v>=0,'Invalid bill value')
            totals[k]+=v; bucket[k]+=v
        for a,b,c in (('gp_head_started','gp_head_completed','gp_head_errors'),('gp_kernel_started','gp_kernel_completed','gp_kernel_errors'),('prediction_started','prediction_completed','prediction_errors')):
            require(d[a]==d[b]+d[c],'Unclosed historical compute bill')
    def evaluation(row, call):
        require(row['call_id']==call,'Call ID differs')
        o=row['observed']; observed_check(o)
        require(row['candidate']==o['candidate'] and row['candidate_id']==o['candidate_id'],'Observation identity differs')
        calls.extend([dict(event='started',call_id=call,task=task,candidate_id=row['candidate_id'],candidate=row['candidate']),dict(event='settled',call_id=call,record_sha256=digest(row))])
        events.extend([dict(event='evaluation_started',call_id=call),dict(event='evaluation_settled',call_id=call,record_sha256=digest(row))])
        totals['math_started']+=1; totals['math_settled']+=1
        return o
    initial_record=mp[prefix+'/initial.json']['value']; require(len(initial_record)==spec['initial'],'Initial size')
    initial=[evaluation(r,name+'/initial/%03d'%i) for i,r in enumerate(initial_record)]
    get(prefix+'/initial.json')
    fit=get(prefix+'/models/initial.json');charge(fit,shared)
    ir=fit['receipt']; require(ir['status']=='OK' and ir['training_sha256']==digest(initial),'Initial model prefix')
    initial_ids={k:v['identity'] for k,v in ir['heads'].items()}
    require(ir['model_id']==digest(dict(training_sha256=digest(initial),seed=u['seed'],risk_only=False,heads=initial_ids)),'Initial model identity')
    histories={a:list(initial) for a in ARMS}; obs={a:[] for a in ARMS}; props={a:[] for a in ARMS}; feedback=[]
    for t in range(spec['steps']):
        poolref=prefix+'/pools/t%03d.json'%t; pool=get(poolref); ids=[digest(c) for c in pool]
        require(len(ids)==spec['pool'] and len(ids)==len(set(ids)),'Pool shape')
        for arm in ARMS:
            h=histories[arm]; ur=arm in (ARMS[1],ARMS[3],ARMS[4]); uo=arm in (ARMS[2],ARMS[3])
            if arm==ARMS[5]: mid=None; modes=None
            elif arm==ARMS[4] and t==0:
                mid=digest({k:initial_ids[k] for k in ('region','resource')})
                modes={k:dict(mode='frozen',identity=initial_ids[k]) for k in ('region','resource')}
            elif t==0 or arm==ARMS[0]:
                mid=ir['model_id']; modes={k:dict(mode='frozen',identity=v) for k,v in initial_ids.items()}
                if 'objective' not in initial_ids: modes['objective']=dict(mode='frozen_unavailable')
            else:
                fit=get(prefix+'/models/'+arm+'/t%03d.json'%t);charge(fit,fits[arm]); rec=fit['receipt'];mid=rec['model_id'];modes=rec['heads']
                require(rec['training_sha256']==digest(h) and rec['seed']==u['seed']+t,'Wrong update prefix/seed')
                require(rec['update_risk'] is ur and rec['update_objective'] is uo,'Wrong update factors')
                require(set(modes)==({'region','resource'} if arm==ARMS[4] else {'region','resource','objective'}),'Head set')
                for head,v in modes.items():
                    update=ur if head!='objective' else uo
                    unavailable=head=='objective' and not any(x['objective']['kind']=='exact' for x in (h if update else initial))
                    expected=('unavailable_no_exact_labels' if update else 'frozen_unavailable') if unavailable else ('updated' if update else 'frozen')
                    require(v['mode']==expected,'Wrong head mode')
                    if expected=='updated': require(v['training_sha256']==digest(h),'Head prefix differs')
                    if expected=='frozen': require(v['identity']==initial_ids[head],'Frozen identity differs')
                headids={k:v['identity'] for k,v in modes.items() if 'identity' in v}
                expectedmid=digest(headids) if arm==ARMS[4] else digest(dict(training_sha256=digest(h),seed=u['seed']+t,update_risk=ur,update_objective=uo,heads=headids))
                require(mid==expectedmid,'Composite identity differs')
            pref=prefix+'/predictions/'+arm+'/t%03d.json'%t; prow=get(pref);charge(prow,fits[arm]);pred=prow['prediction']
            require(prow['model_id']==mid and prow['query_sha256']==digest(pool),'Prediction provenance')
            if pred is not None:
                require(pred['candidate_ids']==ids and pred['model_id']==mid,'Prediction ID alignment')
                for head in ('region','resource','joint'):
                    v=pred['probabilities'][head]
                    require(len(v)==len(ids) and all(finite(p) and 0<=p<=1 for p in v),'Risk probability vector')
                require(all(abs(a*b-c)<=1e-12 for a,b,c in zip(pred['probabilities']['region'],pred['probabilities']['resource'],pred['probabilities']['joint'])),'Joint risk product')
                if pred['objective'] is not None:
                    require(len(pred['objective']['mean'])==len(ids)==len(pred['objective']['variance']),'Objective vector shape')
                    require(all(finite(v) for v in pred['objective']['mean']) and all(finite(v) and v>=0 for v in pred['objective']['variance']),'Objective moments')
            seen={x['candidate_id'] for x in h}; available=[i for i,cid in enumerate(ids) if cid not in seen]
            require(available,'Empty available set')
            decision=selection(arm,ids,available,pred,best(h),[name,t])
            prop=get(prefix+'/trials/'+arm+'/proposal_%03d.json'%t)
            require(prop['history_sha256']==digest(h) and prop['history_information']==information(h),'Proposal prefix')
            require(prop['pool_ref']==poolref and prop['pool_sha256']==digest(pool) and prop['available_indices']==available and prop['prediction_ref']==pref,'Pool/proposal binding')
            require(prop['model_id']==mid and prop['head_modes']==modes and prop['update_factors']==[ur,uo],'Head/proposal binding')
            for key in ('mode','selected_index','selected_candidate_id'):
                require(prop['decision'][key]==decision[key],'Selected action differs')
            require(same(prop['decision'],decision),'Score replay differs')
            ref=prefix+'/trials/'+arm+'/observation_%03d.json'%t
            raw=mp[ref]['value']; j=decision['selected_index']
            require(raw['candidate']==pool[j] and raw['candidate_id']==ids[j],'Wrong evaluated candidate')
            row=evaluation(raw,name+'/'+arm+'/%03d'%t);get(ref)
            r,s,f=observed_check(row); o=pred['objective'] if pred else None
            item=dict(task=task,repeat=repeat,arm=arm,step=t,phase=t//6,branch=row['candidate']['branch'],
                initial_warm=best(initial) is not None,mode=decision['mode'],region_label=r,resource_label=s,joint_label=f,
                region_kind=row['region']['kind'],resource_kind=row['resource']['kind'],objective_kind=row['objective']['kind'],
                objective_y=row['objective'].get('value'),objective_mu=o['mean'][j] if o else None,
                objective_var=o['variance'][j] if o else None,training_exact_objectives=information(h)['exact_objectives'])
            for head in ('region','resource','joint'): item[head+'_p']=pred['probabilities'][head][j] if pred else None
            item['joint_pool_quantile']=quantile_rank(pred['probabilities']['joint'],j,available) if pred else None
            item['ei_pool_quantile']=quantile_rank([ei(m,v,best(h)) for m,v in zip(o['mean'],o['variance'])],j,available) if o and best(h) is not None else None
            feedback.append(item);obs[arm].append(row);props[arm].append(prop);histories[arm].append(row)
    require(len({props[a][0]['decision']['selected_candidate_id'] for a in ARMS[:4]})==1,'Four-cell first action differs')
    trials=[]
    for arm in ARMS:
        calculated=trial_replay(task,repeat,arm,initial,obs[arm],props[arm]);stored=get(prefix+'/trials/'+arm+'/result.json')
        require(same(calculated,stored),'Trial not supported by observations: '+name+'/'+arm)
        trials.append(calculated)
    require(order==[r['ref'] for r in journal],'Journal order/membership')
    return dict(trials=trials,selected=feedback,bill=totals,shared=shared,per_arm_bill=fits,
                events=events,calls=calls,initial_warm=best(initial) is not None)


def effect(values):
    if any(v is None for v in values): return dict(complete=False,risk=None,objective=None,interaction=None)
    a,b,c,d=values
    return dict(complete=True,risk=.5*((b-a)+(d-c)),objective=.5*((c-a)+(d-b)),interaction=d-b-c+a)


def sign_p(w,l):
    n=w+l
    return min(1.,2*sum(math.comb(n,i) for i in range(min(w,l)+1))/2**n) if n else 1.


def effect_report(rows):
    result={}
    for key in ('risk','objective','interaction'):
        vs=[r[key] for r in rows if r['complete']]; w=sum(v< -1e-10 for v in vs);l=sum(v>1e-10 for v in vs)
        result[key]=dict(n=len(vs),missing=len(rows)-len(vs),mean=math.fsum(vs)/len(vs) if vs else None,
            median=statistics.median(vs) if vs else None,wins_losses_ties=[w,l,len(vs)-w-l],two_sided_sign_p=sign_p(w,l),values=vs)
    return result


def holm(p):
    out={}; previous=0.
    for i,k in enumerate(sorted(p,key=lambda k:(p[k],k))):
        previous=max(previous,min(1.,(len(p)-i)*p[k]));out[k]=dict(raw_p=p[k],holm_p=previous,reject=previous<=.05)
    return out


def calculate(root, profile):
    """Core analysis after archive authentication. Safe for synthetic unit tests."""
    full=root/profile
    count=verify_manifest(full, MANIFESTS[profile])
    verify_manifest(root/(profile+'-analysis'))
    spec=read(full/'plan.json');source=read(full/'source_receipt.json');report=read(full/'report.json')
    n=16 if profile=='full' else 1;steps=24 if profile=='full' else 3;block=960000 if profile=='full' else 950000
    units=[dict(task=t,repeat=r,seed=block+1000*i+32*r) for i,t in enumerate(TASKS) for r in range(n)]
    require(spec['units']==units and spec['arms']==list(ARMS) and spec['initial']==48 and spec['steps']==steps and spec['pool']==256 and spec['profile']==profile,'Frozen plan differs')
    require(source['code_sha']==report['code_sha']==CODE_SHA and source['phase']=='C2','Archived code identity')
    require(report['status'] in ('COMPLETE_C2_FACTORIAL','COMPLETE_WITH_MODEL_FALLBACKS'),'Stopped input')
    selected=[];trials=[];bills={k:0 for k in BILL_KEYS};perarm={a:{k:0 for k in BILL_KEYS} for a in ARMS};shared={k:0 for k in BILL_KEYS};progress=[];warm=0
    event_iter=iter(lines(full/'events.jsonl'));call_iter=iter(lines(full/'evaluations.jsonl'))
    for u in units:
        z=audit_unit(full,spec,u);trials.extend(z['trials']);selected.extend(z['selected']);warm+=z['initial_warm']
        for k in bills: bills[k]+=z['bill'][k];shared[k]+=z['shared'][k]
        for a in ARMS:
            for k in bills: perarm[a][k]+=z['per_arm_bill'][a][k]
        for e in z['events']:require(next(event_iter,None)==e,'Merged event order/seal differs')
        for c in z['calls']:require(next(call_iter,None)==c,'Historical evaluation ledger differs')
        progress.append(dict(unit='%s/r%d'%(u['task'],u['repeat']),completed_trials=len(trials),math_settled=bills['math_settled']))
    require(next(event_iter,None) is None and next(call_iter,None) is None,'Extra historical event/call')
    require(list(lines(full/'progress.jsonl'))==progress,'Progress differs')
    require(same(report['summary']['trials'],trials) and report['trial_count']==len(trials),'Full report trial summary differs')
    require(bills==report['bill'] and bills['math_started']==bills['math_settled']==spec['total_math_calls'],'Historical bill differs')
    require(all(bills[k]==0 for k in ('math_errors','vae_fits','fantasy_fits','aircraft_calls','rcaide_calls')),'Forbidden historical calls')
    armrows={}; taskarm={}; compact=[]
    for t in trials:
        c0,c1=t['information_curve'][0],t['information_curve'][-1]
        compact.append(dict(task=t['task'],repeat=t['repeat'],arm=t['arm'],initial_warm=t['initial_warm'],initial_best=t['initial_best'],
            final_best=t['final_best'],final_feasible=t['final_best'] is not None,no_solve=t['final_best'] is None,
            first_feasible_call=t['first_feasible_call'],online_calls=steps,online_failures=steps-(c1['feasible']-c0['feasible']),
            exact_objective_labels=c1['exact_objectives']-c0['exact_objectives'],model_fallbacks=t['model_failure_calls']))
    def aggregate(rr):
        return dict(trials=len(rr),final_feasible=sum(x['final_feasible'] for x in rr),no_solve=sum(x['no_solve'] for x in rr),
            cold=sum(not x['initial_warm'] for x in rr),online_calls=sum(x['online_calls'] for x in rr),
            failures=sum(x['online_failures'] for x in rr),exact_labels=sum(x['exact_objective_labels'] for x in rr))
    for arm in ARMS:
        armrows[arm]=aggregate([r for r in compact if r['arm']==arm]);armrows[arm]['compute_bill']=perarm[arm]
        for task in TASKS:taskarm[task+'/'+arm]=aggregate([r for r in compact if r['arm']==arm and r['task']==task])
    discrepancies=[]
    def compare(name,a,b):
        if not same(a,b):discrepancies.append(dict(field=name,recomputed=a,archived=b))
    analysis=read(root/(profile+'-analysis')/'report.json');effects={};factorials=[];analysis_rows={}
    keyed={(r['task'],r['repeat'],r['arm']):r for r in trials}
    for task in TASKS:
        rows=[];analysis_rows[task]=[]
        for r in range(n):
            cell=[keyed[task,r,a] for a in ARMS[:4]];e=effect([x['final_best'] for x in cell]);
            if cell[0]['initial_warm']: rows.append(e)
            factorials.append(dict(task=task,repeat=r,initial_warm=cell[0]['initial_warm'],**e))
            tt={a:keyed[task,r,a] for a in ARMS}
            analysis_rows[task].append(dict(repeat=r,warm=cell[0]['initial_warm'],effects=e,
                outcomes={a:tt[a]['final_best'] for a in ARMS[:4]},
                infeasible={a:steps-(tt[a]['information_curve'][-1]['feasible']-tt[a]['information_curve'][0]['feasible']) for a in ARMS},
                exact_objectives={a:tt[a]['information_curve'][-1]['exact_objectives']-tt[a]['information_curve'][0]['exact_objectives'] for a in ARMS},
                branch_counts={a:{str(b):tt[a]['information_curve'][-1]['branches'][str(b)]-tt[a]['information_curve'][0]['branches'][str(b)] for b in range(4)} for a in ARMS}))
        effects[task]=effect_report(rows)
        compare('analysis/'+task,effects[task],analysis['task_effects'][task]['effects'])
    compare('analysis/all_task_repeat_costs',analysis_rows,analysis['all_task_repeat_costs'])
    compare('analysis/factorial_trials',analysis_rows,read(root/(profile+'-analysis')/'factorial_trials.json'))
    p={t+'/'+k:v['two_sided_sign_p'] for t,rows in effects.items() for k,v in rows.items()}
    corrected=holm(p) if profile=='full' else None
    compare('analysis/holm_12',corrected,analysis['holm_12'])
    all_final=all(r['final_feasible'] for r in compact)
    if profile=='full':
        summary=read(root/'analysis_summary.json');integrity=read(root/'integrity_check.json')
        with (root/'arm_costs.csv').open(encoding='utf-8',newline='') as f: csvrows={r['arm']:r for r in csv.DictReader(f)}
        require(set(csvrows)==set(ARMS),'Published arm CSV membership')
        for arm,row in armrows.items():
            for k in ('final_feasible','no_solve','cold','failures','exact_labels'):
                compare('summary/'+arm+'/'+k,row[k],int(summary['arms'][arm][k]))
            for k,ck in (('final_feasible','final_feasible'),('no_solve','no_solve'),('cold','cold'),('failures','online_failures'),('exact_labels','online_exact_objective_labels')):
                compare('arm_costs/'+arm+'/'+ck,row[k],int(csvrows[arm][ck]))
        compare('integrity/arms_all_final_feasible',all_final,integrity.get('arms_all_final_feasible'))
        compare('integrity/warm_groups',warm,integrity.get('warm_groups'))
        compare('integrity/cold_groups',len(units)-warm,integrity.get('cold_groups'))
        compare('analysis_summary/bill',bills,summary['bill'])
    groups={}
    for row in selected:
        keys=['arm/'+row['arm'],'task_arm/'+row['task']+'/'+row['arm'],
              'step/'+row['task']+'/'+row['arm']+'/'+str(row['step']),
              'branch/'+row['task']+'/'+row['arm']+'/'+str(row['branch']),
              'phase/'+row['task']+'/'+row['arm']+'/'+str(row['phase'])]
        for k in keys:groups.setdefault(k,[]).append(row)
    metrics={k:diagnostics(rr) for k,rr in sorted(groups.items())}
    summary=dict(status='COMPLETE_WITH_REPORTING_DISCREPANCIES' if discrepancies else 'COMPLETE_PR84_AUDIT',
        profile=profile,profile_manifest_files=count,profile_manifest_sha256=sha(full/'manifest.json'),
        units=len(units),trials=len(trials),selected_observations=len(selected),warm_groups=warm,cold_groups=len(units)-warm,
        all_six_arms_final_feasible=all_final,per_arm=armrows,per_task_arm=taskarm,historical_bill=bills,shared_initial_bill=shared,
        factor_effects=effects,holm_12=corrected,discrepancy_count=len(discrepancies),new_bill=ZERO,
        integrity_flag_source_definition='UNRESOLVED unless original summary-generator source is supplied; discrepancy is not proof of a failed trajectory.',
        limits=['No evaluator or GP call; truth payload ignored for metrics.','Selected-sample diagnostics, not independent global calibration or causal mediation.',
                'Objective errors conditional on exact observed objective; missing labels are not imputed.',
                'Known tasks and existing trajectories only; confirmatory C2 testing family unchanged.'])
    return dict(report=summary,diagnostics=metrics,discrepancies=discrepancies,per_trial=compact,selected=selected,factorials=factorials)


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def write_csv(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def seal(output):
    write_json(output/'manifest.json',dict(schema='sha256_manifest_v1',files={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))


def execute(code_root,expected_sha,evidence_root,profile,output,verify=False):
    code_root=Path(code_root).resolve();output=Path(output).resolve();ev=Path(evidence_root).resolve()
    require(profile in MANIFESTS,'Invalid audit profile')
    require(output!=ev and ev not in output.parents and output!=code_root and code_root not in output.parents,'Output must be outside both worktrees')
    if not verify:require(not output.exists(),'Write-once output exists; do not rerun')
    root,receipt=identity(code_root,expected_sha,ev)
    if verify:verify_manifest(output)
    z=calculate(root,profile);z['report']['input_identity']=receipt
    payloads={'audit_report.json':z['report'],'diagnostics.json':z['diagnostics'],
              'discrepancies.json':z['discrepancies'],'factorial_recomputed.json':z['factorials']}
    if verify:
        for name,value in payloads.items():require(same(read(output/name),value),'Audit output recomputation differs: '+name)
        for name,rows in [('per_trial.csv',z['per_trial']),('selected_feedback.csv',z['selected'])]:
            s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
            require((output/name).read_text(encoding='utf-8')==s.getvalue(),'Audit CSV recomputation differs: '+name)
    else:
        output.mkdir(parents=True)
        for name,value in payloads.items():write_json(output/name,value)
        write_csv(output/'per_trial.csv',z['per_trial']);write_csv(output/'selected_feedback.csv',z['selected']);seal(output)
    identity(code_root,expected_sha,ev)
    return dict(status='VERIFIED_PR84_AUDIT' if verify else z['report']['status'],profile=profile,
                units=z['report']['units'],trials=z['report']['trials'],discrepancies=z['report']['discrepancy_count'],new_bill=ZERO)
