"""C2 matched six-arm online study; independent per-head update factors."""
from copy import deepcopy
from pathlib import Path
import platform,time,traceback
from unittest.mock import patch
from c1_protocol import (SCHEMA as C1_SCHEMA,INITIAL,POOL,digest,best,information,
                         trial_result,feasible,ei)
from c2_protocol import plan,ARMS,select_c2
from c2_native import fit_composite,_head,CompositePosterior
from c1_native import fit as fit_all
from c1_protocol import select as c1_select
from risk_learning.core import candidates,observed,canonical
from risk_learning.io import write,append,seal

FLAGS={
 "risk_frozen_objective_frozen":(False,False),
 "risk_updated_objective_frozen":(True,False),
 "risk_frozen_objective_updated":(False,True),
 "risk_updated_objective_updated":(True,True),
 "risk_only_updated":(True,False),
 "uniform_hash":(False,False)}
def bill():
 return dict(math_started=0,math_settled=0,math_errors=0,gp_head_started=0,gp_head_completed=0,
  gp_head_errors=0,gp_kernel_started=0,gp_kernel_completed=0,gp_kernel_errors=0,
  kernel_candidates_rejected=0,prediction_started=0,prediction_completed=0,prediction_errors=0,
  vae_fits=0,fantasy_fits=0,aircraft_calls=0,rcaide_calls=0)
def run(root,profile,source):
 root=Path(root).resolve()
 if root.exists(): raise FileExistsError("Write-once C2 output")
 root.mkdir(parents=True);spec=plan(profile);b=bill();trials=[];start=time.perf_counter()
 write(root/"plan.json",spec);write(root/"source_receipt.json",source)
 write(root/"environment.json",dict(python=platform.python_version(),platform=platform.platform()))
 events=root/"events.jsonl";evaluations=root/"evaluations.jsonl"
 forbidden=lambda *_a,**_k: (_ for _ in ()).throw(RuntimeError("Truth forbidden in model/selector"))
 def save(rel,value,event):
  journal=root/"/".join(rel.split("/")[:3])/"records.jsonl"
  journal.parent.mkdir(parents=True,exist_ok=True)
  append(journal,dict(ref=rel,value=value));append(events,dict(event=event,ref=rel,record_sha256=digest(value)))
 def evaluate(task,c,call):
  c=canonical(c);cid=digest(c);append(evaluations,dict(event="started",call_id=call,task=task,candidate_id=cid,candidate=c))
  append(events,dict(event="evaluation_started",call_id=call));b["math_started"]+=1
  try:
   from risk_learning.core import truth
   with patch("risk_learning.core.truth",forbidden): target=truth(task,deepcopy(c))
   if set(target)!={"region","resource","objective","feasible"} or target["feasible"]!=(target["region"]<=0 and target["resource"]<=0): raise ValueError("Evaluator contract changed")
   r=dict(call_id=call,candidate=c,candidate_id=cid,truth=target,observed=observed(c,target));digest(r)
  except Exception as e:
   b["math_errors"]+=1;append(evaluations,dict(event="error",call_id=call,kind=type(e).__name__,message=str(e)));raise
  b["math_settled"]+=1;append(evaluations,dict(event="settled",call_id=call,record_sha256=digest(r)))
  append(events,dict(event="evaluation_settled",call_id=call,record_sha256=digest(r)));return r
 def fit_initial(public,seed,rel):
  before=dict(b);model,receipt=fit_all(public,seed,b,risk_only=False)
  row=dict(receipt=receipt,bill_delta={k:b[k]-before[k] for k in b})
  save(rel,row,"initial_composite_sealed")
  if model is None: raise RuntimeError("Initial composite fit failed; stop")
  model.provenance={k:dict(mode="frozen",identity=v.identity) for k,v in model.heads.items()}
  if "objective" not in model.heads:model.provenance["objective"]=dict(mode="frozen_unavailable")
  return model
 def fit_arm(public,seed,arm,frozen,rel):
  ur,uo=FLAGS[arm];before=dict(b)
  if arm=="uniform_hash": return None,dict(status="NO_MODEL")
  if arm=="risk_only_updated":
   heads={};prov={}
   for i,n in enumerate(("region","resource")):
    h,r=_head(public,n,seed+i,b);heads[n]=h;prov[n]=dict(mode="updated",training_sha256=r["training_sha256"],identity=r["head"]["identity"])
   model=CompositePosterior(heads,digest({k:v.identity for k,v in heads.items()}),prov)
   receipt=dict(status="OK",model_id=model.identity,heads=prov,training_sha256=digest(public),seed=seed,update_risk=True,update_objective=False)
  else:
   model,receipt=fit_composite(public,seed,b,update_risk=ur,update_objective=uo,frozen=frozen)
  receipt["seed"]=seed
  row=dict(receipt=receipt,bill_delta={k:b[k]-before[k] for k in b})
  save(rel,row,"arm_composite_sealed")
  return model,receipt
 def predict(model,queries,rel):
  before=dict(b)
  if model is None: pred,err=None,dict(kind="NO_MODEL")
  else: pred,err=model.predict(deepcopy(queries),b)
  row=dict(model_id=None if model is None else model.identity,prediction=pred,error=err,
           query_sha256=digest(queries),bill_delta={k:b[k]-before[k] for k in b})
  save(rel,row,"prediction_sealed");return pred
 try:
  for u in spec["units"]:
   name="%s/r%d"%(u["task"],u["repeat"]);prefix="units/"+name
   initial=[evaluate(u["task"],c,name+"/initial/%03d"%i) for i,c in enumerate(candidates(u["seed"],INITIAL))]
   save(prefix+"/initial.json",initial,"initial_sealed")
   init=[r["observed"] for r in initial]
   frozen=fit_initial(init,u["seed"],prefix+"/models/initial.json")
   histories={a:deepcopy(init) for a in ARMS};observations={a:[] for a in ARMS};proposals={a:[] for a in ARMS}
   risk_frozen=CompositePosterior({k:frozen.heads[k] for k in ("region","resource")},
      digest({k:frozen.heads[k].identity for k in ("region","resource")}),
      {k:dict(mode="frozen",identity=frozen.heads[k].identity) for k in ("region","resource")})
   for t in range(spec["steps"]):
    pool=candidates(u["seed"]+1+t,POOL);poolref=prefix+"/pools/t%03d.json"%t;save(poolref,pool,"pool_sealed")
    ids=[digest(c) for c in pool]
    for a in ARMS:
     if a in ("risk_only_updated","uniform_hash"):
      if a=="risk_only_updated":
       if t==0: model=risk_frozen
       else: model,_=fit_arm(histories[a],u["seed"],a,None,prefix+"/models/"+a+"/t%03d.json"%t)
      else:model=None
     else:
      ur,uo=FLAGS[a]
      if t==0:model=frozen
      elif ur or uo:
       model,_=fit_arm(histories[a],u["seed"],a,frozen,prefix+"/models/"+a+"/t%03d.json"%t)
      else:model=frozen
     predref=prefix+"/predictions/"+a+"/t%03d.json"%t
     pred=predict(model,pool,predref)
     public=histories[a];seen={r["candidate_id"] for r in public}
     available=[i for i,cid in enumerate(ids) if cid not in seen]
     decision=select_c2(a,ids,available,pred,best(public),[name,t])
     proposal=dict(schema="c2_head_factorial_v1",task=u["task"],repeat=u["repeat"],arm=a,iteration=t,
       history_sha256=digest(public),history_information=information(public),pool_ref=poolref,pool_sha256=digest(pool),
       available_indices=available,model_id=None if model is None else model.identity,
       prediction_ref=predref,head_modes=None if model is None else model.provenance,
       decision=decision,update_factors=FLAGS[a])
     propref=prefix+"/trials/"+a+"/proposal_%03d.json"%t;save(propref,proposal,"proposal_sealed");proposals[a].append(proposal)
     r=evaluate(u["task"],pool[decision["selected_index"]],name+"/"+a+"/%03d"%t)
     save(prefix+"/trials/"+a+"/observation_%03d.json"%t,r,"observation_sealed")
     observations[a].append(r["observed"]);histories[a].append(r["observed"])
   for a in ARMS:
    result=trial_result(u["task"],u["repeat"],a,init,observations[a],proposals[a])
    save(prefix+"/trials/"+a+"/result.json",result,"trial_sealed");trials.append(result)
   append(root/"progress.jsonl",dict(unit=name,completed_trials=len(trials),math_settled=b["math_settled"]))
  if b["math_started"]!=spec["total_math_calls"] or b["math_settled"]!=spec["total_math_calls"] or b["math_errors"]:
   raise ValueError("C2 evaluation bill does not close")
  report=dict(schema="c2_head_factorial_v1",code_sha=source["code_sha"],profile=profile,
   status="COMPLETE_C2_FACTORIAL",bill=b,trial_count=len(trials),summary=dict(trials=trials),
   elapsed_seconds=time.perf_counter()-start,
   limits=["Known tasks, not held-out generalization.","Each arm uses only its own settled feedback.",
    "Frozen heads are initial posterior objects; live incumbent still updates.",
    "No missing objective imputation; unavailable objective posterior triggers hash fallback.",
    "No safety claim; infeasible calls are a separate cost endpoint."])
  write(root/"report.json",report);seal(root);return report
 except Exception as e:
  write(root/"failure.json",dict(status="STOPPED_ERROR",kind=type(e).__name__,message=str(e),
   traceback=traceback.format_exc(),bill=b,completed_trials=len(trials)));seal(root);raise
