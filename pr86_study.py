"""PR86-A offline learnability study; no online optimization."""
from copy import deepcopy
from pathlib import Path
import platform,time,traceback
from risk_learning.core import TASKS,candidates,truth,observed,digest,known_label,features
from risk_learning.io import write,append,seal,read,check_manifest
from risk_learning.study import new_bill
from risk_learning.native import predict as native_predict
from pr86_protocol import plan,POLICIES,METHODS,sample_orders,coverage
from pr86_metrics import metrics

def _score(test,pred):
 out={}
 for head in ("region","resource","joint"):
  labels=[(r["truth"]["region"]<=0 and r["truth"]["resource"]<=0) if head=="joint" else r["truth"][head]<=0 for r in test]
  probs=pred["probabilities"][head]
  out[head]=metrics(labels,probs)
  # A joint probability is available, but there is no single joint margin.
  # Therefore margin RMSE is defined only for the two native constraint heads.
  if head in ("region","resource"):
   margin=pred.get("margin_mean",{}).get(head)
   if margin is not None:
    truth_values=[r["truth"][head] for r in test]
    out[head]["margin_rmse"]=(sum((a-b)**2 for a,b in zip(margin,truth_values))/len(test))**.5
 return out

def run(root,profile,source,predictor=None):
 root=Path(root).resolve()
 if root.exists():raise FileExistsError("PR86 write-once output")
 root.mkdir(parents=True);spec=plan(profile);bill=new_bill();rows=[];start=time.perf_counter()
 write(root/"plan.json",spec);write(root/"source_receipt.json",source)
 write(root/"environment.json",dict(python=platform.python_version(),platform=platform.platform()))
 try:
  for unit in spec["units"]:
   task,rep,ts,vs=unit["task"],unit["repeat"],unit["train_seed"],unit["test_seed"]
   name="%s/r%d"%(task,rep);folder=root/"datasets"/name
   train=[]
   for i,c in enumerate(candidates(ts,spec["train_pool"])):
    cid=digest(c);call=name+"/train/%04d"%i
    append(root/"evaluations.jsonl",dict(event="started",call_id=call,role="train",task=task,candidate_id=cid,candidate=c));bill["math_started"]+=1
    y=truth(task,deepcopy(c));r=dict(candidate=c,candidate_id=cid,truth=y,observed=observed(c,y),call_id=call)
    train.append(r);append(root/"evaluations.jsonl",dict(event="settled",call_id=call,record_sha256=digest(r)));bill["math_settled"]+=1
   test=[]
   for i,c in enumerate(candidates(vs,spec["test_count"])):
    cid=digest(c);call=name+"/test/%04d"%i
    append(root/"evaluations.jsonl",dict(event="started",call_id=call,role="test",task=task,candidate_id=cid,candidate=c));bill["math_started"]+=1
    y=truth(task,deepcopy(c));r=dict(candidate=c,candidate_id=cid,truth=y,call_id=call)
    test.append(r);append(root/"evaluations.jsonl",dict(event="settled",call_id=call,record_sha256=digest(r)));bill["math_settled"]+=1
   if set(r["candidate_id"] for r in train)&set(r["candidate_id"] for r in test):raise ValueError("Train/test overlap")
   write(folder/"train_pool.json",train);write(folder/"heldout_test.json",test)
   orders,pool_info=sample_orders([r["candidate"] for r in train],[r["observed"] for r in train],ts+7919)
   write(folder/"sampling_orders.json",dict(orders=orders,pool_info=pool_info))
   for policy,order in orders.items():
    for n in spec["sizes"]:
     idx=order[:n];selected=[train[i] for i in idx]
     public=[deepcopy(r["observed"]) for r in selected]
     labels=[deepcopy(r["truth"]) for r in selected]
     train_cov=coverage(public);label_stats=pool_info
     for method in METHODS:
      before=deepcopy(bill)
      kwargs=dict(privileged=labels) if method=="privileged_exact" else {}
      fn=predictor or native_predict
      pred,receipt=fn(public,[r["candidate"] for r in test],method,ts+n,bill,**kwargs)
      status="OK" if pred is not None else "MODEL_NUMERICAL_FAILURE"
      score=_score(test,pred) if pred is not None else None
      row=dict(schema=spec["schema"],task=task,repeat=rep,policy=policy,n=n,method=method,
       train_indices=idx,train_prefix_sha256=digest([train[i] for i in idx]),
       train_observed_sha256=digest(public),test_set_sha256=digest(test),
       test_candidate_ids=[r["candidate_id"] for r in test],
       train_label_counts=dict(region_exact=sum(r["region"]["kind"]=="exact" for r in public),
        region_interval=sum(r["region"]["kind"]=="interval" for r in public),
        resource_exact=sum(r["resource"]["kind"]=="exact" for r in public),
        objective_exact=sum(r["objective"]["kind"]=="exact" for r in public),
        joint_feasible=sum(all(known_label(r,h) for h in ("region","resource")) for r in public)),
       coverage=train_cov,pool_info=label_stats,receipt=receipt,predictions=pred,
       status=status,metrics=score,bill_delta={k:bill[k]-before[k] for k in bill})
      write(root/"fits"/name/("n%d"%n)/(policy+"__"+method+".json"),row)
      rows.append(row);append(root/"progress.jsonl",dict(event="fit_settled",task=task,repeat=rep,
       policy=policy,n=n,method=method,status=status))
  if bill["math_started"]!=spec["total_math_calls"] or bill["math_settled"]!=spec["total_math_calls"]:
   raise ValueError("PR86 mathematical evaluation bill mismatch")
  failures=sum(r["status"]!="OK" for r in rows)
  report=dict(schema=spec["schema"],code_sha=source["code_sha"],profile=profile,
   status="COMPLETE_PR86_LEARNABILITY" if not failures else "COMPLETE_WITH_MODEL_FAILURES",
   bill=bill,model_rows=len(rows),failed_model_rows=failures,
   summary=dict(task_policy_size_method_rows=len(rows),all_test_sets_shared_within_unit=True,
    nested_sample_prefixes=True,all_test_labels_fit_isolated=True),
   elapsed_seconds=time.perf_counter()-start,
   limits=["Offline supervised learning only; no online optimization.",
    "Test truth is used only by _score after prediction and never passed to fit.",
    "privileged_exact is an information upper-bound diagnostic, not deployable.",
    "Metrics are task-stratified; raw margins are not pooled across tasks.",
    "Sampling contrasts are descriptive; class-balanced design uses observed training labels."])
  write(root/"report.json",report);seal(root);return report
 except Exception as e:
  write(root/"failure.json",dict(status="STOPPED_ERROR",kind=type(e).__name__,message=str(e),
   traceback=traceback.format_exc(),bill=bill,completed_model_rows=len(rows)));seal(root);raise
