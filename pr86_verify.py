"""Independent PR86-A verifier; no truth calls and no model fitting."""
import json,math
from pathlib import Path
from risk_learning.core import candidates,digest,observed,known_label
from risk_learning.io import read,check_manifest
from pr86_protocol import plan,POLICIES,METHODS,sample_orders,coverage
from pr86_metrics import metrics
from risk_learning.native import training_payload

def req(x,m):
 if not x:raise ValueError(m)
def close(a,b,tol=1e-12):
 return abs(a-b)<=tol+tol*max(abs(a),abs(b))
def coverage_equal(a,b):
 if a.get("branch_counts")!=b.get("branch_counts") or a.get("branches_covered")!=b.get("branches_covered"):
  return False
 x=a.get("nearest_neighbor_distance");y=b.get("nearest_neighbor_distance")
 if x is None or y is None:return x is None and y is None
 return all(close(x[k],y[k]) for k in ("mean","median","maximum"))
def verify(root,expected_sha=None):
 root=Path(root).resolve();nfiles=check_manifest(root)
 req(not (root/"failure.json").exists(),"Failed attempt is retained")
 spec=read(root/"plan.json");source=read(root/"source_receipt.json");report=read(root/"report.json")
 req(spec==plan(spec["profile"]),"Frozen plan mismatch")
 req(report["code_sha"]==source["code_sha"] and (expected_sha is None or source["code_sha"]==expected_sha),"Code identity mismatch")
 req(report["status"] in ("COMPLETE_PR86_LEARNABILITY","COMPLETE_WITH_MODEL_FAILURES"),"Unexpected run status")
 expected_files={"plan.json","source_receipt.json","environment.json","report.json","evaluations.jsonl","progress.jsonl"}
 rows=[];bill={k:0 for k in report["bill"]};calls={}
 for u in spec["units"]:
  task,rep,ts,vs=u["task"],u["repeat"],u["train_seed"],u["test_seed"];name="%s/r%d"%(task,rep)
  train=read(root/"datasets"/name/"train_pool.json");test=read(root/"datasets"/name/"heldout_test.json")
  req(len(train)==spec["train_pool"] and len(test)==spec["test_count"],"Dataset count mismatch")
  req([x["candidate"] for x in train]==candidates(ts,spec["train_pool"]),"Train stream mismatch")
  req([x["candidate"] for x in test]==candidates(vs,spec["test_count"]),"Test stream mismatch")
  train_ids=[x["candidate_id"] for x in train];test_ids=[x["candidate_id"] for x in test]
  req(len(set(train_ids))==len(train_ids) and len(set(test_ids))==len(test_ids),"Duplicate candidate")
  req(not set(train_ids)&set(test_ids),"Train/test overlap")
  for role,data in (("train",train),("test",test)):
   for i,r in enumerate(data):
    cid=name+"/"+role+"/%04d"%i
    req(r["candidate_id"]==digest(r["candidate"]) and r["call_id"]==cid,"Candidate/call identity mismatch")
    req(r["truth"]["feasible"]==all(r["truth"][h]<=0 for h in ("region","resource")),"Feasibility mismatch")
    if role=="train":req(r["observed"]==observed(r["candidate"],r["truth"]),"Observation mask mismatch")
    calls[cid]=r
  order_record=read(root/"datasets"/name/"sampling_orders.json")
  orders,info=sample_orders([r["candidate"] for r in train],[r["observed"] for r in train],ts+7919)
  req(order_record==dict(orders=orders,pool_info=info),"Sampling design mismatch")
  expected_files.update(("datasets/"+name+"/train_pool.json","datasets/"+name+"/heldout_test.json","datasets/"+name+"/sampling_orders.json"))
  for policy in POLICIES:
   for n in spec["sizes"]:
    indices=orders[policy][:n];selected=[train[i] for i in indices];public=[x["observed"] for x in selected]
    for method in METHODS:
     rel="fits/"+name+"/n%d/"%n+policy+"__"+method+".json";expected_files.add(rel)
     row=read(root/rel)
     req(row["train_indices"]==indices and row["train_prefix_sha256"]==digest(selected),"Nested prefix mismatch")
     req(row["train_observed_sha256"]==digest(public),"Training observation hash mismatch")
     req(row["test_set_sha256"]==digest(test) and row["test_candidate_ids"]==test_ids,"Test identity mismatch")
     req(coverage_equal(row["coverage"],coverage(public)),"Training coverage mismatch")
     lc=row["train_label_counts"]
     region_pos=sum(known_label(x,"region") for x in public)
     resource_pos=sum(known_label(x,"resource") for x in public)
     joint_pos=sum(all(known_label(x,h) for h in ("region","resource")) for x in public)
     expected_counts=dict(region_exact=sum(x["region"]["kind"]=="exact" for x in public),
      region_interval=sum(x["region"]["kind"]=="interval" for x in public),
      region_feasible=region_pos,region_infeasible=n-region_pos,
      resource_exact=sum(x["resource"]["kind"]=="exact" for x in public),
      resource_feasible=resource_pos,resource_infeasible=n-resource_pos,
      objective_exact=sum(x["objective"]["kind"]=="exact" for x in public),
      joint_feasible=joint_pos,joint_infeasible=n-joint_pos)
     req(lc==expected_counts,"Per-constraint training class counts mismatch")
     rec=row["receipt"]
     req(rec["method"]==method and rec["training_rows"]==n and rec["query_ids"]==test_ids,"Fit receipt mismatch")
     req(rec["privileged_training"]==(method=="privileged_exact"),"Supervision contract mismatch")
     privileged=[x["truth"] for x in selected] if method=="privileged_exact" else None
     expected_payload=training_payload(public,method,privileged)
     req(rec["training_input_sha256"]==digest(expected_payload) and rec["objective_labels_used"] is False,
      "Fit input hash or objective-label isolation mismatch")
     req("test_truth" not in rec and "heldout_truth" not in rec,"Test-label leakage in receipt")
     req(row["status"] in ("OK","MODEL_NUMERICAL_FAILURE"),"Unknown model status")
     if row["status"]=="OK":
      pred=row["predictions"];req(pred["candidate_ids"]==test_ids,"Prediction/test identity mismatch")
      for h in ("region","resource","joint"):
       labels=[(r["truth"]["region"]<=0 and r["truth"]["resource"]<=0) if h=="joint" else r["truth"][h]<=0 for r in test]
       actual=metrics(labels,pred["probabilities"][h]);saved=row["metrics"][h]
       for k,v in actual.items():
        if k=="bins":
         req(len(v)==len(saved[k]) and all(x["bin"]==y["bin"] and x["n"]==y["n"] and
          ((x["mean_probability"] is None and y["mean_probability"] is None) or close(x["mean_probability"],y["mean_probability"])) and
          ((x["observed_rate"] is None and y["observed_rate"] is None) or close(x["observed_rate"],y["observed_rate"])) for x,y in zip(v,saved[k])),"Reliability bins mismatch")
        elif isinstance(v,float):req(saved[k] is not None and close(v,saved[k]),"Heldout metric mismatch: "+k)
        else:req(saved[k]==v,"Heldout metric mismatch: "+k)
       if h in ("region","resource") and h in pred.get("margin_mean",{}):
        mu=pred["margin_mean"][h];truth_margin=[r["truth"][h] for r in test]
        rmse=(sum((a-b)**2 for a,b in zip(mu,truth_margin))/len(test))**.5
        req("margin_rmse" in saved and close(rmse,saved["margin_rmse"]),"Constraint margin RMSE mismatch")
       if h=="joint":req("margin_rmse" not in saved,"Undefined joint margin RMSE must not be reported")
      req(set(pred["gaussian"])=={"region","resource"},"Gaussian predictive moments missing")
      noise=1. if method=="native_sign" else 0.
      for h,mom in pred["gaussian"].items():
       req(mom["observation_noise_variance"]==noise,"Observation-noise contract mismatch")
       for mean,var,p in zip(mom["mean"],mom["variance"],pred["probabilities"][h]):
        expected=.5*math.erfc(mean/math.sqrt(2*(var+noise))) if var+noise>0 else float(mean<=0)
        req(close(expected,p),"Gaussian probability/moment mismatch")
      req(all(close(a*b,j) for a,b,j in zip(pred["probabilities"]["region"],
       pred["probabilities"]["resource"],pred["probabilities"]["joint"])),"Joint probability mismatch")
      for h,mom in pred["gaussian"].items():
       req(len(mom["mean"])==len(test)==len(mom["variance"]),"Gaussian prediction shape mismatch")
       for x,v in zip(mom["mean"],mom["variance"]):req(math.isfinite(x) and math.isfinite(v) and v>=0,"Invalid Gaussian moment")
     else:req(row["predictions"] is None and row["metrics"] is None,"Failed fit has fabricated output")
     for k,v in row["bill_delta"].items():
      req(type(v)is int and v>=0,"Invalid model bill");bill[k]+=v
     if method in ("native_censored","privileged_exact"):
      if row["status"]=="OK":
       req(row["bill_delta"]["gp_head_started"]==2 and row["bill_delta"]["gp_head_completed"]==2
        and row["bill_delta"]["gp_head_errors"]==0 and row["bill_delta"]["gp_kernel_started"]==16
        and row["bill_delta"]["gp_kernel_completed"]==16 and row["bill_delta"]["gp_kernel_errors"]==0,
        "Native GP accounting mismatch")
      else:
       req(row["bill_delta"]["gp_head_errors"]>=1 and
        row["bill_delta"]["gp_head_started"]==row["bill_delta"]["gp_head_completed"]+row["bill_delta"]["gp_head_errors"],
        "Native model-failure accounting mismatch")
     rows.append(row)
 expected_files.update(("evaluations.jsonl","progress.jsonl"))
 actual_files={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name!="manifest.json"}
 req(actual_files==expected_files,"Missing/extra PR86 evidence")
 events=[json.loads(s) for s in (root/"evaluations.jsonl").read_text(encoding="utf-8").splitlines()]
 starts={};settled={}
 for e in events:
  cid=e["call_id"];req(cid in calls,"Unknown call")
  if e["event"]=="started":
   req(cid not in starts and e["candidate_id"]==calls[cid]["candidate_id"],"Duplicate/mismatched start");starts[cid]=e
  elif e["event"]=="settled":
   req(cid in starts and cid not in settled and e["record_sha256"]==digest(calls[cid]),"Mismatched settlement");settled[cid]=e
  else:raise ValueError("Unknown event")
 req(set(starts)==set(settled)==set(calls),"Unsettled evaluation ledger")
 req(len(calls)==spec["total_math_calls"],"Historical math call total mismatch")
 bill["math_started"]=len(calls);bill["math_settled"]=len(calls)
 req(bill==report["bill"],"GP/evaluation bill mismatch")
 failures=sum(r["status"]!="OK" for r in rows)
 req(len(rows)==spec["model_rows"]==report["model_rows"] and failures==report["failed_model_rows"],"Model row summary mismatch")
 return dict(status="VERIFIED_PR86_LEARNABILITY",files=nfiles,model_rows=len(rows),
  historical_math_calls=len(calls),new_math_calls=0,new_gp_fits=0,new_gp_predictions=0,
  heldout_labels_fit_isolated=True,nested_prefixes_verified=True)
