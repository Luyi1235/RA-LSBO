"""Independent PR86 verifier; no truth evaluator, no GP fit."""
import json,math
from pathlib import Path
from risk_learning.core import candidates,digest,observed,known_label,label_counts,features
from risk_learning.io import read,check_manifest
from pr86_protocol import plan,POLICIES,METHODS,coverage,sample_orders
from pr86_metrics import metrics

def req(x,m):
 if not x:raise ValueError(m)
def verify(root,expected_sha=None):
 root=Path(root).resolve();nfiles=check_manifest(root)
 req(not (root/"failure.json").exists(),"Failed PR86 run retained")
 spec=read(root/"plan.json");source=read(root/"source_receipt.json");report=read(root/"report.json")
 req(spec==plan(spec["profile"]),"Frozen PR86 plan mismatch")
 req(source["code_sha"]==report["code_sha"] and (expected_sha is None or source["code_sha"]==expected_sha),"Code identity mismatch")
 req(report["status"] in ("COMPLETE_PR86_LEARNABILITY","COMPLETE_WITH_MODEL_FAILURES"),"Incomplete PR86 status")
 rows=[];expected_paths={"plan.json","source_receipt.json","environment.json","report.json","evaluations.jsonl","progress.jsonl"}
 reconstructed={k:0 for k in report["bill"]}
 for u in spec["units"]:
  task,rep,ts,vs=u["task"],u["repeat"],u["train_seed"],u["test_seed"];name="%s/r%d"%(task,rep)
  train=read(root/"datasets"/name/"train_pool.json");test=read(root/"datasets"/name/"heldout_test.json")
  req(len(train)==spec["train_pool"] and len(test)==spec["test_count"],"Dataset sizes changed")
  req([x["candidate"] for x in train]==candidates(ts,spec["train_pool"]),"Train pool stream mismatch")
  req([x["candidate"] for x in test]==candidates(vs,spec["test_count"]),"Test stream mismatch")
  req(not ({x["candidate_id"] for x in train}&{x["candidate_id"] for x in test}),"Train/test overlap")
  for role,data in (("train",train),("test",test)):
   for i,r in enumerate(data):
    req(r["candidate_id"]==digest(r["candidate"]),"Candidate identity mismatch")
    req(r["truth"]["feasible"]==all(r["truth"][h]<=0 for h in ("region","resource")),"Truth feasibility mismatch")
    if role=="train":req(r["observed"]==observed(r["candidate"],r["truth"]),"Observable mask mismatch")
  order_record=read(root/"datasets"/name/"sampling_orders.json")
  orders,info=sample_orders([r["candidate"] for r in train],[r["observed"] for r in train],ts+7919)
  req(order_record==dict(orders=orders,pool_info=info),"Sampling design changed")
  for policy in POLICIES:
   order=orders[policy]
   for n in spec["sizes"]:
    ix=order[:n];selected=[train[i] for i in ix];public=[x["observed"] for x in selected]
    for method in METHODS:
     rel="fits/"+name+"/n%d/"%n+policy+"__"+method+".json";expected_paths.add(rel)
     row=read(root/rel)
     req(row["train_indices"]==ix and row["train_prefix_sha256"]==digest(selected),"Nested training prefix mismatch")
     req(row["train_observed_sha256"]==digest(public) and row["test_set_sha256"]==digest(test),"Training/test identity mismatch")
     req(row["test_candidate_ids"]==[x["candidate_id"] for x in test],"Test candidate identity mismatch")
     req(row["train_label_counts"]["joint_feasible"]==sum(all(known_label(x,h) for h in ("region","resource")) for x in public),"Training labels mismatch")
     req(row["coverage"]==coverage(public),"Training coverage mismatch")
     rec=row["receipt"]
     req(rec["method"]==method and rec["training_rows"]==n and rec["query_ids"]==[x["candidate_id"] for x in test],"Model receipt mismatch")
     req(rec["privileged_training"]==(method=="privileged_exact"),"Privileged contract mismatch")
     req(row["status"] in ("OK","MODEL_NUMERICAL_FAILURE"),"Unknown model outcome")
     if row["status"]=="OK":
      pred=row["predictions"];req(pred["candidate_ids"]==[x["candidate_id"] for x in test],"Prediction/test identity mismatch")
      for h in ("region","resource","joint"):
       labels=[(x["truth"]["region"]<=0 and x["truth"]["resource"]<=0) if h=="joint" else x["truth"][h]<=0 for x in test]
       req(metrics(labels,pred["probabilities"][h])==row["metrics"][h] or all(
        (metrics(labels,pred["probabilities"][h])[k] is None and row["metrics"][h][k] is None) or
        (isinstance(metrics(labels,pred["probabilities"][h])[k],float) and abs(metrics(labels,pred["probabilities"][h])[k]-row["metrics"][h][k])<1e-12) or
        metrics(labels,pred["probabilities"][h])[k]==row["metrics"][h][k]
        for k in metrics(labels,pred["probabilities"][h]) if k not in ("bins",)),"Heldout metrics mismatch")
      # Test truth is not represented in the model receipt or fit input hash.
      req("test_truth" not in rec and "heldout_truth" not in rec,"Test labels leaked into fit receipt")
     delta=row["bill_delta"]
     for k,v in delta.items():
      req(type(v)is int and v>=0,"Invalid fit bill delta");reconstructed[k]+=v
     rows.append(row)
  expected_paths.add("datasets/"+name+"/train_pool.json");expected_paths.add("datasets/"+name+"/heldout_test.json")
  expected_paths.add("datasets/"+name+"/sampling_orders.json")
 req({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name!="manifest.json"}==expected_paths,"Missing/extra PR86 evidence files")
 events=[json.loads(s) for s in (root/"evaluations.jsonl").read_text(encoding="utf-8").splitlines()]
 starts={};settles={}
 expected_calls=spec["total_math_calls"]
 for e in events:
  cid=e["call_id"]
  if e["event"]=="started":req(cid not in starts,"Duplicate started call");starts[cid]=e
  elif e["event"]=="settled":req(cid in starts and cid not in settles,"Invalid settlement");settles[cid]=e
  else:raise ValueError("Unknown math event")
 req(len(starts)==len(settles)==expected_calls,"Unsettled/extra mathematical calls")
 req(report["bill"]["math_started"]==report["bill"]["math_settled"]==expected_calls,"Math bill mismatch")
 req(reconstructed=={k:v for k,v in report["bill"].items() if k not in ("math_started","math_settled")},"GP bill mismatch")
 req(report["model_rows"]==len(rows)==spec["model_rows"],"Model row count mismatch")
 req(report["failed_model_rows"]==sum(r["status"]!="OK" for r in rows),"Model failure count mismatch")
 return dict(status="VERIFIED_PR86_LEARNABILITY",files=nfiles,model_rows=len(rows),
  historical_math_calls=expected_calls,new_math_calls=0,new_gp_fits=0,new_gp_predictions=0,
  test_label_fit_isolation=True,training_prefixes_verified=True)
