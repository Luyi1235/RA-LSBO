"""Model-free C2 verifier: replay recorded head modes, prefix isolation, scores and ledger."""
import json,math
from pathlib import Path
from c2_protocol import plan,ARMS,digest,select_c2
from c2_protocol import bill as empty_bill
from c1_protocol import candidates,canonical,best,information,trial_result
from risk_learning.io import read,check_manifest
from risk_learning.core import observed

def require(x,m):
 if not x: raise ValueError(m)
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding="utf-8").splitlines()]
def verify(root,expected_sha=None):
 root=Path(root).resolve();nfiles=check_manifest(root)
 require(not (root/"failure.json").exists(),"Stopped C2 evidence")
 spec=read(root/"plan.json");source=read(root/"source_receipt.json");report=read(root/"report.json")
 require(spec==plan(spec["profile"]),"C2 frozen plan changed")
 require(source["code_sha"]==report["code_sha"] and (expected_sha is None or source["code_sha"]==expected_sha),"Code identity mismatch")
 require(source.get("phase")=="C2" and source.get("parent_sha")=="0a532a16e59c0f94e5263ad77a8a7e6f145196d1","C2 parent identity mismatch")
 total=empty_bill();all_trials=[];eval_events=[];event_events=[];progress=[];expected_paths={"plan.json","source_receipt.json","environment.json","report.json","events.jsonl","evaluations.jsonl","progress.jsonl"}
 for u in spec["units"]:
  unit="%s/r%d"%(u["task"],u["repeat"]);prefix="units/"+unit;journal=Path(root/(prefix+"/records.jsonl"))
  expected_paths.add(prefix+"/records.jsonl");rows=lines(journal);mp={x["ref"]:x["value"] for x in rows}
  require(len(mp)==len(rows),"Duplicate record reference");order=[]
  def get(ref):
   require(ref in mp,"Missing record "+ref);order.append(ref);v=mp[ref]
   event_events.append(dict(event="record_sealed",ref=ref,record_sha256=digest(v)));return v
  def addbill(row):
   for k,v in row["bill_delta"].items():total[k]+=v
   d=row["bill_delta"]
   require(d["gp_head_started"]==d["gp_head_completed"]+d["gp_head_errors"],"Head bill unclosed")
   require(d["gp_kernel_started"]==d["gp_kernel_completed"]+d["gp_kernel_errors"],"Kernel bill unclosed")
   require(d["prediction_started"]==d["prediction_completed"]+d["prediction_errors"],"Prediction bill unclosed")
  initial=get(prefix+"/initial.json")
  require(len(initial)==48,"Initial design size changed")
  for i,r in enumerate(initial):
   c=candidates(u["seed"],48)[i]
   require(r["candidate"]==canonical(c) and r["candidate_id"]==digest(c),"Initial candidate changed")
   require(r["observed"]==observed(c,r["truth"]),"Initial observation changed")
   eval_events.extend([dict(event="started",call_id=r["call_id"],task=u["task"],candidate_id=r["candidate_id"],candidate=r["candidate"]),dict(event="settled",call_id=r["call_id"],record_sha256=digest(r))])
   total["math_started"]+=1;total["math_settled"]+=1
  ir=get(prefix+"/models/initial.json");addbill(ir)
  initial_receipt=ir["receipt"];require(initial_receipt["status"]=="OK","Initial composite failed")
  initial_ids={k:v["identity"] for k,v in initial_receipt["heads"].items()}
  require({"region","resource"}.issubset(initial_ids) and set(initial_ids).issubset({"region","resource","objective"}),"Initial risk heads absent/extra")
  histories={a:[r["observed"] for r in initial] for a in ARMS}
  trial_obs={a:[] for a in ARMS};trial_props={a:[] for a in ARMS}
  for t in range(spec["steps"]):
   poolref=prefix+"/pools/t%03d.json"%t;pool=get(poolref)
   require(pool==candidates(u["seed"]+1+t,spec["pool"]),"Pool stream changed")
   ids=[digest(c) for c in pool]
   for a in ARMS:
    modelref=prefix+"/models/"+a+"/t%03d.json"%t
    predref=prefix+"/predictions/"+a+"/t%03d.json"%t
    if a=="uniform_hash":
     model_id=None;head_modes=None
    elif a=="risk_only_updated" and t==0:
     model_id=digest({k:initial_ids[k] for k in ("region","resource")})
     head_modes={k:dict(mode="frozen",identity=initial_ids[k]) for k in ("region","resource")}
    elif a=="risk_frozen_objective_frozen" or (a in ("risk_frozen_objective_updated",
                "risk_updated_objective_frozen","risk_updated_objective_updated") and t==0):
     model_id=initial_receipt["model_id"]
     head_modes={k:dict(mode="frozen",identity=v) for k,v in initial_ids.items()}
     if "objective" not in initial_ids:head_modes["objective"]=dict(mode="frozen_unavailable")
    else:
     if a=="risk_only_updated" or a!="uniform_hash":
      fit=get(modelref);addbill(fit);rec=fit["receipt"];model_id=rec["model_id"];head_modes=rec["heads"]
      require(rec["training_sha256"]==digest(histories[a]),"Fit used wrong arm prefix")
      if a=="risk_only_updated":
       require(set(head_modes)=={"region","resource"},"Risk-only fit contains objective head")
      else:
       ur,uo={"risk_frozen_objective_frozen":(False,False),"risk_updated_objective_frozen":(True,False),
        "risk_frozen_objective_updated":(False,True),"risk_updated_objective_updated":(True,True)}[a]
       for h in ("region","resource"):
        require(head_modes[h]["mode"]==("updated" if ur else "frozen"),"Risk factor mismatch")
        if not ur: require(head_modes[h]["identity"]==initial_ids[h],"Frozen risk identity changed")
       if uo and head_modes["objective"]["mode"]=="unavailable_no_exact_labels":
        require("objective" not in head_modes or head_modes["objective"]["mode"]=="unavailable_no_exact_labels","Objective unavailable fabricated")
       elif uo: require(head_modes["objective"]["mode"]=="updated","Objective factor not updated")
       else:
        require(head_modes["objective"]["mode"] in ("frozen","frozen_unavailable"),"Objective factor mismatch")
        if "objective" in initial_ids:require(head_modes["objective"]["identity"]==initial_ids["objective"],"Frozen objective identity changed")
        else:require(head_modes["objective"]["mode"]=="frozen_unavailable","Missing frozen objective was fabricated")
    predrow=get(predref);addbill(predrow);pred=predrow["prediction"]
    require(predrow["model_id"]==model_id and predrow["query_sha256"]==digest(pool),"Prediction provenance mismatch")
    if pred is not None:
     require(pred["candidate_ids"]==ids and pred["model_id"]==model_id,"Prediction candidate/model identity mismatch")
    seen={r["candidate_id"] for r in histories[a]};available=[i for i,cid in enumerate(ids) if cid not in seen]
    decision=select_c2(a,ids,available,pred,best(histories[a]),[unit,t])
    propref=prefix+"/trials/"+a+"/proposal_%03d.json"%t;proposal=get(propref)
    require(proposal["history_sha256"]==digest(histories[a]) and proposal["history_information"]==information(histories[a]),"Proposal prefix mismatch")
    require(proposal["model_id"]==model_id and proposal["head_modes"]==head_modes,"Proposal head composition mismatch")
    require(proposal["decision"]==decision,"Selected action/score mismatch")
    trial_props[a].append(proposal)
    obsref=prefix+"/trials/"+a+"/observation_%03d.json"%t;obs=get(obsref)
    c=pool[decision["selected_index"]]
    require(obs["candidate"]==canonical(c) and obs["candidate_id"]==digest(c),"Unevaluated/wrong candidate selected")
    require(obs["observed"]==observed(c,obs["truth"]),"Observation differs from truth")
    eval_events.extend([dict(event="started",call_id=obs["call_id"],task=u["task"],candidate_id=obs["candidate_id"],candidate=obs["candidate"]),dict(event="settled",call_id=obs["call_id"],record_sha256=digest(obs))])
    total["math_started"]+=1;total["math_settled"]+=1
    trial_obs[a].append(obs["observed"]);histories[a].append(obs["observed"])
  first_actions=[trial_props[a][0]["decision"]["selected_candidate_id"] for a in ("risk_frozen_objective_frozen","risk_updated_objective_frozen","risk_frozen_objective_updated","risk_updated_objective_updated")]
  require(len(set(first_actions))==1,"Four factorial cells do not share first action")
  for a in ARMS:
   ref=prefix+"/trials/"+a+"/result.json";r=get(ref);all_trials.append(r)
   expected_trial=trial_result(u["task"],u["repeat"],a,[x["observed"] for x in initial],trial_obs[a],trial_props[a])
   require(r==expected_trial,"Trial summary differs from observable trajectory")
  require(order==[x["ref"] for x in rows],"Record order mismatch")
  progress.append(dict(unit=unit,completed_trials=len(all_trials),math_settled=total["math_settled"]))
 require(lines(root/"evaluations.jsonl")==eval_events,"Evaluation ledger mismatch")
 ev=lines(root/"events.jsonl")
 record_events=[x for x in ev if x.get("event")=="record_sealed"]
 expected_records=[]
 for u in spec["units"]:
  jp=root/"units"/u["task"]/("r%d"%u["repeat"])/"records.jsonl"
  expected_records.extend([dict(event="record_sealed",ref=x["ref"],record_sha256=digest(x["value"])) for x in lines(jp)])
 require(record_events==expected_records,"Record seal event sequence/hash differs")
 eval_only=[x for x in ev if x.get("event") in ("evaluation_started","evaluation_settled")]
 expected_eval_only=[]
 for x in eval_events:
  if x["event"]=="started":expected_eval_only.append(dict(event="evaluation_started",call_id=x["call_id"]))
  else:expected_eval_only.append(dict(event="evaluation_settled",call_id=x["call_id"],record_sha256=x["record_sha256"]))
 require(eval_only==expected_eval_only,"Evaluation event order differs")
 require(lines(root/"progress.jsonl")==progress,"Progress mismatch")
 require(set(read(root/"manifest.json")["files"])==expected_paths,"Evidence file set differs")
 require(total["math_started"]==total["math_settled"]==spec["total_math_calls"],"Math budget mismatch")
 require(total==report["bill"],"Bill differs from report")
 require(report["schema"]=="c2_head_factorial_v1" and report["status"]=="COMPLETE_C2_FACTORIAL" and report["trial_count"]==len(all_trials)==spec["trials"],"C2 report status/count differs")
 require(all(total[k]==0 for k in ("math_errors","vae_fits","fantasy_fits","aircraft_calls","rcaide_calls")),"Forbidden/failed call")
 return dict(status="VERIFIED_C2_FACTORIAL",files=nfiles,units=len(spec["units"]),trials=len(all_trials),bill=total,new_math_calls=0)
