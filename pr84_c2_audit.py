"""Read-only audit of archived C2 evidence; no evaluator/model imports."""
import argparse,csv,hashlib,json,math
from pathlib import Path

RESULT_HEAD="ade354de8b35d20a2bb32b469b04bc7da92aa004"
CODE_SHA="975d96caac53d675ebe1b790a0647479a75a2ed5"
ARMS=("risk_frozen_objective_frozen","risk_updated_objective_frozen",
"risk_frozen_objective_updated","risk_updated_objective_updated","risk_only_updated","uniform_hash")
TASKS=("tradeoff_s0","tradeoff_s1","tradeoff_s2","tradeoff_s3")
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def lines(path):return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
def require(ok,msg):
 if not ok:raise ValueError(msg)
def verify_manifest(root):
 m=read(root/"full"/"manifest.json");require(m.get("schema")=="sha256_manifest_v1","Full manifest schema mismatch")
 actual={p.relative_to(root/"full").as_posix() for p in (root/"full").rglob("*") if p.is_file() and p.name!="manifest.json"}
 require(actual==set(m["files"]),"Full manifest file set mismatch")
 bad=[p for p,h in m["files"].items() if sha(root/"full"/p)!=h]
 require(not bad,"Full manifest SHA mismatch: "+",".join(bad[:10]))
 return len(actual),sha(root/"full"/"manifest.json")
def auc(labels,probs):
 pos=[p for y,p in zip(labels,probs) if y];neg=[p for y,p in zip(labels,probs) if not y]
 if not pos or not neg:return None
 return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
def binary(labels,probs):
 if not labels:return dict(n=0,positives=0,negatives=0,brier=None,log_loss=None,auroc=None)
 eps=1e-15
 return dict(n=len(labels),positives=sum(labels),negatives=len(labels)-sum(labels),
  brier=sum((p-float(y))**2 for y,p in zip(labels,probs))/len(labels),
  log_loss=-sum(float(y)*math.log(max(eps,p))+(1-float(y))*math.log(max(eps,1-p)) for y,p in zip(labels,probs))/len(labels),
  auroc=auc(labels,probs))
def audit(root,output):
 root=Path(root).resolve();output=Path(output).resolve()
 require(not output.exists(),"Write-once audit output already exists")
 nfiles,manifest_sha=verify_manifest(root)
 full=root/"full";report=read(full/"report.json");plan=read(full/"plan.json")
 source=read(full/"source_receipt.json");integrity=read(root/"integrity_check.json")
 summary=read(root/"analysis_summary.json")
 require(source.get("code_sha")==CODE_SHA==report.get("code_sha"),"C2 code SHA mismatch")
 require(report.get("status") in ("COMPLETE_C2_FACTORIAL","COMPLETE_WITH_MODEL_FALLBACKS"),"C2 full status incomplete")
 require(plan.get("profile")=="full" and len(plan.get("units",[]))==64,"Unexpected C2 full plan")
 byarm={a:dict(trials=0,final_feasible=0,no_solve=0,online_failures=0,exact_objective_labels=0,
  region_exact=0,region_interval=0,region_missing=0,resource_exact=0,resource_interval=0,resource_missing=0,
  branches={str(b):0 for b in range(4)},risk_labels={"region":[],"resource":[],"joint":[]},
  risk_probs={"region":[],"resource":[],"joint":[]},objective_abs_error=[],objective_sq_error=[],objective_z_abs=[],
  objective_predictions=0,objective_exact=0,model_fallbacks=0) for a in ARMS}
 per_trial=[];per_task={t:{a:[] for a in ARMS} for t in TASKS};online_rows=[]
 warm=cold=0;first_feasible=[]
 for u in plan["units"]:
  task,rep=u["task"],u["repeat"];unit=full/"units"/task/("r%d"%rep)
  journal=lines(unit/"records.jsonl");mp={x["ref"]:x["value"] for x in journal}
  require(len(mp)==len(journal),"Duplicate logical ref: "+str(unit))
  init=mp["units/%s/r%d/initial.json"%(task,rep)]
  init_obs=[x["observed"] for x in init]
  init_feasible=any(x["objective"]["kind"]=="exact" and x["region"]["kind"]=="exact" and x["region"]["value"]<=0 and x["resource"]["kind"]=="exact" and x["resource"]["value"]<=0 for x in init_obs)
  warm+=int(init_feasible);cold+=int(not init_feasible)
  for arm in ARMS:
   result=mp["units/%s/r%d/trials/%s/result.json"%(task,rep,arm)]
   stat=byarm[arm];stat["trials"]+=1
   feasible_final=result.get("final_best") is not None
   stat["final_feasible"]+=int(feasible_final);stat["no_solve"]+=int(not feasible_final)
   stat["model_fallbacks"]+=int(result.get("model_failure_calls",0))
   first_feasible.append(dict(task=task,repeat=rep,arm=arm,first_feasible_call=result.get("first_feasible_call")))
   online=0;exact=0
   for step in range(plan["steps"]):
    prefix="units/%s/r%d/"%(task,rep)
    prop=mp[prefix+"trials/"+arm+"/proposal_%03d.json"%step]
    predrow=mp[prefix+"predictions/"+arm+"/t%03d.json"%step]
    obs=mp[prefix+"trials/"+arm+"/observation_%03d.json"%step]
    o=obs["observed"];truth=obs["truth"];c=obs["candidate"];branch=str(c["branch"])
    stat["branches"][branch]+=1
    for head in ("region","resource"):
     kind=o[head]["kind"];stat[head+"_"+kind]+=1
     if head=="region" and kind=="interval":online+=1
    exact_now=o["objective"]["kind"]=="exact";exact+=int(exact_now)
    p=predrow.get("prediction")
    if p is not None:
     idx=prop["decision"]["selected_index"]
     for head in ("region","resource","joint"):
      if head=="joint":label=bool(truth["region"]<=0 and truth["resource"]<=0)
      else:label=bool(truth[head]<=0)
      prob=p["probabilities"][head][idx]
      stat["risk_labels"][head].append(label);stat["risk_probs"][head].append(prob)
     if exact_now and p.get("objective") is not None:
      mu=p["objective"]["mean"][idx];var=p["objective"]["variance"][idx];y=o["objective"]["value"]
      stat["objective_abs_error"].append(abs(mu-y));stat["objective_sq_error"].append((mu-y)**2)
      stat["objective_predictions"]+=1;stat["objective_exact"]+=1
      if var>0:stat["objective_z_abs"].append(abs(y-mu)/math.sqrt(var))
    online_rows.append(dict(task=task,repeat=rep,arm=arm,step=step,branch=c["branch"],
      region_kind=o["region"]["kind"],resource_kind=o["resource"]["kind"],objective_kind=o["objective"]["kind"],
      objective_exact=exact_now,region_p=(p["probabilities"]["region"][prop["decision"]["selected_index"]] if p else None),
      resource_p=(p["probabilities"]["resource"][prop["decision"]["selected_index"]] if p else None),
      joint_p=(p["probabilities"]["joint"][prop["decision"]["selected_index"]] if p else None),
      objective_mu=(p["objective"]["mean"][prop["decision"]["selected_index"]] if p and p.get("objective") else None),
      objective_var=(p["objective"]["variance"][prop["decision"]["selected_index"]] if p and p.get("objective") else None),
      objective_y=(o["objective"].get("value") if exact_now else None)))
   stat["online_failures"]+=online;stat["exact_objective_labels"]+=exact
   per_task[task][arm].append(dict(repeat=rep,final_feasible=feasible_final,online_failures=online,exact_objective_labels=exact,first_feasible_call=result.get("first_feasible_call")))
   per_trial.append(dict(task=task,repeat=rep,arm=arm,initial_warm=init_feasible,final_feasible=feasible_final,
      no_solve=not feasible_final,online_failures=online,exact_objective_labels=exact,
      first_feasible_call=result.get("first_feasible_call"),reported_final_best=result.get("final_best")))
 # Compare independent recomputation to all published summaries.
 published=summary["arms"]
 comparisons={}
 for arm,s in byarm.items():
  p=published[arm]
  comparisons[arm]={k:dict(recomputed=s[k],published=int(p[k]),match=(s[k]==int(p[k]))) for k in ("trials","final_feasible","no_solve","cold","failures","exact_labels")}
  require(all(v["match"] for v in comparisons[arm].values()),"Published arm summary mismatch: "+arm)
 expected_total=plan["total_math_calls"]
 require(report["bill"]["math_started"]==report["bill"]["math_settled"]==expected_total,"C2 bill not settled")
 all_final=all(x["final_feasible"]==x["trials"] for x in byarm.values())
 result=dict(schema="pr84_c2_readonly_audit_v1",status="COMPLETE_PR84_AUDIT",input_result_head="ade354de8b35d20a2bb32b469b04bc7da92aa004",
  code_sha=CODE_SHA,full_files=nfiles,full_manifest_sha256=manifest_sha,
  math_calls_started=report["bill"]["math_started"],math_calls_settled=report["bill"]["math_settled"],
  new_math_calls=0,new_gp_fits=0,new_vae_fits=0,new_aircraft_calls=0,new_rcaide_calls=0,
  warm_groups=warm,cold_groups=cold,all_six_arms_final_feasible=all_final,
  integrity_check_arms_all_final_feasible=integrity.get("arms_all_final_feasible"),
  integrity_flag_consistent=(integrity.get("arms_all_final_feasible")==all_final),
  integrity_flag_definition="UNRESOLVED: source-generation formula not included in archived evidence",
  arm_recomputed=byarm,published_summary_comparison=comparisons,per_task=per_task,
  risk_metrics={a:{h:binary(s["risk_labels"][h],s["risk_probs"][h]) for h in ("region","resource","joint")} for a,s in byarm.items()},
  objective_metrics={a:dict(n=s["objective_exact"],mae=(sum(s["objective_abs_error"])/len(s["objective_abs_error"]) if s["objective_abs_error"] else None),
    rmse=(math.sqrt(sum(s["objective_sq_error"])/len(s["objective_sq_error"])) if s["objective_sq_error"] else None),
    mean_abs_standardized_residual=(sum(s["objective_z_abs"])/len(s["objective_z_abs"]) if s["objective_z_abs"] else None)) for a,s in byarm.items()},
  selection_feedback_note="Metrics are on policy-selected evaluated candidates only; they are not unbiased full-pool calibration and do not establish causal mediation.",
  per_trial=per_trial,first_feasible=first_feasible)
 output.mkdir(parents=True)
 (output/"audit_report.json").write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 with (output/"per_trial.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(per_trial[0]));w.writeheader();w.writerows(per_trial)
 with (output/"selected_feedback.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(online_rows[0]));w.writeheader();w.writerows(online_rows)
 return result
def main():
 p=argparse.ArgumentParser(description="Read-only audit of archived PR83 C2 evidence")
 p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--expected-result-head",default=RESULT_HEAD);a=p.parse_args()
 require(a.expected_result_head==RESULT_HEAD,"Unexpected PR83 result HEAD")
 r=audit(a.input,a.output);print(json.dumps({k:r[k] for k in ("status","full_files","warm_groups","cold_groups","all_six_arms_final_feasible","integrity_check_arms_all_final_feasible","integrity_flag_consistent","new_math_calls","new_gp_fits")},indent=2))
if __name__=="__main__":main()
