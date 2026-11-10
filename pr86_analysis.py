"""Task-stratified PR86 learning-curve analysis; no fitting or truth calls."""
import csv,json,statistics
from pathlib import Path
from risk_learning.io import read,write,seal,check_manifest,file_sha
from pr86_protocol import plan,POLICIES,METHODS

def analyze(root,output,expected_sha):
 root=Path(root).resolve();output=Path(output).resolve()
 if output.exists():raise FileExistsError("Write-once PR86 analysis output")
 check_manifest(root);spec=read(root/"plan.json");source=read(root/"source_receipt.json")
 if source["code_sha"]!=expected_sha:raise ValueError("Code SHA mismatch")
 grouped={}
 for u in spec["units"]:
  name="%s/r%d"%(u["task"],u["repeat"])
  for n in spec["sizes"]:
   for p in POLICIES:
    for m in METHODS:
     row=read(root/"fits"/name/("n%d"%n)/(p+"__"+m+".json"))
     grouped.setdefault((u["task"],p,n,m),[]).append(row)
 out=[]
 for (task,policy,n,method),rows in sorted(grouped.items()):
  valid=[r for r in rows if r["status"]=="OK"]
  metrics={}
  for head in ("joint","region","resource"):
   for metric in ("brier","log_loss","auroc","auprc","ece","margin_rmse"):
    vals=[r["metrics"][head].get(metric) for r in valid if r["metrics"][head].get(metric) is not None]
    metrics[head+"_"+metric]=dict(n=len(vals),values=vals,mean=statistics.fmean(vals) if vals else None,
      median=statistics.median(vals) if vals else None)
  out.append(dict(task=task,policy=policy,n=n,method=method,repeats=len(rows),
    valid=len(valid),failed=len(rows)-len(valid),metrics=metrics,
    train_joint_feasible_counts=[r["train_label_counts"]["joint_feasible"] for r in rows],
    branches_covered=[r["coverage"]["branches_covered"] for r in rows],
    nearest_neighbor_mean=[r["coverage"]["nearest_neighbor_distance"]["mean"] for r in rows]))
 result=dict(status="COMPLETE_PR86_ANALYSIS",code_sha=expected_sha,input_manifest_sha256=file_sha(root/"manifest.json"),
  task_stratified_curves=out,limits=["No cross-task raw margin pooling.",
   "native_censored is the deployable observation contract.",
   "privileged_exact is an information upper-bound diagnostic only.",
   "Sampling policies are descriptive and class-balanced selection uses labelled training-pool observations.",
   "All held-out test labels are used only for post-fit scoring."])
 output.mkdir(parents=True);write(output/"learning_curves.json",result)
 with (output/"learning_curves.csv").open("w",newline="",encoding="utf-8") as f:
  fields=["task","policy","n","method","repeats","valid","failed","joint_brier_mean","joint_auroc_mean","joint_ece_mean","joint_auprc_mean"]
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in out:w.writerow(dict(task=r["task"],policy=r["policy"],n=r["n"],method=r["method"],
   repeats=r["repeats"],valid=r["valid"],failed=r["failed"],
   joint_brier_mean=r["metrics"]["joint_brier"]["mean"],
   joint_auroc_mean=r["metrics"]["joint_auroc"]["mean"],
   joint_ece_mean=r["metrics"]["joint_ece"]["mean"],
   joint_auprc_mean=r["metrics"]["joint_auprc"]["mean"]))
 seal(output);return dict(status="COMPLETE_PR86_ANALYSIS",rows=len(out))
