"""C2 read-only factorial analysis; no model fitting or evaluator imports."""
from pathlib import Path
from c2_protocol import ARMS,factorial,TASKS
from c2_stats import summarize_factorial,holm_factorial
from risk_learning.io import read,write,seal

CELLS=("risk_frozen_objective_frozen","risk_updated_objective_frozen",
       "risk_frozen_objective_updated","risk_updated_objective_updated")
def analyze(root,output,expected_sha):
 root=Path(root).resolve();output=Path(output).resolve()
 if output.exists():raise FileExistsError("Write-once analysis output")
 spec=read(root/"plan.json");source=read(root/"source_receipt.json")
 if source["code_sha"]!=expected_sha or source["phase"]!="C2":raise ValueError("Wrong C2 source")
 reports={};costs={}
 for task in spec["tasks"]:
  journal=root/"units"/task/"r0"/"records.jsonl"
  # The task has all repeats, each in its own journal.
  rows=[]
  for u in [x for x in spec["units"] if x["task"]==task]:
   p=root/"units"/task/("r%d"%u["repeat"])/"records.jsonl"
   import json
   mp={x["ref"]:x["value"] for x in (json.loads(z) for z in p.read_text(encoding="utf-8").splitlines())}
   trials={a:mp["units/%s/r%d/trials/%s/result.json"%(task,u["repeat"],a)] for a in ARMS}
   warm=trials[CELLS[0]]["initial_warm"]
   ys=[trials[a]["final_best"] for a in CELLS]
   rows.append(dict(repeat=u["repeat"],warm=warm,effects=factorial(*ys),outcomes=dict(zip(CELLS,ys)),
    infeasible={a:trials[a]["online_calls"]-(trials[a]["information_curve"][-1]["feasible"]-trials[a]["information_curve"][0]["feasible"]) for a in ARMS},
    exact_objectives={a:trials[a]["information_curve"][-1]["exact_objectives"]-trials[a]["information_curve"][0]["exact_objectives"] for a in ARMS},
    branch_counts={a:trials[a]["information_curve"][-1]["branches"] for a in ARMS}))
  warmrows=[r for r in rows if r["warm"]]
  reports[task]=summarize_factorial([dict(repeat=r["repeat"],effects=r["effects"]) for r in warmrows])
  costs[task]=rows
 infer=holm_factorial(reports) if all(reports[t]["effects"][k]["n"] or reports[t]["effects"][k]["missing"] for t in reports for k in ("risk","objective","interaction")) else None
 result=dict(status="COMPLETE_C2_FACTORIAL_ANALYSIS",code_sha=expected_sha,input_manifest_sha=read(root/"manifest.json").get("sha256"),
  task_effects=reports,holm_12=infer,all_task_repeat_costs=costs,
  limits=["Factorial effects are intention-to-treat contrasts over adaptive trajectories.",
   "Missing four-cell feasible outcomes are not imputed.",
   "Known tasks only; no held-out task generalization.",
   "Infeasible evaluations and exact objective labels are separate cost outcomes, not safety claims."])
 output.mkdir(parents=True)
 write(output/"report.json",result);write(output/"factorial_trials.json",costs);seal(output)
 return dict(status=result["status"],holm_12=infer)
