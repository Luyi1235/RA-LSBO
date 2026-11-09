"""C2 factorial effects and exact paired direction tests; standard library only."""
import math,statistics
from online_confirm_stats import sign_p,holm

def summarize_factorial(rows):
 if len({r["repeat"] for r in rows})!=len(rows): raise ValueError("Duplicate repeat")
 effects={k:[] for k in ("risk","objective","interaction")}
 missing=0;paired=[]
 for r in rows:
  z=r["effects"]
  if not z["complete"]: missing+=1
  else:
   for k in effects: effects[k].append(z[k])
  paired.append(dict(repeat=r["repeat"],**z))
 out={}
 for k,values in effects.items():
  wins=sum(v< -1e-10 for v in values);losses=sum(v>1e-10 for v in values);ties=len(values)-wins-losses
  out[k]=dict(n=len(values),missing=missing,wins_losses_ties=[wins,losses,ties],
   mean=math.fsum(values)/len(values) if values else None,
   median=statistics.median(values) if values else None,
   two_sided_sign_p=sign_p(wins,losses),values=values)
 return dict(effects=out,complete_four_cell=len(rows)-missing,missing_four_cell=missing,paired=paired)

def holm_factorial(task_reports):
 p={}
 for task,report in task_reports.items():
  for effect,row in report["effects"].items(): p[task+"/"+effect]=row["two_sided_sign_p"]
 if len(p)!=12: raise ValueError("C2 confirmatory family must contain exactly 12 tests")
 adjusted=holm(p)
 return {k:dict(raw_p=v,holm_p=adjusted[k],reject=adjusted[k]<=.05) for k,v in p.items()}
