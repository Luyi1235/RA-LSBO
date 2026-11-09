"""Eight resealed-copy adversarial checks for C2; original evidence is immutable."""
import argparse,json,shutil,tempfile
from pathlib import Path
from c2_verify import verify,lines
from c2_protocol import digest
from risk_learning.io import read,write,seal

def dump(path,rows):path.write_text("".join(json.dumps(x,sort_keys=True,allow_nan=False)+"\n" for x in rows),encoding="utf-8")
def check(root,expected):
 baseline=verify(root,expected);checks=[]
 modes=("factor_mode","prefix","selection","frozen_identity","objective_feedback","ledger","event_order","bill")
 for mode in modes:
  with tempfile.TemporaryDirectory() as td:
   dst=Path(td)/"copy";shutil.copytree(root,dst)
   journals=list((dst/"units").rglob("records.jsonl"));journal=journals[0];rs=lines(journal)
   def mutate(ref,fn):
    row=next(x for x in rs if x["ref"]==ref);fn(row["value"]);dump(journal,rs)
    ev=lines(dst/"events.jsonl")
    for e in ev:
     if e.get("event")=="record_sealed" and e.get("ref")==ref:e["record_sha256"]=digest(row["value"])
    dump(dst/"events.jsonl",ev)
   if mode=="factor_mode":
    entry=next(x for x in rs if "/trials/risk_updated_objective_frozen/proposal_" in x["ref"])
    mutate(entry["ref"],lambda v:v.update(update_factors=[False,False]))
   elif mode=="prefix":
    entry=next(x for x in rs if "/trials/risk_frozen_objective_frozen/proposal_" in x["ref"])
    mutate(entry["ref"],lambda v:v.update(history_sha256="0"*64))
   elif mode=="selection":
    entry=next(x for x in rs if "/trials/risk_frozen_objective_frozen/proposal_" in x["ref"])
    mutate(entry["ref"],lambda v:v["decision"].update(selected_index=(v["decision"]["selected_index"]+1)%256))
   elif mode=="frozen_identity":
    entry=next(x for x in rs if "/trials/risk_frozen_objective_frozen/proposal_" in x["ref"])
    mutate(entry["ref"],lambda v:v["head_modes"]["region"].update(identity="f"*64))
   elif mode=="objective_feedback":
    entry=next(x for x in rs if "/trials/risk_frozen_objective_updated/observation_" in x["ref"])
    mutate(entry["ref"],lambda v:v["observed"]["objective"].update(value=12345.))
   elif mode=="ledger":
    p=dst/"evaluations.jsonl";r=lines(p);r.pop();dump(p,r)
   elif mode=="event_order":
    p=dst/"events.jsonl";r=lines(p);r[0],r[-1]=r[-1],r[0];dump(p,r)
   else:
    p=dst/"report.json";v=read(p);v["bill"]["gp_head_started"]+=1;p.unlink();write(p,v)
   (dst/"manifest.json").unlink();seal(dst)
   try:verify(dst,expected)
   except (ValueError,KeyError,AssertionError):checks.append(mode)
   else:raise AssertionError("Accepted resealed C2 tampering: "+mode)
 return dict(status="REJECTED_ALL_C2_TAMPERING",checks=checks,baseline=baseline,new_math_calls=0,new_gp_fits=0)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--expected-sha",required=True)
 print(json.dumps(check(p.parse_args().input,p.parse_args().expected_sha),indent=2))
