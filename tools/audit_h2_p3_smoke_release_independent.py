import argparse, hashlib, json, math, tempfile, time, urllib.request, zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
URL="https://github.com/Luyi1235/RA-LSBO/releases/download/h2-p3-windows-smoke-20261102/RA-LSBO-H2-P3-smoke-evidence-20261102_01.zip"
ZSHA="134eea36e47c62b61d830aa59ad9c309ae29345a4886dae9e0da4c54ee28bd0f"; ZSIZE=1004176
CODE="57e390fc8f6280db9f201940aec14089e81c104c"; ROOT="RA-LSBO-H2-P3-smoke-output-20261102_01/"; DG="RA-LSBO-H2-P3-smoke-diagnostics-20261102_02/"
CASES=("aircraft_ordinary","aircraft_sparse"); ARMS=("h2_dynamic","h2_fixed50"); SEED=12070
def ck(x,m):
    if not x: raise AssertionError(m)
def sha(b): return hashlib.sha256(b).hexdigest()
def dig(x): return sha(json.dumps(x,sort_keys=True,allow_nan=False).encode())
def j(z,p): return json.loads(z.read(p))
def jl(z,p): return [json.loads(x) for x in z.read(p).splitlines() if x.strip()]
def fsha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--results-dir",required=True,type=Path); ap.add_argument("--output",required=True,type=Path); a=ap.parse_args()
    manifest={}
    for line in (a.results_dir/"SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            h,n,p=line.split(maxsplit=2); ck(len(h)==64 and p not in manifest,"bad manifest row"); manifest[p]=(h.lower(),int(n))
    tmp=tempfile.TemporaryDirectory(prefix="h2p3-smoke-audit-"); zp=Path(tmp.name)/"smoke.zip"; last=None
    for attempt in range(1,4):
        try:
            req=urllib.request.Request(URL,headers={"User-Agent":"RA-LSBO-P3-independent-audit"})
            with urllib.request.urlopen(req,timeout=120) as r,zp.open("wb") as w:
                while True:
                    b=r.read(1048576)
                    if not b: break
                    w.write(b)
            ck(zp.stat().st_size==ZSIZE and fsha(zp)==ZSHA,"Release ZIP size/SHA mismatch"); break
        except Exception as e:
            last=e
            print("download attempt",attempt,type(e).__name__,str(e),flush=True)
            if attempt==3: raise
            time.sleep(attempt*2)
    with zipfile.ZipFile(zp) as z:
        actual={}
        for info in z.infolist():
            if info.is_dir(): continue
            name=info.filename; pp=PurePosixPath(name)
            ck(name not in actual and not pp.is_absolute() and ".." not in pp.parts and "\\" not in name,"unsafe/duplicate member")
            h=hashlib.sha256(); n=0
            with z.open(info) as f:
                for b in iter(lambda:f.read(1048576),b""): h.update(b); n+=len(b)
            ck(n==info.file_size,"member size mismatch"); actual[name]=(h.hexdigest(),n)
        ck(len(manifest)==len(actual)==263 and set(manifest)==set(actual),"ZIP manifest mismatch")
        for p,v in manifest.items(): ck(actual[p]==v,"member SHA mismatch: "+p)
        plan=j(z,ROOT+"plan.json"); reg=j(z,ROOT+"registration.json"); audit=j(z,ROOT+"audit.json"); res=j(z,ROOT+"results.json")
        checks=j(z,DG+"checks.json"); receipt=j(z,DG+"smoke-receipt.json")
        for fn,member in [("plan.json",ROOT+"plan.json"),("registration.json",ROOT+"registration.json"),("audit.json",ROOT+"audit.json"),("results.json",ROOT+"results.json"),("qualification_diagnostics.json",ROOT+"qualification_diagnostics.json"),("checks.json",DG+"checks.json"),("smoke-receipt.json",DG+"smoke-receipt.json")]:
            ck(j(z,member)==json.loads((a.results_dir/fn).read_text()),"PR/ZIP JSON mismatch: "+fn)
        ck(z.read(ROOT+"report.md")== (a.results_dir/"report.md").read_bytes(),"report mismatch")
        ck(plan["source_commit"]==CODE and plan["profile"]["name"]=="smoke" and plan["profile"]["seeds"]==[SEED] and plan["profile"]["horizon"]==2,"plan identity")
        ck(tuple(plan["arms"])==ARMS and tuple(plan["cases"])==CASES,"roster")
        b=plan["budget"]
        ck(b["maximum_model_calls"]==24 and b["maximum_rcaide_mda_calls"]==0 and b["shared_initial"]==12 and b["online"]==8 and b["incumbent_step_verification"]==4,"budget")
        ck(len(plan["sources"])==186 and len(set(plan["sources"].values()))==185,"source count")
        for h in set(plan["sources"].values()): ck(actual.get(ROOT+"source_snapshot/"+h+".source",(None,None))[0]==h,"source blob")
        ck(reg["plan_sha256"]==sha(z.read(ROOT+"plan.json")) and reg["maximum_model_calls"]==24,"registration")
        ck(audit["passed"] and audit["settled_calls"]==24 and audit["checked_proposals"]==8 and audit["trial_count"]==4 and audit["source_bindings"]==186 and audit["H2_controller_replayed"] and not audit["algorithm_replay"] and not audit["physical_validation"] and len(audit["evidence_hashes"])==41,"audit scope")
        for p,h in audit["evidence_hashes"].items(): ck(actual.get(ROOT+p,(None,None))[0]==h,"evidence hash: "+p)
        ck(plan["qualification"]["settled_calls"]==316 and plan["qualification"]["new_evaluations"]==0,"qualification")
        for f in ("load-plan-exit-code.txt","checks-exit-code.txt","run-exit-code.txt","audit-exit-code.txt","report-exit-code.txt","smoke-receipt-exit-code.txt"): ck(z.read(DG+f).decode().strip()=="0","exit code: "+f)
        ck(z.read(DG+"run-stderr.txt")==b"" and all(v.get("status")=="PASS" for v in checks.values()),"stderr/checks")
        ck(receipt["source_commit"]==CODE and receipt["smoke_numerical_gate_only"] and receipt["smoke_no_feasible_trial_count"]==0 and receipt["audit_sha256"]==sha(z.read(ROOT+"audit.json")) and receipt["registration_sha256"]==sha(z.read(ROOT+"registration.json")) and receipt["evidence_hashes"]==audit["evidence_hashes"],"smoke receipt")
        events=jl(z,ROOT+"calls.jsonl"); ck(len(events)==48,"event count")
        tags={}; ids=set(); roles=Counter()
        for i,(s,e) in enumerate(zip(events[::2],events[1::2]),1):
            ck(s["event"]=="started" and e["event"]=="settled" and s["number"]==i,"event order")
            identity={k:s[k] for k in ("number","tag","role","candidate","contract","new_rcaide_mda")}; cid=dig(identity); row=e["record"]
            ck(s["call_id"]==e["call_id"]==cid and s["tag"]==e["tag"] and s["tag"] not in tags and cid not in ids,"call id")
            ck(s["new_rcaide_mda"] is False and row["candidate"]==s["candidate"] and row["observation"]["design_id"]==dig(row["candidate"]) and row["observation"]["contract_id"]==s["contract"] and e["record_digest"]==dig(row),"record digest")
            ck(math.isfinite(e["seconds"]) and e["seconds"]>=0,"duration")
            tags[s["tag"]]=(s,row); ids.add(cid); roles[s["role"]]+=1
        expected={"shared_initial":12,"online":8,"incumbent_step_verification":4}
        ck(dict(roles)==expected==res["charged_by_role"] and len(ids)==res["charged_model_calls"]==24 and res["maximum_model_calls"]==24 and res["unused_calls_not_reassigned"]==0 and res["new_rcaide_mda_calls"]==0,"call accounting")
        rows={(x["case"],x["seed"],x["method"]):x for x in res["summaries"]}; keys={(c,SEED,m) for c in CASES for m in ARMS}
        ck(len(rows)==4 and set(rows)==keys,"trial roster")
        parity={}; online=verified=exact=interval=0
        for case in CASES:
            initial=[tags[f"{case}/{SEED}/initial/{i}"][1] for i in range(6)]
            ck(len({dig(x["candidate"]) for x in initial})==6,"initial candidates")
            trajectories={}
            for arm in ARMS:
                s=rows[(case,SEED,arm)]; folder=f"optimization/{case}/{SEED}_{arm}/"
                ck(j(z,ROOT+folder+"summary.json")==s,"summary")
                mf=j(z,ROOT+folder+"manifest.json")
                ck(mf["initial"]==initial and mf["initial_digest"]==dig(initial) and (mf["case"],mf["seed"],mf["method"])==(case,SEED,arm),"trial manifest")
                props=jl(z,ROOT+folder+"proposals.jsonl"); obs=jl(z,ROOT+folder+"observations.jsonl")
                ck(len(props)==len(obs)==s["online_calls"]==2 and not s["stopped"] and s["stop_reason"] is None,"online")
                hist=list(initial); seen={dig(x["candidate"]) for x in initial}; chosen=[]
                for t,(p,o) in enumerate(zip(props,obs)):
                    ck(p["iteration"]==t and p["prefix_digest"]==dig(hist) and p["candidate"]==o["candidate"],"prefix")
                    ticket,record=tags[f"{case}/{SEED}/{arm}/online/{t}"]; ck(ticket["role"]=="online" and record==o,"online ledger")
                    cd=dig(p["candidate"]); ck(cd not in seen,"duplicate"); seen.add(cd); chosen.append(p["candidate"]); hist.append(o); online+=1
                    kind=dict(o["observation"]["constraints"])["soc"]["kind"]; exact+=kind=="exact"; interval+=kind=="interval"; ck(kind in ("exact","interval","missing"),"SOC kind")
                trajectories[arm]=chosen
                step=s["step_verification"]; ck(step["status"]=="evaluated" and step["verified_usable"] and step["screen"]["status"]=="passed_screen","5s")
                ticket,record=tags[f"{case}/{SEED}/{arm}/verification/5"]; ck(ticket["role"]=="incumbent_step_verification" and j(z,ROOT+folder+"verification.json")==record,"verification ledger")
                verified+=1; ck(s["model_failure_coverage_calls"]==0,"fallback")
                replay=s["H2_controller_replay"]; state=replay["final_state"]
                ck(replay["settled_events"]==replay["scored_events"]==2 and state["settled"]==state["scored"]==2 and not state["pending"],"controller replay")
                ck(state["settings"]["warmup_scored"]==4 and state["settings"]["prior_onehot"]==0.5 and state["onehot_weight"]==0.5,"warmup")
                if arm=="h2_fixed50": ck(state["P3_weight_policy"]=="fixed_0.5","fixed policy")
                else: ck("P3_weight_policy" not in state,"dynamic policy")
            parity[case]=trajectories["h2_dynamic"]==trajectories["h2_fixed50"]
        ck((online,verified,exact,interval)==(8,4,6,2),"trial/SOC totals")
    out={"passed":True,"zip_sha256":ZSHA,"zip_bytes":ZSIZE,"zip_members":263,"manifest_entries":263,"evidence_hashes_checked":41,"source_paths":186,"unique_source_blobs":185,"source_commit":CODE,"settled_calls":24,"roles":dict(roles),"rcaide_calls":0,"trials":4,"online_proposals":8,"verified_5s":4,"exact_soc":6,"interval_soc":2,"effective_weight_updates":0,"identical_candidate_trajectories":parity,"auditor_model_calls":0,"physical_validation":False,"confirmatory_evidence":False,"interpretation":"Two scored observations are below the four-score warmup; smoke validates integration only."}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
    print(json.dumps(out,sort_keys=True)); print("INDEPENDENT P3 SMOKE AUDIT PASSED")
    tmp.cleanup()
if __name__=="__main__": main()
