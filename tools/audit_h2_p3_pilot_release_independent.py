"""P3 pilot: independent original-archive, ledger, causal-weight and paired-result audit.

Read-only numerical postprocessing. No RA-LSBO imports, GP/VAE training, aircraft
evaluation, RCAIDE use, optimizer replay, or changes to frozen evidence.
"""
import argparse, hashlib, json, math, tempfile, time, urllib.request, zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from scipy.special import log_ndtr

URL="https://github.com/Luyi1235/RA-LSBO/releases/download/h2-p3-windows-pilot-20261103/RA-LSBO-H2-P3-pilot-evidence-20261103_01.zip"
ZSHA="bd170195201b45942e4fe2c2e04e1e994263bb0a4566a467b28816cdce0473b4"
ZSIZE=50183941
CODE="57e390fc8f6280db9f201940aec14089e81c104c"
ROOT="RA-LSBO-H2-P3-pilot-output-20261102_01/"
LOGS="RA-LSBO-H2-P3-pilot-logs-20261102_01/"
PKG="RA-LSBO-H2-P3-pilot-package-20261103_01/"
CASES=("aircraft_ordinary","aircraft_sparse")
ARMS=("h2_dynamic","h2_fixed50")
SEEDS=tuple(range(12071,12079))

def ck(ok,msg):
    if not ok: raise AssertionError(msg)
def sha(b): return hashlib.sha256(b).hexdigest()
def digest(x): return sha(json.dumps(x,sort_keys=True,allow_nan=False).encode())
def j(z,p): return json.loads(z.read(p))
def jl(z,p): return [json.loads(b) for b in z.read(p).splitlines() if b.strip()]
def fsha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=3e-8,abs_tol=3e-8)
def weight(scored,clipped,method):
    if method=="h2_fixed50" or scored<4: return .5
    x=.35*clipped/math.sqrt(scored)
    return min(.9,max(.1,1/(1+math.exp(-x))))
def nll(component,obs):
    mean=float(component["mean"]); var=float(component["variance"])
    ck(math.isfinite(mean) and math.isfinite(var) and var>0,"nonfinite Gaussian variance")
    sd=math.sqrt(var)
    if obs["kind"]=="exact":
        return ("exact",.5*math.log(2*math.pi*var)+.5*((obs["value"]-mean)/sd)**2)
    ck(obs["kind"]=="interval","unrecognized observation kind")
    lo=-math.inf if obs["lower"] is None else (obs["lower"]-mean)/sd
    hi=math.inf if obs["upper"] is None else (obs["upper"]-mean)/sd
    ck(lo<hi,"invalid SOC interval")
    if lo==-math.inf: lp=float(log_ndtr(hi))
    elif hi==math.inf: lp=float(log_ndtr(-lo))
    elif lo>=0:
        big,small=float(log_ndtr(-lo)),float(log_ndtr(-hi))
        lp=big+math.log(-math.expm1(small-big))
    else:
        big,small=float(log_ndtr(hi)),float(log_ndtr(lo))
        lp=big+math.log(-math.expm1(small-big))
    ck(math.isfinite(lp),"nonfinite interval evidence")
    return ("interval",-lp)

def download(path):
    last=None
    for attempt in (1,2,3):
        try:
            request=urllib.request.Request(URL,headers={"User-Agent":"RA-LSBO-H2-P3-ReadOnly-Release-Audit"})
            with urllib.request.urlopen(request,timeout=160) as resp,path.open("wb") as target:
                while True:
                    chunk=resp.read(1048576)
                    if not chunk: break
                    target.write(chunk)
            ck(path.stat().st_size==ZSIZE and fsha(path)==ZSHA,"ZIP identity does not match frozen Release")
            print("Release ZIP independently downloaded and SHA256 verified",flush=True)
            return
        except Exception as error:
            last=error
            print("Download attempt",attempt,"failed:",type(error).__name__,str(error),flush=True)
            if attempt!=3: time.sleep(2*attempt)
    raise last

def verify_archive(z,manifest):
    actual={}
    for item in z.infolist():
        if item.is_dir(): continue
        name=item.filename; path=PurePosixPath(name)
        ck(name not in actual and not path.is_absolute() and ".." not in path.parts and "\\" not in name,
           "unsafe or repeated ZIP file")
        h=hashlib.sha256(); n=0
        with z.open(item) as f:
            for part in iter(lambda:f.read(1048576),b""):
                h.update(part); n+=len(part)
        ck(n==item.file_size,"member size mismatch: "+name)
        actual[name]=(h.hexdigest(),n)
    ck(len(manifest)==701,"unexpected original manifest count")
    extra=set(actual)-set(manifest)
    ck(set(manifest).issubset(actual),"missing listed ZIP files")
    ck(extra in (set(),{PKG+"SHA256SUMS.txt"}),"unregistered ZIP files: "+repr(extra))
    for name,identity in manifest.items():
        ck(actual[name]==identity,"sealed file SHA/size mismatch: "+name)
    if PKG+"SHA256SUMS.txt" in actual:
        embedded={}
        for row in z.read(PKG+"SHA256SUMS.txt").decode().splitlines():
            if row.strip():
                h,n,p=row.split(maxsplit=2); embedded[p]=(h.lower(),int(n))
        ck(embedded==manifest,"embedded checksum list differs from PR manifest")
    return actual

def verify_ledger(z,plan,result):
    events=jl(z,ROOT+"calls.jsonl")
    ck(len(events)==1600,"not 800 started/settled call pairs")
    tickets={}; ids=set(); roles=Counter()
    for index,(start,end) in enumerate(zip(events[::2],events[1::2]),1):
        ck(start["event"]=="started" and end["event"]=="settled" and start["number"]==index,
           "non-sequential started/settled ledger")
        key={k:start[k] for k in ("number","tag","role","candidate","contract","new_rcaide_mda")}
        cid=digest(key); row=end["record"]
        ck(start["call_id"]==end["call_id"]==cid and start["tag"]==end["tag"]
           and start["tag"] not in tickets and cid not in ids,"ledger ticket mismatch")
        ck(start["new_rcaide_mda"] is False and row["candidate"]==start["candidate"]
           and row["observation"]["design_id"]==digest(row["candidate"])
           and row["observation"]["contract_id"]==start["contract"]
           and end["record_digest"]==digest(row),"settled observation digest mismatch")
        ck(math.isfinite(end["seconds"]) and end["seconds"]>=0,"invalid settled duration")
        tickets[start["tag"]]=(start,row); ids.add(cid); roles[start["role"]]+=1
    counts={"shared_initial":192,"online":576,"incumbent_step_verification":32}
    ck(dict(roles)==counts==result["charged_by_role"],"call roles differ")
    ck(len(ids)==result["charged_model_calls"]==800
       and result["maximum_model_calls"]==plan["budget"]["maximum_model_calls"]==800
       and result["unused_calls_not_reassigned"]==0 and result["new_rcaide_mda_calls"]==0,
       "truth-budget counters differ")
    return tickets,dict(roles)

def verify_trials(z,tickets,result,weight_summary):
    rows={(r["case"],r["seed"],r["method"]):r for r in result["summaries"]}
    ck(len(rows)==len(result["summaries"])==32,"missing or repeated trial result")
    ck(set(rows)=={(c,s,m) for c in CASES for s in SEEDS for m in ARMS},"trial roster changed")
    all_updates=used_updates=all_scores=exacts=intervals=online_count=verified=0
    divergences={}; paired={}
    logged_weight={(v["case"],v["seed"],v["method"]):v for v in weight_summary["trials"]}
    ck(len(logged_weight)==32,"weight summary trial roster")
    for case in CASES:
        case_deltas=[]
        for seed in SEEDS:
            init=[tickets[f"{case}/{seed}/initial/{i}"][1] for i in range(12)]
            ck(len({digest(x["candidate"]) for x in init})==12,"duplicate shared initial candidate")
            choices={}; outcomes={}
            for method in ARMS:
                row=rows[(case,seed,method)]
                folder=f"optimization/{case}/{seed}_{method}/"
                ck(j(z,ROOT+folder+"summary.json")==row,"trial summary differs from frozen raw")
                mf=j(z,ROOT+folder+"manifest.json")
                ck(mf["initial"]==init and mf["initial_digest"]==digest(init)
                   and (mf["case"],mf["seed"],mf["method"])==(case,seed,method),
                   "shared initial/trial identity mismatch")
                proposals=jl(z,ROOT+folder+"proposals.jsonl")
                observations=jl(z,ROOT+folder+"observations.jsonl")
                ck(len(proposals)==len(observations)==row["online_calls"]==18
                   and not row["stopped"] and row["stop_reason"] is None,"incomplete 18-step trial")
                history=list(init); seen={digest(x["candidate"]) for x in init}; selected=[]
                n_scored=0; clipped=0.; weights=[]; exact=interval=0
                for step,(prop,obsrow) in enumerate(zip(proposals,observations)):
                    ck(prop["iteration"]==step and prop["prefix_digest"]==digest(history)
                       and prop["candidate"]==obsrow["candidate"],"proposal history mismatch")
                    ticket,rawrow=tickets[f"{case}/{seed}/{method}/online/{step}"]
                    ck(ticket["role"]=="online" and rawrow==obsrow,"online ledger mismatch")
                    identifier=digest(prop["candidate"])
                    ck(identifier not in seen,"duplicate candidate within trial")
                    seen.add(identifier); selected.append(identifier); online_count+=1
                    before=prop["metadata"]["H2_controller_before"]
                    commitment=prop["metadata"]["H2_prior_forecast_commitment"]
                    ck(before["settled"]==step and before["scored"]==n_scored
                       and close(before["clipped_sum"],clipped)
                       and close(before["onehot_weight"],weight(n_scored,clipped,method)),
                       "noncausal prior weight or score prefix")
                    ck(commitment["step"]==step and commitment["design_id"]==identifier
                       and commitment["prefix"]==prop["prefix_digest"]
                       and commitment["prior_chain"]==before["event_chain"]
                       and close(commitment["weight_at_proposal"],before["onehot_weight"]),
                       "forecast commitment not prior to evaluation")
                    ck(commitment["commitment"]==digest({k:v for k,v in commitment.items() if k!="commitment"}),
                       "forecast commitment SHA changed")
                    weights.append(before["onehot_weight"])
                    evidence=dict(obsrow["observation"]["constraints"])["soc"]
                    if commitment["skip"] is not None:
                        ck(commitment["skip"] in ("coverage","model_failure","no_forecast")
                           and commitment["onehot"] is None and commitment["latent"] is None,
                           "unregistered unscored outcome")
                    elif evidence["kind"]!="missing":
                        comp=prop["metadata"]["forecasts"]["constraints"]["soc"]["components"]
                        idx=prop["metadata"]["selected_index"]
                        ck(len(comp["weights"])==2
                           and close(comp["weights"][0],before["onehot_weight"])
                           and close(comp["weights"][1],1-before["onehot_weight"]),
                           "SOC candidate mixture weights mismatch")
                        for k,view in enumerate(("onehot","latent")):
                            ck(close(commitment[view]["mean"],comp["mean"][k][idx])
                               and close(commitment[view]["variance"],comp["variance"][k][idx]),
                               "selected component prediction differs from committed")
                        f1,f2=nll(commitment["onehot"],evidence),nll(commitment["latent"],evidence)
                        ck(f1[0]==f2[0],"SOC component evidence-type mismatch")
                        clipped+=min(4.,max(-4.,f2[1]-f1[1])); n_scored+=1
                        if f1[0]=="exact": exact+=1; exacts+=1
                        else: interval+=1; intervals+=1
                    history.append(obsrow)
                last=row["H2_controller_replay"]["final_state"]
                ck(last["settled"]==18 and last["scored"]==n_scored
                   and close(last["clipped_sum"],clipped)
                   and close(last["onehot_weight"],weight(n_scored,clipped,method))
                   and last["pending"] is False,"final controller replay inconsistent")
                ck(row["H2_controller_replay"]["scored_events"]==n_scored
                   and row["H2_controller_replay"]["exact_scores"]==exact
                   and row["H2_controller_replay"]["interval_scores"]==interval,
                   "H2 exact/censored counts differ")
                weights.append(last["onehot_weight"])
                changes=sum(abs(b-a)>1e-12 for a,b in zip(weights,weights[1:]))
                used=sum(abs(b-a)>1e-12 for a,b in zip(weights[:-2],weights[1:-1]))
                w=logged_weight[(case,seed,method)]
                ck(changes==w["weight_changes_at_settlement"]
                   and used==w["changes_used_by_later_proposal"]
                   and close(w["final_weight"],last["onehot_weight"]) and w["scored"]==n_scored,
                   "saved weight trajectory analysis differs from raw")
                if method=="h2_fixed50":
                    ck(changes==used==0 and all(close(x,.5) for x in weights),
                       "fixed mixture weight changed")
                else:
                    all_updates+=changes; used_updates+=used; all_scores+=n_scored
                choices[method]=selected; outcomes[method]=row
                stepcheck=row["step_verification"]
                ck(stepcheck["status"]=="evaluated" and stepcheck["verified_usable"]
                   and stepcheck["screen"]["status"]=="passed_screen","missing 5s check")
                ticket,verification=tickets[f"{case}/{seed}/{method}/verification/5"]
                ck(ticket["role"]=="incumbent_step_verification"
                   and verification==j(z,ROOT+folder+"verification.json"),
                   "5s verification is not bound to ledger")
                verified+=1
                ck(row["model_failure_coverage_calls"]==0,"surrogate fallback not reported")
            differing=[i for i,(a,b) in enumerate(zip(choices["h2_dynamic"],choices["h2_fixed50"])) if a!=b]
            divergences[f"{case}/{seed}"]={"different_proposals":len(differing),
                                           "first_difference_zero_based":differing[0] if differing else None}
            d=outcomes["h2_dynamic"]["best_feasible"]-outcomes["h2_fixed50"]["best_feasible"]
            case_deltas.append({"seed":seed,"delta_Wh":d,"initial_kind":outcomes["h2_dynamic"]["initial_kind"]})
        values=[x["delta_Wh"] for x in case_deltas]
        ordered=sorted(values)
        trimmed=[x for x in values if abs(x)!=max(map(abs,values))]
        paired[case]={"n":len(values),"deltas":case_deltas,"mean_Wh":sum(values)/len(values),
                      "median_Wh":(ordered[3]+ordered[4])/2,"wins":sum(x<0 for x in values),
                      "losses":sum(x>0 for x in values),"leave_largest_abs_out_mean_Wh":sum(trimmed)/len(trimmed)}
    ck((online_count,verified,all_scores,all_updates,used_updates)==(576,32,284,236,220),
       "proposal/verification/weight totals mismatch")
    ck(weight_summary["dynamic_scored_events"]==284
       and weight_summary["dynamic_weight_changes_at_settlement"]==236
       and weight_summary["dynamic_changes_used_by_later_proposal"]==220
       and weight_summary["fixed_weight_changes_at_settlement"]==0,
       "registered weight-summary totals differ")
    return paired,divergences,{"exact_soc_scored_both_arms":exacts,"interval_soc_scored_both_arms":intervals}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-dir",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--zip-file",type=Path)
    args=ap.parse_args()
    manifest={}
    for line in (args.results_dir/"SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            h,size,name=line.split(maxsplit=2)
            ck(name not in manifest and len(h)==64,"malformed checksum manifest")
            manifest[name]=(h.lower(),int(size))
    tmp=None
    if args.zip_file: path=args.zip_file
    else:
        tmp=tempfile.TemporaryDirectory(prefix="h2p3-independent-pilot-")
        path=Path(tmp.name)/"pilot.zip"
        download(path)
    ck(path.stat().st_size==ZSIZE and fsha(path)==ZSHA,"Release ZIP is not the frozen asset")
    with zipfile.ZipFile(path) as z:
        archive_hashes=verify_archive(z,manifest)
        plan=j(z,ROOT+"plan.json"); reg=j(z,ROOT+"registration.json")
        audit=j(z,ROOT+"audit.json"); result=j(z,ROOT+"results.json")
        weights=j(z,LOGS+"weight-analysis.json")
        for fn,part in (("plan.json",ROOT+"plan.json"),("registration.json",ROOT+"registration.json"),
                        ("audit.json",ROOT+"audit.json"),("results.json",ROOT+"results.json"),
                        ("qualification_diagnostics.json",ROOT+"qualification_diagnostics.json"),
                        ("weight-analysis.json",LOGS+"weight-analysis.json")):
            ck(j(z,part)==json.loads((args.results_dir/fn).read_text()),
               "checked-in PR JSON differs from sealed raw: "+fn)
        ck(z.read(ROOT+"report.md")== (args.results_dir/"report.md").read_bytes(),
           "checked-in report differs from sealed raw")
        ck(plan["source_commit"]==CODE and plan["profile"]["name"]=="pilot"
           and tuple(plan["profile"]["seeds"])==SEEDS
           and plan["profile"]["horizon"]==18
           and tuple(plan["arms"])==ARMS and tuple(plan["cases"])==CASES,
           "frozen pilot identity differs")
        ck(plan["budget"]=={"shared_initial":192,"online":576,"incumbent_step_verification":32,
                             "maximum_model_calls":800,"maximum_rcaide_mda_calls":0},
           "registered pilot budget differs")
        ck(len(plan["sources"])==186 and len(set(plan["sources"].values()))==185,
           "registered source path count differs")
        for source in set(plan["sources"].values()):
            ck(archive_hashes[ROOT+"source_snapshot/"+source+".source"][0]==source,
               "content-addressed source has changed")
        ck(reg["plan_sha256"]==sha(z.read(ROOT+"plan.json"))
           and reg["maximum_model_calls"]==800,"plan registration differs")
        ck(audit["passed"] is True and audit["settled_calls"]==800
           and audit["checked_proposals"]==576 and audit["trial_count"]==32
           and audit["source_bindings"]==186 and audit["H2_controller_replayed"] is True
           and audit["algorithm_replay"] is False and audit["physical_validation"] is False
           and len(audit["evidence_hashes"])==485,"original audit scope differs")
        for rel,h in audit["evidence_hashes"].items():
            ck(archive_hashes[ROOT+rel][0]==h,"original audit evidence hash differs: "+rel)
        ck(plan["reviewed_qualification"]["settled_calls"]==316
           and plan["smoke_prerequisite"]["source_commit"]==CODE,
           "316-call or audited P3 smoke prerequisite differs")
        ck(z.read(LOGS+"run-exit.txt").strip()==b"0"
           and z.read(LOGS+"run-stderr.txt")==b"",
           "native Windows run exit/stderr not clean")
        tickets,roles=verify_ledger(z,plan,result)
        paired,divergences,soc_scores=verify_trials(z,tickets,result,weights)
        original_pairs={(x["case"],x["seed"]):x for x in result["ablation_pairs"]}
        for case in CASES:
            for p in paired[case]["deltas"]:
                original=original_pairs[(case,p["seed"])]
                ck(close(original["dynamic_minus_fixed"],p["delta_Wh"]),
                   "paired final energy differs from native report")
    out={"schema":"ra_lsbo_h2_p3_pilot_independent_sealed_audit_v1","passed":True,
         "zip_sha256":ZSHA,"zip_bytes":ZSIZE,"zip_members":len(archive_hashes),
         "manifest_entries":len(manifest),"original_evidence_hashes_checked":485,
         "source_bindings":186,"source_blobs":185,"fixed_source":CODE,"settled_calls":800,
         "roles":roles,"trials":32,"online_proposals":576,"verification_passed":32,
         "causal_weight_changes":236,"applied_weight_changes":220,
         "fixed_weight_changes":0,"paired":paired,"trajectory_divergence":divergences,
         "soc_scores":soc_scores,"auditor_new_aircraft_calls":0,"auditor_new_rcaide_calls":0,
         "GP_or_VAE_refit":False,"optimizer_reexecuted":False,
         "confirmatory":False,"physical_safety_validation":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k not in ("paired","trajectory_divergence")},
                     sort_keys=True))
    print("INDEPENDENT P3 PILOT RAW RELEASE AUDIT PASSED")
    if tmp is not None: tmp.cleanup()

if __name__=="__main__": main()
