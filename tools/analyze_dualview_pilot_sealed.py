"""Independent immutable RA-LSBO-H pilot audit and preregistered paired analysis.

Zero imports from RA-LSBO, zero model/GP/encoder/evaluator/optimizer calls.
Only the exact sealed Windows Release ZIP is read. All statistics are
development-only with N=4 paired seeds per task; not confirmatory inference.

Usage: python tools/analyze_dualview_pilot_sealed.py FIXED_RELEASE_ZIP NEW_DIR
"""
from __future__ import annotations
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
import csv
import json
import math
import statistics
import sys
import zipfile

ARCHIVE_SHA="3d9cb4f684bbc9198c523bb44bf466463c2f85053d87ca1770eca9111b48fed1"
PLAN_SHA="ce59583c1fc482406e709225e7ee2496e85b36c3cbc82df9532366f45f239950"
REG_SHA="f1d9e8d734ca81d2cb0e8c6ea9ec53fbbb7c118661a076046ea5b44b2dccb271"
CODE_SHA="b219163fc565dfc32876f38f0bbf5f4e65abbd36"
CASES=("aircraft_ordinary","aircraft_sparse")
ARMS=("random","onehot_stable","corpus_stable","hybrid_onehot","hybrid_guard")
SEEDS=(9961,9962,9963,9964)
CONTRASTS=(("hybrid_onehot","corpus_stable","constraint_view"),
           ("hybrid_guard","hybrid_onehot","SOC_guard"),
           ("hybrid_onehot","onehot_stable","objective_view"))


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h=sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def avg(x):
    return sum(x)/len(x) if x else None

def median(x):
    return statistics.median(x) if x else None

def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")

def table(path,rows,fields):
    with open(path,"x",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def archive_data(path):
    if file_hash(path)!=ARCHIVE_SHA:
        raise ValueError("GitHub release ZIP SHA-256 is not the pinned Windows archive")
    with zipfile.ZipFile(path) as z:
        members=[v for v in z.infolist() if not v.is_dir()]
        if len(members)!=641 or len({v.filename for v in members})!=641:
            raise ValueError(f"Expected 641 distinct sealed files, got {len(members)}")
        for item in members:
            p=PurePosixPath(item.filename)
            if p.is_absolute() or ".." in p.parts or '\\' in item.filename:
                raise ValueError("Untrusted archive member path")
        contents={v.filename:z.read(v) for v in members}
    audit_paths=[p for p in contents if p.endswith("/audit.json") or p=="audit.json"]
    target=[]
    for ap in audit_paths:
        prefix=ap[:-len("audit.json")]
        if (prefix+"plan.json" in contents and prefix+"registration.json" in contents
                and prefix+"results.json" in contents and prefix+"calls.jsonl" in contents):
            target.append(prefix)
    if len(target)!=1:
        raise ValueError(f"Expected exactly one sealed pilot root, found {target}")
    prefix=target[0]
    raw=lambda path:contents[prefix+path]
    j=lambda path:json.loads(raw(path))
    if sha256(raw("plan.json")).hexdigest()!=PLAN_SHA or sha256(raw("registration.json")).hexdigest()!=REG_SHA:
        raise ValueError("Registered pilot plan or registration bytes have changed")
    plan,reg,result,audit=(j(x) for x in ("plan.json","registration.json","results.json","audit.json"))
    if audit["passed"] is not True or audit["settled_calls"]!=856 or audit["trial_count"]!=40 or audit["checked_proposals"]!=720:
        raise ValueError("The saved audit does not report a completely audited 856-call run")
    if plan["source_commit"]!=CODE_SHA or plan["profile"]["name"]!="pilot":
        raise ValueError("Source commit or experiment profile differs")
    if plan["profile"]["seeds"]!=list(SEEDS) or plan["profile"]["horizon"]!=18 or plan["profile"]["initial"]!=12:
        raise ValueError("Seed, initial, or online horizon differs")
    if (tuple(plan["arms"])!=ARMS or tuple(plan["cases"])!=CASES
        or plan["budget"]["maximum_model_calls"]!=856
        or plan["budget"]["maximum_rcaide_mda_calls"]!=0
        or result["charged_model_calls"]!=856
        or result["new_rcaide_mda_calls"]!=0):
        raise ValueError("Pilot roster, cap, or settled costs differ")
    if reg!={"plan_sha256":PLAN_SHA,"maximum_model_calls":856}:
        raise ValueError("Registration did not pin the original plan")
    expected= audit["evidence_hashes"]
    if audit["source_bindings"]!=171 or len(plan["sources"])!=171:
        raise ValueError("Unexpected source-binding count")
    for rel,h in expected.items():
        if prefix+rel not in contents or sha256(raw(rel)).hexdigest()!=h:
            raise ValueError("Audit evidence file missing or different: "+rel)
    for rel,h in plan["sources"].items():
        if prefix+"source_snapshot/"+h+".source" not in contents:
            raise ValueError("Source snapshot missing "+rel)
        if sha256(raw("source_snapshot/"+h+".source")).hexdigest()!=h:
            raise ValueError("Snapshot bytes changed "+rel)
    if "execution.lock" not in {p[len(prefix):] for p in contents if p.startswith(prefix)}:
        raise ValueError("Execution lock not preserved")

    # Verify 856 independent started->settled pairs from raw journal
    events=[json.loads(x) for x in raw("calls.jsonl").splitlines() if x.strip()]
    if len(events)!=1712:raise ValueError("856-call ledger must have 1712 events")
    role_count=Counter()
    tags=set()
    record_ids=set()
    for i in range(856):
        start,end=events[2*i],events[2*i+1]
        call={key:start[key] for key in ("number","tag","role","candidate","contract","new_rcaide_mda")}
        if (start["event"]!="started" or end["event"]!="settled"
            or start["number"]!=i+1 or start["call_id"]!=digest(call)
            or end["call_id"]!=start["call_id"] or end["tag"]!=start["tag"]
            or end["record_digest"]!=digest(end["record"])
            or end["record"]["candidate"]!=start["candidate"]
            or end["record"]["observation"]["contract_id"]!=start["contract"]
            or end["record"]["observation"]["design_id"]!=digest(start["candidate"])
            or start["tag"] in tags or start["call_id"] in record_ids
            or start["new_rcaide_mda"] is not False):
            raise ValueError("Started/settled ledger mismatch at event "+str(i))
        tags.add(start["tag"])
        record_ids.add(start["call_id"])
        role_count[start["role"]]+=1
    if role_count!={"shared_initial":96,"online":720,"incumbent_step_verification":40}:
        raise ValueError("Unexpected actual ledger role charges")
    if dict(role_count)!=result["charged_by_role"]:
        raise ValueError("Results role charges differ from journal")

    # Verify that all 40 saved summaries are from the exact registered
    # chronologically paired case, seed and arm list, one row per trial.
    rows=result["summaries"]
    expected_keys={(c,s,m) for c in CASES for s in SEEDS for m in ARMS}
    lookup={(r["case"],r["seed"],r["method"]):r for r in rows}
    if len(rows)!=40 or len(lookup)!=40 or set(lookup)!=expected_keys:
        raise ValueError("40 complete case/seed/arm summaries missing or duplicate")
    if len(result.get("timings",[]))!=40:
        raise ValueError("Missing per-trial end-to-end timing records")

    per_trial=[]
    for case,seed,arm in sorted(expected_keys):
        r=lookup[(case,seed,arm)]
        root=f"optimization/{case}/{seed}_{arm}/"
        saved=j(root+"summary.json")
        if saved!=r:
            raise ValueError("Trial summary differs from results.json")
        manifest=j(root+"manifest.json")
        if tuple((manifest["case"],manifest["seed"],manifest["method"]))!=(case,seed,arm):
            raise ValueError("Trial identity differs")
        props=[json.loads(x) for x in raw(root+"proposals.jsonl").splitlines() if x.strip()]
        observations=[json.loads(x) for x in raw(root+"observations.jsonl").splitlines() if x.strip()]
        if (len(props)!=18 or len(observations)!=18 or r["online_calls"]!=18
                or r["stopped"] or not r["has_feasible"] or r["best_feasible"] is None
                or r["step_verification"]["verified_usable"] is not True
                or r["step_verification"]["screen"]["status"]!="passed_screen"):
            raise ValueError("Incomplete or unverified trial "+str((case,seed,arm)))
        for idx,(p,o) in enumerate(zip(props,observations)):
            if (p["iteration"]!=idx or p["candidate"]!=o["candidate"]
                or p["metadata"].get("method")!=arm):
                raise ValueError("Proposal does not match settled online observation")
        initial_ids=[digest(o["candidate"]) for o in manifest["initial"]]
        if len(initial_ids)!=12:raise ValueError("Trial initial sample size differs")
        failure_counts=sum(p["metadata"].get("coverage_reason")=="model_failure" for p in props)
        if failure_counts!=r["model_failure_coverage_calls"]:
            raise ValueError("Failure fallback status did not match saved proposals")
        soc=r.get("soc_prequential",{})
        if soc.get("exact",0)+soc.get("interval",0)+soc.get("missing",0)>18:
            raise ValueError("SOC score count exceeds horizon")
        if r["step_verification"]["status"]!="evaluated":
            raise ValueError("Unexpected trial verification status")
        per_trial.append({
            "case":case,"seed":seed,"method":arm,
            "best_feasible_wh":float(r["best_feasible"]),
            "initial_kind":r["initial_kind"],
            "initial_best_wh":r["initial_best"],
            "first_feasible_online_call":r["first_feasible_online_call"],
            "online_usable":r["online_usable"],
            "verified_5s":r["step_verification"]["verified_usable"],
            "verified_refined_wh":r["step_verification"].get("refined_objective_Wh"),
            "fallbacks":failure_counts,
            "soc_exact_n":soc.get("exact",0),
            "soc_exact_nll_mean":soc.get("exact_density_nll_mean"),
            "soc_exact_95_coverage_num":soc.get("exact_95pct_coverage_numerator",0),
            "soc_interval_n":soc.get("interval",0),
            "soc_interval_nll_mean":soc.get("interval_mass_nll_mean"),
            "soc_skipped":soc.get("skipped_no_forecast",0),
            "vae_fit_seconds":r.get("corpus_training",{}).get("fit_seconds") if r.get("corpus_training") else None,
        })
        # Registered shared initial must be byte-for-byte equivalent between arms.
        for other in ARMS:
            if j(f"optimization/{case}/{seed}_{other}/manifest.json")["initial"]!=manifest["initial"]:
                raise ValueError("Paired arms do not share their 12 identical initial labels")

    pair_records=[]
    for case in CASES:
        for left,right,name in CONTRASTS:
            for seed in SEEDS:
                lhs=lookup[(case,seed,left)]
                rhs=lookup[(case,seed,right)]
                pair_records.append({
                    "case":case,"seed":seed,"contrast":name,
                    "left_method":left,"right_method":right,
                    "left_wh":float(lhs["best_feasible"]),
                    "right_wh":float(rhs["best_feasible"]),
                    "left_minus_right_wh":float(lhs["best_feasible"]-rhs["best_feasible"]),
                    "better":"left" if lhs["best_feasible"]<rhs["best_feasible"] else ("right" if lhs["best_feasible"]>rhs["best_feasible"] else "tie"),
                })

    groups=[]
    for case in CASES:
        for arm in ARMS:
            selected=[x for x in per_trial if x["case"]==case and x["method"]==arm]
            energies=[x["best_feasible_wh"] for x in selected]
            exact=[(x["soc_exact_n"],x["soc_exact_nll_mean"]) for x in selected
                   if x["soc_exact_n"] and x["soc_exact_nll_mean"] is not None]
            interval=[(x["soc_interval_n"],x["soc_interval_nll_mean"]) for x in selected
                      if x["soc_interval_n"] and x["soc_interval_nll_mean"] is not None]
            exact_den=sum(n for n,_ in exact)
            int_den=sum(n for n,_ in interval)
            groups.append({
                "case":case,"method":arm,"n_seeds":4,
                "mean_wh":avg(energies),"median_wh":median(energies),
                "minimum_wh":min(energies),"maximum_wh":max(energies),
                "seed_wh":{str(x["seed"]):x["best_feasible_wh"] for x in selected},
                "online_usable_total":sum(x["online_usable"] for x in selected),
                "same_design_5s_verified":sum(x["verified_5s"] for x in selected),
                "soc_exact_n":exact_den,
                "soc_exact_weighted_nll":sum(n*y for n,y in exact)/exact_den if exact_den else None,
                "soc_exact_95_covered":sum(x["soc_exact_95_coverage_num"] for x in selected),
                "soc_interval_n":int_den,
                "soc_interval_weighted_nll":sum(n*y for n,y in interval)/int_den if int_den else None,
                "soc_skipped":sum(x["soc_skipped"] for x in selected),
            })
    paired=[]
    for case in CASES:
        for _,_,name in CONTRASTS:
            z=[r for r in pair_records if r["case"]==case and r["contrast"]==name]
            d=[r["left_minus_right_wh"] for r in z]
            paired.append({
                "case":case,"contrast":name,
                "difference_direction":"left_minus_right (negative = left lower Wh)",
                "paired_n":4,"seed_differences_wh":{str(x["seed"]):x["left_minus_right_wh"] for x in z},
                "mean_wh":avg(d),"median_wh":median(d),
                "left_better":sum(v<0 for v in d),"right_better":sum(v>0 for v in d),
                "ties":sum(v==0 for v in d),
                "interpretation":"Four exploratory paired seeds only; not an independent superiority test",
            })

    audit_results={
        "schema":"ra_lsbo_h_windows_pilot_sealed_readonly_v1",
        "release_zip_sha256":ARCHIVE_SHA,
        "execution_commit":CODE_SHA,
        "archive_file_count":len(contents),
        "files_sha256_checked":len(contents),
        "pinned_plan_sha256":PLAN_SHA,
        "pinned_registration_sha256":REG_SHA,
        "audit_evidence_member_hashes_checked":len(expected),
        "source_snapshots_verified":len(plan["sources"]),
        "settled_calls":856,
        "charged_by_role":dict(role_count),
        "trial_count":40,"online_proposals_checked":720,
        "rcaide_calls":0,"audited_source_forward_not_training_replay":True,
        "research_design":"two 100/140km missions; 4 paired seeds, 5 arms, 18 online",
        "model_calls_in_this_audit":0,
        "new_physical_truth_calls":0,
    }
    return audit_results,per_trial,pair_records,groups,paired


def main():
    if len(sys.argv)!=3:
        raise SystemExit("usage: analyze_dualview_pilot_sealed.py RELEASE_ZIP NEW_OUTPUT_DIR")
    archive,out=Path(sys.argv[1]),Path(sys.argv[2])
    if out.exists():raise FileExistsError("Read-only audit output directory already exists")
    integrity,trials,rows,groups,contrasts=archive_data(archive)
    out.mkdir(parents=True)
    save_json(out/"integrity.json",integrity)
    save_json(out/"group_descriptives.json",groups)
    save_json(out/"paired_contrasts.json",contrasts)
    table(out/"trial_rows.csv",trials,list(trials[0]))
    table(out/"paired_seed_rows.csv",rows,list(rows[0]))
    print(json.dumps({
        "audit_passed":True,"verified_calls":integrity["settled_calls"],
        "trials":integrity["trial_count"],"archive_files":integrity["archive_file_count"],
        "groups":groups,"contrasts":contrasts,
        "limits":"n=4 paired exploratory seeds; no new evaluator, GP, VAE or optimization calls"
    },allow_nan=False))


if __name__=="__main__":
    main()
