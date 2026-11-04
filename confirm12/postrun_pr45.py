"""Independent read-only post-run accounting and exploratory paired analysis.

This module NEVER evaluates an aircraft or updates an original journal.
It reads the sealed PR45 result directory and optionally its smaller archived
method results, computes seed-paired descriptive comparisons, and separately
tests the raw shared-initial q_A feasibility rate.

All tests are RETROSPECTIVE / PROVISIONAL. They do not retrospectively make the
unqualified 140 km design, CBOB implementation or method mapping confirmatory.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import statistics
import zipfile

from ra_lsbo.budget import digest
from ra_lsbo.partial_bo.records import DesignRecord

CASES=("aircraft_ordinary","aircraft_sparse")
PRIMARY=("ra_lsbo","full_latent")
AUX=("full_onehot","reverse_representation","interval_as_missing","random","cbob_adapted")
EXPECTED_GROUPS=2048
EXPECTED_TRIALS=6656
CALL_CAP=151040
# Exact artifacts verified on the PR45 GitHub result branch / Release.
FROZEN_SOURCE_SHA="386082beeacac8ae43d0dfb1ccda5c5ad3b356cf"
FROZEN_PLAN_SHA256="ac5f1424f3c8d52eff9ffd835cb5004224be965afee8ec15069e1fea38c43012"
FROZEN_REAL_SQLITE_SHA256="a30b0dadb5d619be9b6c0d70c5aa29fe115f86b85d62a449e4acf05325d6f1b7"
FROZEN_METHOD_ARCHIVE_SHA256="02227a77d3ba7b9b0427c99f8f2671fbc7c781e919930539d3f08808c5b71ef1"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _sqlite(path):
    uri=Path(path).resolve().as_uri()+"?mode=ro"
    return sqlite3.connect(uri,uri=True)


def verify_roster(plan):
    """Verify the entire frozen mixed 1024/256 roster without running physics."""
    if plan.get("max_calls")!=CALL_CAP or plan.get("workers")!=12 or set(plan.get("cases",{}))!=set(CASES):
        raise ValueError("Frozen 151040-call/12-worker/two-task roster differs")
    count=0;trials=0
    expected={}
    for case in CASES:
        spec=plan["cases"][case]
        seeds=spec["seeds"]
        if len(seeds)!=1024 or len(set(seeds))!=1024:
            raise ValueError("Primary seed count/uniqueness mismatch")
        arms=spec["arms_by_seed"]
        if set(arms)!={str(s) for s in seeds}:
            raise ValueError("Seed-to-method map drift")
        seven=0
        for seed in seeds:
            labels=arms[str(seed)]
            if labels not in [list(PRIMARY),list(PRIMARY+AUX)]:
                raise ValueError("Unfrozen arm order or auxiliary roster")
            seven+=(len(labels)==7)
            count+=12+19*len(labels)
            trials+=len(labels)
            expected[(case,int(seed))]=tuple(labels)
        if seven!=256:
            raise ValueError("Expected exactly 256 nested auxiliary seeds per task")
    if count!=CALL_CAP or trials!=EXPECTED_TRIALS:
        raise ValueError("Call cap or number of methods changed")
    if set(plan["cases"][CASES[0]]["seeds"]) & set(plan["cases"][CASES[1]]["seeds"]):
        raise ValueError("Cross-task seed overlap")
    return expected


def verify_progress(root, expected):
    """Compare all group receipts with the frozen seed/method registration."""
    path=Path(root)/"progress.jsonl"
    seen=set();total=0;methods=0;counter=set();bad=[]
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip(): continue
        item=json.loads(line)
        key=(item["case"],int(item["seed"]))
        if key not in expected or key in seen:
            bad.append("unknown/duplicate progress group")
            continue
        seen.add(key)
        if item["methods"]!=len(expected[key]) or item["total_groups"]!=EXPECTED_GROUPS:
            bad.append("progress method registration mismatch")
        counter.add(item["completed_groups"])
        total+=int(item["charged_true_calls"])
        methods+=int(item["methods"])
    if seen!=set(expected) or counter!=set(range(1,EXPECTED_GROUPS+1)):
        bad.append("incomplete/duplicate progress sequence")
    if methods!=EXPECTED_TRIALS:
        bad.append("method count mismatch")
    return {"groups":len(seen),"methods":methods,"calls_from_progress":total,"errors":bad}


def verify_ledger_structure(root,plan,*,check_event_chain=True):
    """Check calls and the immutable event-hash chain, without changing DB.

    This independent read uses SQL aggregates (not the prior runner auditor).
    Original RUNNING_REAL metadata is preserved; a postrun attestation is
    written to a separate new directory rather than modifying the source.
    """
    db_path=Path(root)/"real_calls.sqlite"
    with _sqlite(db_path) as db:
        quick=db.execute("PRAGMA quick_check").fetchone()[0]
        meta=dict(db.execute("SELECT key,value FROM meta"))
        by_role={role:n for role,n in db.execute(
            "SELECT role, count(*) FROM calls GROUP BY role")}
        by_state={state:n for state,n in db.execute(
            "SELECT state,count(*) FROM calls GROUP BY state")}
        by_case={case:n for case,n in db.execute(
            "SELECT case_name,count(*) FROM calls GROUP BY case_name")}
        calls=sum(by_role.values())
        dup=db.execute(
            "SELECT count(*) FROM (SELECT id FROM calls GROUP BY id HAVING count(*)>1)"
        ).fetchone()[0]
        head="0"*64
        events=0
        counts=Counter()
        corrupt_at=None
        if check_event_chain:
            for seq,cid,kind,payload,previous,claimed in db.execute(
                    "SELECT seq,call_id,kind,payload,prev,chain FROM events ORDER BY seq"):
                events+=1
                computed=digest({
                    "seq":seq,"call_id":cid,"kind":kind,
                    "payload":payload,"prev":previous})
                if corrupt_at is None and (seq!=events or previous!=head or claimed!=computed):
                    corrupt_at=seq
                counts[kind]+=1
                head=computed
    errors=[]
    if quick!="ok":errors.append("sqlite integrity check failed")
    if dup:errors.append("duplicate call ID")
    if meta.get("scope")!="REAL_AIRCRAFT_USER_INITIATED":
        errors.append("wrong real-call scope")
    if meta.get("plan_sha")!=digest(plan):
        errors.append("SQLite plan hash differs from plan.json")
    if meta.get("plan")!=json.dumps(plan,sort_keys=True,separators=(",",":"),allow_nan=False):
        errors.append("SQLite registered plan JSON differs")
    if calls>CALL_CAP or by_state!={"settled":calls}:
        errors.append("Unsettled or over-budget real model calls")
    if check_event_chain:
        if corrupt_at is not None:errors.append("event-chain corruption")
        if counts!=Counter({"created":1,"started":calls,"settled":calls}):
            errors.append("unexpected number or kind of call events")
    return {"calls":calls,"by_role":by_role,"by_state":by_state,"by_case":by_case,
            "ledger_state_original":meta.get("state"),"quick_check":quick,
            "duplicate_call_ids":dup,"chain_head_recomputed":head,
            "event_count":events if check_event_chain else None,
            "errors":errors}


def _load_methods(root,expected,archive=None):
    """Read all independent final method receipts, using local files or ZIP.

    ZIPs are read in place, not extracted; no unsafe paths are written.
    """
    source=Path(root)
    results={}
    ver_present=set()
    if archive is None:
        for (case,seed),arms in expected.items():
            for arm in arms:
                p=source/"trials"/case/str(seed)/arm
                results[(case,seed,arm)]=_read(p/"result.json")
                if (p/"verification.json").is_file():
                    ver_present.add((case,seed,arm))
    else:
        expression=re.compile(
            r"(?:^|/)trials/(aircraft_ordinary|aircraft_sparse)/(\d+)/([^/]+)/(result|verification)\.json$")
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                m=expression.search(name.replace("\\","/"))
                if not m:continue
                case,seed,arm,kind=m.groups()
                key=(case,int(seed),arm)
                if (case,int(seed)) not in expected or arm not in expected[(case,int(seed))]:
                    raise ValueError("Archive contains an unregistered method")
                if kind=="verification":
                    ver_present.add(key)
                else:
                    if key in results:raise ValueError("Duplicate result.json in archive")
                    results[key]=json.loads(z.read(name))
    expected_keys={(case,seed,arm) for (case,seed),arms in expected.items() for arm in arms}
    if set(results)!=expected_keys:
        raise ValueError(f"Missing/extra method results: expected {len(expected_keys)}, got {len(results)}")
    bad=[];checksums={};nominal_counts=Counter();verification_counts=Counter()
    for (case,seed,arm),row in results.items():
        if (row.get("case"),row.get("seed"),row.get("method"))!=(case,seed,arm):
            bad.append("Wrong case/seed/arm result identity")
        if row.get("online_calls")!=18:
            bad.append("Incomplete online horizon")
        token=row.get("shared_initial_sha256")
        if not isinstance(token,str) or len(token)!=64:
            bad.append("Missing shared initial digest")
        key=(case,seed)
        if key in checksums and checksums[key]!=token:
            bad.append("Different shared initial history across methods")
        checksums[key]=token
        feasible=row.get("nominal_usable_feasible")
        verified=row.get("same_design_5sec_verification")
        if type(feasible) is not bool or type(verified) is not bool:
            bad.append("Invalid nominal/verification flags")
            continue
        if verified!=( (case,seed,arm) in ver_present):
            bad.append("Verification file/summary disagreement")
        if feasible!=verified:
            bad.append("Nominal feasibility must match validation existence")
        energy=row.get("nominal_best_energy_Wh")
        if feasible and (not isinstance(energy,(float,int)) or
                         isinstance(energy,bool) or not math.isfinite(energy)):
            bad.append("Feasible group has no finite best energy")
        if not feasible and energy is not None:
            bad.append("Infeasible group contains false best energy")
        nominal_counts[(case,arm)]+=int(feasible)
        verification_counts[(case,arm)]+=int(verified)
    if bad:raise ValueError("Method archive audit: "+"; ".join(sorted(set(bad))))
    return results,ver_present,nominal_counts,verification_counts


def _interval(values):
    if not values:return [None,None]
    mean=statistics.mean(values)
    sd=statistics.stdev(values) if len(values)>1 else 0.
    # Approximate paired normal CI: descriptive, not a preregistered test.
    half=1.95996398454*sd/math.sqrt(len(values))
    return [mean-half,mean+half]


def _pair_energy(results,case,control,seeds):
    pairs=[]
    for seed in seeds:
        a=results[(case,seed,"ra_lsbo")]
        b=results[(case,seed,control)]
        if a["nominal_usable_feasible"] and b["nominal_usable_feasible"]:
            pairs.append((seed,float(a["nominal_best_energy_Wh"])-
                               float(b["nominal_best_energy_Wh"])))
    values=[x[1] for x in pairs]
    if not values:
        return {"paired_feasible":0,"mean_difference_Wh":None,"median_difference_Wh":None}
    return {"paired_feasible":len(values),
            "difference":"RA-LSBO minus "+control+"; negative favors RA-LSBO",
            "mean_difference_Wh":statistics.mean(values),
            "median_difference_Wh":statistics.median(values),
            "normal_approx_95_ci_Wh":_interval(values),
            "RA_lower":sum(x<0 for x in values),
            "control_lower":sum(x>0 for x in values),
            "exact_equal":sum(x==0 for x in values),
            "energy_comparison_excludes_unpaired_infeasible":True}


def _exact_mcnemar(b,c):
    n=b+c
    if not n:return 1.
    # Two-sided exact conditional binomial test, handles the n=2 case.
    return min(1.,2.*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n)


def _pair_discovery(results,case,control,seeds):
    pairs=[(results[(case,s,"ra_lsbo")]["nominal_usable_feasible"],
            results[(case,s,control)]["nominal_usable_feasible"]) for s in seeds]
    b=sum(a and not z for a,z in pairs)
    c=sum(z and not a for a,z in pairs)
    deltas=[int(a)-int(z) for a,z in pairs]
    # Stable resampling of entire matched seed pairs, never individual arms.
    rng=random.Random(20261104+len(seeds))
    boot=[]
    for _ in range(1200):
        boot.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas)/len(deltas))
    boot.sort()
    return {"paired_seeds":len(pairs),
            "RA_feasible":sum(a for a,z in pairs),
            "control_feasible":sum(z for a,z in pairs),
            "RA_only_feasible":b,"control_only_feasible":c,
            "paired_rate_difference_pp":100*statistics.mean(deltas),
            "bootstrap_95_ci_pp":[100*boot[29],100*boot[1170]],
            "exact_McNemar_two_sided_p_descriptive":_exact_mcnemar(b,c)}


def paired_tables(results,plan):
    primary={};auxiliary={}
    for case in CASES:
        seeds=plan["cases"][case]["seeds"]
        primary[case]={
            "reference":"ra_lsbo - full_latent",
            "energy":_pair_energy(results,case,"full_latent",seeds),
            "feasible_discovery":_pair_discovery(results,case,"full_latent",seeds)}
        nested=[s for s in seeds if len(plan["cases"][case]["arms_by_seed"][str(s)])==7]
        auxiliary[case]={
            arm:{
                "energy":_pair_energy(results,case,arm,nested),
                "feasible_discovery":_pair_discovery(results,case,arm,nested)}
            for arm in AUX}
    return primary,auxiliary


def initial_feasibility(root,plan):
    """Observed q_A single-draw feasibility on 24,576 shared initial calls.

    This diagnostic is *not* independent 300-draw prospective task qualification.
    """
    counts={case:{"n":0,"feasible":0,"unknown":0} for case in CASES}
    from scipy.stats import beta
    with _sqlite(Path(root)/"real_calls.sqlite") as db:
        for case,seed,idx,blob,status in db.execute(
                "SELECT case_name,seed,idx,record,state FROM calls "
                "WHERE role='initial' ORDER BY case_name,seed,idx"):
            if status!="settled" or case not in counts or blob is None:
                raise ValueError("Unsettled/malformed shared initial record")
            rec=DesignRecord.from_dict(json.loads(blob))
            if rec.observation.contract_id!=plan["cases"][case]["contracts"]["20s"]:
                raise ValueError("Wrong 20s physics contract on shared initial")
            row=counts[case];row["n"]+=1
            row["feasible"]+=int(rec.usable_feasible)
            row["unknown"]+=int(rec.physical_feasible is None)
    for case,row in counts.items():
        if row["n"]!=12288:raise ValueError("Expected 12*1024 shared initial records")
        n=row["n"];k=row["feasible"]
        lo=0. if k==0 else float(beta.ppf(.025,k,n-k+1))
        hi=1. if k==n else float(beta.ppf(.975,k+1,n-k))
        row.update({"rate":k/n,
                    "clopper_pearson_95":[lo,hi],
                    "task_diagnostic_passes_original_threshold": (
                        lo>.2 if case=="aircraft_ordinary" else hi<.1),
                    "task_independently_qualified":False,
                    "interpretation":"observed shared-initial q_A draws, not an independently registered qualification sample"})
    return counts


def analyze(root,output,archive=None,*,check_event_chain=True):
    """Write a new report without modifying historical source files or ledger."""
    root=Path(root)
    output=Path(output)
    if output.exists():raise FileExistsError("Choose a new postrun output directory")
    plan=_read(root/"plan.json")
    expected=verify_roster(plan)
    completion=_read(root/"completed.json")
    original_audit=_read(root/"audit.json")
    status=_read(root/"execution_status.json")
    progress=verify_progress(root,expected)
    ledger=verify_ledger_structure(root,plan,check_event_chain=check_event_chain)
    results,verified,nominal,verification=_load_methods(root,expected,archive)
    cap_delta=CALL_CAP-ledger["calls"]
    expected_initial=EXPECTED_GROUPS*12
    expected_online=EXPECTED_TRIALS*18
    expected_ver=len(verified)
    errors=list(ledger["errors"])+list(progress["errors"])
    # Bind local originals to the *already uploaded* external SHA-256 receipts.
    if plan.get("source_commit")!=FROZEN_SOURCE_SHA:
        errors.append("Campaign source differs from uploaded PR45 source")
    if _sha(root/"plan.json")!=FROZEN_PLAN_SHA256:
        errors.append("Campaign plan bytes differ from GitHub preserved plan")
    db_sha=_sha(root/"real_calls.sqlite")
    if db_sha!=FROZEN_REAL_SQLITE_SHA256:
        errors.append("Real-call SQLite bytes differ from Release-verified handoff")
    if archive is not None and _sha(archive)!=FROZEN_METHOD_ARCHIVE_SHA256:
        errors.append("Method result ZIP bytes differ from GitHub preserved archive")
    if (check_event_chain and
            ledger["chain_head_recomputed"]!=original_audit.get("event_chain_head")):
        errors.append("Independent event chain head differs from original evidence")
    if (ledger["by_role"]!={"initial":expected_initial,
                            "online":expected_online,"verification":expected_ver}):
        errors.append("Actual charged calls differ from exact 12/18/optional-verification design")
    if cap_delta!=EXPECTED_TRIALS-expected_ver:
        errors.append("Unused budget is not fully explained by missing optional verifications")
    if (progress["calls_from_progress"]!=ledger["calls"] or
            progress["groups"]!=EXPECTED_GROUPS or
            progress["methods"]!=EXPECTED_TRIALS):
        errors.append("Progress and SQLite results disagree")
    if (completion.get("charged_calls")!=ledger["calls"] or
            completion.get("status")!="ALL_REAL_EVALUATIONS_COMPLETED" or
            status.get("completed_groups")!=EXPECTED_GROUPS or
            status.get("failures")!=[] or
            not original_audit.get("ledger_contract_passed") or
            original_audit.get("group_errors")!=[] or
            original_audit.get("calls")!=ledger["calls"]):
        errors.append("Original final and audit receipts disagree")
    primary,auxiliary=paired_tables(results,plan)
    initial=initial_feasibility(root,plan)
    report={
        "schema":"ra_lsbo_pr45_independent_postrun_v1",
        "source_sha":plan.get("source_commit"),
        "original_result_directory":str(root),
        "original_sqlite_sha256":db_sha,
        "frozen_plan_sha256":_sha(root/"plan.json"),
        "methods":EXPECTED_TRIALS,
        "verifications":expected_ver,
        "unused_optional_verifications":cap_delta,
        "original_ledger_status_unchanged":ledger["ledger_state_original"],
        "independent_postrun_status":(
            "COMPLETE_AUDITED_POSTRUN" if not errors else "INCONSISTENT"),
        "accounting":ledger,
        "progress":progress,
        "shared_initial_qA_feasibility":initial,
        "primary_paired_exploratory":primary,
        "auxiliary_paired_exploratory":auxiliary,
        "scientific_task_qualification":False,
        "formal_seven_arm_equivalence_proven":False,
        "all_p_values_and_intervals_posthoc_descriptive":True,
        "original_run_not_modified":True,
        "errors":errors,
    }
    output.mkdir(parents=True,exist_ok=False)
    with (output/"postrun_report.json").open("x",encoding="utf8") as f:
        json.dump(report,f,indent=2,sort_keys=True,allow_nan=False,ensure_ascii=False)
        f.write("\n")
    with (output/"paired_main_by_seed.csv").open("x",newline="",encoding="utf8") as f:
        writer=csv.writer(f)
        writer.writerow(["case","seed","RA_feasible","FullLatent_feasible",
                         "RA_best_Wh","FullLatent_best_Wh","RA_minus_FullLatent_Wh"])
        for case in CASES:
            for seed in plan["cases"][case]["seeds"]:
                a=results[(case,seed,"ra_lsbo")]
                b=results[(case,seed,"full_latent")]
                x=a["nominal_best_energy_Wh"] if a["nominal_usable_feasible"] else None
                y=b["nominal_best_energy_Wh"] if b["nominal_usable_feasible"] else None
                writer.writerow([case,seed,a["nominal_usable_feasible"],
                    b["nominal_usable_feasible"],x,y,x-y if x is not None and y is not None else None])
    if errors:
        raise RuntimeError("Independent PR45 postrun audit failed; review new report")
    return report
