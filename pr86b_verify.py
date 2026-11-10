"""Independent PR86-B verifier; no evaluator calls or GP fitting."""
import json, math
from pathlib import Path
from risk_learning.core import TASKS, candidates, truth, observed, digest, features
from risk_learning.io import read, check_manifest
from pr86b_protocol import plan, METHODS
from pr86b_acquisition import choose, anchor_prevalence, calibrate_prediction, metrics

def req(ok, message):
    if not ok:
        raise ValueError(message)

def close(a,b):
    return (a is None and b is None) or (isinstance(a,(int,float)) and isinstance(b,(int,float))
        and math.isfinite(float(a)) and math.isfinite(float(b))
        and math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12))

def metric_pack(test_rows, probabilities, baseline):
    out = {}
    for head in ("region","resource","joint"):
        labels = ([bool(r["truth"]["feasible"]) for r in test_rows] if head=="joint"
                  else [bool(r["truth"][head] <= 0) for r in test_rows])
        m = metrics(labels, probabilities[head])
        b = baseline[head]
        m["brier_skill_score"] = None if b <= 0 or m["brier"] is None else 1.0-m["brier"]/b
        out[head] = m
    return out

def numeric_equal(a,b):
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(numeric_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return len(a)==len(b) and all(numeric_equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
        return close(a,b)
    return a==b

def metric_equal(a,b):
    return numeric_equal(a,b)

def feature_coverage(records):
    counts={str(b):sum(r["candidate"]["branch"]==b for r in records) for b in range(4)}
    vectors=[features(r["candidate"]) for r in records]
    nearest=[min(sum((a-b)**2 for a,b in zip(x,y))**0.5
                 for j,y in enumerate(vectors) if i!=j) for i,x in enumerate(vectors)]
    ordered=sorted(nearest)
    return dict(branch_counts=counts,branches_covered=sum(v>0 for v in counts.values()),
        nearest_neighbor_distance=dict(mean=sum(nearest)/len(nearest),
            median=ordered[len(ordered)//2],maximum=max(nearest)))

def verify(root, expected_sha=None):
    root=Path(root).resolve()
    file_count=check_manifest(root)
    req(not (root/"failure.json").exists(),"Failed attempt is retained; do not relabel it complete")
    spec=read(root/"plan.json");source=read(root/"source_receipt.json");report=read(root/"report.json")
    req(spec==plan(spec["profile"]),"Frozen PR86-B plan mismatch")
    req(source["code_sha"]==report["code_sha"] and (expected_sha is None or source["code_sha"]==expected_sha),
        "Code SHA mismatch")
    req(report["status"]=="COMPLETE_PR86B_SEQUENTIAL_RISK_INFORMATION","Incomplete PR86-B status")
    events=[json.loads(s) for s in (root/"evaluations.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    started={};settled={}
    for e in events:
        cid=e["call_id"]
        if e["event"]=="started":
            req(cid not in started,"Duplicate evaluation start");started[cid]=e
        elif e["event"]=="settled":
            req(cid in started and cid not in settled,"Settlement without unique start")
            req(e.get("record_sha256")==digest(e.get("record")),"Settlement record hash mismatch")
            settled[cid]=e
        else:
            raise ValueError("Unknown evaluation ledger event")
    req(set(started)==set(settled),"Unsettled or unstarted mathematical evaluation")
    req(len(started)==spec["total_math_calls"],"Mathematical evaluation count mismatch")
    expected_trials=len(spec["units"])*len(METHODS)
    trials_seen=0
    expected_progress={}
    for unit in spec["units"]:
        task,repeat=unit["task"],unit["repeat"]
        name="%s/r%d"%(task,repeat)
        data=read(root/"units"/name/"unit.json")
        anchors=data["anchors"];test=data["test_rows"];pool=data["pool"]
        req(len(anchors)==spec["anchor_count"] and len(test)==spec["test_count"] and len(pool)==spec["candidate_pool"],
            "Unit data count mismatch")
        req([r["candidate"] for r in anchors]==candidates(unit["anchor_seed"],spec["anchor_count"]),
            "Anchor stream mismatch")
        req([r["candidate"] for r in test]==candidates(unit["test_seed"],spec["test_count"]),
            "Test stream mismatch")
        req(pool==candidates(unit["pool_seed"],spec["candidate_pool"]),"Candidate pool stream mismatch")
        req(calibration_pool==candidates(unit["calibration_seed"],spec["calibration_anchor_count"]),
            "Calibration-anchor pool stream mismatch")
        all_ids=[r["candidate_id"] for r in anchors+test]+[digest(c) for c in pool+calibration_pool]
        req(len(set(all_ids))==len(all_ids),"Anchor/acquisition/calibration/test overlap")
        for role,rows in (("anchor",anchors),("test",test)):
            for i,row in enumerate(rows):
                call=name+"/"+role+"/%04d"%i if role=="test" else name+"/anchor/%03d"%i
                req(row["call_id"]==call and row["candidate_id"]==digest(row["candidate"]),
                    "Dataset call/candidate identity mismatch")
                req(numeric_equal(row["truth"],truth(task,row["candidate"])),"Dataset truth mismatch")
                if role=="anchor":
                    req(row["observed"]==observed(row["candidate"],row["truth"]),"Anchor observation contract mismatch")
                req(call in settled and started[call]["candidate"]==row["candidate"],"Dataset missing from paid ledger")
                req(settled[call]["record_sha256"]==digest(row),"Dataset settlement mismatch")
        rates=anchor_prevalence([r["observed"] for r in anchors])
        req(data["anchor_rates"]==rates,"Anchor prevalence mismatch")
        baseline={}
        for head in ("region","resource","joint"):
            labels=([bool(r["truth"]["feasible"]) for r in test] if head=="joint"
                    else [bool(r["truth"][head]<=0) for r in test])
            q=rates[head]
            baseline[head]=sum((q-float(y))**2 for y in labels)/len(labels)
        req(all(close(baseline[k],data["anchor_baseline_brier"][k]) for k in baseline),
            "Anchor baseline Brier mismatch")
        req(len(data["trials"])==len(METHODS),"Missing method trial")
        for trial in data["trials"]:
            trials_seen+=1
            method=trial["method"]
            req(method in METHODS and trial["task"]==task and trial["repeat"]==repeat,"Trial identity mismatch")
            req(trial["anchor_ids"]==[r["candidate_id"] for r in anchors],"Arms do not share common anchors")
            available=list(pool);calibration_available=list(calibration_pool);history=list(anchors)
            req(len(trial["proposals"])==spec["online_per_method"] and
                len(trial["online_observations"])==spec["online_per_method"],"Online budget mismatch")
            req(trial["model_refits"]==spec["online_per_method"]+1,"Sequential refit count mismatch")
            for step,(proposal,observation) in enumerate(zip(trial["proposals"],trial["online_observations"])):
                req(proposal["step"]==step and proposal["method"]==method and proposal["pre_outcome"] is True,
                    "Proposal identity/order mismatch")
                expected_progress[("proposal",task,repeat,method,step)]=proposal
                expected_progress[("observation",task,repeat,method,step)]=observation
                req(proposal["history_sha256"]==digest([r["observed"] for r in history])
                    and proposal["available_pool_sha256"]==digest(available)
                    and proposal["calibration_available_pool_sha256"]==digest(calibration_available)
                    and proposal["candidate_pool_sha256"]==digest(pool)
                    and proposal["calibration_pool_sha256"]==digest(calibration_pool),
                    "Proposal history/pool binding mismatch")
                req(not any(k in proposal for k in ("truth","label","objective","heldout_truth")),
                    "Proposal contains an outcome or privileged label")
                fit=proposal["fit_receipt"]
                expected_queries=[digest(x) for x in available]+[r["candidate_id"] for r in test]
                req(fit["method"]=="native_censored" and fit["training_rows"]==len(history)
                    and fit["query_ids"]==expected_queries and fit["objective_labels_used"] is False,
                    "Fit receipt is not bound to the observable prefix and unlabeled query pool")
                req("test_truth" not in fit and "heldout_truth" not in fit,
                    "Held-out truth leaked into fit receipt")
                forecast={"gaussian":proposal["forecast_gaussian"],
                          "probabilities":proposal["forecast_probabilities"]}
                idx,receipt=choose(method,available,[r["candidate"] for r in history],forecast,
                                   trial["seed"],step,calibration_candidates=calibration_available)
                if receipt["selected_from"]=="calibration_anchor_pool":
                    selected=calibration_available[idx]
                    selected_pool="calibration_anchor_pool"
                    calibration_available.pop(idx)
                else:
                    selected=available[idx]
                    selected_pool="acquisition_pool"
                    available.pop(idx)
                req(idx==proposal["selected_index"] and selected==proposal["selected_candidate"]
                    and proposal["selected_pool"]==selected_pool
                    and proposal["selected_candidate_sha256"]==digest(selected)
                    and proposal["selected_from"]==receipt["selected_from"]
                    and metric_equal(proposal["scores"],receipt["scores"]) and proposal["rule"]==receipt["rule"],
                    "Acquisition decision cannot be independently replayed")
                rec=observation["record"]
                req(observation["step"]==step and rec["candidate"]==selected
                    and rec["candidate_id"]==digest(selected)
                    and rec["observed"]==observed(selected,rec["truth"])
                    and numeric_equal(rec["truth"],truth(task,selected)),"Online outcome binding mismatch")
                call=name+"/online/"+method+"/%03d"%step
                req(rec["call_id"]==call and call in settled and started[call]["candidate"]==selected
                    and settled[call]["record"]==rec
                    and settled[call]["record_sha256"]==digest(rec),"Online ledger mismatch")
                history.append(rec)
            if method=="calibration_aware_voi":
                req(len(calibration_available)==0,"Reserved calibration-anchor pool not fully used")
            else:
                req(len(calibration_available)==len(calibration_pool),"Non-VOI arm consumed calibration-only candidates")
            req(trial["final_n"]==spec["total_observations_per_method"]
                and trial["online_calls"]==spec["online_per_method"],"Final budget mismatch")
            req(trial["selected_branch_counts"]=={str(b):sum(r["candidate"]["branch"]==b for r in history)
                for b in range(4)},"Branch count mismatch")
            req(metric_equal(trial["feature_coverage"],feature_coverage(history)),
                "Hierarchical feature-space coverage mismatch")
            req(close(trial["gp_fit_seconds"],
                sum(p["fit_receipt"]["elapsed_seconds"] for p in trial["proposals"])
                +trial["final_fit_receipt"]["elapsed_seconds"]),
                "Per-trial GP fit-time accounting mismatch")
            req(len(trial["checkpoints"])==len(spec["checkpoints"]),"Checkpoint count mismatch")
            for cp in trial["checkpoints"]:
                n=cp["n"]
                prefix=history[:n]
                req(n in spec["checkpoints"] and cp["method"]==method
                    and cp["training_prefix_sha256"]==digest([r["observed"] for r in prefix]),
                    "Checkpoint training prefix mismatch")
                req(cp["test_candidate_ids"]==[r["candidate_id"] for r in test]
                    and cp["anchor_rates"]==rates,"Checkpoint test/anchor identity mismatch")
                online_n=n-spec["anchor_count"]
                calibration_rows=list(anchors)
                calibration_ids=[r["candidate_id"] for r in anchors]
                if method=="iid_random":
                    calibration_rows += [o["record"] for o in trial["online_observations"][:online_n]]
                    calibration_ids += [o["candidate_id"] for o in trial["online_observations"][:online_n]]
                elif method=="calibration_aware_voi":
                    for prior,outcome in zip(trial["proposals"][:online_n],
                                             trial["online_observations"][:online_n]):
                        if prior["rule"]=="reserved_random_calibration_anchor":
                            calibration_rows.append(outcome["record"])
                            calibration_ids.append(outcome["candidate_id"])
                calibration_rates=anchor_prevalence([r["observed"] for r in calibration_rows])
                req(cp["calibration_rates"]==calibration_rates
                    and cp["calibration_anchor_ids"]==calibration_ids,
                    "Calibration anchor roster mismatch")
                cpfit=cp["fit_receipt"]
                prefix_ids={r["candidate_id"] for r in history[:n]}
                cp_available=[c for c in pool if digest(c) not in prefix_ids]
                req(cpfit["method"]=="native_censored" and cpfit["training_rows"]==n
                    and cpfit["objective_labels_used"] is False
                    and cpfit["query_ids"]==[digest(c) for c in cp_available]+[r["candidate_id"] for r in test],
                    "Checkpoint fit receipt mismatch")
                raw=cp["test_probabilities_raw"];cal=cp["test_probabilities_anchor_calibrated"]
                recomputed,shifts=calibrate_prediction({"probabilities":raw},calibration_rates)
                req(metric_equal(cal,recomputed) and all(close(cp["calibration_logit_shifts"][h],shifts[h])
                    for h in shifts),"Anchor calibration mismatch")
                req(metric_equal(cp["raw_metrics"],metric_pack(test,raw,baseline))
                    and metric_equal(cp["anchor_calibrated_metrics"],metric_pack(test,cal,baseline)),
                    "Held-out scoring mismatch")
                req("test_truth" not in cp["fit_receipt"] and "heldout_truth" not in cp["fit_receipt"],
                    "Held-out labels leaked into fit receipt")
    progress=[json.loads(s) for s in (root/"progress.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    observed_progress={}
    for event in progress:
        if event["event"] in ("proposal","observation"):
            key=(event["event"],event["task"],event["repeat"],event["method"],event["step"])
            req(key not in observed_progress,"Duplicate incremental progress event")
            payload=event["proposal"] if event["event"]=="proposal" else event["observation"]
            observed_progress[key]=payload
    req(observed_progress==expected_progress,"Incremental proposal/observation trace differs from sealed unit records")
    req(trials_seen==expected_trials and report["trials"]==expected_trials,"Trial roster mismatch")
    req(report["bill"]["math_started"]==spec["total_math_calls"]
        and report["bill"]["math_settled"]==spec["total_math_calls"],"Report math bill mismatch")
    req(report["bill"]["gp_head_errors"]==0 and report["bill"]["gp_kernel_errors"]==0
        and report["bill"]["kernel_candidates_rejected"]==0,"Native GP error/rejection in run")
    req(report["bill"]["gp_head_completed"]==spec["expected_gp_heads"]
        and report["bill"]["gp_kernel_completed"]==spec["expected_gp_kernel_constructions"],
        "GP/kernel construction count mismatch")
    return dict(status="VERIFIED_PR86B_SEQUENTIAL_RISK_INFORMATION",files=file_count,
        trials=trials_seen,math_calls=len(started),new_math_calls=0,new_gp_fits=0,
        test_labels_fit_isolated=True,acquisition_decisions_replayed=True)
