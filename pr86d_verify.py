"""Independent PR86-D verifier for nested anchors, paid ledgers and replayed decisions."""
import json
import math
from pathlib import Path

from risk_learning.core import candidates, observed, digest
from risk_learning.io import read, check_manifest
from risk_learning.native import training_payload
from pr86d_protocol import plan, METHODS, ANCHOR_SIZES
from pr86c_evaluator import truth_for_evaluation
from pr86c_models import outcome_state
from pr86c_acquisition import choose
from pr86b_acquisition import anchor_prevalence
from pr86d_study import _test_metrics, _baseline_brier


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def close(a,b):
    if a is None or b is None:
        return a is None and b is None
    return isinstance(a,(float,int)) and isinstance(b,(float,int)) and (
        math.isfinite(a) and math.isfinite(b) and
        math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-11))


def same_numbers(a,b):
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(
            same_numbers(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return isinstance(b,list) and len(a)==len(b) and all(
            same_numbers(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,int)) and not isinstance(a,bool):
        return close(a,b)
    return a==b


def check_row(row,task,call,has_observation,started,settled):
    need(row["call_id"]==call,"Wrong paid call ID")
    need(row["candidate_id"]==digest(row["candidate"]),"Candidate digest mismatch")
    need(same_numbers(row["truth"],truth_for_evaluation(task,row["candidate"])),
         "Mathematical truth replay mismatch")
    if has_observation:
        need(row.get("observed")==observed(row["candidate"],row["truth"]),
             "Observable censoring contract mismatch")
    else:
        need("observed" not in row,"Heldout row must not contain training observations")
    need(call in started and call in settled,"Missing started/settled ledger event")
    need(started[call]["candidate"]==row["candidate"],"Started candidate mismatch")
    need(settled[call]["record"]==row and
         settled[call]["record_sha256"]==digest(row),"Settled record mismatch")


def check_fit(current,available,tests,risk_receipt,objective_receipt):
    need(risk_receipt["status"]=="OK" and
         not risk_receipt["objective_labels_used"] and
         not risk_receipt["privileged_training"],
         "Risk GP used forbidden objective/privileged labels")
    payload=training_payload(current,"native_censored")
    need(risk_receipt["training_rows"]==len(current) and
         risk_receipt["training_input_sha256"]==digest(payload),
         "Risk GP training prefix mismatch")
    need(risk_receipt["query_ids"]==[digest(c) for c in available]+
         [r["candidate_id"] for r in tests],"Risk GP query identity mismatch")
    need(set(risk_receipt["heads"])=={"region","resource"},"Missing risk head")
    for head in risk_receipt["heads"].values():
        need(len(head["candidates"])==8 and all(
            item["status"]=="ok" for item in head["candidates"]),
            "Risk GP kernel diagnostic failed")

    exact=[r for r in current if r["objective"]["kind"]=="exact"]
    need(objective_receipt["training_rows"]==len(current) and
         objective_receipt["exact_training_rows"]==len(exact) and
         not objective_receipt["privileged_training"] and
         not objective_receipt["missing_labels_imputed"],
         "Objective GP observation contract mismatch")
    need(objective_receipt["training_input_sha256"]==digest([
         (r["candidate_id"],r["objective"]) for r in exact]) and
         objective_receipt["query_ids"]==[digest(c) for c in available],
         "Objective GP query/training identity mismatch")
    if len(exact)<2:
        need(objective_receipt["status"]=="INSUFFICIENT_EXACT_OBJECTIVES",
             "Objective cold fallback status mismatch")
    else:
        need(objective_receipt["status"]=="OK" and
             objective_receipt["objective_fit_only_exact"] and
             len(objective_receipt["head"]["candidates"])==8,
             "Objective GP fit diagnostic mismatch")


def verify(root,expected_sha=None):
    root=Path(root).resolve()
    file_count=check_manifest(root)
    need(not (root/"failure.json").exists(),"Failed attempt cannot be certified")
    spec=read(root/"plan.json")
    source=read(root/"source_receipt.json")
    report=read(root/"report.json")
    need(spec==plan(spec["profile"]),"PR86-D plan differs from frozen protocol")
    need(report["status"]=="COMPLETE_PR86D_INITIAL_SAMPLE_ABLATION",
         "Incomplete PR86-D status")
    need(report["code_sha"]==source["code_sha"] and
         (expected_sha is None or source["code_sha"]==expected_sha),
         "Source SHA mismatch")

    started,settled={},{}
    for line in (root/"evaluations.jsonl").read_text(encoding="utf-8").splitlines():
        event=json.loads(line)
        call=event["call_id"]
        if event["event"]=="started":
            need(call not in started,"Duplicate start")
            started[call]=event
        elif event["event"]=="settled":
            need(call in started and call not in settled,"Unpaired settlement")
            need(event["record_sha256"]==digest(event["record"]),
                 "Settled record digest mismatch")
            settled[call]=event
        else:
            raise ValueError("Unknown ledger event")
    need(set(started)==set(settled) and
         len(started)==spec["total_math_calls"],
         "Started/settled mathematical call count mismatch")

    trials_seen=0
    expected_progress={}
    objective_success_count=0
    objective_receipts_ok=0
    for unit in spec["units"]:
        task,repeat=unit["task"],unit["repeat"]
        name="%s/r%d"%(task,repeat)
        data=read(root/"units"/name/"unit.json")
        anchors,pool,tests=data["anchors"],data["pool"],data["tests"]
        need([r["candidate"] for r in anchors]==candidates(
            unit["anchor_seed"],spec["shared_anchor_count"]),
            "Shared 24-anchor stream mismatch")
        need(pool==candidates(unit["pool_seed"],spec["candidate_pool"]),
             "Candidate pool seed mismatch")
        need([r["candidate"] for r in tests]==candidates(
            unit["test_seed"],spec["test_count"]),
            "Test seed mismatch")
        ids=[r["candidate_id"] for r in anchors+tests]+[digest(c) for c in pool]
        need(len(ids)==len(set(ids)),"Anchor/pool/test candidate overlap")
        for i,row in enumerate(anchors):
            check_row(row,task,name+"/anchor/%03d"%i,True,started,settled)
        for i,row in enumerate(tests):
            check_row(row,task,name+"/test/%04d"%i,False,started,settled)
        need(len(data["trials"])==len(ANCHOR_SIZES)*len(METHODS),
             "Missing anchor-size/method trajectory")

        for cohort_index,anchor_size in enumerate(ANCHOR_SIZES):
            cohort_anchors=anchors[:anchor_size]
            hidden=anchors[anchor_size:]
            rates=anchor_prevalence([r["observed"] for r in cohort_anchors])
            baseline=_baseline_brier(tests,rates)
            for method_index,method in enumerate(METHODS):
                trial=next((t for t in data["trials"] if
                    t["anchor_size"]==anchor_size and t["method"]==method),None)
                need(trial is not None,"Missing registered cohort/method")
                need(trial["task"]==task and trial["repeat"]==repeat,
                     "Trial identity mismatch")
                need(trial["anchor_ids"]==[r["candidate_id"] for r in cohort_anchors],
                     "Cohort anchor prefix mismatch")
                need(trial["masked_anchor_ids"]==[r["candidate_id"] for r in hidden],
                     "Masked anchor identity mismatch")
                initial_state=outcome_state([r["observed"] for r in cohort_anchors])
                need(trial["initial_outcome"]==initial_state,
                     "Initial observable state mismatch")
                online_budget=spec["online_per_method_by_anchor_size"][str(anchor_size)]
                proposals=trial["proposals"]
                online=trial["online_observations"]
                need(len(proposals)==len(online)==online_budget and
                     trial["online_calls"]==online_budget,
                     "Unequal online budget")
                need(trial["risk_refits"]==online_budget+1,
                     "Risk refit count mismatch")
                available=list(pool)
                history=list(cohort_anchors)
                first_objective_fit=None
                first_feasible=(anchor_size if initial_state["feasible_found"] else None)
                for step,(proposal,event) in enumerate(zip(proposals,online)):
                    current=[r["observed"] for r in history]
                    state=outcome_state(current)
                    need(proposal["anchor_size"]==anchor_size and
                         proposal["method"]==method and proposal["step"]==step,
                         "Proposal cohort/method/step mismatch")
                    need(proposal["history_observed_sha256"]==digest(current) and
                         proposal["available_sha256"]==digest(available) and
                         proposal["candidate_pool_sha256"]==digest(pool),
                         "Proposal did not use exact observable prefix/pool")
                    need(proposal["incumbent"]==state["best_exact_feasible"],
                         "Objective incumbent is stale or privileged")
                    need(proposal["pre_outcome"] and
                         proposal["no_test_truth_accessed"] and
                         proposal["no_unselected_truth_accessed"],
                         "Proposal provenance flags invalid")
                    check_fit(current,available,tests,
                              proposal["method_fit_risk"],
                              proposal["method_fit_objective"])
                    if proposal["method_fit_objective"]["status"]=="OK" and first_objective_fit is None:
                        first_objective_fit=len(history)
                    arm_seed=unit["pool_seed"]+1009*method_index
                    index,replay=choose(method,available,
                        proposal["candidate_forecast"],proposal["objective_forecast"],
                        state["best_exact_feasible"],arm_seed,step)
                    need(digest(proposal["scores"])==proposal["score_digest"],
                         "Stored acquisition score digest mismatch")
                    for key in ("method","rule","step","selected_index",
                                "selected_candidate_sha256","scores",
                                "cold_fallback","incumbent"):
                        need(same_numbers(replay[key],proposal[key]),
                             "Acquisition replay mismatch: "+key)
                    need(index==proposal["selected_index"],"Selected index mismatch")
                    selected=available.pop(index)
                    need(selected==proposal["selected_candidate"],
                         "Selected candidate mismatch")
                    record=event["record"]
                    call=(name+"/online/n0%02d/"%anchor_size+
                          method+"/%03d"%step)
                    need(event["step"]==step and
                         event["candidate_id"]==digest(selected) and
                         record["candidate"]==selected,
                         "Paid online observation differs from proposal")
                    check_row(record,task,call,True,started,settled)
                    history.append(record)
                    state_after=outcome_state([r["observed"] for r in history])
                    if first_feasible is None and state_after["feasible_found"]:
                        first_feasible=len(history)
                    expected_progress[("proposal",task,repeat,anchor_size,method,step)]=proposal
                    expected_progress[("observation",task,repeat,anchor_size,method,step)]={
                        "step":step,"candidate_id":record["candidate_id"],
                        "call_id":call,"observed":record["observed"]}

                objective_success_count += sum(p["method_fit_objective"]["status"]=="OK" for p in proposals)
                final_current=[r["observed"] for r in history]
                final_available=list(available)
                final_state=outcome_state(final_current)
                need(trial["final_n"]==spec["total_observations_per_arm"] and
                     trial["final_outcome"]==final_state,
                     "Final outcome mismatch")
                check_fit(final_current,final_available,tests,
                          trial["final_risk_fit_receipt"],
                          trial["final_objective_fit_receipt"])
                if trial["final_objective_fit_receipt"]["status"]=="OK":
                    objective_success_count += 1
                    if first_objective_fit is None:
                        first_objective_fit=len(history)
                need(trial["first_objective_gp_fit_n"]==first_objective_fit and
                     trial["first_feasible_n"]==first_feasible,
                     "First objective-GP/feasible discovery timing mismatch")
                expected_checkpoints=spec["checkpoints_by_anchor_size"][str(anchor_size)]
                need([cp["n"] for cp in trial["checkpoints"]]==expected_checkpoints,
                     "Checkpoint sequence mismatch")
                for cp in trial["checkpoints"]:
                    n=cp["n"]
                    prefix=history[:n]
                    prefix_observed=[r["observed"] for r in prefix]
                    state=outcome_state(prefix_observed)
                    need(cp["observed_prefix_sha256"]==digest(prefix_observed) and
                         cp["outcome"]==state,"Checkpoint prefix/outcome mismatch")
                    need(cp["initial_outcome"]==initial_state,
                         "Checkpoint initial state mismatch")
                    online_prefix=n-anchor_size
                    available_cp=list(pool)
                    for prior in trial["online_observations"][:online_prefix]:
                        cid=prior["candidate_id"]
                        selected_idx=next((i for i,x in enumerate(available_cp)
                                           if digest(x)==cid),None)
                        need(selected_idx is not None,"Checkpoint selected point not in pool")
                        available_cp.pop(selected_idx)
                    check_fit(prefix_observed,available_cp,tests,
                              cp["risk_fit_receipt"],cp["objective_fit_receipt"])
                    cp_fit_sizes=[anchor_size+p["step"] for p in proposals
                                  if anchor_size+p["step"]<=n and
                                  p["method_fit_objective"]["status"]=="OK"]
                    if cp["objective_fit_receipt"]["status"]=="OK":
                        cp_fit_sizes.append(n)
                    expected_first_fit=min(cp_fit_sizes) if cp_fit_sizes else None
                    feasible_sizes=[m for m in range(anchor_size,n+1)
                        if outcome_state([r["observed"] for r in history[:m]])["feasible_found"]]
                    expected_first_feasible=min(feasible_sizes) if feasible_sizes else None
                    need(cp["first_objective_gp_fit_n"]==expected_first_fit and
                         cp["first_feasible_n"]==expected_first_feasible,
                         "Checkpoint warm-start/feasible timing mismatch")
                    need(cp["test_candidate_ids"]==[r["candidate_id"] for r in tests] and
                         cp["anchor_prevalence"]==rates and
                         cp["anchor_baseline_brier"]==baseline,
                         "Checkpoint test/anchor baseline mismatch")
                    expected_metrics=_test_metrics(tests,cp["test_probabilities_raw"],baseline)
                    need(same_numbers(cp["risk_metrics"],expected_metrics),
                         "Heldout risk metrics cannot be recomputed from stored probabilities")
                trial_count=1
                trials_seen+=trial_count
                expected_progress[("trial_complete",task,repeat,anchor_size,method,None)]={
                    "initial_outcome":trial["initial_outcome"],
                    "final_outcome":trial["final_outcome"]}

    progress=[json.loads(line) for line in
        (root/"progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    actual_progress={}
    for event in progress:
        key=(event["event"],event["task"],event["repeat"],
             event["anchor_size"],event["method"],event.get("step"))
        need(key not in actual_progress,"Duplicate progress event")
        if event["event"]=="proposal":
            payload=event["proposal"]
        elif event["event"]=="observation":
            payload={k:event[k] for k in ("step","candidate_id","call_id","observed")}
        elif event["event"]=="trial_complete":
            payload={k:event[k] for k in ("initial_outcome","final_outcome")}
        else:
            raise ValueError("Unknown progress event")
        actual_progress[key]=payload
    need(actual_progress==expected_progress,
         "Incremental progress trace differs from sealed unit records")
    expected_trials=len(spec["units"])*len(METHODS)*len(ANCHOR_SIZES)
    need(trials_seen==expected_trials and report["trials"]==expected_trials,
         "Trial roster mismatch")
    bill=report["bill"]
    need(bill["math_started"]==len(started)==bill["math_settled"],
         "Math bill differs from ledger")
    need(bill["gp_head_started"]==bill["gp_head_completed"]==spec["expected_risk_heads"] and
         bill["gp_kernel_started"]==bill["gp_kernel_completed"]==
             spec["expected_risk_kernel_constructions"],
         "Risk GP head/kernel bill mismatch")
    need(bill["gp_head_errors"]==bill["gp_kernel_errors"]==
         bill["kernel_candidates_rejected"]==0,
         "Risk GP error/rejection")
    need(bill["objective_gp_head_started"]==bill["objective_gp_head_completed"]==
         objective_success_count and
         bill["objective_gp_kernel_started"]==bill["objective_gp_kernel_completed"] and
         bill["objective_gp_kernel_started"]==8*bill["objective_gp_head_started"],
         "Objective GP bill mismatch")
    need(bill["objective_gp_head_errors"]==bill["objective_gp_kernel_errors"]==
         bill["objective_gp_kernel_candidates_rejected"]==0,
         "Objective GP error/rejection")
    return dict(status="VERIFIED_PR86D_INITIAL_SAMPLE_ABLATION",
        files=file_count,trials=trials_seen,math_calls=len(started),
        nested_anchor_masking_verified=True,acquisition_decisions_replayed=True,
        objective_missing_not_imputed=True,test_labels_fit_isolated=True,
        new_math_calls=0,new_gp_fits=0)
