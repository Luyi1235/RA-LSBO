"""Independent PR86-C audit: verify paid calls, observable training, and replay decisions.

No surrogate fitting and no additional paid objective evaluations.
"""
import json
import math
from pathlib import Path

from risk_learning.core import candidates, observed, digest
from risk_learning.io import read, check_manifest
from risk_learning.native import training_payload
from pr86c_protocol import plan, METHODS
from pr86c_evaluator import truth_for_evaluation
from pr86c_models import outcome_state
from pr86c_acquisition import choose
from pr86b_acquisition import anchor_prevalence


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def close(a,b):
    if a is None or b is None:
        return a is None and b is None
    return isinstance(a,(float,int)) and isinstance(b,(float,int)) and (
        math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-11))


def same_numbers(a,b):
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(
            same_numbers(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return isinstance(b,list) and len(a)==len(b) and all(
            same_numbers(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and not isinstance(a,bool):
        return close(a,b)
    return a==b


def _check_row(row,task,expected_call,has_observation,started,settled):
    need(row["call_id"]==expected_call,"Wrong call ID")
    need(row["candidate_id"]==digest(row["candidate"]),"Candidate ID mismatch")
    need(same_numbers(row["truth"],truth_for_evaluation(task,row["candidate"])),
         "Mathematical result mismatch")
    if has_observation:
        need(row.get("observed")==observed(row["candidate"],row["truth"]),
             "Broken censoring/observation contract")
    else:
        need("observed" not in row,"Test-truth packet must not carry training observation")
    need(expected_call in started and expected_call in settled,
         "Missing paid call in ledger")
    need(started[expected_call]["candidate"]==row["candidate"],
         "Started candidate differs from settled result")
    need(settled[expected_call]["record"]==row and
         settled[expected_call]["record_sha256"]==digest(row),
         "Settled row/digest mismatch")


def _validate_fit_receipts(current,available,tests,decision):
    rr=decision["method_fit_risk"]
    need(rr["status"]=="OK" and not rr["objective_labels_used"] and
         not rr["privileged_training"],"Risk fit used privileged/objective labels")
    risk_inputs=training_payload(current,"native_censored")
    need(rr["training_input_sha256"]==digest(risk_inputs) and
         rr["training_rows"]==len(current),"Risk GP fit prefix mismatch")
    need(rr["query_ids"]==[digest(c) for c in available]+
         [r["candidate_id"] for r in tests],"Risk forecast query identity mismatch")
    need(set(rr["heads"])=={"region","resource"},"Risk heads missing")
    for h in rr["heads"].values():
        need(len(h["candidates"])==8 and all(
            c["status"]=="ok" for c in h["candidates"]), "Risk kernel candidate failed")

    orr=decision["method_fit_objective"]
    exact=[r for r in current if r["objective"]["kind"]=="exact"]
    need(orr["training_rows"]==len(current) and
         orr["exact_training_rows"]==len(exact) and
         not orr["privileged_training"] and not orr["missing_labels_imputed"],
         "Objective GP label or fit count mismatch")
    need(orr["training_input_sha256"]==digest([
         (r["candidate_id"],r["objective"]) for r in exact]) and
         orr["query_ids"]==[digest(c) for c in available],
         "Objective GP training/query identity mismatch")
    if len(exact)<2:
        need(orr["status"]=="INSUFFICIENT_EXACT_OBJECTIVES" and
             decision["objective_forecast"] is None,
             "Objective cold fallback invalid")
    else:
        need(orr["status"]=="OK" and orr["objective_fit_only_exact"] and
             len(orr["head"]["candidates"])==8,"Objective head missing or failed")
    cand=decision["candidate_forecast"]["probabilities"]
    need(all(len(cand[h])==len(available) for h in ("region","resource","joint")),
         "Candidate forecast length mismatch")
    need(all(close(a*b,c) for a,b,c in zip(
          cand["region"],cand["resource"],cand["joint"])),
         "Joint product assumption violated")
    need(decision["forecast_sha256"]==digest(decision["candidate_forecast"]),
         "Risk forecast SHA mismatch")


def verify(root,expected_sha=None):
    root=Path(root).resolve()
    files=check_manifest(root)
    need(not (root/"failure.json").exists(),"Failed run must not be certified")
    spec=read(root/"plan.json")
    source=read(root/"source_receipt.json")
    report=read(root/"report.json")
    need(spec==plan(spec["profile"]),"Protocol not frozen")
    need(report["status"]=="COMPLETE_PR86C_OBJECTIVE_AWARE","Incomplete PR86-C")
    need(report["code_sha"]==source["code_sha"] and
         (expected_sha is None or source["code_sha"]==expected_sha),"Source SHA differs")

    started,settled={},{}
    for line in (root/"evaluations.jsonl").read_text(encoding="utf-8").splitlines():
        event=json.loads(line)
        call=event["call_id"]
        if event["event"]=="started":
            need(call not in started,"Duplicated started call")
            started[call]=event
        elif event["event"]=="settled":
            need(call in started and call not in settled,"Unpaired settled call")
            need(event["record_sha256"]==digest(event["record"]),"Wrong settled digest")
            settled[call]=event
        else:
            raise ValueError("Unknown ledger event")
    need(set(started)==set(settled) and len(started)==spec["total_math_calls"],
         "Unsettled calls or incorrect paid count")

    trial_count=0
    expected_progress=0
    for unit in spec["units"]:
        task,repeat=unit["task"],unit["repeat"]
        name="%s/r%d"%(task,repeat)
        data=read(root/"units"/name/"unit.json")
        anchors,pool,tests=data["anchors"],data["pool"],data["tests"]
        need([r["candidate"] for r in anchors]==candidates(unit["anchor_seed"],
             spec["anchor_count"]),"Anchor seed mismatch")
        need(pool==candidates(unit["pool_seed"],spec["candidate_pool"]),
             "Candidate seed mismatch")
        need([r["candidate"] for r in tests]==candidates(unit["test_seed"],
             spec["test_count"]),"Test seed mismatch")
        ids=[r["candidate_id"] for r in anchors+tests]+[digest(c) for c in pool]
        need(len(ids)==len(set(ids)),"Candidate identity overlap")
        need(len(data["trials"])==len(METHODS),"Missing arm")
        for i,row in enumerate(anchors):
            _check_row(row,task,name+"/anchor/%03d"%i,True,started,settled)
        for i,row in enumerate(tests):
            _check_row(row,task,name+"/test/%04d"%i,False,started,settled)
        rates=anchor_prevalence([r["observed"] for r in anchors])
        need(rates==data["anchor_prevalence"],"Anchor prevalence mismatch")
        for m_idx,method in enumerate(METHODS):
            trial=data["trials"][m_idx]
            need(trial["method"]==method and trial["task"]==task and trial["repeat"]==repeat,
                 "Method/trial alignment mismatch")
            need(trial["anchor_ids"]==[r["candidate_id"] for r in anchors],
                 "Arms do not share identical anchors")
            online=trial["online_observations"]
            proposals=trial["proposals"]
            need(len(online)==len(proposals)==spec["online_per_method"] and
                 trial["online_calls"]==len(online),"Unequal online budget")
            need(trial["risk_refits"]==len(proposals)+1,"Risk fit count mismatch")
            available=list(pool)
            history=list(anchors)
            for step,(proposal,event) in enumerate(zip(proposals,online)):
                current=[r["observed"] for r in history]
                state=outcome_state(current)
                need(proposal["step"]==step and proposal["method"]==method and
                     proposal["history_observed_sha256"]==digest(current),
                     "Proposal did not use current observed history")
                need(proposal["available_sha256"]==digest(available) and
                     proposal["candidate_pool_sha256"]==digest(pool),
                     "Proposal pool changed")
                need(proposal["incumbent"]==state["best_exact_feasible"],
                     "Privileged or stale objective incumbent")
                need(proposal["pre_outcome"] and proposal["no_test_truth_accessed"] and
                     proposal["no_unselected_truth_accessed"],
                     "Proposal provenance flag invalid")
                _validate_fit_receipts(current,available,tests,proposal)
                arm_seed=unit["pool_seed"]+1009*m_idx
                index,replay=choose(method,available,
                    proposal["candidate_forecast"],proposal["objective_forecast"],
                    state["best_exact_feasible"],arm_seed,step)
                for k in ("method","rule","step","selected_index",
                          "selected_candidate_sha256","score_digest","scores",
                          "cold_fallback","incumbent"):
                    need(same_numbers(replay[k],proposal[k]),
                         "Acquisition replay mismatch: "+k)
                need(index==proposal["selected_index"],"Selected index differs")
                selected=available.pop(index)
                need(selected==proposal["selected_candidate"],
                     "Selected candidate differs")
                record=event["record"]
                call=name+"/online/"+method+"/%03d"%step
                need(event["step"]==step and event["candidate_id"]==digest(selected) and
                     record["candidate"]==selected,"Observation out of sequence")
                _check_row(record,task,call,True,started,settled)
                history.append(record)
                expected_progress+=2
            state=outcome_state([r["observed"] for r in history])
            need(trial["final_n"]==spec["total_observations_per_method"] and
                 trial["final_outcome"]==state,"Final true-objective endpoint mismatch")
            need([r["observed"] for r in history][:spec["anchor_count"]]==
                 [r["observed"] for r in anchors],"Anchor prefix changed")
            need([cp["n"] for cp in trial["checkpoints"]]==spec["checkpoints"],
                 "Checkpoint sequence mismatch")
            for cp in trial["checkpoints"]:
                current=[r["observed"] for r in history[:cp["n"]]]
                need(cp["observed_prefix_sha256"]==digest(current) and
                     cp["outcome"]==outcome_state(current),
                     "Checkpoint includes non-observable objective truth")
                need(cp["test_candidate_ids"]==[r["candidate_id"] for r in tests] and
                     cp["anchor_prevalence"]==rates,
                     "Test identity/prevalence mismatch")
                need(set(cp["risk_metrics"])=={"region","resource","joint"},
                     "Missing risk diagnostic")
            trial_count+=1
            expected_progress+=1
    progress=[json.loads(line) for line in
              (root/"progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    need(len(progress)==expected_progress,"Proposal/settlement progress event count mismatch")
    need(trial_count==len(spec["units"])*len(METHODS) and report["trials"]==trial_count,
         "Trial budget mismatches")
    need(report["bill"]["math_started"]==len(started) and
         report["bill"]["math_settled"]==len(started),"Bill differs from journal")
    need(report["bill"]["objective_gp_head_started"]==
         report["bill"]["objective_gp_head_completed"] and
         report["bill"]["objective_gp_kernel_started"]==
         report["bill"]["objective_gp_kernel_completed"],"Unsettled objective head")
    return dict(status="VERIFIED_PR86C_OBJECTIVE_AWARE",
                files=files,trials=trial_count,math_calls=len(started),
                acquisition_decisions_replayed=True,
                exact_objective_only_verified=True,
                test_labels_fit_isolated=True,
                new_math_calls=0,new_gp_fits=0)
