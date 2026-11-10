"""PR86-B sequential, paid risk-information acquisition study."""
from copy import deepcopy
from pathlib import Path
import platform, time, traceback

from risk_learning.core import TASKS, candidates, truth, observed, digest, features
from risk_learning.io import write, append, seal
from risk_learning.study import new_bill
from risk_learning.native import predict as native_predict
from pr86b_protocol import plan, METHODS
from pr86b_acquisition import choose, anchor_prevalence, calibrate_prediction, metrics, brier_skill_score

def _evaluate(task, candidate, call_id, role, unit_name, root, bill):
    cid = digest(candidate)
    append(root/"evaluations.jsonl", dict(event="started", call_id=call_id, role=role,
        task=task, unit=unit_name, candidate_id=cid, candidate=candidate))
    bill["math_started"] += 1
    target = truth(task, deepcopy(candidate))
    record = dict(candidate=candidate, candidate_id=cid, truth=target, call_id=call_id)
    if role in ("anchor", "online"):
        record["observed"] = observed(candidate, target)
    append(root/"evaluations.jsonl", dict(event="settled", call_id=call_id,
        record_sha256=digest(record)))
    bill["math_settled"] += 1
    return record

def _feature_coverage(records):
    branches = {str(b): sum(r["candidate"]["branch"] == b for r in records) for b in range(4)}
    vectors = [features(r["candidate"]) for r in records]
    nearest = []
    for i, x in enumerate(vectors):
        nearest.append(min(sum((a-b)**2 for a,b in zip(x,y))**0.5
                           for j,y in enumerate(vectors) if i != j))
    ordered = sorted(nearest)
    return dict(branch_counts=branches, branches_covered=sum(v>0 for v in branches.values()),
                nearest_neighbor_distance=dict(mean=sum(nearest)/len(nearest),
                    median=ordered[len(ordered)//2], maximum=max(nearest)))

def _metric_pack(test_rows, probabilities, baseline):
    result = {}
    for head in ("region", "resource", "joint"):
        if head == "joint":
            labels = [bool(r["truth"]["feasible"]) for r in test_rows]
        else:
            labels = [bool(r["truth"][head] <= 0) for r in test_rows]
        m = metrics(labels, probabilities[head])
        m["brier_skill_score"] = brier_skill_score(m["brier"], baseline[head])
        result[head] = m
    return result

def _baseline_briers(test_rows, anchor_rates):
    out = {}
    for head in ("region", "resource", "joint"):
        labels = ([bool(r["truth"]["feasible"]) for r in test_rows] if head == "joint"
                  else [bool(r["truth"][head] <= 0) for r in test_rows])
        q = anchor_rates[head]
        out[head] = sum((q-float(y))**2 for y in labels)/len(labels)
    return out

def run(output, profile, source):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("PR86-B is write-once; existing/failed output must be inspected, never overwritten")
    spec = plan(profile)
    output.mkdir(parents=True)
    bill = new_bill()
    trials = []
    started = time.perf_counter()
    write(output/"plan.json", spec)
    write(output/"source_receipt.json", source)
    write(output/"environment.json", dict(python=platform.python_version(), platform=platform.platform()))
    try:
        for unit in spec["units"]:
            task, repeat = unit["task"], unit["repeat"]
            unit_name = "%s/r%d" % (task, repeat)
            anchors = [_evaluate(task, c, unit_name+"/anchor/%03d" % i, "anchor",
                                 unit_name, output, bill)
                       for i,c in enumerate(candidates(unit["anchor_seed"], spec["anchor_count"]))]
            pool = candidates(unit["pool_seed"], spec["candidate_pool"])
            calibration_pool = candidates(unit["calibration_seed"], spec["calibration_anchor_count"])
            test_rows = [_evaluate(task, c, unit_name+"/test/%04d" % i, "test",
                                   unit_name, output, bill)
                         for i,c in enumerate(candidates(unit["test_seed"], spec["test_count"]))]
            ids = [r["candidate_id"] for r in anchors + test_rows]
            all_candidate_ids = ids + [digest(c) for c in pool] + [digest(c) for c in calibration_pool]
            if len(set(all_candidate_ids)) != len(all_candidate_ids):
                raise ValueError("Anchor/acquisition/calibration/test candidate identity overlap")
            rates = anchor_prevalence([r["observed"] for r in anchors])
            baseline_brier = _baseline_briers(test_rows, rates)
            unit_trials = []
            for method_index, method in enumerate(METHODS):
                history = list(anchors)
                available = list(pool)
                calibration_available = list(calibration_pool)
                proposals, observations, checkpoints = [], [], []
                last_fit_receipt = None
                arm_seed = unit["pool_seed"] + 1009*method_index
                online_limit = spec["online_per_method"]
                for step in range(online_limit + 1):
                    observed_rows = [deepcopy(r["observed"]) for r in history]
                    query_candidates = available + [r["candidate"] for r in test_rows]
                    fit_seed = unit["pool_seed"] + 1009 * len(history)
                    forecast, receipt = native_predict(observed_rows, query_candidates,
                                                       "native_censored", fit_seed, bill)
                    last_fit_receipt = receipt
                    if forecast is None or receipt.get("status") != "OK":
                        raise RuntimeError("Native GP fit failed; preserve failure evidence and stop: %r" % receipt)
                    nc = len(available)
                    candidate_forecast = dict(
                        probabilities={h: list(forecast["probabilities"][h][:nc]) for h in ("region","resource","joint")},
                        gaussian={h: dict(mean=list(forecast["gaussian"][h]["mean"][:nc]),
                                          variance=list(forecast["gaussian"][h]["variance"][:nc]),
                                          observation_noise_variance=forecast["gaussian"][h]["observation_noise_variance"])
                                  for h in ("region","resource")})
                    test_forecast = dict(probabilities={h: list(forecast["probabilities"][h][nc:]) for h in ("region","resource","joint")})
                    nobs = len(history)
                    if nobs in spec["checkpoints"]:
                        calibration_records = list(anchors)
                        calibration_anchor_ids = [r["candidate_id"] for r in anchors]
                        if method == "iid_random":
                            calibration_records += [o["record"] for o in observations]
                            calibration_anchor_ids += [o["candidate_id"] for o in observations]
                        elif method == "calibration_aware_voi":
                            for prior, outcome in zip(proposals, observations):
                                if prior["rule"] == "reserved_random_calibration_anchor":
                                    calibration_records.append(outcome["record"])
                                    calibration_anchor_ids.append(outcome["candidate_id"])
                        calibration_rates = anchor_prevalence([r["observed"] for r in calibration_records])
                        calibrated, shifts = calibrate_prediction(test_forecast, calibration_rates)
                        checkpoints.append(dict(n=nobs, method=method,
                            training_prefix_sha256=digest([r["observed"] for r in history]),
                            anchor_rates=rates, calibration_rates=calibration_rates,
                            calibration_anchor_ids=calibration_anchor_ids,
                            anchor_baseline_brier=baseline_brier,
                            raw_metrics=_metric_pack(test_rows, test_forecast["probabilities"], baseline_brier),
                            anchor_calibrated_metrics=_metric_pack(test_rows, calibrated, baseline_brier),
                            calibration_logit_shifts=shifts,
                            test_candidate_ids=[r["candidate_id"] for r in test_rows],
                            test_probabilities_raw=test_forecast["probabilities"],
                            test_probabilities_anchor_calibrated=calibrated,
                            fit_receipt=receipt))
                    if nobs >= spec["total_observations_per_method"]:
                        break
                    available_before_sha = digest(available)
                    calibration_available_before_sha = digest(calibration_available)
                    index, decision = choose(method, available,
                        [r["candidate"] for r in history], candidate_forecast,
                        arm_seed, step, calibration_candidates=calibration_available)
                    if decision["selected_from"] == "calibration_anchor_pool":
                        selected = calibration_available[index]
                        del calibration_available[index]
                        selected_pool = "calibration_anchor_pool"
                    else:
                        selected = available[index]
                        del available[index]
                        selected_pool = "acquisition_pool"
                    decision.update(fit_receipt=receipt, candidate_pool_sha256=digest(pool),
                                    calibration_pool_sha256=digest(calibration_pool),
                                    available_pool_sha256=available_before_sha,
                                    calibration_available_pool_sha256=calibration_available_before_sha,
                                    selected_pool=selected_pool,
                                    history_sha256=digest([r["observed"] for r in history]),
                                    forecast_gaussian=candidate_forecast["gaussian"],
                                    forecast_probabilities=candidate_forecast["probabilities"],
                                    selected_candidate=selected,
                                    pre_outcome=True, no_unselected_labels_accessed=True)
                    proposals.append(decision)
                    call_id = unit_name+"/online/"+method+"/%03d" % step
                    record = _evaluate(task, selected, call_id, "online", unit_name, output, bill)
                    observations.append(dict(step=step, candidate_id=record["candidate_id"],
                                              record=record))
                    history.append(record)
                trial = dict(task=task, repeat=repeat, method=method, seed=arm_seed,
                    anchor_ids=[r["candidate_id"] for r in anchors],
                    online_observations=observations, proposals=proposals,
                    checkpoints=checkpoints, final_n=len(history),
                    selected_branch_counts={str(b):sum(r["candidate"]["branch"]==b for r in history) for b in range(4)},
                    feature_coverage=_feature_coverage(history),
                    final_fit_receipt=last_fit_receipt,
                    gp_fit_seconds=sum(p["fit_receipt"]["elapsed_seconds"] for p in proposals) + last_fit_receipt["elapsed_seconds"],
                    online_calls=len(observations), model_refits=len(proposals)+1)
                unit_trials.append(trial)
                trials.append(trial)
                append(output/"progress.jsonl", dict(task=task, repeat=repeat, method=method,
                    online_calls=len(observations), final_n=len(history)))
            write(output/"units"/unit_name/ "unit.json", dict(task=task, repeat=repeat,
                anchors=anchors, test_rows=test_rows, pool=pool, calibration_pool=calibration_pool, anchor_rates=rates,
                anchor_baseline_brier=baseline_brier, trials=unit_trials))
        if bill["math_started"] != spec["total_math_calls"] or bill["math_settled"] != spec["total_math_calls"]:
            raise ValueError("PR86-B math evaluation bill mismatch")
        expected_fits = spec["expected_model_refits"]
        if bill["gp_head_completed"] != expected_fits*2 or bill["gp_head_errors"] or bill["gp_kernel_errors"] or bill["kernel_candidates_rejected"]:
            raise ValueError("PR86-B native GP bill failed registered zero-error gate")
        report = dict(schema=spec["schema"], code_sha=source["code_sha"], profile=profile,
            status="COMPLETE_PR86B_SEQUENTIAL_RISK_INFORMATION", bill=bill,
            units=len(spec["units"]), trials=len(trials), model_refits=expected_fits,
            checkpoints=list(spec["checkpoints"]), trials_summary=[
                {k:t[k] for k in ("task","repeat","method","online_calls","final_n","model_refits","selected_branch_counts")}
                for t in trials],
            interpretation="Risk learnability and acquisition only; not aircraft optimization or safety evidence",
            elapsed_seconds=time.perf_counter()-started)
        write(output/"report.json", report)
        seal(output)
        return report
    except BaseException as error:
        write(output/"failure.json", dict(code_sha=source.get("code_sha"),
            error_type=type(error).__name__, message=str(error),
            math_started=bill["math_started"], math_settled=bill["math_settled"],
            unknown_calls=bill["math_started"]-bill["math_settled"],
            completed_trials=len(trials), traceback=traceback.format_exc()))
        seal(output)
        raise
