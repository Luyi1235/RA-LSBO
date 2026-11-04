"""PR53: phase and strictly prefix-only risk calibration on native PR45 decisions.

Reads only the byte-fixed PR52 frozen online forecast CSV and JSON.
Does NOT run aircraft physics, GP, VAE, or any optimization proposal.
A pooled/phase calibrator uses OTHER seeds retrospectively; a within-seed
prefix calibrator consumes ONLY earlier selected outcomes and predictions.
Neither demonstrates a counterfactual online-optimization benefit or safety.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .feasibility_risk_calibration import _calibrate, _logit, _score, fit_intercept
from .risk_event_semantics import sha_file

SCHEMA = "ra_lsbo_pr53_phase_and_prefix_selected_risk_v1"
SOURCE_SCHEMA = "ra_lsbo_pr52_original_online_prequential_risk_v1"
SOURCE_PR52_RESULT_COMMIT = "694b098cacba75bf5168eec3a1c55dbf84289051"
SOURCE_CSV_GIT_BLOB_SHA = "d1a32240bfdd40bfa52987c93438f5b35879920b"
SOURCE_REPORT_GIT_BLOB_SHA = "0a254889624ef43fb3236a19dedabadd8ecb108a"
SOURCE_PR45_SHA = "386082beeacac8ae43d0dfb1ccda5c5ad3b356cf"
SOURCE_PR45_SQLITE_SHA = "a30b0dadb5d619be9b6c0d70c5aa29fe115f86b85d62a449e4acf05325d6f1b7"
EXPECTED_ROWS = 9216
CASES = ("aircraft_ordinary", "aircraft_sparse")
ARMS = ("ra_lsbo", "full_latent")
SPLITS = ("calibration", "test")
PROXIES = {
    "feasible": ("predicted_constraint_product", "feasible_true"),
    "mission": ("predicted_soc", "mission_true"),
}
PRED_FIELDS = ("predicted_constraint_product", "predicted_soc",
               "predicted_record_completion", "predicted_legacy_completion_product")
ROW_FIELDS = (
    "case", "arm", "seed", "iteration", "mode", "phase", "target_event",
    "prediction_available", "prediction_skip_reason", "true_event",
    "raw_probability", "pooled_cross_seed_probability",
    "phase_cross_seed_probability", "prefix_only_probability",
    "prefix_fit_active", "prefix_prior_scored", "prefix_logit_intercept",
)
MIN_PREFIX = 6
PREFIX_RIDGE = .25
PHASE_RIDGE = .10
MIN_PHASE_CAL = 32
MIN_PHASE_SEEDS = 8
BUDGETS = (4., 6., 8.)
THRESHOLDS = (.5, .8, .9)


def git_blob_sha(path):
    """Verify exact historical Git BLOB bytes including CRLF, not just content."""
    path = Path(path)
    sha = hashlib.sha1()
    sha.update(("blob %d\0" % path.stat().st_size).encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def phase_of(mode):
    return "boundary" if mode == "budgeted_boundary_information" else "other"


def bit(s):
    if s not in ("0", "1"):
        raise ValueError("Outcome is not a trusted binary label")
    return int(s)


def prob(s):
    if s in ("", None):
        return None
    p = float(s)
    if not math.isfinite(p) or not 0. <= p <= 1.:
        raise ValueError("Nonfinite or out-of-range native risk probability")
    return p


def _read_source(root):
    """Validate every PR52 call and source byte before partitioning or fitting."""
    root = Path(root).resolve()
    csv_path = root / "prequential_risk_predictions.csv"
    report_path = root / "prequential_risk_report.json"
    if git_blob_sha(csv_path) != SOURCE_CSV_GIT_BLOB_SHA:
        raise ValueError("PR52 original CSV Git blob identity mismatch")
    if git_blob_sha(report_path) != SOURCE_REPORT_GIT_BLOB_SHA:
        raise ValueError("PR52 original report Git blob identity mismatch")
    report = json.loads(report_path.read_text(encoding="utf8"))
    if (report.get("schema") != SOURCE_SCHEMA or
        report.get("status") != "COMPLETE_NATIVE_PREQUENTIAL_RISK_AUDIT" or
        report.get("source_sha") != SOURCE_PR45_SHA or
        report.get("original_sqlite_sha256_unchanged") != SOURCE_PR45_SQLITE_SHA or
        report.get("per_block") != 64 or
        report.get("total_original_online_selected_calls_scanned") != EXPECTED_ROWS or
        report.get("new_aircraft_calls") != 0 or report.get("new_gp_fits") != 0 or
        report.get("split_seed_indices") != {
            "calibration": [640, 703], "test": [704, 767]}):
        raise ValueError("Unexpected PR52 original native-prequential report scope")
    rows = []
    registered = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf8") as stream:
        reader = csv.DictReader(stream)
        if not set(("case", "arm", "split", "seed", "iteration", "mode",
                    "prediction_available", "prediction_skip_reason",
                    "mission_true", "feasible_true", "soc_true",
                    "predicted_constraint_product", "predicted_soc",
                    "predicted_record_completion", "predicted_legacy_completion_product")
                   ) <= set(reader.fieldnames or []):
            raise ValueError("PR52 original CSV columns changed")
        for raw in reader:
            case, arm, split = raw["case"], raw["arm"], raw["split"]
            if case not in CASES or arm not in ARMS or split not in SPLITS:
                raise ValueError("Unregistered PR52 task/arm/split")
            seed = int(raw["seed"])
            iteration = int(raw["iteration"])
            if not 0 <= iteration < 18:
                raise ValueError("Non-online iteration")
            if raw["prediction_available"] not in ("True", "False"):
                raise ValueError("Prediction coverage boolean missing")
            available = raw["prediction_available"] == "True"
            y_mission, y_feasible, y_soc = (
                bit(raw["mission_true"]), bit(raw["feasible_true"]),
                bit(raw["soc_true"]))
            if y_feasible > y_mission or y_mission != y_soc:
                raise ValueError("PR50 audited mission/SOC/feasibility relationship drift")
            p = {name: prob(raw[name]) for name in PRED_FIELDS}
            if available:
                if any(value is None for value in p.values()):
                    raise ValueError("Model-selected call lacks recorded pre-outcome probabilities")
                if abs(p["predicted_legacy_completion_product"] -
                       p["predicted_record_completion"] *
                       p["predicted_constraint_product"]) > 5e-8:
                    raise ValueError("Original GP probability product identity differs")
            elif (any(x is not None for x in p.values()) or
                  not raw["prediction_skip_reason"]):
                raise ValueError("Model-free coverage call has invented risk probability")
            row = {
                "case": case, "arm": arm, "split": split,
                "seed": seed, "iteration": iteration,
                "mode": raw["mode"], "phase": phase_of(raw["mode"]),
                "prediction_available": available,
                "prediction_skip_reason": raw["prediction_skip_reason"],
                "feasible_true": y_feasible, "mission_true": y_mission,
                **p,
            }
            registered[(case,arm,split,seed)].append(iteration)
            rows.append(row)
    if len(rows) != EXPECTED_ROWS or len(registered) != 2*2*2*64:
        raise ValueError("Not all 9,216 original online proposals are present")
    for iterations in registered.values():
        if sorted(iterations) != list(range(18)):
            raise ValueError("Duplicate, reordered or absent original online decision")
    for case in CASES:
        expected_base = 500001 if case == "aircraft_ordinary" else 600001
        for arm in ARMS:
            for split, offset in (("calibration", 640), ("test", 704)):
                actual = {seed for c,a,s,seed in registered if (c,a,s)==(case,arm,split)}
                expected = set(range(expected_base+offset, expected_base+offset+64))
                if actual != expected:
                    raise ValueError("Seed selection deviates from frozen PR52 roster")
    return rows, report, {"source_csv_sha256": sha_file(csv_path),
                          "source_report_sha256": sha_file(report_path)}


def select_blocks(rows, per_block):
    """Fixed low-index prefixes only for smoke; full 64/64 is the study."""
    if type(per_block) is not int or not 1 <= per_block <= 64:
        raise ValueError("per_block must be 1..64")
    retained = []
    for row in rows:
        base = 500001 if row["case"] == "aircraft_ordinary" else 600001
        lo = 640 if row["split"] == "calibration" else 704
        if base + lo <= row["seed"] < base + lo + per_block:
            retained.append(row)
    if len(retained) != per_block * 2 * 2 * 2 * 18:
        raise ValueError("Selected calibration/test seed population incomplete")
    return retained


def _sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1./(1.+e)
    e = math.exp(z)
    return e/(1.+e)


def fit_centered(probs, truth, global_shift, ridge=PHASE_RIDGE):
    """Phase shift shrinks toward pooled-only calibration, not toward test data."""
    if len(probs) != len(truth) or not probs or not math.isfinite(global_shift):
        raise ValueError("Invalid phase-local calibration information")
    logits = [_logit(p) for p in probs]
    value = global_shift
    for _ in range(60):
        fitted = [_sigmoid(x + value) for x in logits]
        derivative = statistics.mean(a-y for a,y in zip(fitted,truth)) + ridge*(value-global_shift)
        curvature = statistics.mean(a*(1.-a) for a in fitted) + ridge
        revised = min(8., max(-8., value - derivative/curvature))
        if abs(revised-value) < 1e-12:
            value = revised
            break
        value = revised
    return value


def cross_seed_calibrators(rows, reference_shifts=None, full=True):
    """Only separate original calibration seeds; no test outcome in any fit."""
    cal = defaultdict(list)
    for row in rows:
        if row["split"] != "calibration" or not row["prediction_available"]:
            continue
        for event, (column, _) in PROXIES.items():
            cal[(row["case"],row["arm"],event)].append(
                (row["seed"],row["phase"],row[column],row[f"{event}_true"]))
    models, coverage = {}, {}
    for key, records in sorted(cal.items()):
        ps = [p for _,_,p,_ in records]
        ys = [y for *_,y in records]
        pooled_shift = fit_intercept(ps, ys)
        case,arm,event = key
        # Reconstruct PR52's intercept for exact same calibration data.
        if full and reference_shifts is not None:
            original_proxy = "constraint_product" if event == "feasible" else "soc_mission"
            registered = reference_shifts[f"{case}/{arm}/{original_proxy}"]
            if abs(registered-pooled_shift) > 1e-9:
                raise ValueError("Cross-seed calibration no longer matches frozen PR52")
        phase_models = {}
        for phase in ("boundary", "other"):
            sub = [(seed,p,y) for seed,v,p,y in records if v == phase]
            independent = len({seed for seed,_,_ in sub})
            supported = (len(sub) >= MIN_PHASE_CAL and
                         independent >= MIN_PHASE_SEEDS and
                         len({y for _,_,y in sub}) == 2)
            b = fit_centered([p for _,p,_ in sub],
                             [y for _,_,y in sub],pooled_shift) if supported else pooled_shift
            phase_models[phase] = b
            coverage["/".join((*key,phase))] = {
                "n": len(sub), "seed_groups": independent,
                "positives": sum(y for _,_,y in sub),
                "negatives": len(sub)-sum(y for _,_,y in sub),
                "phase_fit_supported": supported,
                "phase_shift": b, "pooled_shift": pooled_shift,
            }
        models[key] = {"pooled_shift": pooled_shift, "phase_shifts": phase_models}
    if len(models) != len(CASES)*len(ARMS)*len(PROXIES):
        raise ValueError("A cross-seed risk calibrator head is missing")
    return models, coverage


def prefix_predictions(rows, event):
    """Each next selected point uses ONLY its own seed's prior online outcomes.

    No initial-sample risk forecasts exist to fit a calibration intercept.
    Do not pull the cross-seed calibrator into this online-available estimator.
    """
    column,_ = PROXIES[event]
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[(row["case"],row["arm"],row["split"],row["seed"])].append(row)
    result = {}
    for key, sequence in by_seed.items():
        history = []
        for row in sorted(sequence,key=lambda r:r["iteration"]):
            if not row["prediction_available"]:
                continue
            x = row[column]
            active = (len(history) >= MIN_PREFIX and
                      len({y for _,y in history}) == 2)
            intercept = fit_intercept(
                [p for p,_ in history],
                [y for _,y in history],ridge=PREFIX_RIDGE) if active else 0.
            value = _calibrate(x,intercept)
            point_key = (*key,row["iteration"])
            result[point_key] = {
                "probability": value, "prefix_fit_active": active,
                "prefix_prior_scored": len(history),
                "prefix_logit_intercept": intercept,
            }
            # Append outcome ONLY AFTER constructing the prediction for this step.
            history.append((x,row[f"{event}_true"]))
    return result


def threshold_metrics(points, key, threshold):
    scored = [r for r in points if r[key] is not None]
    accepted = [r for r in scored if r[key] >= threshold]
    return {
        "threshold": threshold, "accepted": len(accepted),
        "accepted_fraction_of_all_original_calls": len(accepted)/len(points),
        "false_accepted": sum(r["true_event"] == 0 for r in accepted),
        "observed_success_rate_when_accepted": (
            statistics.mean(r["true_event"] for r in accepted) if accepted else None),
        "original_calls_not_accepted_or_unscored": len(points)-len(accepted),
    }


def cluster_delta(points, better, reference):
    per_seed = defaultdict(list)
    for r in points:
        if r[better] is None or r[reference] is None:
            continue
        y = r["true_event"]
        per_seed[r["seed"]].append((r[better]-y)**2-(r[reference]-y)**2)
    means = [statistics.mean(values) for values in per_seed.values()]
    if not means:
        return {"clusters": 0, "mean_brier_delta": None, "cluster_normal_95_ci": None}
    mean = statistics.mean(means)
    se = statistics.stdev(means)/math.sqrt(len(means)) if len(means)>1 else None
    return {"clusters": len(means), "mean_brier_delta": mean,
            "cluster_normal_95_ci": [mean-1.96*se,mean+1.96*se] if se is not None else None,
            "descriptive_not_confirmatory": True}


def scoring(rows):
    grouped = defaultdict(list)
    for row in rows:
        for phase in ("all",row["phase"]):
            grouped[(row["case"],row["arm"],row["target_event"],phase)].append(row)
    result = {}
    for key, points in sorted(grouped.items()):
        modeled = [r for r in points if r["prediction_available"]]
        record = {
            "total_original_selected_calls": len(points),
            "model_probabilities_available": len(modeled),
            "model_free_or_no_probability": len(points)-len(modeled),
            "seed_groups": len({r["seed"] for r in points}),
            "true_successes_all_original_calls": sum(r["true_event"] for r in points),
            "strategies": {},
            "paired_delta": {},
        }
        if modeled:
            labels = [r["true_event"] for r in modeled]
            for method,col in (
                ("raw","raw_probability"),
                ("pooled_other_seeds","pooled_cross_seed_probability"),
                ("phase_other_seeds","phase_cross_seed_probability"),
                ("prefix_only","prefix_only_probability"),
            ):
                predictions = [r[col] for r in modeled]
                if any(v is None for v in predictions):
                    raise ValueError("Modeled selected call lacks a required strategy prediction")
                record["strategies"][method] = {
                    **_score(predictions,labels),
                    "thresholds": {str(t): threshold_metrics(points,col,t) for t in THRESHOLDS},
                    "expected_failures_on_scored_original_calls": sum(1-p for p in predictions),
                    "realized_failures_on_scored_original_calls": len(labels)-sum(labels),
                }
            for method,col in (
                ("pooled_minus_raw","pooled_cross_seed_probability"),
                ("phase_minus_pooled","phase_cross_seed_probability"),
                ("prefix_minus_raw","prefix_only_probability"),
            ):
                ref = "pooled_cross_seed_probability" if method=="phase_minus_pooled" else "raw_probability"
                record["paired_delta"][method] = cluster_delta(modeled,col,ref)
            record["prefix_fit_supported_calls"] = sum(r["prefix_fit_active"] for r in modeled)
        result["/".join(key)] = record
    return result


def budget_diagnostic(rows):
    """Original selected trajectories ONLY; no new policy decisions or guarantees."""
    sequences = defaultdict(list)
    for row in rows:
        if row["target_event"] != "feasible":
            continue
        sequences[(row["case"],row["arm"],row["seed"])].append(row)
    by_arm = defaultdict(list)
    for (case,arm,seed),seq in sorted(sequences.items()):
        ordered = sorted(seq,key=lambda r:r["iteration"])
        if [r["iteration"] for r in ordered] != list(range(18)):
            raise ValueError("Budget diagnostic needs all 18 original calls")
        raw = sum(1-r["raw_probability"] if r["raw_probability"] is not None else 1.
                  for r in ordered)
        within = sum(1-r["prefix_only_probability"]
                     if r["prefix_only_probability"] is not None else 1.
                     for r in ordered)
        actual = sum(1-r["true_event"] for r in ordered)
        by_arm[(case,arm)].append({"seed":seed,"predicted_raw_debit":raw,
                                   "predicted_prefix_debit":within,
                                   "realized_nonfeasible":actual,
                                   "model_free_calls_worst_case_debit":sum(
                                       r["raw_probability"] is None for r in ordered)})
    output = {}
    for (case,arm),seeds in sorted(by_arm.items()):
        output[f"{case}/{arm}"] = {
            "seeds":len(seeds),
            "n_original_calls":18*len(seeds),
            "mean_raw_predicted_failure_debit":statistics.mean(s["predicted_raw_debit"] for s in seeds),
            "mean_prefix_predicted_failure_debit":statistics.mean(s["predicted_prefix_debit"] for s in seeds),
            "mean_realized_nonfeasible":statistics.mean(s["realized_nonfeasible"] for s in seeds),
            "mode_free_worst_case_debits":sum(s["model_free_calls_worst_case_debit"] for s in seeds),
            "budget_thresholds": {
                str(B):{
                    "actual_exceed":sum(s["realized_nonfeasible"]>B for s in seeds),
                    "raw_predicted_exceed":sum(s["predicted_raw_debit"]>B for s in seeds),
                    "prefix_predicted_exceed":sum(s["predicted_prefix_debit"]>B for s in seeds),
                    "raw_within_but_actual_exceeds":sum(
                        s["predicted_raw_debit"]<=B<s["realized_nonfeasible"] for s in seeds),
                    "prefix_within_but_actual_exceeds":sum(
                        s["predicted_prefix_debit"]<=B<s["realized_nonfeasible"] for s in seeds),
                } for B in BUDGETS},
            "not_a_counterfactual_stop_policy":True,
            "no_risk_or_failure_budget_guarantee":True,
        }
    return output


def run(source_dir, output_dir, *, per_block=64):
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source==output or source in output.parents or output.exists():
        raise ValueError("Use new output directory outside the original PR52 result directory")
    rows, report, identity = _read_source(source)
    selected = select_blocks(rows,per_block)
    models,coverage = cross_seed_calibrators(
        selected,report["calibration_logit_shifts"],full=(per_block==64))
    prefix = {event:prefix_predictions(selected,event) for event in PROXIES}
    test = []
    for row in selected:
        if row["split"] != "test":
            continue
        for event,(column,_) in PROXIES.items():
            k = (row["case"],row["arm"],row["split"],row["seed"],row["iteration"])
            fit = models[(row["case"],row["arm"],event)]
            raw = row[column]
            pre = prefix[event].get(k)
            if (raw is None) != (pre is None):
                raise ValueError("Prefix prediction coverage does not match original")
            test.append({
                "case":row["case"],"arm":row["arm"],"seed":row["seed"],
                "iteration":row["iteration"],"mode":row["mode"],"phase":row["phase"],
                "target_event":event,"prediction_available":row["prediction_available"],
                "prediction_skip_reason":row["prediction_skip_reason"],
                "true_event":row[f"{event}_true"],
                "raw_probability":raw,
                "pooled_cross_seed_probability":(
                    _calibrate(raw,fit["pooled_shift"]) if raw is not None else None),
                "phase_cross_seed_probability":(
                    _calibrate(raw,fit["phase_shifts"][row["phase"]])
                    if raw is not None else None),
                "prefix_only_probability":pre["probability"] if pre else None,
                "prefix_fit_active":pre["prefix_fit_active"] if pre else False,
                "prefix_prior_scored":pre["prefix_prior_scored"] if pre else None,
                "prefix_logit_intercept":pre["prefix_logit_intercept"] if pre else None,
            })
    if len(test) != per_block*2*2*18*2:
        raise ValueError("Incomplete heldout row count")
    summary = scoring(test)
    budgets = budget_diagnostic(test)
    # Provenance byte checks after all computation, prior to writing any output.
    if git_blob_sha(source/"prequential_risk_predictions.csv") != SOURCE_CSV_GIT_BLOB_SHA or \
       git_blob_sha(source/"prequential_risk_report.json") != SOURCE_REPORT_GIT_BLOB_SHA:
        raise RuntimeError("Original PR52 data changed during analysis")
    output.mkdir(parents=True,exist_ok=False)
    with (output/"phase_prefix_predictions.csv").open("x",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows([{k:r[k] for k in ROW_FIELDS} for r in test])
    result = {
        "schema": SCHEMA,
        "status": ("COMPLETE_HELDOUT_PHASE_PREFIX_RISK_DIAGNOSTIC"
                   if per_block==64 else "PARTIAL_SMOKE_ONLY"),
        "source_pr52_result_commit": SOURCE_PR52_RESULT_COMMIT,
        "source_pr52_code_sha": "9a3879d45f05718d7356545d54d8835628d1bf99",
        "source_pr45_sha": SOURCE_PR45_SHA,
        "original_pr45_sqlite_sha256": SOURCE_PR45_SQLITE_SHA,
        "source_git_blob_sha1": {
            "prequential_risk_predictions.csv": SOURCE_CSV_GIT_BLOB_SHA,
            "prequential_risk_report.json": SOURCE_REPORT_GIT_BLOB_SHA},
        **identity,
        "source_read_only":True,"new_aircraft_calls":0,"new_rcaide_calls":0,
        "new_gp_or_vae_fits":0,"new_optimizer_proposals":0,
        "per_seed_block":per_block,"n_test_event_rows":len(test),
        "n_test_original_selected_calls":len(test)//len(PROXIES),
        "cross_seed_calibration":coverage,
        "scoring_on_heldout_selected_calls":summary,
        "budget_diagnostics_on_original_18_call_paths":budgets,
        "frozen_protocol":{
            "phase_groups":["boundary","other"],
            "phase_ridge":PHASE_RIDGE,"phase_min_calibration_labels":MIN_PHASE_CAL,
            "phase_min_seed_groups":MIN_PHASE_SEEDS,
            "prefix_min_prior_scored":MIN_PREFIX,"prefix_ridge":PREFIX_RIDGE,
            "thresholds":list(THRESHOLDS),"budget_sensitivity":list(BUDGETS),
        },
        "limitations":[
            "All labels and decisions are retrospective PR45 native selected designs, not new policy outcomes.",
            "Pooled and phase calibrators use DIFFERENT calibration seeds and were NEVER available to the original PR45 optimizer.",
            "The prefix-only calibrator is mathematically based on earlier within-run outcomes, but is evaluated only on PR45's fixed original selected path.",
            "The phase separation was proposed after inspecting PR52 findings; it is exploratory and not independent scientific confirmation.",
            "Any new risk-based selection would change future data, and requires separate prospective evaluation.",
            "Model-free proposals remain as original calls with no fake probabilities and worst-case diagnostic debit.",
            "Threshold coverage and expected-debit budgets are observational; no safe BO or failure-count guarantee.",
            "100/140km aircraft models are reduced screening cases; independent 140km sparse qualification remains unresolved.",
        ],
    }
    with (output/"phase_prefix_report.json").open("x",encoding="utf8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
    return result
