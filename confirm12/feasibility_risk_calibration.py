"""PR51: honest risk-probability diagnostic on disjoint PR45 qA seed blocks.

No new aircraft/RCAIDE calls, changes to PR45 observations, or optimizer edits.
Old GP completion = evaluator-record status (all 1 in PR45), NOT mission success.
The exact PR50 data audit found mission success == SOC constraint satisfaction
throughout this historical campaign; this is an empirical task-specific fact,
not a generic causal identity. Avoid treating both as independent events.
"""
from collections import defaultdict
from contextlib import ExitStack
import csv
import json
import math
from pathlib import Path
import statistics
import traceback
from unittest.mock import patch

from .risk_event_semantics import label_record, sha_file, digest

SCHEMA = "ra_lsbo_pr51_fixed_qa_feasibility_risk_calibration_v1"
FROZEN_SOURCE = "386082beeacac8ae43d0dfb1ccda5c5ad3b356cf"
FROZEN_PLAN_SHA = "ac5f1424f3c8d52eff9ffd835cb5004224be965afee8ec15069e1fea38c43012"
FROZEN_SQLITE_SHA = "a30b0dadb5d619be9b6c0d70c5aa29fe115f86b85d62a449e4acf05325d6f1b7"
SIZES = (12, 24, 48, 96, 192)
SPLIT_TRAIN_BLOCK = 16
SPLIT_CALIBRATION_OFFSET = 384
SPLIT_TEST_OFFSET = 512
SEEDS_CALIBRATION = 8
SEEDS_TEST = 8
VIEWS = ("latent", "onehot")
PROXIES = (
    "mission_status_proxy",             # diagnostic of the old semantic mistake
    "mission_soc_proxy",                # same physical event as observed PR50
    "feasible_legacy_completion_product", # old status head times every constraint
    "feasible_constraint_product",      # do not double-count status
    "feasible_soc_stall_bending",        # PR50 exploratory reduced-head sensitivity
)
RISK_TARGET = {
    "mission_status_proxy": "mission",
    "mission_soc_proxy": "mission",
    "feasible_legacy_completion_product": "feasible",
    "feasible_constraint_product": "feasible",
    "feasible_soc_stall_bending": "feasible",
}
PREDICTION_FIELDS = (
    "case", "fold", "training_size", "view", "proxy", "target_event",
    "target_seed", "target_index", "candidate_sha256", "true_event",
    "raw_probability", "calibrated_probability", "train_prevalence_baseline",
    "calibration_logit_intercept",
)
EPS = 1e-7
REG = 0.02


def fixed_splits(plan, folds=2):
    """Outcome-blind nonoverlapping training/calibration/test qA seed blocks.

    Prior PR48 holdouts were at offsets 128-135, PR49 at 256 onward.
    Here calibration starts at 384, test at 512, and all folds use disjoint
    16-seed training blocks. The same test designs are used for every n/view.
    """
    if type(folds) is not int or not 1 <= folds <= 8:
        raise ValueError("folds must be 1..8")
    out = []
    for case in ("aircraft_ordinary", "aircraft_sparse"):
        seeds = sorted(plan["cases"][case]["seeds"])
        if len(seeds) != 1024 or len(set(seeds)) != 1024:
            raise ValueError("Requires all 1024 original task seed identities")
        for fold in range(folds):
            training = seeds[fold * SPLIT_TRAIN_BLOCK:(fold + 1) * SPLIT_TRAIN_BLOCK]
            calibration = seeds[
                SPLIT_CALIBRATION_OFFSET + fold * SEEDS_CALIBRATION:
                SPLIT_CALIBRATION_OFFSET + (fold + 1) * SEEDS_CALIBRATION]
            testing = seeds[
                SPLIT_TEST_OFFSET + fold * SEEDS_TEST:
                SPLIT_TEST_OFFSET + (fold + 1) * SEEDS_TEST]
            if (len(training) != 16 or len(calibration) != 8 or len(testing) != 8 or
                    len(set(training + calibration + testing)) != 32):
                raise ValueError("Training/calibration/test seeds not disjoint")
            out.append(dict(case=case, fold=fold, train=training,
                            calibration=calibration, test=testing))
    for case in ("aircraft_ordinary", "aircraft_sparse"):
        selections = [x for x in out if x["case"] == case]
        train = [seed for x in selections for seed in x["train"]]
        cal = [seed for x in selections for seed in x["calibration"]]
        tst = [seed for x in selections for seed in x["test"]]
        if len(set(train + cal + tst)) != len(train + cal + tst):
            raise ValueError("Cross-fold within-task seed reuse")
    return out


def _event(record):
    """Use PR50 physical tri-state evidence, never fabricate unknown negatives."""
    d = label_record(record.to_dict())
    if (d["mission_completed"] is None or d["physical_feasibility"] is None or
            d["status_completion"] != 1 or d["semantic_issues"]):
        raise ValueError("PR45 observed risk-event contract differs from audited PR50 semantics")
    soc_sign = d["constraint_heads"]["soc"][1]
    if soc_sign is not d["mission_completed"]:
        raise ValueError("PR50 SOC/mission empirical relationship not found; stop, do not relabel")
    if d["usable_feasible"] != d["mission_and_usable_feasible"]:
        raise ValueError("PR50 usable/mission overlap is not identical")
    return {"mission": int(d["mission_completed"]),
            "feasible": int(d["usable_feasible"])}


def _sigmoid(t):
    if t >= 0.0:
        e = math.exp(-t)
        return 1.0 / (1.0 + e)
    e = math.exp(t)
    return e / (1.0 + e)


def _logit(p):
    q = max(EPS, min(1.0 - EPS, float(p)))
    if not math.isfinite(q):
        raise ValueError("Nonfinite probability")
    return math.log(q / (1.0 - q))


def _calibrate(p, intercept):
    return _sigmoid(_logit(p) + intercept)


def fit_intercept(probs, truth, ridge=REG):
    """Single monotone log-odds shift, ridge-regularized; no temperature search.

    Fits ONLY a disjoint calibration seed block. Never consumes test labels.
    """
    if (len(probs) != len(truth) or not probs or
            any(type(y) is not int or y not in (0, 1) for y in truth) or
            any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in probs) or
            not 0 < ridge <= 1):
        raise ValueError("Invalid calibration data")
    logits = [_logit(p) for p in probs]
    b = 0.0
    for _ in range(60):
        ps = [_sigmoid(z + b) for z in logits]
        grad = (sum(p - y for p, y in zip(ps, truth)) / len(ps)) + ridge * b
        hess = (sum(p * (1.0 - p) for p in ps) / len(ps)) + ridge
        next_b = max(-8.0, min(8.0, b - grad / hess))
        if abs(next_b - b) < 1e-12:
            b = next_b
            break
        b = next_b
    return b


def laplace_prevalence(labels):
    if not labels or any(type(x) is not int or x not in (0, 1) for x in labels):
        raise ValueError("Binary training labels required")
    return (sum(labels) + 1.0) / (len(labels) + 2.0)


def probabilities(status, heads):
    """All versions share precisely the same GP predictions.

    The reduced three-head variant was chosen AFTER seeing PR50 audit labels,
    and is exploratory, not a pre-registered discovery or an optimizer.
    """
    if not heads or not {"soc", "stall", "root_bending"} <= set(heads):
        raise ValueError("Missing non-analytic aircraft constraint head")
    if (not math.isfinite(status) or not 0 <= status <= 1 or
            any(not math.isfinite(p) or not 0 <= p <= 1 for p in heads.values())):
        raise ValueError("Invalid risk GP marginal probabilities")
    complete = math.prod(heads.values())
    three = heads["soc"] * heads["stall"] * heads["root_bending"]
    return {
        "mission_status_proxy": status,
        "mission_soc_proxy": heads["soc"],
        "feasible_legacy_completion_product": status * complete,
        "feasible_constraint_product": complete,
        "feasible_soc_stall_bending": three,
    }


def _score(probs, truth):
    if not probs or len(probs) != len(truth):
        raise ValueError("Aligned probability and truth arrays required")
    brier = statistics.mean((p - y) ** 2 for p, y in zip(probs, truth))
    log_loss = statistics.mean(
        -y * math.log(max(EPS, min(1.0 - EPS, p))) -
        (1-y) * math.log(max(EPS, min(1.0 - EPS, 1.0-p)))
        for p, y in zip(probs, truth)
    )
    bins = []
    for j in range(5):
        rows = [(p, y) for p, y in zip(probs, truth)
                if (j / 5 <= p < (j + 1) / 5 if j < 4 else j / 5 <= p <= 1.0)]
        if rows:
            bins.append(dict(bin=j, count=len(rows),
                             mean_forecast=statistics.mean(p for p, _ in rows),
                             observed_rate=statistics.mean(y for _, y in rows)))
    ece = sum((v["count"] / len(probs)) *
              abs(v["mean_forecast"] - v["observed_rate"]) for v in bins)
    return {"brier":brier, "log_loss":log_loss, "ece_5_bin":ece,
            "false_positive_at_0_5":sum(p >= .5 and y == 0 for p, y in zip(probs, truth)),
            "known_negatives":sum(y == 0 for y in truth),
            "known_positives":sum(y == 1 for y in truth),
            "bins":bins}


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        key = (row["case"], row["training_size"], row["target_event"],
               row["proxy"], row["view"])
        buckets[key].append(row)
    out = {}
    for (case, n, target, proxy, view), values in sorted(buckets.items()):
        truth = [row["true_event"] for row in values]
        raw = [row["raw_probability"] for row in values]
        calibrated = [row["calibrated_probability"] for row in values]
        baseline = [row["train_prevalence_baseline"] for row in values]
        out[f"{case}/{n}/{target}/{proxy}/{view}"] = {
            "n_test": len(values),
            "test_seeds": len({(row["fold"], row["target_seed"]) for row in values}),
            "folds": len({row["fold"] for row in values}),
            "raw": _score(raw, truth),
            "calibrated": _score(calibrated, truth),
            "training_prevalence_baseline": _score(baseline, truth),
        }
    return out


def _validate_disjoint(training, calibration, testing):
    identities = [[digest(r.candidate) for r in block]
                  for block in (training, calibration, testing)]
    if set(identities[0]) & set(identities[1]) or \
       set(identities[0]) & set(identities[2]) or \
       set(identities[1]) & set(identities[2]):
        raise ValueError("Overlap across training, calibration or test design identities")
    # Within-block qA designs may repeat: charge each recorded evaluation,
    # but never let a repeated design cross a partition.
    return [digest(ids) for ids in identities]


def _gp_probabilities(forecasts, count, completion_probability, gaussian_probability):
    """Convert PR45 heads to risk proxies without inspecting target outcomes."""
    status = completion_probability(forecasts["completion"])
    marginals = {name: gaussian_probability(pred["mean"], pred["variance"])
                 for name, pred in forecasts["constraints"].items()}
    if any(len(values) != count for values in [status] + list(marginals.values())):
        raise ValueError("Incorrect number of candidate predictions")
    out = []
    for index in range(count):
        heads = {name: float(values[index]) for name, values in marginals.items()}
        out.append(probabilities(float(status[index]), heads))
    return out


def run(source, output, *, folds=2, sizes=SIZES):
    """Immutable original SQLite, freshly fitted real GP/VAE; zero truth calls."""
    # Late imports allow tiny pure-Python CI without Torch/RCAIDE.
    from .postrun_pr45 import (
        _sqlite, _read, verify_roster, FROZEN_REAL_SQLITE_SHA256,
        FROZEN_PLAN_SHA256, FROZEN_SOURCE_SHA,
    )
    from .matched_gp_mechanism import (
        _read_initial, _train_views, _completion_probability, _gaussian_probability,
    )
    from ra_lsbo.experiments.core.problems import AircraftCase
    from ra_lsbo.partial_bo.problems import SimplifiedElectricAircraft

    if (tuple(sizes) != tuple(sorted(set(sizes))) or not sizes or
            any(type(n) is not int or n not in SIZES for n in sizes)):
        raise ValueError("Use sorted nonempty subset of 12,24,48,96,192")
    root, dest = Path(source).resolve(), Path(output).resolve()
    if dest == root or root in dest.parents or dest.exists():
        raise ValueError("Refuse existing or original-result output directory")
    planpath, dbpath = root / "plan.json", root / "real_calls.sqlite"
    if (sha_file(planpath) != FROZEN_PLAN_SHA256 != FROZEN_PLAN_SHA or
            FROZEN_SOURCE_SHA != FROZEN_SOURCE or
            FROZEN_REAL_SQLITE_SHA256 != FROZEN_SQLITE_SHA):
        raise ValueError("Source constants drift")
    plan = _read(planpath)
    if plan["source_commit"] != FROZEN_SOURCE:
        raise ValueError("Wrong frozen PR45 campaign")
    verify_roster(plan)
    before_hash = sha_file(dbpath)
    before_mtime = dbpath.stat().st_mtime_ns
    if before_hash != FROZEN_SQLITE_SHA:
        raise ValueError("Wrong PR45 read-only SQLite")
    splits = fixed_splits(plan, folds)
    dest.mkdir(parents=True, exist_ok=False)
    rows, receipts, errors = [], [], []
    with ExitStack() as stack:
        def prohibited(*_args, **_kwargs):
            raise RuntimeError("PR51 PROHIBITS ALL AIRCRAFT EVALUATIONS")
        stack.enter_context(patch.object(AircraftCase, "evaluate", prohibited))
        stack.enter_context(patch.object(SimplifiedElectricAircraft, "evaluate", prohibited))
        with _sqlite(dbpath) as db:
            for spec in splits:
                case, fold = spec["case"], spec["fold"]
                contract = plan["cases"][case]["contracts"]["20s"]
                requested = spec["train"] + spec["calibration"] + spec["test"]
                data = {seed: _read_initial(db, case, seed, contract)
                        for seed in requested}
                calibration = [row for seed in spec["calibration"] for row in data[seed]]
                testing = [row for seed in spec["test"] for row in data[seed]]
                cal_truth = [_event(record) for record in calibration]
                test_truth = [_event(record) for record in testing]
                if len(calibration) != 96 or len(testing) != 96:
                    raise ValueError("Expected 96 calibration and 96 testing truth observations")
                for n in sizes:
                    training = [row for seed in spec["train"][:n // 12] for row in data[seed]]
                    ids = _validate_disjoint(training, calibration, testing)
                    training_truth = [_event(record) for record in training]
                    baselines = {event: laplace_prevalence([r[event] for r in training_truth])
                                 for event in ("mission", "feasible")}
                    try:
                        # No calibration/test information enters GP, VAE or kernels.
                        models, identity = _train_views(
                            __problem_for(case), spec["train"][0], training)
                        candidates = [r.candidate for r in calibration + testing]
                        scored_views = {
                            view: _gp_probabilities(
                                models[view].predict(candidates), len(candidates),
                                _completion_probability, _gaussian_probability)
                            for view in VIEWS
                        }
                    except Exception as exc:
                        errors.append(dict(case=case, fold=fold, n=n,
                                           error_type=type(exc).__name__,
                                           message=str(exc),
                                           traceback=traceback.format_exc()))
                        print("GP fit/predict FAILED:", case, fold, n, type(exc).__name__, flush=True)
                        continue
                    receipt = {
                        "case": case, "fold": fold, "n": n,
                        "train_seeds": spec["train"][:n // 12],
                        "calibration_seeds": spec["calibration"],
                        "test_seeds": spec["test"],
                        "design_partition_sha256": ids,
                        "training_labels": {event: {
                            "positive": sum(v[event] for v in training_truth),
                            "negative": len(training_truth) - sum(v[event] for v in training_truth)}
                            for event in ("mission", "feasible")},
                        "calibration_labels": {event: {
                            "positive": sum(v[event] for v in cal_truth),
                            "negative": len(cal_truth) - sum(v[event] for v in cal_truth)}
                            for event in ("mission", "feasible")},
                        "test_labels": {event: {
                            "positive": sum(v[event] for v in test_truth),
                            "negative": len(test_truth) - sum(v[event] for v in test_truth)}
                            for event in ("mission", "feasible")},
                        "original_status_completion_training_is_all_one": True,
                        "source_GP_identity": identity,
                        "shifts": {},
                        "same_test_designs_for_all_sizes_views": True,
                    }
                    for view in VIEWS:
                        scored = scored_views[view]
                        for proxy in PROXIES:
                            event = RISK_TARGET[proxy]
                            shift = fit_intercept(
                                [p[proxy] for p in scored[:len(calibration)]],
                                [r[event] for r in cal_truth])
                            receipt["shifts"][f"{view}/{proxy}"] = shift
                            for index, (prediction, truth) in enumerate(
                                    zip(scored[len(calibration):], test_truth)):
                                seed = spec["test"][index // 12]
                                rows.append({
                                    "case": case, "fold": fold, "training_size": n,
                                    "view": view, "proxy": proxy, "target_event": event,
                                    "target_seed": seed, "target_index": index % 12,
                                    "candidate_sha256": digest(testing[index].candidate),
                                    "true_event": truth[event],
                                    "raw_probability": prediction[proxy],
                                    "calibrated_probability": _calibrate(prediction[proxy], shift),
                                    "train_prevalence_baseline": baselines[event],
                                    "calibration_logit_intercept": shift,
                                })
                    receipts.append(receipt)
                    print("PR51:", case, "fold", fold, "train", n,
                          "test", len(testing), "rows", len(rows), flush=True)
    if sha_file(dbpath) != before_hash or dbpath.stat().st_mtime_ns != before_mtime:
        raise RuntimeError("Original PR45 SQLite changed")
    with (dest / "feasibility_risk_predictions.csv").open(
            "x", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    report = {
        "schema": SCHEMA,
        "status": "COMPLETE_RETROSPECTIVE_RISK_DIAGNOSTIC" if not errors else
                  "INCOMPLETE_GP_FITS_RETAINED",
        "source_SHA": FROZEN_SOURCE,
        "frozen_plan_sha256": FROZEN_PLAN_SHA,
        "original_sqlite_sha256_unchanged": before_hash,
        "new_aircraft_calls": 0, "new_rcaide_calls": 0,
        "original_source_read_only": True,
        "pr50_result_commit": "ec76c4102a19c751081dcfaa92c18b702aba26f3",
        "sizes": list(sizes), "folds_per_task": folds,
        "n_model_configs_complete": len(receipts),
        "n_model_configs_expected": len(splits) * len(sizes),
        "n_test_prediction_rows": len(rows),
        "split_identity": splits,
        "fit_receipts": receipts,
        "fit_errors": errors,
        "probability_scores": summarize(rows),
        "limitations": [
            "PR45 qA seeds are retrospective; not a prospective online optimizer trial.",
            "The evaluation-record status GP has no failed class in PR45.",
            "SOC sign equals mission completion in this archive, not a general universal equivalence.",
            "Marginal constraint products assume independence; they are not calibrated joint probabilities.",
            "The three-head selection is exploratory AFTER the PR50 whole-archive audit.",
            "Per-proxy intercept calibration fits disjoint qA seeds only; no safety/coverage guarantee.",
            "n>12 uses more paid-for PR45 truth evaluations than a single 12+18 online arm.",
            "The repeated test sets across sizes/views are paired; p-values and naive IID CIs are prohibited.",
            "This code never updates the original optimizer or claims improved online BO performance.",
            "The 140km task remains unqualified as a strictly sparse feasible domain.",
        ],
    }
    with (dest / "feasibility_risk_report.json").open("x", encoding="utf8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return report


def __problem_for(case):
    # Wrapper enables tests to validate pure helpers without creating aircraft.
    from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
    return problem_for(case)
