"""PR52: audit PR45's NATIVE online selected-candidate risk forecasts.

Every forecast was stored BEFORE its actual outcome. No GP/VAE retraining, no
new aircraft calls, and no counterfactual outcome imputation. Model-free
coverage/fallback proposals stay in the denominator but are NEVER invented
as predictions. Independent seed blocks separate retrospective calibration
from final heldout testing. This is not a new online risk-control algorithm.
"""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from .postrun_pr45 import (
    FROZEN_SOURCE_SHA, FROZEN_PLAN_SHA256, FROZEN_REAL_SQLITE_SHA256,
    _sqlite, _read, verify_roster,
)
from .feasibility_risk_calibration import fit_intercept, _calibrate, _score
from .risk_event_semantics import label_record, sha_file, digest

SCHEMA = "ra_lsbo_pr52_original_online_prequential_risk_v1"
CASES = ("aircraft_ordinary", "aircraft_sparse")
ARMS = ("ra_lsbo", "full_latent")
HORIZON = 18
CAL_START = 640
TEST_START = 704
MAX_SEEDS_PER_BLOCK = 64
PROXIES = ("constraint_product", "legacy_completion_product", "soc_mission", "record_completion_as_mission")
TARGETS = {"constraint_product": "feasible", "legacy_completion_product": "feasible",
           "soc_mission": "mission", "record_completion_as_mission": "mission"}
EXTRA_HEADS = ("soc", "stall", "root_bending")
FIELDS = (
    "case", "arm", "split", "seed", "iteration", "mode", "cold_start",
    "proposal_sha256", "prediction_available", "prediction_skip_reason",
    "mission_true", "feasible_true", "soc_true", "stall_true", "root_bending_true",
    "historical_feasible_prevalence", "historical_mission_prevalence",
    "predicted_record_completion", "predicted_constraint_product",
    "predicted_legacy_completion_product", "predicted_soc",
    "predicted_stall", "predicted_root_bending",
    "online_calibrated_constraint_product", "online_calibrated_legacy_completion_product",
    "online_calibrated_soc_mission", "online_calibrated_record_completion_as_mission",
)


def fixed_splits(plan, per_block=MAX_SEEDS_PER_BLOCK):
    """128 seed groups/task maximum; completely disjoint from PR48/49/51 panels."""
    if type(per_block) is not int or not 1 <= per_block <= MAX_SEEDS_PER_BLOCK:
        raise ValueError("per_block must be in 1..64")
    verify_roster(plan)
    selections = []
    for case in CASES:
        seeds = sorted(plan["cases"][case]["seeds"])
        cal = seeds[CAL_START:CAL_START + per_block]
        test = seeds[TEST_START:TEST_START + per_block]
        if (len(cal) != per_block or len(test) != per_block or
                len(set(cal + test)) != 2 * per_block):
            raise ValueError("Unregistered overlapping calibration/test seed blocks")
        for seed in cal:
            selections.append((case, "calibration", seed))
        for seed in test:
            selections.append((case, "test", seed))
    return selections


def event(record):
    d = label_record(record)
    if (d["mission_completed"] is None or d["physical_feasibility"] is None or
            d["status_completion"] != 1 or d["semantic_issues"] or
            d["mission_and_usable_feasible"] != d["usable_feasible"]):
        raise ValueError("Non-PR50 physical risk-event semantics")
    return {
        "mission": int(d["mission_completed"]),
        "feasible": int(d["usable_feasible"]),
        **{name: None if d["constraint_heads"][name][1] is None
           else int(d["constraint_heads"][name][1]) for name in EXTRA_HEADS},
    }


def _verified_record(row, contract):
    cid, idx, candidate_hash, saved_contract, state, blob, record_hash = row
    if state != "settled" or blob is None or saved_contract != contract:
        raise ValueError("Missing or foreign settled original call")
    original = json.loads(blob)
    if (record_hash != digest(original) or
            candidate_hash != digest(original["candidate"]) or
            original["observation"]["design_id"] != candidate_hash or
            original["observation"]["contract_id"] != contract or
            original["diagnostics"].get("confirm12_real_call", {}).get("call_id") != cid):
        raise ValueError("Original ticket/design/record identity changed")
    return idx, candidate_hash, original


def load_records(db, case, seed, role, arm, contract, count):
    sql = ("SELECT id,idx,candidate_hash,contract,state,record,record_hash "
           "FROM calls WHERE case_name=? AND seed=? AND role=? AND arm=? ORDER BY idx")
    found = [_verified_record(row, contract) for row in db.execute(sql, (case, seed, role, arm))]
    if [i for i, _, _ in found] != list(range(count)):
        raise ValueError("Incomplete original role/seed/arm call sequence")
    return found


def _prob(value):
    f = float(value)
    if not math.isfinite(f) or f < -1e-12 or f > 1.0 + 1e-12:
        raise ValueError("Nonfinite/out-of-range stored posterior probability")
    return min(1.0, max(0.0, f))


def selected_probabilities(metadata, candidate_sha):
    """Use pre-outcome acquisition probabilities at the ACTUAL selected index.

    The original recorded joint is checked against the independent product
    that the original acquisition uses. Never use future outcomes or an
    alternative optimization trajectory. Coverage with skipped fit is null.
    """
    if metadata.get("probability_guarantee") is not False:
        raise ValueError("Original proposal missing no-guarantee disclosure")
    i = metadata.get("selected_index")
    pool = metadata.get("pool")
    if type(i) is not int or not isinstance(pool, list) or not 0 <= i < len(pool):
        raise ValueError("Invalid stored selected-index candidate pool")
    if digest(pool[i]) != candidate_sha:
        raise ValueError("Stored proposed design differs from selected-index pool")
    if metadata.get("posterior_fit_skipped") or metadata.get("posterior_fit_failed"):
        return None, ("model_failure" if metadata.get("posterior_fit_failed")
                      else "posterior_fit_skipped")
    completion = metadata.get("completion_probability")
    heads = metadata.get("per_constraint_probability")
    joint = metadata.get("joint_probability_independence_proxy")
    if (not isinstance(completion, list) or not isinstance(heads, dict) or
            not isinstance(joint, list) or not set(EXTRA_HEADS) <= set(heads)):
        raise ValueError("Required original GP probabilities missing")
    status = _prob(completion[i])
    factors = {}
    for name, array in heads.items():
        if not isinstance(array, list) or len(array) != len(pool):
            raise ValueError("Nonaligned original constraint posterior")
        factors[name] = _prob(array[i])
    p_all = math.prod(factors.values())
    p_legacy = status * p_all
    if len(joint) != len(pool) or abs(_prob(joint[i]) - p_legacy) > 5e-8:
        raise ValueError("Original acquisition joint probability differs from stored factors")
    return {
        "constraint_product": p_all,
        "legacy_completion_product": p_legacy,
        "soc_mission": factors["soc"],
        "record_completion_as_mission": status,
        "predicted_record_completion": status,
        "predicted_soc": factors["soc"],
        "predicted_stall": factors["stall"],
        "predicted_root_bending": factors["root_bending"],
    }, None


def _read_proposals(path):
    with Path(path).open("r", encoding="utf8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Blank proposal row")
            yield json.loads(line)


def _prior(history, kind):
    events = [event(r)[kind] for r in history]
    return (sum(events) + 1.0) / (len(events) + 2.0)


def read_arm(db, root, case, split, seed, arm, contract):
    """Verify every initial/online ticket and prefix before any scoring."""
    initial = load_records(db, case, seed, "initial", "", contract, 12)
    online = load_records(db, case, seed, "online", arm, contract, HORIZON)
    history = [record for _, _, record in initial]
    proposals = _read_proposals(
        root / "trials" / case / str(seed) / arm / "proposals.jsonl")
    collected = []
    for iteration in range(HORIZON):
        item = next(proposals, None)
        if item is None or item.get("iteration") != iteration:
            raise ValueError("Missing or reordered original prequential proposal")
        _, candidate_hash, actual = online[iteration]
        chosen = item.get("candidate")
        if (candidate_hash != digest(chosen) or
                item.get("candidate_sha256") != candidate_hash or
                actual["candidate"] != chosen):
            raise ValueError("Original next design/result/proposal identity mismatch")
        prefix = digest(history)
        if item.get("prefix_digest") != prefix:
            raise ValueError("Stored proposal contains a different training prefix")
        meta = item.get("metadata")
        if not isinstance(meta, dict) or meta.get("raw_history_digest") != prefix:
            raise ValueError("Stored risk model does not bind to the actual past-only history")
        predicted, why = selected_probabilities(meta, candidate_hash)
        outcome = event(actual)
        row = {
            "case": case, "arm": arm, "split": split, "seed": seed,
            "iteration": iteration, "mode": meta.get("mode", "unknown"),
            "cold_start": bool(meta.get("cold_start")),
            "proposal_sha256": candidate_hash, "prediction_available": predicted is not None,
            "prediction_skip_reason": why,
            "mission_true": outcome["mission"], "feasible_true": outcome["feasible"],
            "soc_true": outcome["soc"], "stall_true": outcome["stall"],
            "root_bending_true": outcome["root_bending"],
            "historical_feasible_prevalence": _prior(history, "feasible"),
            "historical_mission_prevalence": _prior(history, "mission"),
            "predicted_record_completion": None,
            "predicted_constraint_product": None,
            "predicted_legacy_completion_product": None,
            "predicted_soc": None, "predicted_stall": None, "predicted_root_bending": None,
            "online_calibrated_constraint_product": None,
            "online_calibrated_legacy_completion_product": None,
            "online_calibrated_soc_mission": None,
            "online_calibrated_record_completion_as_mission": None,
        }
        if predicted is not None:
            row.update(
                predicted_record_completion=predicted["predicted_record_completion"],
                predicted_constraint_product=predicted["constraint_product"],
                predicted_legacy_completion_product=predicted["legacy_completion_product"],
                predicted_soc=predicted["soc_mission"],
                predicted_stall=predicted["predicted_stall"],
                predicted_root_bending=predicted["predicted_root_bending"],
            )
        collected.append(row)
        history.append(actual)
    if next(proposals, None) is not None:
        raise ValueError("More than 18 original proposal decisions for this arm")
    return collected


def _proxy(row, proxy):
    return {
        "constraint_product": row["predicted_constraint_product"],
        "legacy_completion_product": row["predicted_legacy_completion_product"],
        "soc_mission": row["predicted_soc"],
        "record_completion_as_mission": row["predicted_record_completion"],
    }[proxy]


def calibrate(calibration, test):
    """Fit a SINGLE intercept for each case/arm/proxy on disjoint seeds.

    A separate model is not tuned on test labels. This is an OFFLINE diagnostic
    and cannot be silently replayed as a policy available to the PR45 runner.
    """
    by_cal = defaultdict(list)
    by_test = defaultdict(list)
    for row in calibration:
        by_cal[row["case"], row["arm"]].append(row)
    for row in test:
        by_test[row["case"], row["arm"]].append(row)
    shifts, coverage = {}, {}
    for (case, arm), train in sorted(by_cal.items()):
        checks = by_test[case, arm]
        if not checks or {r["seed"] for r in train} & {r["seed"] for r in checks}:
            raise ValueError("Calibration/test seed disjointness failure")
        for proxy, target in TARGETS.items():
            applicable = [r for r in train if r["prediction_available"]]
            if not applicable:
                shifts[f"{case}/{arm}/{proxy}"] = None
                coverage[f"{case}/{arm}/{proxy}"] = {"calibration_predictions": 0}
                continue
            shift = fit_intercept(
                [_proxy(r, proxy) for r in applicable],
                [r[f"{target}_true"] for r in applicable])
            shifts[f"{case}/{arm}/{proxy}"] = shift
            coverage[f"{case}/{arm}/{proxy}"] = {
                "calibration_predictions": len(applicable),
                "calibration_seed_groups": len({r["seed"] for r in applicable}),
                "positive": sum(r[f"{target}_true"] for r in applicable),
                "negative": sum(1 - r[f"{target}_true"] for r in applicable),
            }
            for r in checks:
                if r["prediction_available"]:
                    r["online_calibrated_" + proxy] = _calibrate(_proxy(r, proxy), shift)
    return shifts, coverage


def _rate_threshold(rows, column, target, threshold):
    accepted = [r for r in rows if r[column] is not None and r[column] >= threshold]
    return {
        "threshold": threshold,
        "accepted_selected_calls": len(accepted),
        "false_accepted": sum(r[f"{target}_true"] == 0 for r in accepted),
        "observed_success_rate": (sum(r[f"{target}_true"] for r in accepted) / len(accepted)
                                  if accepted else None),
    }


def _cluster_delta(rows, a_col, b_col, target):
    # Paired changes on identical outcomes, independent seed groups, not 18x IID.
    clusters = defaultdict(list)
    for r in rows:
        a, b = r[a_col], r[b_col]
        if a is None or b is None:
            continue
        y = r[f"{target}_true"]
        clusters[r["seed"]].append((a-y)**2 - (b-y)**2)
    values = [statistics.mean(v) for v in clusters.values() if v]
    if not values:
        return {"clusters": 0, "mean_brier_delta": None, "descriptive_cluster_95_interval": None}
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return {"clusters": len(values), "mean_brier_delta": mean,
            "descriptive_cluster_95_interval": [mean-1.96*se, mean+1.96*se]
            if se is not None else None}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["split"] != "test":
            continue
        groups[row["case"], row["arm"], "all"].append(row)
        phase = "boundary" if row["mode"] == "budgeted_boundary_information" else (
            "coverage" if "coverage" in row["mode"] else
            "discovery" if row["mode"] in ("feasible_discovery", "posterior_support_restoration") else
            "improvement" if "improvement" in row["mode"] else "other")
        groups[row["case"], row["arm"], phase].append(row)
        groups[row["case"], row["arm"], "first_online_step" if row["iteration"] == 0
               else "subsequent_steps"].append(row)
    summary = {}
    for (case, arm, phase), group in sorted(groups.items()):
        modeled = [r for r in group if r["prediction_available"]]
        data = {
            "total_selected_calls": len(group),
            "model_scored_calls": len(modeled),
            "no_model_probability_calls": len(group) - len(modeled),
            "seed_groups": len({r["seed"] for r in group}),
            "mode_counts": dict(Counter(r["mode"] for r in group)),
            "observed_mission_rate_all_calls": statistics.mean(r["mission_true"] for r in group),
            "observed_feasible_rate_all_calls": statistics.mean(r["feasible_true"] for r in group),
            "models": {},
        }
        for proxy, target in TARGETS.items():
            prefix = {
                "constraint_product": "predicted_constraint_product",
                "legacy_completion_product": "predicted_legacy_completion_product",
                "soc_mission": "predicted_soc",
                "record_completion_as_mission": "predicted_record_completion",
            }[proxy]
            good = [r for r in modeled if r[prefix] is not None]
            if not good:
                data["models"][proxy] = {"n": 0}
                continue
            truth = [r[f"{target}_true"] for r in good]
            raw = [r[prefix] for r in good]
            updated = [r["online_calibrated_"+proxy] for r in good]
            prior_col = ("historical_feasible_prevalence" if target == "feasible"
                         else "historical_mission_prevalence")
            priors = [r[prior_col] for r in good]
            metrics = {
                "n": len(good),
                "raw": _score(raw, truth),
                "online_calibrated": _score(updated, truth) if all(x is not None for x in updated) else None,
                "history_only_prequential_baseline": _score(priors, truth),
                "paired_calibrated_minus_raw_brier": _cluster_delta(
                    good, "online_calibrated_"+proxy, prefix, target),
                "risk_charge": {
                    "expected_raw_failure_sum": sum(1-p for p in raw),
                    "observed_failure_sum": sum(1-y for y in truth),
                    "mean_raw_minus_observed_failure": statistics.mean(
                        (1-p)-(1-y) for p, y in zip(raw, truth)),
                    "no_failure_budget_guarantee": True,
                },
                "threshold_0_8_raw": _rate_threshold(good, prefix, target, .8),
                "threshold_0_8_calibrated": _rate_threshold(
                    good, "online_calibrated_"+proxy, target, .8),
            }
            data["models"][proxy] = metrics
        summary["/".join((case,arm,phase))] = data
    return summary


def run(source, output, *, per_block=64):
    """Independent calibration/test on saved original online decisions, no model calls."""
    root, dest = Path(source).resolve(), Path(output).resolve()
    if dest.exists() or root == dest or root in dest.parents:
        raise ValueError("Output must be a fresh directory outside the original evidence")
    plan_path, db_path = root/"plan.json", root/"real_calls.sqlite"
    if sha_file(plan_path) != FROZEN_PLAN_SHA256:
        raise ValueError("Wrong frozen PR45 plan")
    plan = _read(plan_path)
    if plan.get("source_commit") != FROZEN_SOURCE_SHA:
        raise ValueError("Wrong original PR45 source SHA")
    selections = fixed_splits(plan, per_block)
    before_sha = sha_file(db_path)
    if before_sha != FROZEN_REAL_SQLITE_SHA256:
        raise ValueError("Wrong PR45 original SQLite")
    before_mtime = db_path.stat().st_mtime_ns
    dest.mkdir(parents=True, exist_ok=False)
    cal, test = [], []
    with _sqlite(db_path) as db:
        for case, split, seed in selections:
            for arm in ARMS:
                if arm not in plan["cases"][case]["arms_by_seed"][str(seed)]:
                    raise ValueError("Missing registered primary method")
                contract = plan["cases"][case]["contracts"]["20s"]
                records = read_arm(db, root, case, split, seed, arm, contract)
                (cal if split == "calibration" else test).extend(records)
            print("native online:", case, split, seed, flush=True)
    if sha_file(db_path) != before_sha or db_path.stat().st_mtime_ns != before_mtime:
        raise RuntimeError("PR45 original SQLite bytes/mtime changed")
    shifts, coverage = calibrate(cal, test)
    rows = cal + test
    with (dest/"prequential_risk_predictions.csv").open("x", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for item in rows:
            writer.writerow({name: item[name] for name in FIELDS})
    report = {
        "schema": SCHEMA, "status": "COMPLETE_NATIVE_PREQUENTIAL_RISK_AUDIT",
        "source_sha": FROZEN_SOURCE_SHA, "original_sqlite_sha256_unchanged": before_sha,
        "plan_sha256": FROZEN_PLAN_SHA256,
        "new_aircraft_calls": 0, "new_rcaide_calls": 0, "new_gp_fits": 0,
        "original_source_read_only": True, "per_block": per_block,
        "split_seed_indices": {"calibration": [CAL_START, CAL_START+per_block-1],
                               "test": [TEST_START, TEST_START+per_block-1]},
        "total_original_online_selected_calls_scanned": len(rows),
        "calibration_seed_calls": len(cal), "test_seed_calls": len(test),
        "calibration_scored_calls": sum(r["prediction_available"] for r in cal),
        "test_scored_calls": sum(r["prediction_available"] for r in test),
        "calibration_logit_shifts": shifts, "calibration_coverage": coverage,
        "test_scores": summarize(test),
        "limitations": [
            "PR45 recorded proposals are model-selected; never IID qA truth or unchosen-candidate counterfactuals.",
            "Original RA-LSBO and FullLatent choose DIFFERENT designs; no causal representation comparison from their scored trajectories.",
            "A model-free coverage/fallback choice has no valid probability and is counted separately.",
            "Online logit calibration uses already observed outcomes of OTHER seeds; retrospective only, unavailable to the PR45 online policy.",
            "Model forecasts are scored on their own SELECTED designs, never refitted using target responses.",
            "The previous PR51 qA n=192 results use more real training data and are NOT original 12+18 online forecasts.",
            "An independently registered mission-risk model, safety guarantee and joint probability model have not been established.",
            "A predicted risk budget debit is NOT an observed-failure bound; threshold analysis does not imply counterfactual policy benefits.",
            "The 140km task has not passed independent sparse task qualification.",
        ],
    }
    with (dest/"prequential_risk_report.json").open("x",encoding="utf8") as f:
        json.dump(report,f,indent=2,sort_keys=True,allow_nan=False)
    return report
