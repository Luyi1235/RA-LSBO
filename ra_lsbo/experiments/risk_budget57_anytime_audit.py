"""PR57: fixed-effort anytime audit of PR54 and corrected PR55 ON-POLICY paths.

This is a retrospective, outcome-aware *analysis* of actual observed prefixes.
At k=6 every arm has at least six charged online evaluations even with the
shared hard failure cap of six. Never extrapolate beyond a stopped trajectory.
Within-pair common-prefix k=min(N_a,N_b) is DESCRIPTIVE and depends on stopping.
No new candidate, GP/VAE fit, analytical/aircraft/RCAIDE call, or report rewrite.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SCHEMA = "ra_lsbo_pr57_fixed_effort_anytime_pairing_v1"
SOURCES = {
    "pr54": {
        "blob_sha1": "7f94ed040ee88c2ae552f704516fcea003193ded",
        "source_commit": "fd5492d33b21421ff8f54ad8bba453543caf39fe",
        "result_commit": "41db5094f29645cf5afd17b08e65ff62544d3f41",
        "status": "COMPLETE_ON_POLICY_EXPLORATORY_MATH",
        "cases": ("branin_ordinary", "branin_sparse"),
        "seeds": tuple(range(995401, 995409)),
        "trial_key": "per_arm_results",
        "registration_key": "registration",
        "n_analytical": 696,
        "n_trials": 48,
    },
    "pr55_corrected_pr56": {
        "blob_sha1": "be839a224ef56868f048cfb161fc54065c2eea8b",
        "source_commit": "d947db5f0b544739c9db286a75d99c0308713fdb",
        "result_commit": "cdd02b551d3233d8262d5adbf46db207d6703f93",
        "status": "COMPLETE_INDEPENDENT_ON_POLICY_ANALYTICAL_VALIDATION",
        "cases": ("branin_ordinary", "branin_sparse",
                  "rosenbrock_ordinary", "rosenbrock_sparse"),
        "seeds": tuple(range(996101, 996117)),
        "trial_key": "trial_details",
        "registration_key": "plan",
        "n_analytical": 2838,
        "n_trials": 192,
    },
}
ARMS = ("legacy", "semantic_only", "budgeted")
PAIRS = (("budgeted", "semantic_only"),
         ("semantic_only", "legacy"),
         ("budgeted", "legacy"))
FIXED_HORIZONS = (0, 6, 8, 10, 12)
HARD_FAILURE_STOP = 6
ONLINE_HORIZON = 12
FIELDS = (
    "study", "case", "seed", "candidate_arm", "control_arm",
    "candidate_online_calls", "control_online_calls", "candidate_total_failures",
    "control_total_failures", "common_online_calls",
    "initial_best", "candidate_at_six", "control_at_six",
    "delta_at_six_if_both_feasible", "candidate_at_common",
    "control_at_common", "delta_at_common_if_both_feasible",
    "candidate_final", "control_final", "delta_final_if_both_feasible",
    "candidate_extra_improvement_beyond_common",
    "control_extra_improvement_beyond_common",
    "candidate_extra_online_calls_beyond_common",
    "control_extra_online_calls_beyond_common",
    "candidate_failures_in_first_six", "control_failures_in_first_six",
    "candidate_reached_six_failure_stop", "control_reached_six_failure_stop",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    """Exactly verify Git's blob bytes, including line endings."""
    path = Path(path)
    h = hashlib.sha1()
    h.update(("blob %d\0" % path.stat().st_size).encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_or_none(value):
    return value is None or (type(value) in (int, float) and math.isfinite(value))


def _validate_trial(record):
    if not isinstance(record, dict):
        raise ValueError("Non-dictionary original trial")
    n = record.get("online_calls")
    failed = record.get("online_nonfeasible")
    if (type(n) is not int or not 6 <= n <= ONLINE_HORIZON or
            type(failed) is not int or not 0 <= failed <= min(HARD_FAILURE_STOP, n)):
        raise ValueError("Invalid online length or observed-failure count")
    curve = record.get("best_feasible_after_each_actual_online_call")
    if not isinstance(curve, list) or len(curve) != n + 1:
        raise ValueError("Online actual-evaluation incumbent path length changed")
    if (curve[0] != record.get("initial_best") or
            curve[-1] != record.get("final_best") or
            any(not finite_or_none(x) for x in curve)):
        raise ValueError("Initial or final incumbent/finite evidence mismatch")
    for earlier, later in zip(curve, curve[1:]):
        if earlier is not None and (
                later is None or later > earlier + 1e-9):
            raise ValueError("Historical best-so-far objective increased")
    receipts = record.get("pre_evaluation_risk_receipts")
    if not isinstance(receipts, list) or len(receipts) != n:
        raise ValueError("Charged online call receipts incomplete")
    if [r.get("iteration") for r in receipts] != list(range(n)):
        raise ValueError("Misordered actual online evaluation receipt")
    if any(type(r.get("realized_nonfeasible")) is not int or
           r["realized_nonfeasible"] not in (0, 1) for r in receipts):
        raise ValueError("No exact observed online nonfeasibility bit")
    if sum(r["realized_nonfeasible"] for r in receipts) != failed:
        raise ValueError("Realized failures disagree with the trial record")
    if n < ONLINE_HORIZON and failed != HARD_FAILURE_STOP:
        raise ValueError("Trial truncated before six observed failures")
    if (record.get("initial_best") is None and
            record.get("final_best") is not None):
        first = next((i for i, value in enumerate(curve[1:])
                      if value is not None), None)
        if first != record.get("first_online_feasible_iteration"):
            raise ValueError("First discovery index not aligned to actual prefix")
    if record.get("first_online_feasible_iteration") is not None and (
            type(record["first_online_feasible_iteration"]) is not int or
            not 0 <= record["first_online_feasible_iteration"] < n):
        raise ValueError("Malformed first-online-feasible index")
    return record


def load_study(path, study):
    """Production entry always checks the exact original Git blob."""
    if study not in SOURCES:
        raise ValueError("Unknown frozen PR54/PR56 study source")
    expected = SOURCES[study]
    path = Path(path).resolve()
    if git_blob_sha1(path) != expected["blob_sha1"]:
        raise ValueError("Original study report Git blob SHA1 differs")
    data = json.loads(path.read_text(encoding="utf8"))
    if (data.get("status") != expected["status"] or
            data.get("source_sha") != expected["source_commit"] or
            data.get("new_aircraft_evaluations", data.get("new_aircraft_calls")) != 0 or
            data.get("new_rcaide_calls") != 0):
        raise ValueError("Original study identity or zero-physical-call claim changed")
    field = ("actual_new_analytical_evaluations" if study == "pr54"
             else "real_new_analytical_calls")
    if data.get(field) != expected["n_analytical"]:
        raise ValueError("Frozen original analytical evaluation total changed")
    registration = data[expected["registration_key"]]
    if (registration.get("profile") != "full" or
            tuple(registration.get("seeds", [])) != expected["seeds"] or
            tuple(registration.get("cases",
                registration.get("analytical_cases", []))) != expected["cases"] or
            tuple(registration["arms"]) != ARMS or
            registration.get("online_horizon") != ONLINE_HORIZON):
        raise ValueError("Original registered task/arm/seed roster drift")
    trials = data.get(expected["trial_key"])
    if not isinstance(trials, list) or len(trials) != expected["n_trials"]:
        raise ValueError("Original on-policy trial population changed")
    if study == "pr55_corrected_pr56":
        audit = data.get("report_aggregation_audit", {})
        if (audit.get("trial_details_count") != 192 or
                audit.get("new_analytical_evaluations") != 0 or
                audit.get("aggregation_only") is not True or
                audit.get("original_report_sha256") !=
                "583d21106615cf05d8ec6c0a96a20d5a55f01de341efe7af76ee441139aa23f9"):
            raise ValueError("PR56 corrected report provenance changed")
    found = {}
    for t in trials:
        _validate_trial(t)
        key = (t["case"], t["seed"], t["arm"])
        if key in found:
            raise ValueError("Repeated original trial identity")
        found[key] = t
    expected_keys = {(case, seed, arm) for case in expected["cases"]
                     for seed in expected["seeds"] for arm in ARMS}
    if set(found) != expected_keys:
        raise ValueError("Missing or unregistered original on-policy trials")
    for case in expected["cases"]:
        for seed in expected["seeds"]:
            initials = [found[(case, seed, arm)]["initial_best"] for arm in ARMS]
            if initials[0] != initials[1] or initials[1] != initials[2]:
                raise ValueError("Paired arms did not share the same initial objective")
    return found, {"study": study, "source_path": str(path),
                   "source_sha256": sha256(path),
                   "source_git_blob_sha1": expected["blob_sha1"],
                   "source_commit": expected["source_commit"],
                   "result_commit": expected["result_commit"],
                   "trial_count": len(trials)}


def _delta(candidate, control):
    return (candidate - control if candidate is not None and
            control is not None else None)


def _mean(values):
    return statistics.mean(values) if values else None


def _median(values):
    return statistics.median(values) if values else None


def _improvement(start, final):
    if start is None or final is None:
        return None
    d = final - start
    if d > 1e-8:
        raise ValueError("Best-so-far increased after an extra actual call")
    return d


def paired_at_horizon(data, arm, control, case, seeds, *, horizon):
    """Strictly score ONLY seeds where BOTH arms reached the actual call k.

    At k=6 this should be ALL predeclared seeds, because the failure stop
    cannot trigger before six nonfeasible online evaluations.
    At k>6 coverage is treatment-dependent: report the denominator and do
    not silently treat early stops as if extra evaluations had happened.
    """
    if horizon not in FIXED_HORIZONS:
        raise ValueError("Unregistered fixed effort horizon")
    total = len(seeds)
    reached = []
    paired = []
    candidate_only = control_only = neither = 0
    for seed in seeds:
        a = data[case, seed, arm]
        b = data[case, seed, control]
        if a["online_calls"] < horizon or b["online_calls"] < horizon:
            continue
        reached.append(seed)
        aa = a["best_feasible_after_each_actual_online_call"][horizon]
        bb = b["best_feasible_after_each_actual_online_call"][horizon]
        d = _delta(aa, bb)
        if d is not None:
            paired.append(d)
        elif aa is not None:
            candidate_only += 1
        elif bb is not None:
            control_only += 1
        else:
            neither += 1
    return {
        "horizon_online_calls_per_arm": horizon,
        "predeclared_seed_groups": total,
        "both_arms_observed_through_horizon": len(reached),
        "pairs_not_observed_at_this_horizon_due_to_stop": total - len(reached),
        "fully_observed_seed_list": reached,
        "both_feasible": len(paired),
        "candidate_only_feasible": candidate_only,
        "control_only_feasible": control_only,
        "neither_feasible": neither,
        "paired_mean_objective_delta_if_both_feasible": _mean(paired),
        "paired_median_objective_delta_if_both_feasible": _median(paired),
        "paired_better": sum(d < -1e-10 for d in paired),
        "paired_worse": sum(d > 1e-10 for d in paired),
        "paired_tied": sum(abs(d) <= 1e-10 for d in paired),
        "no_imputed_post_stop_outcomes": True,
        "coverage_selection_bias_for_k_gt_6": horizon > 6,
    }


def _per_seed(study, data, case, seeds, arm, control):
    out = []
    for seed in seeds:
        a, b = data[case, seed, arm], data[case, seed, control]
        n1, n2 = a["online_calls"], b["online_calls"]
        k = min(n1, n2)
        curve_a = a["best_feasible_after_each_actual_online_call"]
        curve_b = b["best_feasible_after_each_actual_online_call"]
        if k < 6:
            raise ValueError("Observed common prefix less than the fixed six calls")
        aa, bb = curve_a[k], curve_b[k]
        da = _improvement(aa, a["final_best"])
        db = _improvement(bb, b["final_best"])
        first6a = sum(t["realized_nonfeasible"] for t in
                      a["pre_evaluation_risk_receipts"][:6])
        first6b = sum(t["realized_nonfeasible"] for t in
                      b["pre_evaluation_risk_receipts"][:6])
        return_row = {
            "study": study, "case": case, "seed": seed,
            "candidate_arm": arm, "control_arm": control,
            "candidate_online_calls": n1, "control_online_calls": n2,
            "candidate_total_failures": a["online_nonfeasible"],
            "control_total_failures": b["online_nonfeasible"],
            "common_online_calls": k,
            "initial_best": curve_a[0],
            "candidate_at_six": curve_a[6], "control_at_six": curve_b[6],
            "delta_at_six_if_both_feasible": _delta(curve_a[6],curve_b[6]),
            "candidate_at_common": aa, "control_at_common": bb,
            "delta_at_common_if_both_feasible": _delta(aa,bb),
            "candidate_final": a["final_best"],"control_final": b["final_best"],
            "delta_final_if_both_feasible": _delta(a["final_best"], b["final_best"]),
            "candidate_extra_improvement_beyond_common": da,
            "control_extra_improvement_beyond_common": db,
            "candidate_extra_online_calls_beyond_common": n1-k,
            "control_extra_online_calls_beyond_common": n2-k,
            "candidate_failures_in_first_six": first6a,
            "control_failures_in_first_six": first6b,
            "candidate_reached_six_failure_stop":
                a["online_nonfeasible"] == HARD_FAILURE_STOP,
            "control_reached_six_failure_stop":
                b["online_nonfeasible"] == HARD_FAILURE_STOP,
        }
        if (da is not None and db is not None and
                return_row["delta_at_common_if_both_feasible"] is not None):
            if not math.isclose(
                    return_row["delta_final_if_both_feasible"],
                    return_row["delta_at_common_if_both_feasible"]+da-db,
                    abs_tol=1e-9, rel_tol=1e-9):
                raise AssertionError("Objective difference failed exact continuation decomposition")
        out.append(return_row)
    return out


def _summary_per_seed(rows, label):
    diffs = [r[label] for r in rows if r[label] is not None]
    return {
        "predeclared_seed_groups":len(rows),
        "both_feasible":len(diffs),
        "candidate_only_feasible":sum(r[label] is None and
            r["candidate_final"] is not None and r["control_final"] is None
            for r in rows) if label == "delta_final_if_both_feasible" else None,
        "control_only_feasible":sum(r[label] is None and
            r["candidate_final"] is None and r["control_final"] is not None
            for r in rows) if label == "delta_final_if_both_feasible" else None,
        "paired_mean_delta":_mean(diffs),
        "paired_median_delta":_median(diffs),
        "paired_better":sum(d < -1e-10 for d in diffs),
        "paired_worse":sum(d > 1e-10 for d in diffs),
        "paired_tied":sum(abs(d) <= 1e-10 for d in diffs),
    }


def study_summary(study, found):
    info = SOURCES[study]
    output = {}
    rows = []
    for case in info["cases"]:
        for arm, control in PAIRS:
            per_seed = _per_seed(study, found, case, info["seeds"], arm, control)
            rows.extend(per_seed)
            common = _summary_per_seed(per_seed,"delta_at_common_if_both_feasible")
            final = _summary_per_seed(per_seed,"delta_final_if_both_feasible")
            fixed = {
                str(k):paired_at_horizon(found,arm,control,case,info["seeds"],horizon=k)
                for k in FIXED_HORIZONS
            }
            if fixed["6"]["both_arms_observed_through_horizon"] != len(info["seeds"]):
                raise ValueError("Not every seeded control had six actual online evaluations")
            continuation = [r for r in per_seed if
                            r["delta_at_common_if_both_feasible"] is not None and
                            r["delta_final_if_both_feasible"] is not None]
            key = "/".join((study,case,arm+"_vs_"+control))
            output[key] = {
                "call_synchronized_fixed_horizons":fixed,
                "same_observed_prefix_effort_per_seed": {
                    **common,
                    "per_seed_common_horizon_min_calls": [
                        {"seed":r["seed"],"k":r["common_online_calls"]}
                        for r in per_seed],
                    "horizon_selected_after_original_early_stop":True,
                },
                "original_full_stop_censored_endpoints":final,
                "continuation_decomposition_only_where_both_feasible_at_common": {
                    "n_pairs":len(continuation),
                    "mean_common_objective_delta":_mean([
                        r["delta_at_common_if_both_feasible"] for r in continuation]),
                    "mean_candidate_extra_improvement":_mean([
                        r["candidate_extra_improvement_beyond_common"] for r in continuation]),
                    "mean_control_extra_improvement":_mean([
                        r["control_extra_improvement_beyond_common"] for r in continuation]),
                    "mean_final_objective_delta":_mean([
                        r["delta_final_if_both_feasible"] for r in continuation]),
                    "mean_candidate_extra_calls":_mean([
                        r["candidate_extra_online_calls_beyond_common"] for r in continuation]),
                    "mean_control_extra_calls":_mean([
                        r["control_extra_online_calls_beyond_common"] for r in continuation]),
                    "decomposition_is_identity_not_causal_attribution":True,
                },
                "actual_full_trajectory_counts": {
                    "mean_candidate_online_calls":_mean([
                        r["candidate_online_calls"] for r in per_seed]),
                    "mean_control_online_calls":_mean([
                        r["control_online_calls"] for r in per_seed]),
                    "mean_candidate_failures":_mean([
                        r["candidate_total_failures"] for r in per_seed]),
                    "mean_control_failures":_mean([
                        r["control_total_failures"] for r in per_seed]),
                    "mean_first_six_failure_difference":_mean([
                        r["candidate_failures_in_first_six"] -
                        r["control_failures_in_first_six"] for r in per_seed]),
                    "candidate_reached_failure_stop":sum(
                        r["candidate_reached_six_failure_stop"] for r in per_seed),
                    "control_reached_failure_stop":sum(
                        r["control_reached_six_failure_stop"] for r in per_seed),
                },
                "interpretation": (
                    "Fixed k=6: every arm genuinely observed exactly six online calls;"
                    " later fixed k horizons: available-only descriptive subsets;"
                    " same min(N_a,N_b): post-treatment effort matching;"
                    " full endpoints: unequal actual calls may affect quality."
                ),
            }
    return output,rows


def run(pr54,pr56,out_dir):
    """Read exact frozen Git-blob inputs; write ONLY a new output directory."""
    paths=(Path(pr54).resolve(),Path(pr56).resolve())
    out=Path(out_dir).resolve()
    if out.exists() or any(out==p.parent or p.parent in out.parents for p in paths):
        raise ValueError("Output must be a fresh directory outside both source result folders")
    if paths[0] == paths[1]:
        raise ValueError("Two independent original report inputs required")
    a,a_ref=load_study(paths[0],"pr54")
    b,b_ref=load_study(paths[1],"pr55_corrected_pr56")
    result_a,rows_a=study_summary("pr54",a)
    result_b,rows_b=study_summary("pr55_corrected_pr56",b)
    # Before writing, assert neither input's bytes changed during scoring.
    if (git_blob_sha1(paths[0])!=SOURCES["pr54"]["blob_sha1"] or
            git_blob_sha1(paths[1])!=SOURCES["pr55_corrected_pr56"]["blob_sha1"]):
        raise RuntimeError("Historical original report changed while scoring")
    rows=rows_a+rows_b
    out.mkdir(parents=True,exist_ok=False)
    with (out/"fixed_effort_paired_seeds.csv").open("x",encoding="utf8",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k:item[k] for k in FIELDS} for item in rows)
    report={
        "schema":SCHEMA,
        "status":"COMPLETE_RETROSPECTIVE_FIXED_EFFORT_ANYTIME_AUDIT",
        "input_identifiers":[a_ref,b_ref],
        "n_original_trial_trajectories":len(a)+len(b),
        "n_paired_method_seed_rows":len(rows),
        "new_analytical_evaluations":0,
        "new_aircraft_evaluations":0,
        "new_rcaide_evaluations":0,
        "new_gp_vae_fits":0,
        "new_candidate_proposals":0,
        "original_files_read_only":True,
        "paired_studies":{**result_a,**result_b},
        "scientific_limits":[
            "PR54 and PR55 are already observed, and the equal-effort audit was devised AFTER seeing their outcomes: exploratory only.",
            "At fixed k=6 ALL compared arms have six actual online evaluations and the same 12 shared initial designs.",
            "At k=8/10/12 only seed pairs whose BOTH real paths reached k are compared: no post-stop truth is imputed; reported coverage is informative treatment-dependent censoring.",
            "At per-seed k=min(actual calls) the effort is matched within each pair but the horizon was selected after seeing the original stop time: descriptive, NOT a randomized controlled equal-work causal estimate.",
            "At full horizon real on-policy methods have different online call counts due to the common hard observed-six-failure stop.",
            "Conditional objective comparisons exclude both-infeasible or singly feasible pairs, which are always disclosed.",
            "Endpoint improvement decomposition is an algebraic identity; it does NOT establish a causal effect of budget on extra calls.",
            "The budgeted policy is an uncalibrated risk heuristic. No safe-BO, physical safety, real aircraft or independently certified sparse-domain claim.",
            "Branin/Rosenbrock custom analytical task variants are not original upstream benchmark replicas.",
            "No confirmatory statistical significance or adjusted multiple-comparison claim is made.",
        ],
    }
    with (out/"fixed_effort_anytime_report.json").open("x",encoding="utf8") as stream:
        json.dump(report,stream,indent=2,sort_keys=True,ensure_ascii=False,
                  allow_nan=False)
    return report
