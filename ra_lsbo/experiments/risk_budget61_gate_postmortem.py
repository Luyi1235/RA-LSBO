"""PR61: exact-source PR60 branch-Beta risk-floor and fallback-collapse audit.

NO new optimizer, GP/VAE, numerical evaluator, aircraft or RCAIDE calls.
Uses ONLY the frozen report of PR60's 288 real on-policy mathematical trials.
This posthoc mechanism analysis describes selected-point associations, NOT
same-candidate causal outcomes or a formally valid Bayesian credible bound.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SCHEMA = "ra_lsbo_pr61_branch_beta_floor_and_gate_saturation_v1"
ORIGINAL_REPORT_GIT_BLOB = "508ba9bbf32715009791c2079caa81f125c35d2d"
ORIGINAL_EXPERIMENT_SOURCE = "8675ee94b23bc06779788907dafa3a068c226c29"
ORIGINAL_RESULT_COMMIT = "b175daf7527b13f9311b95fda8083253a14b14a8"
ORIGINAL_REPORT_SHA256 = "18032f4540f520e6ca87e2a71e973e6b95716f8beb97254ea1bf1aaa2f0acea9"
SOURCE_SCHEMA = "ra_lsbo_pr60_branch_evidence_dispersion_onpolicy_v1"
CASES = ("branin_ordinary","branin_sparse",
         "rosenbrock_ordinary","rosenbrock_sparse")
ARMS = ("legacy","semantic_only","budgeted",
        "branch_mean","branch_dispersion","branch_explore")
BRANCH_ARMS = ("branch_mean","branch_dispersion","branch_explore")
SEEDS = tuple(range(997101,997113))
CALLS = 6
INITIAL = 12
NUMERIC_TOL = 1.e-9
COLUMNS = (
    "case","arm","seed","iteration","mode","forecast_available",
    "actual_nonfeasible","gp_failure","selected_proxy_failure",
    "branch_n","branch_failures","branch_successes",
    "branch_beta_mean","branch_beta_sd","branch_mean_plus_sd",
    "beta_floor_bind","beta_risk_increment",
    "gate_fallback","gate_passed","selection_rule",
    "branch_small_n_le_3","risk_is_heuristic_not_a_safety_bound",
)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    path=Path(path)
    h=hashlib.sha1()
    h.update(("blob %d\0"%path.stat().st_size).encode("ascii"))
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def probability(x):
    return type(x) in (float,int) and math.isfinite(x) and 0<=x<=1


def close(x,y):
    return math.isclose(x,y,rel_tol=NUMERIC_TOL,abs_tol=NUMERIC_TOL)


def parse_original(path):
    """Verify source provenance and original pre-outcome risk arithmetic."""
    path=Path(path).resolve()
    if git_blob_sha1(path)!=ORIGINAL_REPORT_GIT_BLOB:
        raise ValueError("PR60 original report Git blob SHA1 changed")
    if sha256(path)!=ORIGINAL_REPORT_SHA256:
        raise ValueError("PR60 original report SHA256 changed")
    d=json.loads(path.read_text(encoding="utf8"))
    plan=d.get("plan",{})
    expected = {
        "schema":SOURCE_SCHEMA,
        "status":"COMPLETE_FRESH_FIXED_SIX_ON_POLICY",
        "code_sha":ORIGINAL_EXPERIMENT_SOURCE,
        "n_real_analytical_calls":2304,
        "n_new_on_policy_trials":288,
        "new_aircraft_calls":0,
        "new_rcaide_calls":0,
    }
    if any(d.get(k)!=v for k,v in expected.items()):
        raise ValueError("PR60 exact completed scientific scope changed")
    if (tuple(plan.get("cases",[]))!=CASES or
        tuple(plan.get("arms",[]))!=ARMS or
        tuple(plan.get("seeds",[]))!=SEEDS or
        plan.get("source_sha")!=ORIGINAL_EXPERIMENT_SOURCE or
        plan.get("profile")!="full" or plan.get("initial")!=INITIAL or
        plan.get("actual_online_calls_per_arm")!=CALLS or
        plan.get("engine_internal_horizon_unmodified")!=12):
        raise ValueError("PR60 frozen seed/arm/online-work registration changed")
    trials=d.get("trial_details")
    if not isinstance(trials,list) or len(trials)!=len(CASES)*len(ARMS)*len(SEEDS):
        raise ValueError("PR60 trial count/roster incomplete")
    mapped={}
    rows=[]
    for trial in trials:
        case,arm,seed=trial.get("case"),trial.get("arm"),trial.get("seed")
        key=(case,seed,arm)
        if (case not in CASES or arm not in ARMS or
            type(seed) is not int or seed not in SEEDS or key in mapped or
            type(trial.get("online_calls")) is not int or
            trial["online_calls"]!=CALLS or
            type(trial.get("observed_nonfeasible")) is not int):
            raise ValueError("PR60 invalid or duplicate real trial")
        mapped[key]=trial
        receipts=trial.get("risk_receipts")
        if not isinstance(receipts,list) or len(receipts)!=CALLS:
            raise ValueError("PR60 original online selected receipt coverage changed")
        prior_fail=0
        for i,r in enumerate(receipts):
            label=r.get("realized_nonfeasible")
            have=r.get("forecast_available_for_selected")
            if (r.get("iteration")!=i or
                type(label) is not int or label not in (0,1) or
                type(have) is not bool or
                r.get("observed_online_nonfeasible_before")!=prior_fail or
                r.get("outcome_not_available_when_selected") is not True):
                raise ValueError("PR60 pre-outcome time order/observed failure ledger mismatch")
            prior_fail+=label
            model_p=r.get("predicted_nonfeasible")
            if have and not probability(model_p):
                raise ValueError("Model selected prediction absent/invalid")
            if not have and model_p is not None:
                raise ValueError("Coverage selected point has invented GP probability")
            if (arm=="legacy" and have) or (arm!="legacy" and not have and
                    r.get("selected_branch_posterior") is not None):
                raise ValueError("PR60 legacy/coverage probability semantics changed")
            gp=None
            n=failures=successes=mean=sd=mean_plus_sd=increment=None
            floor_bind=None
            if arm in BRANCH_ARMS and have:
                gp=r.get("original_gp_predicted_failure")
                b=r.get("selected_branch_posterior")
                if not probability(gp) or not isinstance(b,dict):
                    raise ValueError("Missing original GP/branch risk before selection")
                n=b.get("n_settled_in_branch")
                failures=b.get("observed_failures")
                successes=b.get("observed_successes")
                if (type(n) is not int or type(failures) is not int or
                        type(successes) is not int or n<0 or
                        failures<0 or successes<0 or failures+successes!=n or
                        n>INITIAL+i):
                    raise ValueError("Branch history counts contain unobserved/future records")
                a=1.+failures
                bb=1.+successes
                total=a+bb
                mean=a/total
                sd=math.sqrt(a*bb/(total*total*(total+1)))
                mean_plus_sd=min(1.,mean+sd)
                registered=(b.get("posterior_failure_mean"),
                            b.get("posterior_failure_sd"),
                            b.get("posterior_failure_mean_plus_sd"))
                if any(not probability(v) for v in registered):
                    raise ValueError("Invalid original branch posterior arithmetic")
                if not all(close(x,y) for x,y in zip(
                        registered,(mean,sd,mean_plus_sd))):
                    raise ValueError("PR60 branch posterior mean/variance doesn't match receipt")
                threshold=mean if arm=="branch_mean" else mean_plus_sd
                if not close(model_p,max(gp,threshold)):
                    raise ValueError("Selected PR60 risk floor doesn't match registered rule")
                increment=max(0.,model_p-gp)
                floor_bind=increment>NUMERIC_TOL
            elif arm not in BRANCH_ARMS and have and r.get("selected_branch_posterior") is not None:
                raise ValueError("Original controls unexpectedly have branch posteriors")
            rule=r.get("budget_rule")
            is_fallback=have and isinstance(rule,str) and rule.startswith(
                "no_candidate_below_allowance_")
            if arm in BRANCH_ARMS and have:
                allowed=("empirical_branch_risk_under_predicted_allowance",
                         "no_candidate_below_allowance_max_branch_information"
                         if arm=="branch_explore" else
                         "no_candidate_below_allowance_min_corrected_proxy")
                if rule not in allowed:
                    raise ValueError("Unknown PR60 branch risk acquisition path")
                if r.get("predicted_gate_passed") is not (not is_fallback):
                    raise ValueError("Branch gate recorded passing an infeasible proxy allowance")
            if arm=="budgeted" and have and rule not in (
                    "predicted_allowance_filter",
                    "no_candidate_below_allowance_min_predicted_failure"):
                raise ValueError("Unknown original budgeted selection rule")
            rows.append({
                "case":case,"arm":arm,"seed":seed,"iteration":i,"mode":r.get("mode"),
                "forecast_available":have,"actual_nonfeasible":label,
                "gp_failure":gp,"selected_proxy_failure":model_p,
                "branch_n":n,"branch_failures":failures,
                "branch_successes":successes,"branch_beta_mean":mean,
                "branch_beta_sd":sd,"branch_mean_plus_sd":mean_plus_sd,
                "beta_floor_bind":floor_bind,"beta_risk_increment":increment,
                "gate_fallback":bool(is_fallback),"gate_passed":r.get("predicted_gate_passed"),
                "selection_rule":rule,
                "branch_small_n_le_3":bool(n<=3) if n is not None else None,
                "risk_is_heuristic_not_a_safety_bound":True,
            })
        if prior_fail!=trial["observed_nonfeasible"]:
            raise ValueError("Observed true failed evaluations don't match trial ledger")
    if len(mapped)!=len(CASES)*len(SEEDS)*len(ARMS):
        raise ValueError("PR60 missing registered real trials")
    for case in CASES:
        for seed in SEEDS:
            init={mapped[case,seed,arm].get("initial_prefix_sha256") for arm in ARMS}
            if len(init)!=1 or None in init:
                raise ValueError("Six paired methods have different initial physical evidence")
    if len(rows)!=len(CASES)*len(SEEDS)*len(ARMS)*CALLS:
        raise ValueError("Original 1,728 charged online observations incomplete")
    return rows, {
        "source_code_sha":ORIGINAL_EXPERIMENT_SOURCE,
        "result_commit":ORIGINAL_RESULT_COMMIT,
        "report_git_blob_sha1":ORIGINAL_REPORT_GIT_BLOB,
        "report_sha256":ORIGINAL_REPORT_SHA256,
        "trial_count":len(trials),
    }


def mean(values):
    return statistics.mean(values) if values else None


def group_statistics(rows):
    model=[r for r in rows if r["forecast_available"]]
    branch=[r for r in model if r["branch_n"] is not None]
    fallback=[r for r in model if r["gate_fallback"]]
    floor=[r for r in branch if r["beta_floor_bind"]]
    small=[r for r in branch if r["branch_small_n_le_3"]]
    return {
        "n_actual_selected_calls":len(rows),
        "n_arms_seed_groups":len({r["seed"] for r in rows}),
        "actual_nonfeasible_all":sum(r["actual_nonfeasible"] for r in rows),
        "forecasted_calls":len(model),
        "model_free_coverage_calls":len(rows)-len(model),
        "model_free_coverage_failures":sum(r["actual_nonfeasible"] for r in rows
                                           if not r["forecast_available"]),
        "forecasted_risk_mean":mean([r["selected_proxy_failure"] for r in model]),
        "actual_nonfeasible_on_forecasted":mean([r["actual_nonfeasible"] for r in model]),
        "n_with_branch_beta":len(branch),
        "n_branch_posterior_floor_bind":len(floor),
        "floor_bind_fraction_among_scored_branch_calls":
            len(floor)/len(branch) if branch else None,
        "mean_gp_failure_on_branch_scored":mean([r["gp_failure"] for r in branch]),
        "mean_risk_increment_due_to_floor":mean(
            [r["beta_risk_increment"] for r in branch]),
        "mean_branch_history_n":mean([r["branch_n"] for r in branch]),
        "selected_branch_n_le_3":len(small),
        "fallback_calls":len(fallback),
        "fallback_actual_nonfeasible":sum(r["actual_nonfeasible"] for r in fallback),
        "fallback_failure_fraction":mean([r["actual_nonfeasible"] for r in fallback]),
        "fallback_with_small_branch_history_n_le_3":sum(
            r["branch_small_n_le_3"] is True for r in fallback),
        "fallback_with_beta_floor_bind":sum(r["beta_floor_bind"] is True for r in fallback),
        "n_predicted_gate_passed":sum(r["gate_passed"] is True for r in model),
        "n_predicted_gate_failed":sum(r["gate_passed"] is False for r in model),
        "no_probabilistic_safety_or_causal_guarantee":True,
    }


def aggregate(rows):
    by=defaultdict(list)
    for r in rows:
        by[r["arm"]+"/all"].append(r)
        by[r["case"]+"/"+r["arm"]].append(r)
        by[r["case"]+"/"+r["arm"]+"/"+str(r["mode"])].append(r)
        by[r["arm"]+"/"+str(r["mode"])].append(r)
    return {key:group_statistics(value) for key,value in sorted(by.items())}


def run(original, output):
    """Read original bytes twice; write never into PR60 result directory."""
    original=Path(original).resolve()
    output=Path(output).resolve()
    if output.exists() or original.parent==output or original.parent in output.parents:
        raise ValueError("PR61 output must be fresh and outside PR60 original result folder")
    rows, identity=parse_original(original)
    counts=aggregate(rows)
    for arm,n,fb,failed,floor in (
        ("branch_mean",265,109,92,213),
        ("branch_dispersion",264,175,131,231),
        ("branch_explore",264,231,213,214),
        ("budgeted",266,22,19,None),
    ):
        summary=counts[arm+"/all"]
        if (summary["forecasted_calls"]!=n or summary["fallback_calls"]!=fb
                or summary["fallback_actual_nonfeasible"]!=failed or
                (floor is not None and
                summary["n_branch_posterior_floor_bind"]!=floor)):
            raise ValueError("Frozen PR60 branch floor/fallback scientific anchors drifted")
    if (git_blob_sha1(original)!=ORIGINAL_REPORT_GIT_BLOB or
            sha256(original)!=ORIGINAL_REPORT_SHA256):
        raise RuntimeError("PR60 immutable source changed while auditing")
    output.mkdir(parents=True,exist_ok=False)
    with (output/"branch_floor_selected_receipts.csv").open(
            "x",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({k:r[k] for k in COLUMNS} for r in rows)
    report={
        "schema":SCHEMA,
        "status":"COMPLETE_RETROSPECTIVE_BRANCH_BETA_GATE_COLLAPSE_AUDIT",
        "source":identity,
        "n_original_new_onpolicy_trials":288,
        "n_original_online_receipts":len(rows),
        "n_new_gp_vae_fits":0,
        "n_new_analytical_calls":0,
        "n_new_aircraft_calls":0,
        "n_new_rcaide_calls":0,
        "original_data_modified":False,
        "summary_by_arm_case_and_mode":counts,
        "scientific_limits":[
            "Only actual PR60 original selected designs and labels are evaluated; no unseen candidate outcome or counterfactual selection is inferred.",
            "Beta(1,1) branch counts are adaptively selected and include different continuous designs; their posterior mean/SD cannot be interpreted as a per-design calibrated failure-probability bound.",
            "This audit verifies original recorded Beta arithmetic, NOT independently reconstructs branch membership from full trial ZIPs.",
            "The observed correlation of beta floor, no-pass gate fallback and actual failures does not by itself prove causal mediation.",
            "All PR60 new algorithm claims failed the equal-six-call comparison; do not retrofit another threshold to the same samples and claim independent validation.",
            "Custom mathematical Branin/Rosenbrock hierarchy, not real RCAIDE or 100/140km aircraft validation; no flight safety guarantee.",
        ],
    }
    with (output/"branch_floor_gate_audit.json").open("x",encoding="utf8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
    return report
