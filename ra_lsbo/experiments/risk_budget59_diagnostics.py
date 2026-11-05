"""PR59 frozen retrospective risk-overconfidence diagnosis from REAL PR58 trajectories.

NO new algorithm, no surrogate fitting, no analytical/aircraft/RCAIDE calls.
The source is the exact PR58 full-report Git blob; replayed decisions are NOT
counterfactuals. Seed-cluster bootstrap intervals describe sampling variation
of THIS selected-policy distribution, not reliable probability or safe BO.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

SCHEMA = "ra_lsbo_pr59_selected_risk_failure_mechanism_diagnostic_v1"
INPUT_SHA1 = "d5029c505b738ec3da471e5bc2b38e8dd0fefadd"
INPUT_RESULT_COMMIT = "84c8a786c907cad5e2b333964c318fac9fdea25d"
SOURCE_CODE_SHA = "7d8e5edb0cd9db6bc6525a8f9989591388c71af5"
INPUT_SCHEMA = "ra_lsbo_pr58_fresh_fixed_six_onpolicy_v1"
TASKS = ("branin_ordinary","branin_sparse",
         "rosenbrock_ordinary","rosenbrock_sparse")
ARMS = ("legacy","semantic_only","budgeted")
SEEDS = tuple(range(997001,997025))
N_CALLS = 6
BOOTSTRAP_SEED = 20261105
BOOTSTRAP_DRAWS = 1000
RISK_BINS = ((0.,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.))
FIELDS = ("case","arm","seed","iteration","mode","phase","has_prediction",
          "predicted_nonfeasible","actual_nonfeasible","budget_rule",
          "budget_gate_passed","forecast_error","brier","high_confidence_safe",
          "high_confidence_safe_failure","model_free_or_coverage")


def git_blob_sha1(path):
    path = Path(path)
    h = hashlib.sha1()
    h.update(("blob %d\0" % path.stat().st_size).encode("ascii"))
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def _finite_probability(v):
    return type(v) in (float,int) and math.isfinite(v) and 0.<=v<=1.


def _input(path):
    path=Path(path).resolve()
    if git_blob_sha1(path)!=INPUT_SHA1:
        raise ValueError("PR58 source report Git blob is not the frozen genuine original")
    d=json.loads(path.read_text(encoding="utf8"))
    p=d.get("plan") or {}
    if (d.get("schema")!=INPUT_SCHEMA or
            d.get("status")!="COMPLETE_FRESH_FIXED_SIX_ON_POLICY" or
            d.get("code_sha")!=SOURCE_CODE_SHA or
            d.get("n_real_analytical_calls")!=2880 or
            d.get("n_original_on_policy_trials")!=288 or
            d.get("new_aircraft_calls")!=0 or d.get("new_rcaide_calls")!=0 or
            tuple(p.get("cases",[]))!=TASKS or
            tuple(p.get("arms",[]))!=ARMS or
            tuple(p.get("seeds",[]))!=SEEDS or
            p.get("actual_online_calls_per_arm")!=N_CALLS or
            p.get("initial")!=12 or
            p.get("engine_internal_horizon_unmodified")!=12 or
            p.get("source_sha")!=SOURCE_CODE_SHA):
        raise ValueError("Frozen PR58 scientific or source registry differs")
    trials=d.get("trial_details")
    if not isinstance(trials,list) or len(trials)!=288:
        raise ValueError("PR58 trial evidence count differs")
    by_key={}
    for trial in trials:
        case,arm,seed=trial.get("case"),trial.get("arm"),trial.get("seed")
        if (case not in TASKS or arm not in ARMS or seed not in SEEDS or
                type(trial.get("online_calls")) is not int or
                trial["online_calls"]!=N_CALLS or
                type(trial.get("observed_nonfeasible")) is not int):
            raise ValueError("Nonregistered PR58 six-call trial")
        key=case,seed,arm
        if key in by_key:
            raise ValueError("Duplicate PR58 trial")
        by_key[key]=trial
    if len(by_key)!=len(TASKS)*len(SEEDS)*len(ARMS):
        raise ValueError("Incomplete PR58 trial grid")
    rows=[]
    for (case,seed,arm),trial in by_key.items():
        rs=trial.get("risk_receipts")
        if not isinstance(rs,list) or len(rs)!=6:
            raise ValueError("Incomplete original six true pre-outcome receipts")
        initial_digest=trial.get("initial_prefix_sha256")
        if not isinstance(initial_digest,str) or len(initial_digest)!=64:
            raise ValueError("Shared original initial hash missing")
        expected_failed=0
        for i,r in enumerate(rs):
            if (r.get("iteration")!=i or
                    type(r.get("realized_nonfeasible")) is not int or
                    r["realized_nonfeasible"] not in (0,1) or
                    type(r.get("forecast_available_for_selected")) is not bool or
                    r.get("outcome_not_available_when_selected") is not True):
                raise ValueError("True settled receipt/temporal identity differs")
            expected_failed+=r["realized_nonfeasible"]
            if r.get("observed_online_nonfeasible_before")!=expected_failed-r["realized_nonfeasible"]:
                raise ValueError("Budget policy read an inconsistent historical prefix")
            have=r["forecast_available_for_selected"]
            prob=r.get("predicted_nonfeasible")
            rule=r.get("budget_rule")
            if have and not _finite_probability(prob):
                raise ValueError("Selected risk prediction invalid or absent")
            if not have and prob is not None:
                raise ValueError("Model-free coverage has invented risk prediction")
            if arm=="legacy" and have:
                raise ValueError("Legacy arm has no selected GP risk receipt in original trial")
            if arm=="budgeted" and have and rule not in (
                    "predicted_allowance_filter",
                    "no_candidate_below_allowance_min_predicted_failure"):
                raise ValueError("Unregistered budgeted risk selection")
            if arm=="semantic_only" and have and rule!="constraint_only_ranking":
                raise ValueError("Unregistered semantic-only risk rule")
            failed=r["realized_nonfeasible"]
            very_safe=have and prob<=.2
            mode=r["mode"]
            rows.append({
                "case":case, "arm":arm, "seed":seed,"iteration":i,
                "mode":mode,"phase":"boundary" if mode=="budgeted_boundary_information" else
                               "discovery" if mode=="feasible_discovery" else
                               "exploitation" if mode=="probability_weighted_feasible_improvement" else
                               "coverage",
                "has_prediction":have,
                "predicted_nonfeasible":prob,
                "actual_nonfeasible":failed,
                "budget_rule":rule,"budget_gate_passed":r.get("predicted_gate_passed"),
                "forecast_error":prob-failed if have else None,
                "brier":(prob-failed)**2 if have else None,
                "high_confidence_safe":very_safe,
                "high_confidence_safe_failure":very_safe and bool(failed),
                "model_free_or_coverage":not have,
            })
        if expected_failed!=trial["observed_nonfeasible"]:
            raise ValueError("Original six-call recorded failures do not match true receipts")
    for case in TASKS:
        for seed in SEEDS:
            identities={by_key[case,seed,arm]["initial_prefix_sha256"] for arm in ARMS}
            if len(identities)!=1:
                raise ValueError("Paired PR58 methods lack identical shared initial evidence")
    return rows,{"pr58_result_commit":INPUT_RESULT_COMMIT,"pr58_original_code_sha":SOURCE_CODE_SHA,
                 "report_git_blob_sha1":INPUT_SHA1,"report_sha256":sha256(path)}


def _mean(xs):
    return statistics.mean(xs) if xs else None


def _percentile(xs,p):
    xs=sorted(xs)
    if not xs:return None
    idx=(len(xs)-1)*p
    i=int(idx)
    return xs[i]*(1-(idx-i))+xs[min(i+1,len(xs)-1)]*(idx-i)


def bootstrap_bias(rows):
    """Resample whole independent seed clusters, never individual online calls."""
    by_seed=defaultdict(list)
    for r in rows:
        if r["has_prediction"]:
            by_seed[r["seed"]].append(r)
    seeds=sorted(by_seed)
    if len(seeds)<2:return {"n_seed_clusters":len(seeds),"interval":None}
    rng=random.Random(BOOTSTRAP_SEED)
    values=[]
    for _ in range(BOOTSTRAP_DRAWS):
        selected=[rng.choice(seeds) for _ in seeds]
        copies=[r for s in selected for r in by_seed[s]]
        values.append(_mean([r["forecast_error"] for r in copies]))
    return {"n_seed_clusters":len(seeds),
            "bootstrap_draws":BOOTSTRAP_DRAWS,
            "bias_interval_95_descriptive":[_percentile(values,.025),_percentile(values,.975)],
            "seed_resampling_not_onpolicy_causal_ci":True}


def _score(rows):
    model=[r for r in rows if r["has_prediction"]]
    n=len(model)
    probs=[r["predicted_nonfeasible"] for r in model]
    real=[r["actual_nonfeasible"] for r in model]
    safe=[r for r in model if r["high_confidence_safe"]]
    fallbacks=[r for r in model if r["budget_rule"]==
               "no_candidate_below_allowance_min_predicted_failure"]
    bins=[]
    for lo,hi in RISK_BINS:
        selected=[r for r in model if lo<=r["predicted_nonfeasible"]<
                  hi or (hi==1 and r["predicted_nonfeasible"]==1)]
        bins.append({"predicted_failure_range":[lo,hi],
                     "n":len(selected),
                     "predicted_mean":_mean([r["predicted_nonfeasible"] for r in selected]),
                     "observed_failure_rate":_mean([r["actual_nonfeasible"] for r in selected])})
    ece=sum((b["n"]/n)*abs(b["predicted_mean"]-b["observed_failure_rate"])
            for b in bins if b["n"]) if n else None
    loss=_mean([-y*math.log(max(p,1e-12))-
                (1-y)*math.log(max(1-p,1e-12)) for p,y in zip(probs,real)])
    return {
        "all_actual_selected_calls":len(rows),
        "forecasted_calls":n,
        "no_forecast_calls":len(rows)-n,
        "actual_failure_count_all":sum(r["actual_nonfeasible"] for r in rows),
        "actual_failure_rate_all":_mean([r["actual_nonfeasible"] for r in rows]),
        "predicted_failure_mean_scored":_mean(probs),
        "actual_failure_rate_scored":_mean(real),
        "signed_probability_bias_predicted_minus_realized":_mean([p-y for p,y in zip(probs,real)]),
        "brier":_mean([r["brier"] for r in model]),
        "log_loss":loss,
        "five_bin_ECE_descriptive":ece,
        "reliability_bins":bins,
        "high_confidence_safe_threshold":.2,
        "predicted_low_failure_calls":len(safe),
        "observed_failures_among_predicted_low_failure_calls":sum(
            r["actual_nonfeasible"] for r in safe),
        "observed_failure_rate_predicted_low_failure":_mean([
            r["actual_nonfeasible"] for r in safe]),
        "budget_gate_fallback_count":len(fallbacks),
        "realized_failure_count_in_fallback":sum(r["actual_nonfeasible"] for r in fallbacks),
        "realized_failure_rate_in_fallback":_mean([r["actual_nonfeasible"] for r in fallbacks]),
        "mode_counts":dict(sorted(Counter(r["mode"] for r in rows).items())),
        "descriptive_cluster_bias":bootstrap_bias(rows) if n else None,
    }


def analyze(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["case"],r["arm"],"all"].append(r)
        groups[r["case"],r["arm"],r["phase"]].append(r)
        groups[r["case"],r["arm"],"iteration_"+str(r["iteration"])].append(r)
    return {"/".join(key):_score(values) for key,values in sorted(groups.items())}


def run(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or source.parent==output or source.parent in output.parents:
        raise ValueError("PR59 output must be NEW and outside original PR58 evidence")
    rows,identity=_input(source)
    report={
        "schema":SCHEMA,
        "status":"COMPLETE_RETROSPECTIVE_PR58_SELECTED_RISK_DIAGNOSIS",
        "immutable_input":identity,
        "total_original_six_call_receipts":len(rows),
        "expected_original_receipts":len(TASKS)*len(SEEDS)*len(ARMS)*N_CALLS,
        "new_gp_vae_fits":0,"new_analytical_calls":0,
        "new_aircraft_calls":0,"new_rcaide_calls":0,
        "new_optimizer_proposals":0,"source_report_read_only":True,
        "group_scores":analyze(rows),
        "interpretation":[
            "The original PR58 comparison found budgeted vs semantic_only 6 wins / 5 losses / 85 ties over 96 paired seeds. Current simple gate has NO demonstrated stable equal-cost benefit.",
            "Branin sparse and Rosenbrock sparse forecasted risk is lower than selected-candidate observed failures. The marginal-GP product is NOT calibrated joint feasibility.",
            "Error rates for model-selected designs are conditional on each policy's own chosen distribution; across policies different designs are evaluated.",
            "The seed-cluster bootstrap intervals are exploratory descriptive uncertainties. No deployment-time guarantee, no safety bound, no counterfactual candidate outcome imputation.",
            "Probability bins on tiny phase/iteration groups are high variance. No cross-seed calibrator is inserted into PR58's frozen optimization.",
            "These are custom mathematical hierarchical tasks, NOT aircraft or RCAIDE and NOT full official benchmark replicas.",
            "No further tweaking of PR54's risk threshold should be described as confirmed generalization from these historical results.",
        ],
    }
    if len(rows)!=report["expected_original_receipts"]:
        raise RuntimeError("PR58 missing genuine charged online decisions")
    if git_blob_sha1(source)!=INPUT_SHA1:
        raise RuntimeError("Original PR58 report changed during scoring")
    output.mkdir(parents=True,exist_ok=False)
    with (output/"selected_risk_receipts.csv").open("x",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        w.writeheader()
        w.writerows({key:r[key] for key in FIELDS} for r in rows)
    with (output/"risk_overconfidence_report.json").open("x",encoding="utf8") as f:
        json.dump(report,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
    return report
