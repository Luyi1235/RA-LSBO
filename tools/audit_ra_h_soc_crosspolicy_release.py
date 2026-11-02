"""Independent read-only audit of RA-LSBO-H 24-prefix cross-policy SOC panel.

No RA-LSBO imports, GP fits, VAE training, acquisition optimization, aircraft
evaluator or RCAIDE. Recomputes pointwise likelihood/Brier scores directly
from the ORIGINAL 856-call Windows observations, not the new panel's labels.

Usage: python tools/audit_ra_h_soc_crosspolicy_release.py PANEL_ZIP PILOT_ZIP NEW_DIR
"""
from __future__ import annotations

from collections import defaultdict, Counter
from hashlib import sha256
from pathlib import Path, PurePosixPath
from statistics import median
import csv
import json
import math
import sys
import zipfile

import numpy as np
from scipy.special import ndtr, log_ndtr

PANEL_SHA="a8b177391b5c6fdbbd2f8a00f60f6f2501191f0c87341762c5daef825b5cbe01"
PILOT_SHA="3d9cb4f684bbc9198c523bb44bf466463c2f85053d87ca1770eca9111b48fed1"
PANEL_CODE_SHA="4e864d400b363cdb06821b3872cb2d4c7a039958"
ORIGINAL_CODE_SHA="b219163fc565dfc32876f38f0bbf5f4e65abbd36"
CASES=("aircraft_ordinary","aircraft_sparse")
SEEDS=(9961,9962,9963,9964)
TIMES=(1,9,17)
POLICIES=("random","onehot_stable","corpus_stable","hybrid_onehot","hybrid_guard")
VIEWS=("hierarchical_onehot","frozen_corpus_latent")


def hash_path(path):
    h=sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()


def almost(a,b,*,atol=2e-7,rtol=1e-8):
    return math.isclose(float(a),float(b),abs_tol=atol,rel_tol=rtol)


def log_interval_probability(lo,hi):
    if not lo<hi:
        raise ValueError("Invalid interval")
    if lo==-math.inf:return float(log_ndtr(hi))
    if hi==math.inf:return float(log_ndtr(-lo))
    if lo>=0:
        large,small=float(log_ndtr(-lo)),float(log_ndtr(-hi))
    else:
        large,small=float(log_ndtr(hi)),float(log_ndtr(lo))
    return large+float(np.log(-np.expm1(small-large)))


class SealedZip:
    def __init__(self,path,expected_hash,file_count,requirements):
        self.path=Path(path)
        if hash_path(path)!=expected_hash:
            raise ValueError("Archived source ZIP SHA-256 differs: "+str(path))
        self.z=zipfile.ZipFile(path)
        entries=[p.filename for p in self.z.infolist() if not p.is_dir()]
        if len(entries)!=file_count or len(set(entries))!=file_count:
            raise ValueError(f"Unexpected sealed member count {len(entries)} != {file_count}")
        for name in entries:
            p=PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name:
                raise ValueError("Unsafe archive member path")
        self.names=set(entries)
        roots=[]
        for name in self.names:
            if not name.endswith("plan.json"):continue
            pre=name[:-len("plan.json")]
            if all(pre+x in self.names for x in requirements):
                roots.append(pre)
        if len(roots)!=1:
            raise ValueError("Ambiguous archive root: "+str(roots))
        self.root=roots[0]

    def raw(self,name):
        if self.root+name not in self.names:
            raise ValueError("Missing archive member: "+name)
        return self.z.read(self.root+name)

    def obj(self,name):
        return json.loads(self.raw(name))

    def jsonl(self,name):
        return [json.loads(x) for x in self.raw(name).splitlines() if x.strip()]

    def __enter__(self):return self
    def __exit__(self,*a):self.z.close()


def outcome_SOC(record):
    constraints=record["observation"]["constraints"]
    matches=[v for name,v in constraints if name=="soc"]
    if len(matches)!=1:
        raise ValueError("Missing or duplicate SOC output in frozen evaluator record")
    return matches[0]


def actual_state(o):
    if o["kind"]=="exact":
        return o["value"]<=0
    if o["kind"]=="interval":
        if o["upper"] is not None and o["upper"]<=0:
            return True
        if (o["lower"] is not None
            and (o["lower"]>0 or
                 o["lower"]==0 and not o["lower_inclusive"])):
            return False
    return None


def independent_score(mu,var,obs):
    if not math.isfinite(mu) or not math.isfinite(var) or var<=0:
        raise ValueError("Nonfinite or degenerate saved prediction")
    sd=math.sqrt(var)
    probability=float(ndtr(-mu/sd))
    state=actual_state(obs)
    out={"feasibility_p":probability,
         "resolved_feasible":state,
         "brier":(probability-float(state))**2 if state is not None else None}
    if obs["kind"]=="exact":
        z=(obs["value"]-mu)/sd
        nll=.5*math.log(2*math.pi*var)+.5*z*z
        out.update(kind="exact",score_nll=nll,central_95_covered=abs(z)<=1.959963984540054)
    elif obs["kind"]=="interval":
        lower=-math.inf if obs["lower"] is None else (obs["lower"]-mu)/sd
        upper=math.inf if obs["upper"] is None else (obs["upper"]-mu)/sd
        out.update(kind="interval",score_nll=-log_interval_probability(lower,upper),
                   central_95_covered=None)
    else:raise ValueError("Unexpected missing SOC in scored frozen panel")
    if not math.isfinite(out["score_nll"]):
        raise ValueError("Independent proper NLL became nonfinite")
    return out


def check_pair(actual,pred,view):
    if pred.get("status")!="ok":
        raise ValueError("Claimed 120 paired records have invalid GP predictions "+view)
    for k in ("score_nll","feasibility_p","brier"):
        value=actual[k]
        if value is None:
            if pred.get(k) is not None:raise ValueError("Invented Brier evidence")
        elif pred.get(k) is None or not almost(value,pred[k],atol=1e-7):
            raise ValueError(f"Independent historical {k} differs for view {view}: {value} != {pred.get(k)}")
    if actual["kind"]=="exact":
        if pred.get("central_95_covered") is not actual["central_95_covered"]:
            raise ValueError("95% interval indicator changed")
    elif pred.get("central_95_covered") is not None:
        raise ValueError("Censored interval falsely has 95% exact coverage")


def write_json(path,payload):
    with Path(path).open("x",encoding="utf8") as f:
        json.dump(payload,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")


def write_csv(path,rows,fields):
    with open(path,"x",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(panel_zip,pilot_zip):
    with SealedZip(panel_zip,PANEL_SHA,87,("plan.json","registration.json","result.json","prefix_attempts.jsonl","paired_score_summary.json")) as panel, \
         SealedZip(pilot_zip,PILOT_SHA,641,("plan.json","registration.json","results.json","calls.jsonl","audit.json")) as pilot:
        plan=panel.obj("plan.json")
        reg=panel.obj("registration.json")
        result=panel.obj("result.json")
        precomputed=panel.obj("paired_score_summary.json")
        records=panel.jsonl("prefix_attempts.jsonl")
        reference=pilot.obj("plan.json")
        pilot_audit=pilot.obj("audit.json")
        if (plan["source_pilot_zip_sha256"]!=PILOT_SHA
            or plan["source_pilot_code_commit"]!=ORIGINAL_CODE_SHA
            or plan["analysis_code_commit"]!=PANEL_CODE_SHA
            or reg!={"plan_sha256":sha256(panel.raw("plan.json")).hexdigest(),
                     "maximum_GP_kernel_candidate_attempts":384}):
            raise ValueError("Frozen source identity or immutable panel registration changed")
        if (reference["source_commit"]!=ORIGINAL_CODE_SHA
            or pilot_audit.get("passed") is not True
            or pilot_audit.get("settled_calls")!=856):
            raise ValueError("Source Windows pilot not the registered fully audited 856-call evidence")
        if (result.get("passed") is not True
            or result.get("GP_head_fits")!=48
            or result.get("GP_kernel_candidates")!=384
            or result.get("all_original_onehot_sentinels_reproduced") is not True
            or result.get("frozen_archive_sha256_after")!=PILOT_SHA
            or not all(result.get(k) is True for k in (
                "no_aircraft_evaluations","no_VAE_training",
                "no_optimizer_invocation","no_RCAIDE"))):
            raise ValueError("Full successful GP-only panel contract not established")
        keys={(c,s,t) for c in CASES for s in SEEDS for t in TIMES}
        if (len(records)!=24 or len(plan["groups"])!=24
            or {(x["case"],x["seed"],x["iteration"]) for x in records}!=keys):
            raise ValueError("24 frozen prefix identities not complete")
        if (sum(len(x["targets"]) for x in records)!=120):
            raise ValueError("Not 120 paired policy observations")
        rows=[]
        fit_rows=[]
        distinct_occurrences=0
        observed=Counter()
        old=Counter()
        for expected,rec in zip(plan["groups"],records):
            case,seed,t=rec["case"],rec["seed"],rec["iteration"]
            if ((case,seed,t)!=(expected["case"],expected["seed"],expected["iteration"])
                or rec["training_digest"]!=expected["training_digest"]
                or rec["training_rows"]!=12+t
                or rec["checkpoint_iteration"]!=t-1
                or rec["original_onehot_forecast_sentinel"].get("onehot_reproduced") is not True):
                raise ValueError("Frozen prefix, encoder chronology or historical forecast sentinel differs")
            # Retain exact sorted five arms rather than rows independent of labels.
            if set(x["policy"] for x in rec["targets"])!=set(POLICIES):
                raise ValueError("Target-policy roster differs")
            if len(rec["targets"])!=5:
                raise ValueError("Five policy targets are required on every common prefix")
            source_initial=pilot.obj(f"optimization/{case}/{seed}_hybrid_onehot/manifest.json")["initial"]
            source_prev=pilot.jsonl(f"optimization/{case}/{seed}_hybrid_onehot/observations.jsonl")[:t]
            if digest(source_initial+source_prev)!=rec["training_digest"]:
                raise ValueError("GP panel prefix includes or excludes historical observations")
            already={digest(x["candidate"]) for x in source_initial+source_prev}
            distinct_occurrences+=len({x["candidate_id"] for x in rec["targets"]})
            for view in VIEWS:
                fit=rec["head_fits"][view]
                attempts=fit["candidate_attempts"]
                if (fit.get("status")!="ok" or len(attempts)!=8
                    or not all(a.get("status")=="ok" for a in attempts)):
                    raise ValueError("An EP candidate did not succeed despite 384/384 success claim")
                fit_rows.append({"case":case,"seed":seed,"iteration":t,"view":view,
                                 "fit_seconds":fit["fit_seconds"],
                                 "kernel_c":fit["selected_kernel"]["continuous_length"],
                                 "kernel_d":fit["selected_kernel"]["discrete_length"]})
            for x in rec["targets"]:
                arm=x["policy"]
                source=pilot.jsonl(f"optimization/{case}/{seed}_{arm}/observations.jsonl")[t]
                prop=pilot.jsonl(f"optimization/{case}/{seed}_{arm}/proposals.jsonl")[t]
                if (digest(source)!=x["original_observation_digest"]
                    or source["candidate"]!=prop["candidate"]
                    or digest(prop["candidate"])!=x["candidate_id"]
                    or x["frozen_outcome_digest"]!=digest(source)
                    or x.get("excluded_as_training_design") is not False
                    or x["candidate_id"] in already):
                    raise ValueError("Frozen candidate outcome or historical anti-leakage relationship changed")
                obs=outcome_SOC(source)
                if obs["kind"]!=x["soc_evidence_kind"]:
                    raise ValueError("Reported SOC kind differs from original historical settled label")
                observed[obs["kind"]]+=1
                paired=x["predictions"]
                for view in VIEWS:
                    pred=paired[view]
                    actual=independent_score(float(pred["mean"]),float(pred["variance"]),obs)
                    check_pair(actual,pred,view)
                one=paired["hierarchical_onehot"]
                lat=paired["frozen_corpus_latent"]
                old["resolved"]+=int(one["resolved_feasible"] is not None)
                rows.append({
                    "case":case,"seed":seed,"iteration":t,
                    "policy":arm,"candidate_id":x["candidate_id"],
                    "kind":obs["kind"],
                    "onehot_mean":one["mean"],"latent_mean":lat["mean"],
                    "onehot_var":one["variance"],"latent_var":lat["variance"],
                    "onehot_nll":one["score_nll"],"latent_nll":lat["score_nll"],
                    "latent_minus_onehot_nll":lat["score_nll"]-one["score_nll"],
                    "onehot_brier":one["brier"],"latent_brier":lat["brier"],
                    "onehot_95hit":one["central_95_covered"],
                    "latent_95hit":lat["central_95_covered"]})
        if (dict(observed)!={"exact":89,"interval":31} or distinct_occurrences!=116
            or len(rows)!=120 or old["resolved"]!=120):
            raise ValueError("Expected 89 exact/31 interval, 116 unique within-prefix candidates, 120 resolved feasibility")
        for kind in ("exact","interval"):
            z=[x for x in rows if x["kind"]==kind]
            # Independently recompute the descriptive figures from raw forecasts.
            target=("exact" if kind=="exact" else "interval")
            print(json.dumps({"kind":target,"n":len(z),
              "onehot_nll_mean":sum(r["onehot_nll"] for r in z)/len(z),
              "latent_nll_mean":sum(r["latent_nll"] for r in z)/len(z),
              "mean_delta":sum(r["latent_minus_onehot_nll"] for r in z)/len(z),
              "onehot_95_covered":sum(r["onehot_95hit"] is True for r in z),
              "latent_95_covered":sum(r["latent_95hit"] is True for r in z)}))
        # Reproduce the output's precomputed seed+policy table as another
        # defense against misaligned model/target scores. Dataset is
        # ordered by case, SOC evidence type, policy, seed.
        key_groups=defaultdict(list)
        for r in rows:
            key_groups[(r["case"],r["kind"],r["policy"],r["seed"])].append(
                r["latent_minus_onehot_nll"])
        for (case,kind,arm,seed),diffs in key_groups.items():
            group=precomputed["groups"][f"{case}/{kind}/{arm}"]
            entry=group["per_seed"][str(seed)]
            if (entry["n"]!=len(diffs) or not almost(entry["mean_latent_minus_onehot_NLL"],
                            sum(diffs)/len(diffs),atol=1e-7)):
                raise ValueError("Precomputed per-seed/policy NLL difference misaligned")
        # Bootstrap/p-value absent intentionally: four *adaptive* seed
        # clusters per task do not justify an independent population claim.
        summary=[]
        group=defaultdict(list)
        for r in rows:
            group[(r["case"],r["kind"],r["seed"])].append(r)
        for case in CASES:
            for kind in ("exact","interval"):
                cases=[r for r in rows if r["case"]==case and r["kind"]==kind]
                seeds={}
                for seed in SEEDS:
                    subset=group.get((case,kind,seed),[])
                    if subset:
                        diffs=[r["latent_minus_onehot_nll"] for r in subset]
                        seeds[str(seed)]={
                            "n":len(subset),"mean_delta":sum(diffs)/len(diffs),
                            "median_delta":median(diffs),
                            "onehot_better_count":sum(v>0 for v in diffs),
                            "latent_better_count":sum(v<0 for v in diffs)}
                summary.append({"case":case,"kind":kind,
                                "n":len(cases),
                                "mean_delta_pooled":(sum(r["latent_minus_onehot_nll"] for r in cases)/len(cases)) if cases else None,
                                "n_independent_seed_clusters":len(seeds),
                                "per_seed":seeds,
                                "seed_mean_delta":(sum(x["mean_delta"] for x in seeds.values())/len(seeds)) if seeds else None,
                                "onehot_better_rows":sum(r["latent_minus_onehot_nll"]>0 for r in cases),
                                "latent_better_rows":sum(r["latent_minus_onehot_nll"]<0 for r in cases)})
        # Influence analysis stays descriptive; it exposes whether the
        # average is carried by an isolated extreme censored observation.
        influence=[]
        for case in CASES:
            for kind in ("exact","interval"):
                group_rows=[r for r in rows if r["case"]==case and r["kind"]==kind]
                diffs=[r["latent_minus_onehot_nll"] for r in group_rows]
                brier_diffs=[r["latent_brier"]-r["onehot_brier"] for r in group_rows]
                largest=max(diffs) if diffs else None
                influence.append({
                    "case":case,"kind":kind,"n":len(diffs),
                    "median_latent_minus_onehot_nll":median(diffs) if diffs else None,
                    "onehot_lower_nll_count":sum(v>0 for v in diffs),
                    "latent_lower_nll_count":sum(v<0 for v in diffs),
                    "mean_delta_after_removing_largest_positive":(
                        (sum(diffs)-largest)/(len(diffs)-1)
                        if len(diffs)>1 else None),
                    "largest_positive_delta":largest,
                    "largest_delta_share_of_net_sum":(
                        largest/sum(diffs) if diffs and sum(diffs)>0 else None),
                    "onehot_brier_mean":sum(r["onehot_brier"] for r in group_rows)/len(group_rows) if group_rows else None,
                    "latent_brier_mean":sum(r["latent_brier"] for r in group_rows)/len(group_rows) if group_rows else None,
                    "mean_delta_brier":sum(brier_diffs)/len(brier_diffs) if brier_diffs else None,
                    "duplicate_policy_records":len(group_rows)-len({
                        (r["case"],r["seed"],r["iteration"],r["candidate_id"])
                        for r in group_rows}),
                    "sample_dependence":"Four independent seeds maximum; repeated and adaptively selected target designs",
                })
        integrity={"panel_zip_sha256":PANEL_SHA,"pilot_zip_sha256":PILOT_SHA,
                    "audited_panel_files":87,"audited_pilot_files":641,
                    "prefixes":24,"GP_heads":48,"GP_kernel_attempts":384,
                    "onehot_historical_sentinels":24,"candidate_policy_records":120,
                    "unique_within_prefix":116,
                    "exact":89,"interval":31,"resolved_feasibility":120,
                    "recalculated_from_original_truth":True,
                    "new_model_fits":0,"new_truth_calls":0,"new_optimizer_calls":0}
        return integrity,rows,fit_rows,summary,influence


def main():
    if len(sys.argv)!=4:raise SystemExit("usage: audit_ra_h_soc_crosspolicy_release.py PANEL_ZIP PILOT_ZIP NEW_DIR")
    out=Path(sys.argv[3])
    if out.exists():raise FileExistsError("Analysis output must be a NEW write-once directory")
    integrity,rows,fits,summary,influence=audit(sys.argv[1],sys.argv[2])
    out.mkdir(parents=True)
    write_json(out/"integrity.json",integrity)
    write_json(out/"case_seed_scores.json",summary)
    write_json(out/"stratified_influence_and_brier.json",influence)
    write_csv(out/"paired_120_historical_recomputed.csv",rows,list(rows[0]))
    write_csv(out/"fit_48_heads.csv",fits,list(fits[0]))
    print(json.dumps({"independent_audit_passed":True,"integrity":integrity,
                      "case_seed_scores":summary,
                      "stratified_influence_and_brier":influence},allow_nan=False))


if __name__=="__main__":
    main()
