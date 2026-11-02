"""Execute the registered RA-LSBO-H common-prefix SOC cross-policy panel.

Only original 856-call Release evidence is used. At 24 preregistered
anchor prefixes, load an already trained, frozen 2,000-corpus VAE checkpoint
(no training) and build onehot features without VAE. Fit exactly eight
stable-tail EP GP kernel candidates for each of the two SOC views (384
candidate attempts total). Predict up to five saved next-design candidates
per prefix, then and ONLY then inspect their historically settled labels.

The saved original hybrid_onehot SOC forecast is independently reproduced
at the same prefix and candidate as a stringent no-leakage identity gate.
Never launches any aircraft evaluator, optimizer, VAE training or RCAIDE.

Commands:
  python tools/ra_h_soc_panel_run.py plan --archive ZIP --output NEW_DIR
  python tools/ra_h_soc_panel_run.py plan --archive ZIP --output NEW_DIR --cross-platform-check
  python tools/ra_h_soc_panel_run.py run --archive ZIP --output EXISTING_PLAN_DIR
  python tools/ra_h_soc_panel_run.py audit --archive ZIP --output COMPLETE_DIR
"""
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.special import ndtr, log_ndtr

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.ra_h_soc_panel_protocol import (
    RELEASE_SHA, ORIGINAL_CODE_SHA,
    CASES, SEEDS, TARGET_ARMS, ANCHOR, STEPS, MODEL_VIEWS,
    SealedPilot, read_only_registration, hash_file, digest, save_exclusive,
)
from ra_lsbo.experiments.core.protocol import BASE
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.features import load_base
from ra_lsbo.structured_encoder import hierarchy_space
from ra_lsbo.structure_corpus import (
    CorpusRepresentation, native_observations, feature_snapshot)
from ra_lsbo.partial_bo.models import PrefixFeatures, log_interval_probability
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.experiments.simplified_aircraft.dualview_optimizer import (
    StableGroupedHead, ModelNumericalFailure)

ROOT=Path(__file__).resolve().parents[1]
VIEWS=("hierarchical_onehot","frozen_corpus_latent")
UNEXPECTED_LARGEST_KERNEL_FITS=384


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _failure(label,exc):
    return {
        "view":label,"status":"failed_all_kernels",
        "exception":type(exc).__name__,
        "candidate_attempts":exc.failures,
        "fit_seconds":None,
        "selected_kernel":None,
        "feature_identity":None,
    }


def _fit(view,features,history,seed,problem):
    start=time.perf_counter()
    input_X=features.transform([r.candidate for r in history])
    evidence=[r.observation.constraint("soc") for r in history]
    fixed_seed=seed+problem.required_constraints.index("soc")+1
    try:
        head=StableGroupedHead(
            input_X,evidence,
            continuous_count=len(problem.space.continuous),
            seed=fixed_seed,
            name="constraint:soc" if view=="hierarchical_onehot"
                 else "constraint:soc:latent",
            continuous_grid=(.05,.15,.4,1.),
            discrete_grid=(.5,1.5))
    except ModelNumericalFailure as error:
        return None,{
            "view":view,"status":"failed_all_kernels",
            "candidate_attempts":error.failures,
            "fit_seconds":float(time.perf_counter()-start),
            "selected_kernel":None,
            "feature_identity":features.identity,
        }
    return head,{
        "view":view,"status":"ok",
        "candidate_attempts":head.candidates,
        "fit_seconds":float(time.perf_counter()-start),
        "selected_kernel":head.candidates[head.selected_index],
        "feature_identity":features.identity,
        "output_center":head.center,"output_scale":head.scale,
    }


def _score(analysis, observation):
    """Outcome evaluated strictly AFTER both predictions were constructed."""
    if analysis is None:
        return {"status":"head_unavailable","score_nll":None}
    mu=float(analysis["mean"])
    var=float(analysis["variance"])
    if not all(map(math.isfinite,(mu,var))) or var<=0:
        return {"status":"zero_or_invalid_variance","score_nll":None}
    sd=math.sqrt(var)
    p=float(ndtr(-mu/sd))
    state=observation.constraint_satisfaction()
    result={"status":"ok","observation_kind":observation.kind,
            "feasibility_p":p,"resolved_feasible":state,
            "brier":(p-float(state))**2 if state is not None else None}
    if observation.kind=="exact":
        z=(observation.value-mu)/sd
        nll=.5*math.log(2*math.pi*var)+.5*z*z
        result.update(
            score_type="exact_SOC_density_NLL",
            standardized_error=z if math.isfinite(z) else None,
            central_95_covered=(abs(z)<=1.959963984540054),
            score_nll=nll if math.isfinite(nll) else None,
            nll_nonfinite=not math.isfinite(nll),
        )
    elif observation.kind=="interval":
        lo=(-math.inf if observation.lower is None else (observation.lower-mu)/sd)
        hi=(math.inf if observation.upper is None else (observation.upper-mu)/sd)
        if not lo<hi:
            raise ValueError("Invalid historical standardized SOC bounds")
        logprob=(
            float(log_ndtr(-lo)) if hi==math.inf else
            float(log_ndtr(hi)) if lo==-math.inf else
            log_interval_probability(lo,hi))
        nll=-float(logprob)
        result.update(score_type="interval_SOC_mass_NLL",
                      central_95_covered=None, standardized_error=None,
                      score_nll=nll if math.isfinite(nll) else None,
                      nll_nonfinite=not math.isfinite(nll))
    else:
        result.update(status="unknown_SOC_evidence",score_type=None,
                      score_nll=None,central_95_covered=None,standardized_error=None)
    return result


def _checkpoint_extract(z,group,out):
    path=group["checkpoint_relative_path"]
    cache=out/"checkpoint_bytes"/group["case"]/str(group["seed"])/f"iter_{group['checkpoint_iteration']:03d}"
    cache.mkdir(parents=True,exist_ok=False)
    for filename,expected in group["checkpoint_files_sha256"].items():
        if filename not in {"metadata.json","state.npz","diagnostics.json"}:
            raise ValueError("Non-whitelisted sealed checkpoint member")
        raw=z.raw(path+filename)
        if sha256(raw).hexdigest()!=expected:
            raise ValueError("Frozen checkpoint member identity changed")
        with (cache/filename).open("xb") as stream:
            stream.write(raw)
    return cache


def _frozen_views(z,group,out,history):
    case,seed,t=group["case"],group["seed"],group["iteration"]
    problem=problem_for(case)
    if any(x.observation.contract_id!=problem.contract_id for x in history):
        raise ValueError("Physical contract differs from frozen prefix evidence")
    n=group["checkpoint_iteration"]
    modules,_=load_base(BASE)
    from dataclasses import asdict
    space=hierarchy_space(modules,[asdict(v) for v in problem.space.variables])
    prior_training=history[:12+n]
    path=_checkpoint_extract(z,group,out)
    obs=native_observations(prior_training,modules,space)
    session=CorpusRepresentation.load_checkpoint(path,space,obs)
    # This reconstructs an already trained encoder. It does not call fit().
    latent=feature_snapshot(problem,BASE,history,session)
    onehot=PrefixFeatures(problem,BASE,history,seed,False,30)
    saved_latent=group["saved_anchor_latent_feature_identity"]
    saved_onehot=group["saved_anchor_onehot_feature_identity"]
    if saved_latent is not None and latent.identity!=saved_latent:
        raise RuntimeError("Frozen latent encoder prefix identity did not reproduce")
    if saved_onehot is not None and onehot.identity!=saved_onehot:
        raise RuntimeError("Frozen onehot prefix identity did not reproduce")
    return problem,{"hierarchical_onehot":onehot,"frozen_corpus_latent":latent}


def _original_forecast_sentinel(z,group,onehot,record):
    if not group["anchor_original_forecast_available"]:
        return {"original_forecast_available":False}
    t=group["iteration"]
    metadata=z.trial(group["case"],group["seed"],ANCHOR)[1][t]["metadata"]
    saved=metadata["forecasts"]["constraints"]["soc"]
    index=metadata["selected_index"]
    if metadata["pool"][index]!=record["candidate"]:
        raise ValueError("Historical selected-index/actual-candidate mismatch")
    if onehot is None:
        raise RuntimeError("Original successful onehot SOC head failed to reproduce")
    import numpy as np
    # Onehot head only: original hybrid_onehot uses the same onehot GP
    # for SOC under this source commit and training prefix.
    prediction=onehot.predict(
        # caller has already encoded the original selected design
        record["onehot_encoded"]
    )
    mu,var=float(prediction["mean"][0]),float(prediction["variance"][0])
    orig_mu,orig_var=float(saved["mean"][index]),float(saved["variance"][index])
    if (not np.isclose(mu,orig_mu,atol=1e-7,rtol=1e-6)
        or not np.isclose(var,orig_var,atol=1e-9,rtol=1e-5)):
        raise RuntimeError(
            "Original Windows prefix forecast did not reproduce: "
            f"mean {mu} vs {orig_mu}; variance {var} vs {orig_var}")
    return {"original_forecast_available":True,
            "onehot_reproduced":True,
            "original_mean":orig_mu,"original_variance":orig_var,
            "recomputed_mean":mu,"recomputed_variance":var}


def _prefix(z,group,out):
    case,seed,t=group["case"],group["seed"],group["iteration"]
    manifest,anchor_props,anchor_online=z.trial(case,seed,ANCHOR)
    prefix_dicts=manifest["initial"]+anchor_online[:t]
    if len(prefix_dicts)!=group["training_rows"] or digest(prefix_dicts)!=group["training_digest"]:
        raise ValueError("Frozen history prefix differs")
    history=[DesignRecord.from_dict(x) for x in prefix_dicts]
    problem,features=_frozen_views(z,group,out,history)
    targets=[]
    seen=set(digest(x["candidate"]) for x in prefix_dicts)
    for entry in group["targets"]:
        arm=entry["policy"]
        _,proposals,online=z.trial(case,seed,arm)
        target=proposals[t]["candidate"]
        if (digest(target)!=entry["candidate_id"]
            or digest(online[t])!=entry["frozen_outcome_digest"]
            or (digest(target) in seen)!=entry["excluded_as_training_design"]):
            raise ValueError("Frozen heldout candidate or exclusion rule changed")
        targets.append({**entry,"candidate":target})
    eligible={x["candidate_id"]:x["candidate"] for x in targets
              if not x["excluded_as_training_design"]}
    predicted={}
    trained={}
    for view in VIEWS:
        head,diag=_fit(view,features[view],history,seed,problem)
        trained[view]=diag
        if len(diag["candidate_attempts"])!=8:
            raise RuntimeError("Unexpected GP kernel attempt budget")
        if head is not None and eligible:
            xs=features[view].transform(list(eligible.values()))
            try:
                preds=head.predict(xs)
                predicted[view]={
                    key:{"mean":float(preds["mean"][i]),
                         "variance":float(preds["variance"][i])}
                    for i,key in enumerate(eligible)}
            except ModelNumericalFailure as error:
                diag["prediction_status"]="failed"
                diag["prediction_error"]=error.failures
                predicted[view]={}
        else:
            predicted[view]={}
    if group["anchor_original_forecast_available"]:
        kernel=trained["hierarchical_onehot"].get("selected_kernel")
        expected=group["saved_anchor_onehot_selected_kernel"]
        if (kernel is None or
            [kernel["continuous_length"],kernel["discrete_length"]]!=expected):
            raise RuntimeError("Original saved onehot SOC kernel was not reproduced")
    anchor_target=next(x for x in targets if x["policy"]==ANCHOR)
    reference={"candidate":anchor_target["candidate"],
               "onehot_encoded":features["hierarchical_onehot"].transform(
                   [anchor_target["candidate"]])}
    reproduction=_original_forecast_sentinel(
        z,group,
        None if trained["hierarchical_onehot"]["status"]!="ok" else
            # Only the selected full model is needed to check the sentinel;
            # already computed forecasts suffice and prevent an extra GP fit.
            _OriginalPredictionProxy(predicted["hierarchical_onehot"].get(
                anchor_target["candidate_id"])),
        {**reference})
    # Held-out outcomes are read only below, AFTER both heads were fully
    # fitted and their prediction dictionaries fixed for this group.
    results=[]
    for entry in targets:
        arm=entry["policy"]
        observed=z.trial(case,seed,arm)[2][t]
        record=DesignRecord.from_dict(observed)
        observation=record.observation.constraint("soc")
        row={k:v for k,v in entry.items() if k!="candidate"}
        row["soc_evidence_kind"]=observation.kind
        row["excluded_as_training_design"]=bool(entry["excluded_as_training_design"])
        row["original_observation_digest"]=digest(observed)
        row["predictions"]={}
        for view in VIEWS:
            pred=predicted[view].get(entry["candidate_id"])
            if entry["excluded_as_training_design"]:
                row["predictions"][view]={"status":"excluded_training_overlap"}
            else:
                scored=_score(pred,observation)
                row["predictions"][view]={**(pred or {}),**scored}
        results.append(row)
    return {
        "case":case,"seed":seed,"iteration":t,
        "training_rows":len(history),"training_digest":group["training_digest"],
        "checkpoint_iteration":group["checkpoint_iteration"],
        "features":{key:f.identity for key,f in features.items()},
        "head_fits":trained,"original_onehot_forecast_sentinel":reproduction,
        "unique_eligible_targets":len(eligible),"targets":results,
        "new_aircraft_calls":0,"new_VAE_fits":0,"new_optimizer_calls":0,
    }


class _OriginalPredictionProxy:
    """Sentinel adapter using the already calculated GP prediction, not refit."""
    def __init__(self, prediction):
        self.prediction=prediction

    def predict(self,_encoded):
        if self.prediction is None:
            raise RuntimeError("Missing anchor candidate prediction in GP-only sentinel")
        return {"mean":[self.prediction["mean"]],"variance":[self.prediction["variance"]]}


def _paired_report(records):
    by_group=defaultdict(list)
    unavailable=0
    for group in records:
        for item in group["targets"]:
            if item["excluded_as_training_design"]:
                continue
            for kind in ("exact","interval"):
                if item["soc_evidence_kind"]!=kind:continue
                left=item["predictions"]["frozen_corpus_latent"]
                right=item["predictions"]["hierarchical_onehot"]
                if (left.get("status")!="ok" or right.get("status")!="ok"
                    or left.get("score_nll") is None or right.get("score_nll") is None):
                    unavailable+=1
                    continue
                by_group[(group["case"],kind,item["policy"],group["seed"])].append(
                    left["score_nll"]-right["score_nll"])
    across=defaultdict(dict)
    for (case,kind,policy,seed),values in by_group.items():
        across[(case,kind,policy)][str(seed)]={
            "n":len(values),"mean_latent_minus_onehot_NLL":sum(values)/len(values)}
    return {
        "schema":"ra_h_same_prefix_crosspolicy_scores_v1",
        "direction":"latent NLL - onehot NLL; positive favors onehot",
        "groups":{f"{case}/{kind}/{policy}":{
            "per_seed":seed_dict,
            "independent_seed_clusters":len(seed_dict),
            "pooled_matched_records":sum(x["n"] for x in seed_dict.values()),
            "seed_mean_of_mean_differences":sum(
                x["mean_latent_minus_onehot_NLL"] for x in seed_dict.values())/len(seed_dict),
            "no_confirmatory_inference":True}
            for (case,kind,policy),seed_dict in sorted(across.items())},
        "unavailable_matched_scores":unavailable,
        "scoring_note":"Exact density and interval mass NLL never pooled; heldout policies are adaptive and correlated",
    }


def _load_plan(archive,out):
    out=Path(out)
    plan=_json(out/"plan.json")
    reg=_json(out/"registration.json")
    if reg!={"plan_sha256":hash_file(out/"plan.json"),
             "maximum_GP_kernel_candidate_attempts":384}:
        raise ValueError("Analysis registration file does not match immutable plan")
    if plan.get("cross_platform_preview_only") or not plan.get("contract_match"):
        raise ValueError("Plan-only Linux preview cannot authorize GP fits")
    expected=read_only_registration(archive,cross_platform_check=False)
    if plan!=expected:
        raise ValueError("Plan, code, source or environment identity has changed")
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("plan","run","audit"))
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cross-platform-check",action="store_true",
                   help="Non-authorizing Linux plan-only preview, no GP fits")
    args=p.parse_args()
    if args.cross_platform_check and args.action!="plan":
        p.error("--cross-platform-check is plan-only, never authorizes model fitting")
    if args.action=="plan":
        if args.output.exists():
            raise FileExistsError("Never overwrite existing analysis output")
        spec=read_only_registration(args.archive,
                                    cross_platform_check=args.cross_platform_check)
        args.output.mkdir(parents=True)
        save_exclusive(args.output/"plan.json",spec)
        save_exclusive(args.output/"registration.json",{
            "plan_sha256":hash_file(args.output/"plan.json"),
            "maximum_GP_kernel_candidate_attempts":384})
        print(json.dumps({"registered_prefixes":24,"target_records":120,
                          "GP_kernel_candidate_limit":384,
                          "preview_only":args.cross_platform_check}))
        return
    plan=_load_plan(args.archive,args.output)
    if args.action=="audit":
        if not (args.output/"result.json").is_file():
            raise ValueError("No completed GP-only panel result")
        result=_json(args.output/"result.json")
        records=[json.loads(x) for x in (args.output/"prefix_attempts.jsonl").read_bytes().splitlines() if x.strip()]
        if (len(records)!=24 or not result.get("passed")
            or result.get("GP_kernel_candidates")!=384):
            raise ValueError("Incomplete GP-only panel")
        for actual,expected in zip(records,plan["groups"]):
            if ((actual["case"],actual["seed"],actual["iteration"])!=
                (expected["case"],expected["seed"],expected["iteration"])
                or actual["training_digest"]!=expected["training_digest"]
                or len(actual["targets"])!=5):
                raise ValueError("A saved prefix/target group changed")
            for v in VIEWS:
                if len(actual["head_fits"][v]["candidate_attempts"])!=8:
                    raise ValueError("Missing EP kernel attempt")
        if hash_file(args.archive)!=RELEASE_SHA:
            raise ValueError("Original sealed pilot was mutated")
        print(json.dumps({"audited_prefixes":24,"kernel_attempts":384,
                          "new_aircraft_evaluations":0,"VAE_fits":0}))
        return

    out=args.output
    if (out/"run.lock").exists() or (out/"prefix_attempts.jsonl").exists():
        raise FileExistsError("Never reuse an attempted or completed model panel")
    # Create immutable lock before any checkpoint reconstruction or GP fit.
    with (out/"run.lock").open("x",encoding="utf8") as f:
        f.write("single-pass frozen RA-LSBO-H crosspolicy SOC GP panel\n")
    rows=[]
    with SealedPilot(args.archive) as z, (out/"prefix_attempts.jsonl").open("x",encoding="utf8") as journal:
        for g in plan["groups"]:
            row=_prefix(z,g,out)
            rows.append(row)
            journal.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
            journal.flush()
            os.fsync(journal.fileno())
    result={
        "schema":"ra_h_soc_commonprefix_crosspolicy_results_v1",
        "passed":True,"prefix_fits":24,
        "GP_kernel_candidates":sum(
            len(row["head_fits"][view]["candidate_attempts"])
            for row in rows for view in VIEWS),
        "GP_head_fits":48,
        "GP_fit_seconds":sum(
            row["head_fits"][view]["fit_seconds"] for row in rows for view in VIEWS),
        "no_aircraft_evaluations":True,"no_VAE_training":True,
        "no_optimizer_invocation":True,"no_RCAIDE":True,
        "all_original_onehot_sentinels_reproduced":all(
            not r["original_onehot_forecast_sentinel"]["original_forecast_available"]
            or r["original_onehot_forecast_sentinel"]["onehot_reproduced"]
            for r in rows),
        "frozen_archive_sha256_after":hash_file(args.archive),
    }
    if result["GP_kernel_candidates"]!=384 or result["frozen_archive_sha256_after"]!=RELEASE_SHA:
        raise RuntimeError("Registered GP attempt count/source integrity violated")
    save_exclusive(out/"paired_score_summary.json",_paired_report(rows))
    save_exclusive(out/"result.json",result)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
