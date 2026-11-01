"""Single D28->D29 prequential SOC correctness gate, NOT a BO run.

On original Windows/Rcaide_vsp only:
 python tools/prequential_soc_D28_gate.py plan --archive ORIGINAL_INTERRUPTED_ZIP --output UNUSED_DIR
 python tools/prequential_soc_D28_gate.py run --archive ORIGINAL_INTERRUPTED_ZIP --output SAME_DIR

Use 40 prior records to train exactly one old30 VAE and 16 total GP kernel
candidates (8 latent stable-tail, 8 onehot stable-tail). The 41st SOC label
never enters the VAE or GP; it is used only at the final scoring step.
The original old30 historical forecast is read rather than re-fitted.
No simulator, RCAIDE, optimizer or pilot resume calls.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ra_lsbo.partial_bo.models import PrefixFeatures, EvidenceGP
from ra_lsbo.risk_core.mixed_gp import MixedGPSettings
from ra_lsbo.risk_core.observations import OutputObservation
from ra_lsbo.partial_bo.records import DesignRecord
from ra_lsbo.partial_refinement.posterior import group_divisors
from ra_lsbo.experiments.core.protocol import BASE, ROOT
from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
from ra_lsbo.budget import digest
from tools.probe_ep_stability_D29 import load_input, sha256_stream, ARCHIVE_SHA, SOURCE_SHA, SEED, CASE

EXPECTED_D28 = "75aa389992a576bacc3af2c0abe77526e7338bac01c1c12ea2b1525cc6c9ebfa"
GRID = [(c,d) for c in (.05,.15,.4,1.) for d in (.5,1.5)]


def save_new(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def prior_data(archive):
    manifest, proposals, history = load_input(archive)
    proposal = proposals[28]
    target = history[40]
    if (proposal["iteration"] != 28 or proposal["candidate"] != target.candidate
            or proposal["metadata"]["posterior_fit_skipped"]):
        raise ValueError("The prospective D28 forecast is unavailable")
    selected = proposal["metadata"]["selected_index"]
    if proposal["metadata"]["pool"][selected] != target.candidate:
        raise ValueError("Stored original predictive pool selection changed")
    if proposal["metadata"]["feature_identity"] != EXPECTED_D28:
        raise ValueError("D28 original encoder identity changed")
    if target.observation.constraint("soc").kind != "exact":
        raise ValueError("Fixed 41st future SOC observation is not exact")
    pre=history[:40]
    if proposal["prefix_digest"]!=digest([r.to_dict() for r in pre]):
        raise ValueError("D28 prefix identity changed")
    return pre, proposal, target, selected


def protocol(archive):
    if sha256_stream(archive) != ARCHIVE_SHA:
        raise ValueError("Original pilot ZIP identity changed")
    pre, proposal, target, index = prior_data(archive)
    problem = problem_for(CASE)
    if (len(problem.space.continuous) != 6
            or any(r.observation.contract_id != problem.contract_id for r in pre+[target])):
        raise ValueError("Physical contract or feature layout changed; original Windows required")
    head_names = [n for n in problem.required_constraints if n not in problem.analytical_constraints]
    if head_names != ["stall","root_bending","soc","current","voltage","power"]:
        raise ValueError("Response head order changed")
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if subprocess.run(["git","merge-base","--is-ancestor", SOURCE_SHA, rev],cwd=ROOT).returncode:
        raise ValueError("Experiment code is not descended from fixed original")
    return {
        "schema":"fixed_D28_to_D29_SOC_prequential_one_point_v1",
        "original_archive_sha256":ARCHIVE_SHA,"source_commit":SOURCE_SHA,
        "diagnostic_commit":rev,
        "D28_original_feature_identity":EXPECTED_D28,
        "original_prior_prefix":proposal["prefix_digest"],
        "target_candidate_id":target.observation.design_id,
        "future_outcome_sha256":digest(target.to_dict()),
        "case":CASE,"seed":SEED,"iteration":28,
        "training_rows":len(pre),"withheld_rows":1,
        "shared_kernel_grid":GRID,
        "representation_comparison":["stable_tail_latent30","stable_tail_onehot"],
        "reference_forecast":"original_saved_old30_no_refit",
        "max_new_VAE_fits":1,"max_new_GP_candidates":16,
        "new_evaluator_calls":0,"new_RCAIDE_calls":0,"new_BO_calls":0,
        "scope":"one adaptive historical target, only a correctness gate; no independent calibration claim"
    }


def fit_head(training_X, test_X, obs, continuous_count, arm, out_path):
    values=np.asarray([x.value for x in obs if x.kind=="exact"],float)
    scale=max(float(np.median(np.abs(values))),.05) if len(values) else 1.
    normalized=[
        replace(o,value=o.value/scale) if o.kind=="exact" else
        replace(o,lower=None if o.lower is None else o.lower/scale,
                  upper=None if o.upper is None else o.upper/scale)
        if o.kind=="interval" else o for o in obs]
    records=[]
    first=time.perf_counter()
    with Path(out_path).open("x",encoding="utf-8") as stream:
        for j,(c,d) in enumerate(GRID):
            divisor=group_divisors(training_X.shape[1],continuous_count,c,d)
            row={"arm":arm,"kernel":(c,d),"kernel_index":j,"status":None,
                 "output_scale":scale}
            begin=time.perf_counter()
            try:
                model=EvidenceGP(training_X/divisor, normalized,
                    seed=SEED+5,
                    settings=MixedGPSettings(length_scale=1.,exact_noise=1e-3,
                        interval_noise=0.,jitter=1e-8,stable_tail_moments=True))
                evidence=float(model.log_evidence())
                prediction=model.predict(test_X/divisor)
                mu=float(prediction["mean"][0]*scale)
                variance=float(prediction["variance"][0]*scale*scale)
                if not all(math.isfinite(v) for v in (evidence,mu,variance)) or variance<=0:
                    raise FloatingPointError("Invalid predictive distribution")
                row.update(status="ok",log_evidence=evidence,mean=mu,variance=variance,
                           sweeps=model.diagnostics["sweeps"])
            except (np.linalg.LinAlgError,FloatingPointError,RuntimeError) as error:
                row.update(status="failed",error_type=type(error).__name__,
                           error_message=str(error),traceback=traceback.format_exc())
            row["fit_seconds"]=float(time.perf_counter()-begin)
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
            records.append(row)
    viable=[r for r in records if r["status"]=="ok"]
    return {"arm":arm,"candidate_count":len(records),"valid_candidates":len(viable),
            "total_fit_seconds":time.perf_counter()-first,
            "selected":max(viable,key=lambda r:r["log_evidence"]) if viable else None}


def run(archive,out):
    out=Path(out)
    plan=json.loads((out/"plan.json").read_text(encoding="utf8"))
    if plan!=protocol(archive):
        raise ValueError("Plan or environment modified; stop")
    with (out/"diagnostic_execution.lock").open("x") as stream:
        stream.write("fixed-D28-SOC-prequential")
    pre, proposal, target, ix=prior_data(archive)
    problem=problem_for(CASE)
    saved=proposal["metadata"]["forecasts"]["constraints"]["soc"]
    old={"arm":"original_saved_old30","mean":float(saved["mean"][ix]),
         "variance":float(saved["variance"][ix]),"fit_count_this_run":0}
    if old["variance"]<=0 or not all(math.isfinite(v) for v in (old["mean"],old["variance"])):
        raise ValueError("Historical saved GP forecast invalid")
    features=PrefixFeatures(problem,BASE,pre,SEED,True,30)
    if features.identity != EXPECTED_D28:
        raise ValueError("D28 encoder identity did not reproduce; GP fitting prohibited")
    save_new(out/"feature_identity.json",{"latent_D28":features.identity,
                                          "prefix":proposal["prefix_digest"]})
    onehot=PrefixFeatures(problem,BASE,pre,SEED,False,30)
    observations=[x.observation.constraint("soc") for x in pre]
    if (sum(o.kind=="exact" for o in observations)!=16
            or sum(o.kind=="interval" for o in observations)!=24):
        raise ValueError("Expected D28 16 exact+24 interval SOC data")
    trained=[]
    for arm, f in (("stable_tail_latent30",features),
                   ("stable_tail_onehot",onehot)):
        X=f.transform([r.candidate for r in pre])
        Xtest=f.transform([target.candidate])
        if not np.isfinite(X).all() or not np.isfinite(Xtest).all():
            raise ValueError("Nonfinite prefix-only features")
        trained.append(fit_head(X,Xtest,observations,
            len(problem.space.continuous),arm,out/(arm+"_candidates.jsonl")))
    # Only NOW unseal the held-out target for scoring. No future label enters
    # encoder/GP model construction or hyperparameter selection.
    label=float(target.observation.constraint("soc").value)
    comparisons=[]
    for candidate in [old]+[r["selected"] for r in trained]:
        if candidate is None:
            comparisons.append({"status":"no_valid_GP","proper_nll":None})
            continue
        mu=float(candidate["mean"])
        var=float(candidate["variance"])
        z=(label-mu)/math.sqrt(var)
        comparisons.append({"arm":candidate["arm"],"status":"scored",
            "observed_soc":label,"mean":mu,"variance":var,
            "z_error":z,"proper_exact_NLL":float(-norm.logpdf(label,loc=mu,scale=math.sqrt(var)))})
    save_new(out/"result.json",{
        "scope":"single held-out historically adaptive D28->D29 SOC forecast, not calibration",
        "train_prefix_count":40,"withheld_observation_count":1,
        "original_old30_refit":False,
        "new_VAE_fits":1,"new_GP_candidate_attempts":sum(a["candidate_count"] for a in trained),
        "arm_fit_summaries":trained,"comparison":comparisons,
        "new_model_truth_calls":0,"new_RCAIDE_calls":0,"new_optimizer_calls":0,
        "original_interrupt_zip_unchanged":sha256_stream(archive)==ARCHIVE_SHA})
    if sha256_stream(archive)!=ARCHIVE_SHA:
        raise ValueError("Interrupted archive modified")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","run"))
    parser.add_argument("--archive",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.command=="plan":
        if args.output.exists():
            raise FileExistsError("Output directory exists")
        p=protocol(args.archive)
        args.output.mkdir(parents=True)
        save_new(args.output/"plan.json",p)
        print("Registered: 1 encoder +16 SOC GP candidate fits; zero evaluator calls")
    else:
        run(args.archive,args.output)
        print("Single point forward SOC scoring finished; no BO or evaluator run")

if __name__=="__main__":
    main()
