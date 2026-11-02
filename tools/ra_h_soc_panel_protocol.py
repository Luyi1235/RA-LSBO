"""Frozen RA-LSBO-H common-prefix, cross-policy SOC panel registration.

Pure standard-library plan generation; this module NEVER fits a surrogate,
trains an encoder, launches an optimizer or invokes an aircraft evaluator.

Fixed anchor: hybrid_onehot. For each case/seed at online t=1,9,17, both
latent and hierarchical-onehot SOC surrogates will train on exactly the
same 12+t anchor records. The frozen VAE checkpoint comes from t=0,8,16.
Five same-iteration target policies provide up to 120 held-out observations.
No target label or post-t record is ever an input to either fitted model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

RELEASE_SHA="3d9cb4f684bbc9198c523bb44bf466463c2f85053d87ca1770eca9111b48fed1"
ORIGINAL_PLAN_SHA="ce59583c1fc482406e709225e7ee2496e85b36c3cbc82df9532366f45f239950"
ORIGINAL_REG_SHA="f1d9e8d734ca81d2cb0e8c6ea9ec53fbbb7c118661a076046ea5b44b2dccb271"
ORIGINAL_CODE_SHA="b219163fc565dfc32876f38f0bbf5f4e65abbd36"
CASES=("aircraft_ordinary","aircraft_sparse")
SEEDS=(9961,9962,9963,9964)
TARGET_ARMS=("random","onehot_stable","corpus_stable","hybrid_onehot","hybrid_guard")
ANCHOR="hybrid_onehot"
STEPS=(1,9,17)
MODEL_VIEWS=("hierarchical_onehot","frozen_corpus_latent")
SCHEMA="ra_lsbo_h_soc_commonprefix_crosspolicy_v1"
THIS_FILE=Path(__file__).resolve()
ROOT=THIS_FILE.parents[1]


def hash_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def save_exclusive(path,obj):
    with Path(path).open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(obj,handle,sort_keys=True,indent=2,allow_nan=False)
        handle.write("\n")


class SealedPilot:
    """Read-only wrapper; no paths from ZIP are extracted automatically."""
    def __init__(self,archive):
        self.archive=Path(archive)
        if hash_file(archive)!=RELEASE_SHA:
            raise ValueError("Fixed 856-call pilot archive SHA-256 mismatch")
        self.z=zipfile.ZipFile(archive,"r")
        infos=[x for x in self.z.infolist() if not x.is_dir()]
        if len(infos)!=641 or len({x.filename for x in infos})!=641:
            raise ValueError("Expected exactly 641 unique sealed evidence files")
        for info in infos:
            p=PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename:
                raise ValueError("Unsafe archive member path")
        self.names=set(x.filename for x in infos)
        prefixes=[n[:-len("audit.json")] for n in self.names
                  if n.endswith("audit.json") and
                  all(n[:-len("audit.json")]+q in self.names
                      for q in ("plan.json","registration.json","results.json","calls.jsonl"))]
        if len(prefixes)!=1:
            raise ValueError("Ambiguous fixed Windows pilot root")
        self.prefix=prefixes[0]
        if (hashlib.sha256(self.raw("plan.json")).hexdigest()!=ORIGINAL_PLAN_SHA
            or hashlib.sha256(self.raw("registration.json")).hexdigest()!=ORIGINAL_REG_SHA):
            raise ValueError("Original pilot plan/registration bytes do not match seal")
        self.plan=self.json("plan.json")
        self.audit=self.json("audit.json")
        if (self.plan["source_commit"]!=ORIGINAL_CODE_SHA
            or self.plan["profile"]["seeds"]!=list(SEEDS)
            or self.plan["profile"]["horizon"]!=18
            or tuple(self.plan["arms"])!=TARGET_ARMS
            or tuple(self.plan["cases"])!=CASES
            or self.audit["passed"] is not True
            or self.audit["settled_calls"]!=856
            or self.audit["trial_count"]!=40
            or self.audit["checked_proposals"]!=720):
            raise ValueError("Original pilot registry or audited execution differs")
        self._trial_cache={}

    def close(self):
        self.z.close()

    def __enter__(self):
        return self

    def __exit__(self,*_args):
        self.close()

    def raw(self,relative):
        path=self.prefix+relative
        if path not in self.names:
            raise ValueError("Missing frozen evidence member: "+relative)
        return self.z.read(path)

    def json(self,relative):
        return json.loads(self.raw(relative))

    def trial(self,case,seed,arm):
        key=(case,seed,arm)
        if key not in self._trial_cache:
            root=f"optimization/{case}/{seed}_{arm}/"
            manifest=self.json(root+"manifest.json")
            proposals=[json.loads(x) for x in self.raw(root+"proposals.jsonl").splitlines() if x.strip()]
            observations=[json.loads(x) for x in self.raw(root+"observations.jsonl").splitlines() if x.strip()]
            if ((manifest["case"],manifest["seed"],manifest["method"])!=key
                or len(manifest["initial"])!=12
                or len(proposals)!=18 or len(observations)!=18):
                raise ValueError("Trial identity or 12+18 observation schedule changed: "+str(key))
            for i,(p,o) in enumerate(zip(proposals,observations)):
                if p["iteration"]!=i or p["candidate"]!=o["candidate"]:
                    raise ValueError("Frozen proposal/outcome identity mismatch: "+str(key))
            self._trial_cache[key]=(manifest,proposals,observations)
        return self._trial_cache[key]


def read_only_registration(archive,*,cross_platform_check=False):
    with SealedPilot(archive) as z:
        groups=[]
        seen_total=0
        for case in CASES:
            initial_reference=None
            for seed in SEEDS:
                manifest,proposal,online=z.trial(case,seed,ANCHOR)
                initial=manifest["initial"]
                if initial_reference is None:
                    initial_reference=True
                for other in TARGET_ARMS:
                    if z.trial(case,seed,other)[0]["initial"]!=initial:
                        raise ValueError("Anchor and target arms do not share initial labels")
                for t in STEPS:
                    checkpoint=t-1
                    prefix=initial+online[:t]
                    before=digest(prefix)
                    anchor_proposal=proposal[t]
                    if anchor_proposal["prefix_digest"]!=before:
                        raise ValueError("Anchor history prefix identity has changed")
                    metadata=anchor_proposal["metadata"]
                    corpus=metadata.get("corpus_training")
                    if not isinstance(corpus,dict) or not isinstance(corpus.get("latest"),dict):
                        raise ValueError("Missing persistent VAE event for anchor prefix")
                    latest=corpus["latest"]
                    if (latest["iteration"]!=checkpoint
                        or latest["raw_training_prefix"]!=digest(initial+online[:checkpoint])
                        or latest["checkpoint"]!=f"representation/iteration_{checkpoint:03d}"):
                        raise ValueError("Checkpoint is not trained on registered past-only observations")
                    path=f"optimization/{case}/{seed}_{ANCHOR}/"+latest["checkpoint"]+"/"
                    for n in ("metadata.json","state.npz","diagnostics.json"):
                        if hashlib.sha256(z.raw(path+n)).hexdigest()!=latest["files"][n]:
                            raise ValueError("Sealed checkpoint hash mismatch")
                    if metadata.get("posterior_fit_skipped") is False:
                        if (not metadata.get("feature_identity") or
                            not metadata.get("risk_feature_identity") or
                            not isinstance(metadata.get("forecasts"),dict)):
                            raise ValueError("Expected saved prior-to-evaluation GP identities/forecast")
                    targets=[]
                    training_designs={digest(x["candidate"]) for x in prefix}
                    for arm in TARGET_ARMS:
                        _,p,y=z.trial(case,seed,arm)
                        target=p[t]["candidate"]
                        identity=digest(target)
                        in_prefix=identity in training_designs
                        seen_total+=int(in_prefix)
                        targets.append({
                            "policy":arm,
                            "candidate_id":identity,
                            "frozen_outcome_digest":digest(y[t]),
                            "excluded_as_training_design":in_prefix,
                        })
                    groups.append({
                        "case":case,"seed":seed,
                        "anchor_arm":ANCHOR,"iteration":t,
                        "training_rows":len(prefix),"training_digest":before,
                        "checkpoint_iteration":checkpoint,
                        "checkpoint_relative_path":path,
                        "checkpoint_files_sha256":latest["files"],
                        "saved_anchor_latent_feature_identity":metadata.get("feature_identity"),
                        "saved_anchor_onehot_feature_identity":metadata.get("risk_feature_identity"),
                        "anchor_original_forecast_available":metadata.get("posterior_fit_skipped") is False,
                        "targets":targets,
                    })
        if len(groups)!=24 or sum(len(x["targets"]) for x in groups)!=120:
            raise AssertionError("Fixed 24 prefixes x 5 held-out policies not registered")
        revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        if subprocess.run(["git","merge-base","--is-ancestor",ORIGINAL_CODE_SHA,revision],
                          cwd=ROOT,capture_output=True).returncode:
            raise ValueError("Research code does not descend from frozen original pilot")
        # The Windows model contract can differ in a preview-only Linux CI.
        from ra_lsbo.experiments.simplified_aircraft.transition import problem_for
        contract_match=all(
            problem_for(case).contract_id==
                z.plan["tasks"][case]["contracts"]["20.0"]
            for case in CASES)
        if not contract_match and not cross_platform_check:
            raise ValueError("Aircraft physical contract differs from frozen Windows evaluator")
        return {
            "schema":SCHEMA,
            "source_pilot_zip_sha256":RELEASE_SHA,
            "source_pilot_code_commit":ORIGINAL_CODE_SHA,
            "analysis_code_commit":revision,
            "analysis_source_file_sha256":hash_file(THIS_FILE),
            "contract_match":bool(contract_match),
            "cross_platform_preview_only":bool(cross_platform_check),
            "anchor_arm":ANCHOR,
            "window_iterations":list(STEPS),
            "five_target_policies":list(TARGET_ARMS),
            "views":list(MODEL_VIEWS),
            "model":"original fixed grouped Matern5/2 stable-tail EP",
            "continuous_grid":[.05,.15,.4,1.],
            "discrete_grid":[.5,1.5],
            "ep_max_sweeps":60,"ep_damping":.7,"ep_tolerance":1e-8,
            "no_VAE_train":True,
            "no_aircraft_evaluations":True,
            "no_RCAIDE":True,"no_optimizer":True,
            "number_of_prefix_fits":24,
            "number_of_heads":48,
            "maximum_GP_kernel_candidate_attempts":384,
            "number_of_target_policy_records":120,
            "targets_in_training_prefix_to_exclude":seen_total,
            "primary_scores":["exact_SOC_density_NLL","interval_SOC_mass_NLL"],
            "secondary_scores":["resolved_feasibility_Brier","95pct_exact_coverage"],
            "comparison":"latent-minus-onehot on identical prefix, candidate and truth",
            "inference_limit":"adaptive cross-policy panel, 4 clustered seeds per task; not iid confirmation",
            "groups":groups,
        }
