"""Read-only descriptive cohort audit of the interrupted five-arm aircraft pilot.

The partial pilot is NOT a completed or auditable 80-trial experiment. This
tool may describe only case/seed/method groups with all five arms and the
complete registered 30-call horizon. No extrapolation to unrun trials,
no cross-case final ranking or model evaluations.

Input: original 116MB release ZIP, SHA256 pinned to interrupted record.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

EXPECTED="aa7f7b618eb126bcb0f7fa67c886664f235a7432ad43c269ce833262c05bd2e9"
ARMS=("random","old30","old480","corpus_persistent","onehot")
CASES=("aircraft_ordinary","aircraft_sparse")
SEEDS=tuple(range(9931,9939))
RE=re.compile(r"(?:^|/)optimization/(aircraft_ordinary|aircraft_sparse)/(\d+)_(random|old30|old480|corpus_persistent|onehot)/summary\.json$")


def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def audit(zip_path):
    if file_hash(zip_path)!=EXPECTED:
        raise ValueError("The interrupted pilot evidence ZIP changed")
    with zipfile.ZipFile(zip_path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or len([x for x in names if not x.endswith("/")])!=978:
            raise ValueError("Input archive member list differs")
        result={}
        partial=[]
        for name in names:
            match=RE.search(name)
            if not match:continue
            case,seed,method=match.group(1),int(match.group(2)),match.group(3)
            if seed not in SEEDS:
                # The immutable archive also contains its earlier 9930 smoke,
                # which is not part of the 9931-9938 interrupted pilot cohort.
                continue
            row=json.loads(archive.read(name))
            if ((row["case"],row["seed"],row["method"])!=(case,seed,method)
                    or row["online_calls"]!=30 or row["stopped"]
                    or len(row.get("curve",[]))!=31):
                raise ValueError("Completed trial summary is not at full horizon")
            key=(case,seed,method)
            if key in result:raise ValueError("Duplicate completed trial")
            result[key]=row
        for name in names:
            if not name.endswith("/proposals.jsonl"):continue
            if not "/optimization/" in name:continue
            location=name.split("/optimization/",1)[-1]
            fields=location.split("/")
            if len(fields)!=3 or fields[0] not in CASES:continue
            case=fields[0]
            if not fields[1].split("_",1)[0].isdigit():continue
            seed=int(fields[1].split("_",1)[0])
            method=fields[1][len(str(seed))+1:]
            if method not in ARMS or seed not in SEEDS:continue
            proposals=[json.loads(s) for s in archive.read(name).splitlines() if s.strip()]
            if (case,seed,method) not in result:
                partial.append({"case":case,"seed":seed,"method":method,
                                "saved_proposals":len(proposals)})
        # Distinct full five-arm matched sets, no missing-arm imputation.
        complete_groups={}
        for case in CASES:
            for seed in SEEDS:
                arm_rows=[result.get((case,seed,arm)) for arm in ARMS]
                if all(arm_rows):
                    complete_groups[(case,seed)]={arm:row for arm,row in zip(ARMS,arm_rows)}
        cohorts=[]
        for case in CASES:
            seeds=[seed for c,seed in complete_groups if c==case]
            per_arm={}
            for arm in ARMS:
                rows=[complete_groups[(case,seed)][arm] for seed in seeds]
                best=[r["best_feasible"] for r in rows if r["has_feasible"]]
                verified=sum(r["step_verification"]["verified_usable"] for r in rows)
                per_arm[arm]={
                    "n_completed_paired_seeds":len(seeds),
                    "final_feasible":len(best),
                    "nominal_final_feasible_objective_wh":best,
                    "nominal_final_feasible_mean_wh":sum(best)/len(best) if best else None,
                    "online_usable_total":sum(r["online_usable"] for r in rows),
                    "same_design_5s_verified":verified,
                    "new_gp_fits":0,
                }
            # Pair with onehot, preserve censored feasibility statuses.
            differences={}
            for arm in ARMS:
                if arm=="onehot":continue
                both=[(complete_groups[(case,seed)]["onehot"],
                       complete_groups[(case,seed)][arm]) for seed in seeds]
                shared=[(a["best_feasible"],b["best_feasible"]) for a,b in both
                        if a["has_feasible"] and b["has_feasible"]]
                differences[arm]={
                    "paired_seeds":len(seeds),"both_nominal_feasible":len(shared),
                    "difference_onehot_minus_comparator_mean_wh":
                        (sum(a-b for a,b in shared)/len(shared)) if shared else None,
                    "inference":"Conditional on both feasible; descriptive, no superiority test"}
            cohorts.append({"case":case,"paired_seeds":seeds,
                            "per_arm":per_arm,"paired_onehot_comparison":differences})
        return {
            "schema":"interrupted_corpus_case_complete_readonly_v1",
            "original_archive_sha256":EXPECTED,
            "original_plan": "2026-10-31 five-arm, 8 seeds/case, 30 online",
            "total_completed_trial_summaries":len(result),
            "partial_trials":partial,
            "complete_case_seed_groups":len(complete_groups),
            "cohorts":cohorts,
            "all_trial_count_target":len(CASES)*len(SEEDS)*len(ARMS),
            "completed_full_pilot":False,
            "publication_ranking_permitted":False,
            "new_evaluator_calls":0,"new_GP_or_VAE_fits":0,
            "limits":"Only full matched 5-arm seed groups; unrun trials are missing, not failures or successes. Group order fixed before outcomes, but 140km incomplete. Descriptive and adaptive, no confirmatory causal ranking."
        }


if __name__=="__main__":
    if len(sys.argv)!=3:raise SystemExit("usage: audit_interrupted_pilot_cohorts.py ORIGINAL_ZIP NEW_OUTPUT_DIR")
    path,directory=Path(sys.argv[1]),Path(sys.argv[2])
    if directory.exists():raise FileExistsError("Read-only audit output must be new")
    report=audit(path)
    directory.mkdir(parents=True)
    (directory/"cohort_report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps({"completed_trials":report["total_completed_trial_summaries"],
                     "complete_five_arm_seed_groups":report["complete_case_seed_groups"],
                     "by_case":[(x["case"],x["paired_seeds"]) for x in report["cohorts"]],
                     "partial_trials":report["partial_trials"]},indent=2))
