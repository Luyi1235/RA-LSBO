"""Read-only explanation diagnostics for RA-LSBO-H frozen 40-trial results.

Inspect saved per-proposal dual SOC switching, SOC disagreement, same-seed
candidate divergence and full nominal feasible energy trajectories. This is
post-hoc mechanism description, NOT outcome-dependent model tuning and NOT
evidence for causal calibration or risk guarantees.
"""
from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import statistics
import sys
import zipfile

from analyze_dualview_pilot_sealed import ARCHIVE_SHA,file_hash,save_json,table,CASES,ARMS,SEEDS


def inspect(archive):
    if file_hash(archive)!=ARCHIVE_SHA:
        raise ValueError("Wrong sealed pilot ZIP")
    with zipfile.ZipFile(archive) as z:
        paths=set(z.namelist())
        roots=[p[:-len("results.json")] for p in paths if p.endswith("results.json")
               and p[:-len("results.json")]+"plan.json" in paths
               and p[:-len("results.json")]+"audit.json" in paths]
        if len(roots)!=1:
            raise ValueError("Pilot root ambiguity")
        pre=roots[0]
        read=lambda name:json.loads(z.read(pre+name))
        props={}
        summaries={}
        curves=[]
        guard=[]
        for case in CASES:
            for seed in SEEDS:
                for method in ARMS:
                    root=f"optimization/{case}/{seed}_{method}/"
                    p=[json.loads(x) for x in z.read(pre+root+"proposals.jsonl").splitlines() if x.strip()]
                    if len(p)!=18:
                        raise ValueError("Not the fixed 18-step trajectory")
                    props[(case,seed,method)]=p
                    summary=read(root+"summary.json")
                    summaries[(case,seed,method)]=summary
                    if len(summary["curve"])!=19:raise ValueError("Wrong 0..18 incumbent trajectory")
                    for c in summary["curve"]:
                        curves.append({"case":case,"seed":seed,"method":method,
                            "online_budget":c["online_budget"],
                            "has_feasible":c["has_feasible"],
                            "best_feasible_wh":c["best_feasible"]})
                    if method=="hybrid_guard":
                        switch=0
                        forecasts=0
                        disagreements=[]
                        tail=[]
                        for row in p:
                            m=row["metadata"]
                            f=m.get("forecasts")
                            if f is None:
                                continue
                            so=f["constraints"]["soc"]
                            index=m["selected_index"]
                            select=so.get("selected_latent")
                            disagreement=so.get("view_disagreement_feasibility")
                            if select is None or disagreement is None:
                                raise ValueError("Missing dual-view route or gap on fitted proposal")
                            flag=int(select[index])
                            d=float(disagreement[index])
                            if flag not in (0,1) or not 0<=d<=1 or not math.isfinite(d):
                                raise ValueError("Invalid latent switching or disagreement")
                            switch+=flag
                            forecasts+=1
                            disagreements.append(d)
                            tail.append({"iter":row["iteration"],
                                         "latent_view_chosen":flag,"abs_probability_gap":d})
                        guard.append({"case":case,"seed":seed,
                            "forecasted":forecasts,"latent_view_chosen":switch,
                            "latent_fraction":switch/forecasts if forecasts else None,
                            "probability_disagreement_mean":sum(disagreements)/len(disagreements) if disagreements else None,
                            "probability_disagreement_median":statistics.median(disagreements) if disagreements else None,
                            "probability_disagreement_max":max(disagreements) if disagreements else None,
                            "decision_trace":tail})
        divergences=[]
        for case in CASES:
            for seed in SEEDS:
                pairs=[("hybrid_guard","hybrid_onehot"),("hybrid_onehot","corpus_stable"),
                       ("hybrid_onehot","onehot_stable")]
                for a,b in pairs:
                    left,right=props[(case,seed,a)],props[(case,seed,b)]
                    n_identical=sum(x["candidate"]==y["candidate"] for x,y in zip(left,right))
                    first=next((i for i,(x,y) in enumerate(zip(left,right))
                                if x["candidate"]!=y["candidate"]),None)
                    divergences.append({"case":case,"seed":seed,
                        "left":a,"right":b,"same_selected_candidates":n_identical,
                        "first_different_iteration":first})
        return curves,guard,divergences


if __name__=="__main__":
    if len(sys.argv)!=3:
        raise SystemExit("usage: analyze_dualview_guard_mechanism.py ORIGINAL_RELEASE_ZIP NEW_OUTPUT_DIR")
    archive,out=Path(sys.argv[1]),Path(sys.argv[2])
    if out.exists():raise FileExistsError("New write-once analysis directory required")
    curve,guard,diff=inspect(archive)
    out.mkdir(parents=True)
    save_json(out/"soc_dual_guard_mechanism.json",guard)
    save_json(out/"same_seed_policy_divergence.json",diff)
    table(out/"incumbent_trajectories.csv",curve,list(curve[0]))
    for name in CASES:
        group=[r for r in guard if r["case"]==name]
        n=sum(x["forecasted"] for x in group)
        selected=sum(x["latent_view_chosen"] for x in group)
        print(json.dumps({"case":name,"guard_forecasted":n,
            "guard_latent_selected":selected,"guard_fraction":selected/n if n else None,
            "differences": [r for r in diff if r["case"]==name and r["left"]=="hybrid_guard"
                            and r["right"]=="hybrid_onehot"]},allow_nan=False))
