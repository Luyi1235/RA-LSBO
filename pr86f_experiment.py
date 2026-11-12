"""PR86-F runner, read-only verifier and task-stratified analysis.

Model code is imported lazily and is unchanged. Tests inject an explicitly marked
synthetic backend. CLI execution accepts the frozen native backend only.
"""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time
import traceback
from copy import deepcopy

from pr86f_protocol import BASE_SHA, METHODS, plan
from pr86f_acquisition import choose


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def append(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")
        f.flush()


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seal(root):
    write(root/"manifest.json", {p.relative_to(root).as_posix():hash_file(p)
         for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"})


def check_manifest(root):
    manifest = read(root/"manifest.json")
    paths = [p for p in root.rglob("*") if p.is_file() and p != root/"manifest.json"]
    require(not any(p.is_symlink() for p in root.rglob("*")), "Symlink in evidence")
    require(set(manifest) == {p.relative_to(root).as_posix() for p in paths}, "Evidence roster mismatch")
    for p in paths:
        require(hash_file(p) == manifest[p.relative_to(root).as_posix()], "Evidence hash mismatch")


def numeric_equal(a,b):
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(numeric_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):
        return isinstance(b,list) and len(a)==len(b) and all(numeric_equal(x,y) for x,y in zip(a,b))
    if isinstance(a,bool) or a is None or isinstance(a,str):
        return type(a) is type(b) and a==b
    return isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-11)


def packet(record):
    """Return only observable labels. No learner sees a truth dictionary."""
    obs = record["observed"]
    require(obs["candidate_id"]==record["candidate_id"]==digest(record["candidate"]), "Candidate identity mismatch")
    require(obs["candidate"]==record["candidate"], "Observable candidate mismatch")
    region, resource, objective = (obs[h] for h in ("region","resource","objective"))
    exact = region["kind"]=="exact"
    if exact:
        require(math.isfinite(region["value"]) and region["value"]<=0, "Invalid exact region")
    else:
        require(region==dict(kind="interval",lower=0.0,lower_inclusive=False), "Invalid censored region")
    require(resource["kind"]=="exact" and math.isfinite(resource["value"]), "Invalid resource observation")
    require(objective["kind"]==("exact" if exact else "missing"), "Invalid objective observability")
    if exact:
        require(math.isfinite(objective["value"]), "Nonfinite objective")
    else:
        require("value" not in objective, "Missing objective must not be imputed")
    return obs


def state(observed):
    exact = [r for r in observed if r["objective"]["kind"]=="exact"]
    feasible = [r for r in exact if r["region"]["value"]<=0 and r["resource"]["value"]<=0]
    best = min((r["objective"]["value"] for r in feasible), default=None)
    return dict(n=len(observed), exact_objective_count=len(exact),
        exact_feasible_count=len(feasible), feasible_found=bool(feasible),
        best_exact_feasible=best, objective_trainable=len(exact)>=2,
        incumbent_ids=sorted(r["candidate_id"] for r in feasible if abs(r["objective"]["value"]-best)<=1e-12))


def risk_metrics(tests, probabilities):
    result = {}
    for head in ("region","resource","joint"):
        labels = [bool(r["truth"]["feasible"]) if head=="joint" else r["truth"][head]<=0 for r in tests]
        ps = probabilities[head]
        require(len(ps)==len(labels) and all(math.isfinite(p) and 0<=p<=1 for p in ps), "Bad risk probabilities")
        n, pos = len(labels), sum(labels)
        result[head] = dict(n=n, positives=pos,
            brier=math.fsum((p-y)**2 for p,y in zip(ps,labels))/n,
            false_feasible=sum(p>=.5 and not y for p,y in zip(ps,labels))/(n-pos) if n>pos else None,
            feasible_recall=sum(p>=.5 and y for p,y in zip(ps,labels))/pos if pos else None)
    return result


BILL_KEYS = ("math_started","math_settled","gp_head_started","gp_head_completed",
    "gp_kernel_started","gp_kernel_completed","gp_head_errors","gp_kernel_errors",
    "kernel_candidates_rejected","objective_gp_head_started","objective_gp_head_completed",
    "objective_gp_kernel_started","objective_gp_kernel_completed","objective_gp_head_errors",
    "objective_gp_kernel_errors","objective_gp_kernel_candidates_rejected",
    "aircraft_calls","rcaide_calls","vae_fits","fantasy_fits")


class NativeBackend:
    kind = "frozen_native"

    def candidates(self, seed, count):
        from risk_learning.core import candidates
        return candidates(seed,count)

    def evaluate(self, task, candidate):
        from pr86c_evaluator import truth_for_evaluation
        from risk_learning.core import observed
        target = truth_for_evaluation(task,candidate)
        return target, observed(candidate,target)

    def risk(self, observed, queries, seed, bill):
        from risk_learning.native import predict
        forecast, receipt = predict(observed,queries,"native_censored",seed,bill)
        require(forecast is not None and receipt["status"]=="OK", "Native risk model failed")
        return forecast, receipt

    def objective(self, observed, queries, seed, bill):
        from pr86c_models import predict_objective
        return predict_objective(observed,queries,seed,bill)


def evaluate(backend,root,bill,task,candidate,call,role):
    append(root/"evaluations.jsonl",dict(event="started",call_id=call,role=role,
        task=task,candidate=candidate,candidate_id=digest(candidate)))
    bill["math_started"] += 1
    target, obs = backend.evaluate(task,deepcopy(candidate))
    row = dict(call_id=call,candidate=candidate,candidate_id=digest(candidate),truth=target)
    if role != "test":
        row["observed"] = obs
        packet(row)
    append(root/"evaluations.jsonl",dict(event="settled",call_id=call,record=row,record_sha256=digest(row)))
    bill["math_settled"] += 1
    return row


def assert_bill(bill,spec,objective_fits):
    expected = dict(math_started=spec["math_calls"],math_settled=spec["math_calls"],
        gp_head_started=spec["risk_heads"],gp_head_completed=spec["risk_heads"],
        gp_kernel_started=spec["risk_kernels"],gp_kernel_completed=spec["risk_kernels"],
        objective_gp_head_started=objective_fits,objective_gp_head_completed=objective_fits,
        objective_gp_kernel_started=8*objective_fits,objective_gp_kernel_completed=8*objective_fits)
    require(set(bill)==set(BILL_KEYS), "Unexpected bill fields")
    for k in BILL_KEYS:
        require(bill[k]==expected.get(k,0), "Bill mismatch or error: "+k)


def run(root,profile,source,backend=None):
    root = Path(root).resolve()
    require(not root.exists(), "Output exists; do not overwrite or retry")
    backend = backend or NativeBackend()
    spec, bill = plan(profile), {k:0 for k in BILL_KEYS}
    root.mkdir(parents=True)
    write(root/"plan.json",spec)
    write(root/"source_receipt.json",source)
    write(root/"environment.json",dict(python=platform.python_version(),platform=platform.platform(),backend=backend.kind))
    started, objective_fits, completed = time.perf_counter(),0,0
    try:
        for u in spec["units"]:
            name = "%s/r%d" % (u["task"],u["repeat"])
            anchors = [evaluate(backend,root,bill,u["task"],c,name+"/anchor/%03d"%i,"anchor")
                for i,c in enumerate(backend.candidates(u["anchor_seed"],spec["anchors"]))]
            pool = backend.candidates(u["pool_seed"],spec["candidate_pool"])
            tests = [evaluate(backend,root,bill,u["task"],c,name+"/test/%04d"%i,"test")
                for i,c in enumerate(backend.candidates(u["test_seed"],spec["test_count"]))]
            ids = [r["candidate_id"] for r in anchors+tests]+[digest(c) for c in pool]
            require(len(ids)==len(set(ids)), "Anchor/pool/test overlap")
            trials = []
            for method in METHODS:
                history, available, frames, outcomes = list(anchors),list(pool),[],[]
                for step in range(spec["online_per_arm"]+1):
                    current = [deepcopy(packet(r)) for r in history]
                    status = state(current)
                    seed = u["pool_seed"]+1009*len(current)
                    guess, rr = backend.risk(current,available+[r["candidate"] for r in tests],seed,bill)
                    objective, orr = backend.objective(current,available,seed+4096,bill)
                    require((objective is not None)==(status["exact_objective_count"]>=2)==(orr["status"]=="OK"), "Objective availability mismatch")
                    objective_fits += orr["status"]=="OK"
                    nc = len(available)
                    forecast = dict(probabilities={h:guess["probabilities"][h][:nc] for h in ("region","resource","joint")})
                    frame = dict(n=len(current),state=status,history_sha256=digest(current),
                        available_sha256=digest(available),risk_receipt=rr,objective_receipt=orr)
                    if len(current) in spec["checkpoints"]:
                        ps = {h:guess["probabilities"][h][nc:] for h in ("region","resource","joint")}
                        frame["checkpoint"] = dict(test_ids=[r["candidate_id"] for r in tests],probabilities=ps,metrics=risk_metrics(tests,ps))
                    if step < spec["online_per_arm"]:
                        index, decision = choose(method,available,forecast,objective,status["best_exact_feasible"])
                        decision.update(candidate_forecast=forecast,objective_forecast=objective)
                        frame["decision"] = decision
                    frames.append(frame)
                    # Persist complete pre-outcome fit/prediction evidence, not just completion flags.
                    append(root/"progress.jsonl",dict(unit=name,method=method,step=step,frame=frame))
                    if step==spec["online_per_arm"]:
                        break
                    selected = available.pop(index)
                    row = evaluate(backend,root,bill,u["task"],selected,name+"/online/"+method+"/%03d"%step,"online")
                    outcomes.append(row)
                    history.append(row)
                trials.append(dict(method=method,frames=frames,online=outcomes,final_state=state([packet(r) for r in history])))
                completed += 1
            write(root/"units"/name/"unit.json",dict(task=u["task"],repeat=u["repeat"],anchors=anchors,pool=pool,tests=tests,trials=trials))
        assert_bill(bill,spec,objective_fits)
        require(completed==spec["trajectories"], "Missing trajectories")
        result = dict(status="COMPLETE_PR86F",code_sha=source["code_sha"],backend=backend.kind,
            profile=profile,bill=bill,trajectories=completed,risk_refits=spec["risk_refits"],
            objective_fits=objective_fits,elapsed_seconds=time.perf_counter()-started)
        write(root/"report.json",result)
        seal(root)
        return result
    except BaseException as exc:
        write(root/"failure.json",dict(type=type(exc).__name__,message=str(exc),bill=bill,
            completed_trajectories=completed,unknown_calls=bill["math_started"]-bill["math_settled"],traceback=traceback.format_exc()))
        seal(root)
        raise


def check_fit(current,available,tests,frame):
    rr, orr = frame["risk_receipt"],frame["objective_receipt"]
    payload = [{k:r[k] for k in ("candidate","candidate_id","region","resource")} for r in current]
    require(rr["status"]=="OK" and rr["method"]=="native_censored" and not rr["objective_labels_used"] and not rr["privileged_training"], "Invalid risk fit contract")
    require(rr["training_rows"]==len(current) and rr["training_input_sha256"]==digest(payload), "Risk training prefix mismatch")
    require(rr["query_ids"]==[digest(c) for c in available]+[r["candidate_id"] for r in tests], "Risk query identity mismatch")
    require(set(rr["heads"])=={"region","resource"}, "Risk head roster mismatch")
    for h in rr["heads"].values():
        require(len(h["candidates"])==8 and all(c["status"]=="ok" for c in h["candidates"]), "Risk kernel failure")
    exact = [r for r in current if r["objective"]["kind"]=="exact"]
    require(orr["training_rows"]==len(current) and orr["exact_training_rows"]==len(exact), "Objective label count mismatch")
    require(not orr["privileged_training"] and not orr["missing_labels_imputed"], "Objective labels imputed or privileged")
    require(orr["training_input_sha256"]==digest([(r["candidate_id"],r["objective"]) for r in exact]), "Objective training prefix mismatch")
    require(orr["query_ids"]==[digest(c) for c in available], "Objective query identity mismatch")
    ready = len(exact)>=2
    require(orr["status"]==("OK" if ready else "INSUFFICIENT_EXACT_OBJECTIVES"), "Objective cold state mismatch")
    if ready:
        require(orr["objective_fit_only_exact"] and len(orr["head"]["candidates"])==8 and all(c["status"]=="ok" for c in orr["head"]["candidates"]), "Objective kernel failure")
    return ready


def verify(root,expected_sha,allow_synthetic=False):
    root = Path(root).resolve()
    check_manifest(root)
    require(not (root/"failure.json").exists(), "Failed run cannot be certified")
    spec, source, report = read(root/"plan.json"),read(root/"source_receipt.json"),read(root/"report.json")
    require(spec==plan(spec["profile"]), "Unregistered protocol")
    require(source["code_sha"]==report["code_sha"]==expected_sha and report["status"]=="COMPLETE_PR86F", "Source/status mismatch")
    require(report["backend"]=="frozen_native" or (allow_synthetic and report["backend"]=="synthetic_test"), "Synthetic results are not scientific evidence")
    if report["backend"]=="frozen_native":
        require(source.get("parent_sha")==BASE_SHA and source.get("frozen_models_unchanged") is True, "Frozen model identity missing")
    started, settled = {},{}
    with (root/"evaluations.jsonl").open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line); key = e["call_id"]
            if e["event"]=="started":
                require(key not in started,"Duplicate started call"); started[key]=e
            else:
                require(e["event"]=="settled" and key in started and key not in settled,"Invalid settlement")
                require(e["record_sha256"]==digest(e["record"]),"Settlement hash mismatch"); settled[key]=e
    require(set(started)==set(settled) and len(started)==spec["math_calls"], "Unsettled calls/budget mismatch")
    consumed, expected_progress, objective_fits, frames_count = set(),{},0,0

    def bind(row,call,role,task):
        require(call not in consumed and row["call_id"]==call and call in settled, "Wrong/duplicate paid record")
        consumed.add(call)
        require(started[call]["role"]==role and started[call]["task"]==task and started[call]["candidate"]==row["candidate"], "Ledger start binding mismatch")
        require(settled[call]["record"]==row and row["candidate_id"]==digest(row["candidate"]), "Ledger record binding mismatch")
        if role!="test":
            packet(row)
            from risk_learning.core import observed
            require(row["observed"]==observed(row["candidate"],row["truth"]), "Observed/truth contract mismatch")
        else:
            require("observed" not in row, "Heldout truth is labeled as training")

    for u in spec["units"]:
        name = "%s/r%d"%(u["task"],u["repeat"])
        d = read(root/"units"/name/"unit.json")
        from risk_learning.core import candidates
        require((d["task"],d["repeat"])==(u["task"],u["repeat"]), "Unit mismatch")
        for kind,seed,count in (("anchors","anchor_seed",spec["anchors"]),("tests","test_seed",spec["test_count"])):
            require([r["candidate"] for r in d[kind]]==candidates(u[seed],count), "Frozen sampling stream mismatch")
            for i,row in enumerate(d[kind]):
                role = "anchor" if kind=="anchors" else "test"
                call = name+("/anchor/%03d"%i if role=="anchor" else "/test/%04d"%i)
                bind(row,call,role,u["task"])
        require(d["pool"]==candidates(u["pool_seed"],spec["candidate_pool"]), "Frozen candidate pool mismatch")
        ids = [r["candidate_id"] for r in d["anchors"]+d["tests"]]+[digest(c) for c in d["pool"]]
        require(len(ids)==len(set(ids)), "Candidate overlap")
        require([t["method"] for t in d["trials"]]==list(METHODS), "Missing/duplicated arm")
        for t in d["trials"]:
            available, current = list(d["pool"]),[packet(r) for r in d["anchors"]]
            require(len(t["online"])==spec["online_per_arm"] and len(t["frames"])==spec["online_per_arm"]+1,"Wrong arm budget")
            for step,frame in enumerate(t["frames"]):
                require(frame["n"]==len(current) and frame["history_sha256"]==digest(current) and frame["available_sha256"]==digest(available), "Frame prefix mismatch")
                require(frame["state"]==state(current), "Observable endpoint mismatch")
                ready = check_fit(current,available,d["tests"],frame)
                objective_fits += ready; frames_count += 1
                expected_progress[(name,t["method"],step)] = digest(frame)
                require(("checkpoint" in frame)==(len(current) in spec["checkpoints"]), "Checkpoint roster mismatch")
                if "checkpoint" in frame:
                    cp = frame["checkpoint"]
                    require(cp["test_ids"]==[r["candidate_id"] for r in d["tests"]], "Heldout identity mismatch")
                    require(numeric_equal(cp["metrics"],risk_metrics(d["tests"],cp["probabilities"])), "Risk metric replay mismatch")
                if step==spec["online_per_arm"]:
                    require("decision" not in frame, "Unexpected final decision"); break
                q = frame["decision"]
                require((q["objective_forecast"] is not None)==ready, "Objective posterior availability mismatch")
                if q["objective_forecast"] is not None:
                    obj=q["objective_forecast"]
                    require(set(obj)=={"mean","variance"} and all(len(obj[k])==len(available) for k in obj), "Objective posterior shape mismatch")
                    require(all(math.isfinite(v) for v in obj["mean"]+obj["variance"]) and all(v>=0 for v in obj["variance"]), "Invalid objective posterior")
                index,replayed = choose(t["method"],available,q["candidate_forecast"],q["objective_forecast"],frame["state"]["best_exact_feasible"])
                require(digest(q["scores"])==q["score_digest"], "Stored score hash mismatch")
                for k in replayed:
                    if k!="score_digest":
                        require(numeric_equal(q[k],replayed[k]), "Decision replay mismatch: "+k)
                row = t["online"][step]
                require(row["candidate"]==available.pop(index), "Selected candidate mismatch")
                bind(row,name+"/online/"+t["method"]+"/%03d"%step,"online",u["task"])
                current.append(packet(row))
            require(t["final_state"]==state(current),"Final outcome mismatch")
    require(consumed==set(settled), "Unaccounted paid calls")
    with (root/"progress.jsonl").open(encoding="utf-8") as f:
        for line in f:
            e=json.loads(line); key=(e["unit"],e["method"],e["step"])
            require(key in expected_progress and expected_progress.pop(key)==digest(e["frame"]),"Pre-outcome journal mismatch")
    require(not expected_progress and frames_count==spec["risk_refits"],"Missing fit journal")
    assert_bill(report["bill"],spec,objective_fits)
    require(report["trajectories"]==spec["trajectories"] and report["risk_refits"]==frames_count and report["objective_fits"]==objective_fits,"Report count mismatch")
    return dict(status="VERIFIED_PR86F",backend=report["backend"],trajectories=spec["trajectories"],
        math_calls=spec["math_calls"],decisions=spec["online_calls"],new_evaluator_calls=0,new_gp_fits=0)


def compare(a,b):
    x,y = a["best_exact_feasible"],b["best_exact_feasible"]
    if x is None or y is None:
        return (0 if x is None and y is None else -1 if x is None else 1),None
    delta = x-y
    return (1 if delta < -1e-12 else -1 if delta > 1e-12 else 0),delta


def analyze(root,output,expected_sha,allow_synthetic=False):
    root,output = Path(root).resolve(),Path(output).resolve()
    require(not output.exists(), "Analysis output exists")
    verification = verify(root,expected_sha,allow_synthetic)
    spec, outcomes, pairs, phases = read(root/"plan.json"),[],[],[]
    for u in spec["units"]:
        name="%s/r%d"%(u["task"],u["repeat"])
        d=read(root/"units"/name/"unit.json")
        for t in d["trials"]:
            counts={"no_objective_gp":0,"no_feasible_incumbent":0,"ready":0}
            for frame in t["frames"]:
                s=frame["state"]
                if "decision" in frame:
                    phase="no_objective_gp" if not s["objective_trainable"] else "no_feasible_incumbent" if not s["feasible_found"] else "ready"
                    counts[phase]+=1
                if "checkpoint" in frame:
                    outcomes.append(dict(task=u["task"],repeat=u["repeat"],method=t["method"],n=frame["n"],state=s,risk=frame["checkpoint"]["metrics"]))
            phases.append(dict(task=u["task"],repeat=u["repeat"],method=t["method"],**counts))
        for n in spec["checkpoints"]:
            by={r["method"]:r["state"] for r in outcomes if r["task"]==u["task"] and r["repeat"]==u["repeat"] and r["n"]==n}
            for left,right in ((METHODS[2],METHODS[1]),(METHODS[2],METHODS[0]),(METHODS[1],METHODS[0])):
                c,delta=compare(by[left],by[right])
                pairs.append(dict(task=u["task"],repeat=u["repeat"],n=n,left=left,right=right,comparison=c,
                    objective_delta_if_both_feasible=delta,left_feasible=by[left]["feasible_found"],right_feasible=by[right]["feasible_found"]))
    summary=[]
    for task,n,left,right in sorted({(r["task"],r["n"],r["left"],r["right"]) for r in pairs}):
        rows=[r for r in pairs if (r["task"],r["n"],r["left"],r["right"])==(task,n,left,right)]
        delta=[r["objective_delta_if_both_feasible"] for r in rows if r["objective_delta_if_both_feasible"] is not None]
        summary.append(dict(task=task,n=n,left=left,right=right,pairs=len(rows),wins=sum(r["comparison"]==1 for r in rows),losses=sum(r["comparison"]==-1 for r in rows),ties=sum(r["comparison"]==0 for r in rows),both_feasible=len(delta),mean_delta_within_task=math.fsum(delta)/len(delta) if delta else None))
    output.mkdir(parents=True)
    result=dict(status="COMPLETE_PR86F_ANALYSIS",code_sha=expected_sha,verification=verification,
        input_manifest_sha256=hash_file(root/"manifest.json"),outcomes=outcomes,pairs=pairs,
        task_stratified=summary,phase_counts=phases,
        limits=["Known task family; three repeat units per task, not independent decisions.",
        "Do not pool raw objective deltas across tasks.","Null objectives are not penalties.",
        "Scores are heuristics, not exact VOI or safety guarantees.","No aircraft/VAE/completion-risk validation."])
    write(output/"analysis.json",result)
    objective_rows=[dict(task=r["task"],repeat=r["repeat"],method=r["method"],n=r["n"],
        feasible_found=r["state"]["feasible_found"],best_exact_feasible=r["state"]["best_exact_feasible"],
        exact_objective_count=r["state"]["exact_objective_count"],exact_feasible_count=r["state"]["exact_feasible_count"])
        for r in outcomes]
    risk_rows=[dict(task=r["task"],repeat=r["repeat"],method=r["method"],n=r["n"],head=h,**{k:v for k,v in r["risk"][h].items() if k!="n"})
               for r in outcomes for h in ("region","resource","joint")]
    for filename,rows in (("paired_objectives.csv",pairs),("phase_counts.csv",phases),("task_summary.csv",summary),
                          ("objective_learning_curves.csv",objective_rows),("risk_learning_curves.csv",risk_rows)):
        with (output/filename).open("x",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    seal(output)
    return dict(status=result["status"],outcome_rows=len(outcomes),paired_rows=len(pairs),task_summary_rows=len(summary))
